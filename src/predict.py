import argparse
import json
import torch
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification
from src.labels import ID2LABEL, PII_MAPPING

def get_regex_spans(text):
    spans = []
    # Simple patterns for noisy data
    # Credit Card: 13-19 digits, maybe spaces/dashes
    # We also need to handle "one two" etc if we want to be perfect, but let's stick to digits for now as per data_gen
    # Actually data_gen uses "one", "two". My regex won't catch those easily without a huge map.
    # But the model should catch them.
    # Let's focus on boundaries.
    
    # Email: "at", "dot"
    email_pat = r'\b[\w\.-]+(?:\s*@\s*|\s+at\s+)[\w\.-]+(?:\s*\.\s*|\s+dot\s+)[\w]{2,}\b'
    for m in re.finditer(email_pat, text, re.IGNORECASE):
        spans.append({"start": m.start(), "end": m.end(), "label": "EMAIL"})
        
    # Phone: 7-15 digits/words
    # This is hard with words.
    
    # Date: YYYY-MM-DD
    date_pat = r'\b\d{4}-\d{2}-\d{2}\b'
    for m in re.finditer(date_pat, text):
        spans.append({"start": m.start(), "end": m.end(), "label": "DATE"})
        
    return spans

def expand_entity(text, entity, regex_spans):
    # If entity overlaps with a regex span of same type, take the regex span
    ent_range = set(range(entity['start'], entity['end']))
    
    for r_span in regex_spans:
        if r_span['label'] == entity['label']:
            r_range = set(range(r_span['start'], r_span['end']))
            if not ent_range.isdisjoint(r_range):
                # Overlap found, use regex span
                entity['start'] = r_span['start']
                entity['end'] = r_span['end']
                return entity
                
    return entity

def predict(model, tokenizer, text, max_len=128):
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors="pt"
    )
    
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    offset_mapping = encoding["offset_mapping"][0].tolist()
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
    predictions = torch.argmax(logits, dim=2)[0].tolist()
    
    # Decode spans
    entities = []
    current_entity = None
    
    for idx, (pred, (start, end)) in enumerate(zip(predictions, offset_mapping)):
        if start == end: continue # Special token
        
        label_str = ID2LABEL[pred]
        
        if label_str == "O":
            if current_entity:
                entities.append(current_entity)
                current_entity = None
            continue
            
        prefix, label_type = label_str.split("-", 1)
        
        if prefix == "B":
            if current_entity:
                entities.append(current_entity)
            current_entity = {
                "label": label_type,
                "start": start,
                "end": end
            }
        elif prefix == "I":
            if current_entity and current_entity["label"] == label_type:
                current_entity["end"] = end
            else:
                # Treat as B if mismatch or no previous B
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    "label": label_type,
                    "start": start,
                    "end": end
                }
    
    if current_entity:
        entities.append(current_entity)
        
    # Post-processing
    regex_spans = get_regex_spans(text)
    
    final_entities = []
    for ent in entities:
        # Try to expand
        ent = expand_entity(text, ent, regex_spans)
        
        ent["pii"] = PII_MAPPING.get(ent["label"], False)
        final_entities.append(ent)
        
    return final_entities

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    
    torch.set_num_threads(1)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    model.eval()
    
    # Quantize
    model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    
    predictions = []
    with open(args.input, 'r') as f:
        for line in f:
            data = json.loads(line)
            text = data["text"]
            entities = predict(model, tokenizer, text)
            predictions.append({
                "id": data["id"],
                "text": text,
                "entities": entities
            })
            
    with open(args.output, 'w') as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")

if __name__ == "__main__":
    main()
