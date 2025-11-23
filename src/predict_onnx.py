import argparse
import json
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from src.labels import ID2LABEL, PII_MAPPING
import re

def get_regex_spans(text):
    spans = []
    # Email: "at", "dot"
    email_pat = r'\b[\w\.-]+(?:\s*@\s*|\s+at\s+)[\w\.-]+(?:\s*\.\s*|\s+dot\s+)[\w]{2,}\b'
    for m in re.finditer(email_pat, text, re.IGNORECASE):
        spans.append({"start": m.start(), "end": m.end(), "label": "EMAIL"})
        
    # Date: YYYY-MM-DD
    date_pat = r'\b\d{4}-\d{2}-\d{2}\b'
    for m in re.finditer(date_pat, text):
        spans.append({"start": m.start(), "end": m.end(), "label": "DATE"})
        
    return spans

def expand_entity(text, entity, regex_spans):
    ent_range = set(range(entity['start'], entity['end']))
    for r_span in regex_spans:
        if r_span['label'] == entity['label']:
            r_range = set(range(r_span['start'], r_span['end']))
            if not ent_range.isdisjoint(r_range):
                entity['start'] = r_span['start']
                entity['end'] = r_span['end']
                return entity
    return entity

def predict_onnx(session, tokenizer, text, max_len=128):
    # Tokenize
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors="np"
    )
    
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    offset_mapping = encoding["offset_mapping"][0]
    
    ort_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }
    
    # Inference
    ort_outs = session.run(None, ort_inputs)
    logits = ort_outs[0]
    
    predictions = np.argmax(logits, axis=2)[0]
    
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
                "start": int(start),
                "end": int(end)
            }
        elif prefix == "I":
            if current_entity and current_entity["label"] == label_type:
                current_entity["end"] = int(end)
            else:
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    "label": label_type,
                    "start": int(start),
                    "end": int(end)
                }
    
    if current_entity:
        entities.append(current_entity)
        
    # Post-processing
    regex_spans = get_regex_spans(text)
    
    final_entities = []
    for ent in entities:
        ent = expand_entity(text, ent, regex_spans)
        ent["pii"] = PII_MAPPING.get(ent["label"], False)
        final_entities.append(ent)
        
    return final_entities

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    session = ort.InferenceSession(args.model_path, providers=["CPUExecutionProvider"])
    
    predictions = []
    with open(args.input, 'r') as f:
        for line in f:
            data = json.loads(line)
            text = data["text"]
            entities = predict_onnx(session, tokenizer, text)
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
