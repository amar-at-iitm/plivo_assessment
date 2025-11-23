import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from src.labels import LABEL2ID

class PIIDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len=128):
        self.data = []
        with open(file_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        entities = item['entities']

        # Tokenize
        encoding = self.tokenizer(
            text,
            return_offsets_mapping=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True
        )
        
        labels = [LABEL2ID["O"]] * len(encoding["input_ids"])
        offset_mapping = encoding["offset_mapping"]

        # Align labels
        # Create a character-level label map first
        char_labels = ["O"] * len(text)
        for ent in entities:
            label = ent['label']
            start = ent['start']
            end = ent['end']
            # Basic check to ensure indices are within bounds
            if start < len(text) and end <= len(text):
                char_labels[start] = f"B-{label}"
                for i in range(start + 1, end):
                    char_labels[i] = f"I-{label}"

        # Map character labels to token labels
        for i, (start, end) in enumerate(offset_mapping):
            if start == end: # Special tokens
                labels[i] = -100
                continue
            
            # If the token spans multiple characters, we take the label of the first character
            # This is a simplification; for more robust handling we might check overlap
            token_label = char_labels[start]
            
            # If it's inside an entity but the previous token was not the same entity, it should be B-
            # But here we just map directly from char_labels which already has B/I
            
            # However, subword tokens should be I- if the previous was B- or I- of the same type
            # The standard way is: First token of word gets B-, subsequent get I- (if word is split)
            # But here we are dealing with character offsets directly.
            
            # Let's refine:
            # If the character range [start, end) overlaps with an entity, assign label.
            # We use the label of the start character.
            
            if token_label.startswith("B-") or token_label.startswith("I-"):
                # Check if this is a continuation of the previous token's entity
                # For simplicity in this starter, we just trust the char_labels logic
                # But we need to handle the case where a B- starts in the middle of a token? 
                # Unlikely with standard tokenizers unless noisy text is weird.
                
                # If the token starts with I-, but it's the first token of the entity in the sequence (e.g. previous was O),
                # it should probably be B-?
                # Actually, if char_labels[start] is I-, it means the entity started earlier. 
                # If the previous token covered the B-, then this is correct.
                # If the B- was skipped (e.g. truncation), then this might be an issue, but we ignore truncation for now.
                
                labels[i] = LABEL2ID[token_label]
            else:
                labels[i] = LABEL2ID["O"]

        return {
            "input_ids": torch.tensor(encoding["input_ids"]),
            "attention_mask": torch.tensor(encoding["attention_mask"]),
            "labels": torch.tensor(labels)
        }
