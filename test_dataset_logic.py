import unittest

# Copy of LABEL2ID from src/labels.py
LABELS = [
    "O",
    "B-CREDIT_CARD", "I-CREDIT_CARD",
    "B-PHONE", "I-PHONE",
    "B-EMAIL", "I-EMAIL",
    "B-PERSON_NAME", "I-PERSON_NAME",
    "B-DATE", "I-DATE",
    "B-CITY", "I-CITY",
    "B-LOCATION", "I-LOCATION"
]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}

# Mocking the tokenizer and dataset logic for testing
class MockTokenizer:
    def __call__(self, text, return_offsets_mapping=True, max_length=128, padding='max_length', truncation=True):
        # Simple whitespace tokenizer for testing
        tokens = text.split()
        # Mock input_ids
        input_ids = [101] + [hash(t) % 1000 for t in tokens] + [102] # CLS + tokens + SEP
        
        offsets = []
        current_pos = 0
        
        # CLS
        offsets.append((0, 0))
        
        for t in tokens:
            start = text.find(t, current_pos)
            end = start + len(t)
            offsets.append((start, end))
            current_pos = end
            
        # SEP
        offsets.append((0, 0))
        
        # Padding
        while len(input_ids) < max_length:
            input_ids.append(0)
            offsets.append((0, 0))
            
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(tokens) + [0] * (max_length - len(tokens)), # Simplified
            "offset_mapping": offsets
        }

class TestDatasetLogic(unittest.TestCase):
    def test_alignment(self):
        text = "my credit card is one two three"
        # "one two three" -> 18-31
        entities = [{"start": 18, "end": 31, "label": "CREDIT_CARD"}]
        
        # Tokenizer mock
        # tokens: ["my", "credit", "card", "is", "one", "two", "three"]
        # offsets: (0,2), (3,9), (10,14), (15,17), (18,21), (22,25), (26,31)
        
        tokenizer = MockTokenizer()
        encoding = tokenizer(text)
        offset_mapping = encoding["offset_mapping"]
        
        labels = [LABEL2ID["O"]] * len(encoding["input_ids"])
        
        char_labels = ["O"] * len(text)
        for ent in entities:
            label = ent['label']
            start = ent['start']
            end = ent['end']
            if start < len(text) and end <= len(text):
                char_labels[start] = f"B-{label}"
                for i in range(start + 1, end):
                    char_labels[i] = f"I-{label}"
                    
        for i, (start, end) in enumerate(offset_mapping):
            if start == end: 
                labels[i] = -100
                continue
            
            token_label = char_labels[start]
            if token_label.startswith("B-") or token_label.startswith("I-"):
                labels[i] = LABEL2ID[token_label]
            else:
                labels[i] = LABEL2ID["O"]
                
        # Check "one" (18-21) -> Should be B-CREDIT_CARD
        # Check "two" (22-25) -> Should be I-CREDIT_CARD
        # Check "three" (26-31) -> Should be I-CREDIT_CARD
        
        # Indices in input_ids: 0=CLS, 1=my, 2=credit, 3=card, 4=is, 5=one, 6=two, 7=three
        
        self.assertEqual(labels[5], LABEL2ID["B-CREDIT_CARD"])
        self.assertEqual(labels[6], LABEL2ID["I-CREDIT_CARD"])
        self.assertEqual(labels[7], LABEL2ID["I-CREDIT_CARD"])
        
        print("Test passed!")

if __name__ == "__main__":
    unittest.main()
