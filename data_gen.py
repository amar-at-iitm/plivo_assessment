import json
import random
from faker import Faker
import os

fake = Faker()

# Entity types
ENTITIES = [
    "CREDIT_CARD",
    "PHONE",
    "EMAIL",
    "PERSON_NAME",
    "DATE",
    "CITY",
    "LOCATION"
]

def noisy_number(number_str):
    # Convert digits to words sometimes
    digit_map = {
        '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
        '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'
    }
    res = []
    for char in number_str:
        if char.isdigit() and random.random() < 0.3:
            res.append(digit_map[char])
        else:
            res.append(char)
    return "".join(res)

def noisy_email(email):
    # Replace @ with " at ", . with " dot "
    email = email.replace("@", " at ").replace(".", " dot ")
    return email

def generate_example(id_num):
    text_content = ""
    entities = []
    
    # Construct a sentence with 1-3 entities
    num_entities = random.randint(1, 3)
    
    # Add some filler text at start
    start_filler = fake.sentence(nb_words=random.randint(2, 5)).lower().replace(".", "")
    text_content += start_filler
    
    for _ in range(num_entities):
        entity_type = random.choice(ENTITIES)
        value = ""
        
        if entity_type == "CREDIT_CARD":
            value = fake.credit_card_number()
            value = noisy_number(value)
        elif entity_type == "PHONE":
            value = fake.phone_number()
            value = noisy_number(value)
        elif entity_type == "EMAIL":
            value = fake.email()
            value = noisy_email(value)
        elif entity_type == "PERSON_NAME":
            value = fake.name().lower()
        elif entity_type == "DATE":
            value = fake.date()
            # Make date noisy? e.g. "january first"
            if random.random() < 0.5:
                try:
                    d = fake.date_object()
                    value = d.strftime("%B %d %Y").lower()
                except:
                    pass
        elif entity_type == "CITY":
            value = fake.city().lower()
        elif entity_type == "LOCATION":
            value = fake.address().lower()
            
        # Clean value (remove punctuation often found in fake data)
        value = value.replace("\n", " ").strip()
        
        # Add separator/filler before entity
        prefix = random.choice([" and ", " ", " also ", " my ", " is ", " , "])
        # simple STT often lacks punctuation, but let's keep it simple
        prefix = prefix.replace(",", "") 
        
        text_content += prefix
        
        start = len(text_content)
        text_content += value
        end = len(text_content)
        
        entities.append({
            "start": start,
            "end": end,
            "label": entity_type
        })
        
    # Add some filler text at end
    end_filler = fake.sentence(nb_words=random.randint(1, 4)).lower().replace(".", "")
    text_content += " " + end_filler
    
    return {
        "id": f"utt_{id_num:04d}",
        "text": text_content,
        "entities": entities
    }

def generate_dataset(num_examples, output_file):
    with open(output_file, 'w') as f:
        for i in range(num_examples):
            example = generate_example(i)
            f.write(json.dumps(example) + "\n")

if __name__ == "__main__":
    if not os.path.exists("data"):
        os.makedirs("data")
    
    print("Generating train.jsonl...")
    generate_dataset(1000, "data/train.jsonl")
    
    print("Generating dev.jsonl...")
    generate_dataset(200, "data/dev.jsonl")
    
    print("Generating test.jsonl...")
    generate_dataset(50, "data/test.jsonl")
    print("Done.")
