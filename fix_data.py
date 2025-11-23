import json
import sys

def fix_file(input_file, output_file):
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            data = json.loads(line)
            # The bug was: current_pos += len(start_filler) + 1
            # But start_filler was appended without space.
            # So all entities are shifted by +1.
            # We need to subtract 1 from start and end.
            
            new_entities = []
            for ent in data['entities']:
                ent['start'] -= 1
                ent['end'] -= 1
                new_entities.append(ent)
            
            data['entities'] = new_entities
            f_out.write(json.dumps(data) + "\n")

if __name__ == "__main__":
    fix_file("data/dev.jsonl", "data/dev_fixed.jsonl")
    fix_file("data/train.jsonl", "data/train_fixed.jsonl")
