import argparse
import json
from collections import defaultdict
from src.labels import PII_MAPPING

def calculate_metrics(gold_file, pred_file):
    gold_data = {}
    with open(gold_file, 'r') as f:
        for line in f:
            item = json.loads(line)
            gold_data[item['id']] = item['entities']
            
    pred_data = {}
    with open(pred_file, 'r') as f:
        for line in f:
            item = json.loads(line)
            pred_data[item['id']] = item['entities']
            
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    
    pii_tp = 0
    pii_fp = 0
    pii_fn = 0
    
    for doc_id, gold_ents in gold_data.items():
        pred_ents = pred_data.get(doc_id, [])
        
        # Convert to tuple sets for easy comparison
        # (label, start, end)
        gold_set = set((e['label'], e['start'], e['end']) for e in gold_ents)
        pred_set = set((e['label'], e['start'], e['end']) for e in pred_ents)
        
        for ent in gold_set:
            if ent in pred_set:
                tp[ent[0]] += 1
                if PII_MAPPING.get(ent[0], False):
                    pii_tp += 1
            else:
                fn[ent[0]] += 1
                if PII_MAPPING.get(ent[0], False):
                    pii_fn += 1
                    
        for ent in pred_set:
            if ent not in gold_set:
                fp[ent[0]] += 1
                if PII_MAPPING.get(ent[0], False):
                    pii_fp += 1
                    
    # Calculate per-entity metrics
    print(f"{'Entity':<15} {'Prec':<10} {'Recall':<10} {'F1':<10}")
    print("-" * 45)
    
    all_labels = set(tp.keys()) | set(fp.keys()) | set(fn.keys())
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for label in sorted(all_labels):
        t = tp[label]
        f = fp[label]
        n = fn[label]
        
        total_tp += t
        total_fp += f
        total_fn += n
        
        prec = t / (t + f) if (t + f) > 0 else 0.0
        rec = t / (t + n) if (t + n) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        
        print(f"{label:<15} {prec:<10.4f} {rec:<10.4f} {f1:<10.4f}")
        
    print("-" * 45)
    
    # Overall
    overall_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = 2 * overall_prec * overall_rec / (overall_prec + overall_rec) if (overall_prec + overall_rec) > 0 else 0.0
    
    print(f"{'Overall':<15} {overall_prec:<10.4f} {overall_rec:<10.4f} {overall_f1:<10.4f}")
    
    # PII Metrics
    pii_prec = pii_tp / (pii_tp + pii_fp) if (pii_tp + pii_fp) > 0 else 0.0
    pii_rec = pii_tp / (pii_tp + pii_fn) if (pii_tp + pii_fn) > 0 else 0.0
    pii_f1 = 2 * pii_prec * pii_rec / (pii_prec + pii_rec) if (pii_prec + pii_rec) > 0 else 0.0
    
    print("-" * 45)
    print(f"PII Precision: {pii_prec:.4f}")
    print(f"PII Recall:    {pii_rec:.4f}")
    print(f"PII F1:        {pii_f1:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=str, required=True)
    parser.add_argument("--pred", type=str, required=True)
    args = parser.parse_args()
    
    calculate_metrics(args.gold, args.pred)

if __name__ == "__main__":
    main()
