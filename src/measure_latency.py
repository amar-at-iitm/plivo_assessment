import argparse
import time
import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification

def measure_latency(model_dir, input_file, runs=50):
    torch.set_num_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    model.eval()
    
    # Dynamic quantization
    model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    
    # Load data
    texts = []
    with open(input_file, 'r') as f:
        for line in f:
            texts.append(json.loads(line)['text'])
            
    # Warmup
    print("Warming up...")
    for _ in range(10):
        _ = tokenizer(texts[0], return_tensors="pt")
        
    latencies = []
    
    print(f"Measuring latency over {runs} runs...")
    
    # We measure per-utterance latency
    # To be realistic, we should pick random utterances or iterate through them
    
    for i in range(runs):
        text = texts[i % len(texts)]
        
        start_time = time.time()
        
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt")
        
        # Inference
        with torch.no_grad():
            _ = model(**inputs)
            
        end_time = time.time()
        latencies.append((end_time - start_time) * 1000) # ms
        
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    
    print(f"p50: {p50:.2f} ms")
    print(f"p95: {p95:.2f} ms")
    
    return p50, p95

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--runs", type=int, default=50)
    args = parser.parse_args()
    
    measure_latency(args.model_dir, args.input, args.runs)

if __name__ == "__main__":
    main()
