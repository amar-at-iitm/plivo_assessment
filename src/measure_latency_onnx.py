import argparse
import time
import json
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

def measure_latency_onnx(model_path, input_file, tokenizer_path, runs=50):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    
    # Load data
    texts = []
    with open(input_file, 'r') as f:
        for line in f:
            texts.append(json.loads(line)['text'])
            
    # Warmup
    print("Warming up...")
    inputs = tokenizer(texts[0], return_tensors="np")
    ort_inputs = {k: v for k, v in inputs.items()}
    _ = session.run(None, ort_inputs)
        
    latencies = []
    
    print(f"Measuring latency over {runs} runs...")
    
    for i in range(runs):
        text = texts[i % len(texts)]
        
        start_time = time.time()
        
        # Tokenize (include in latency as it's part of pipeline)
        inputs = tokenizer(text, return_tensors="np")
        ort_inputs = {k: v for k, v in inputs.items()}
        
        # Inference
        _ = session.run(None, ort_inputs)
            
        end_time = time.time()
        latencies.append((end_time - start_time) * 1000) # ms
        
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    
    print(f"p50: {p50:.2f} ms")
    print(f"p95: {p95:.2f} ms")
    
    return p50, p95

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--runs", type=int, default=50)
    args = parser.parse_args()
    
    measure_latency_onnx(args.model_path, args.input, args.tokenizer_path, args.runs)

if __name__ == "__main__":
    main()
