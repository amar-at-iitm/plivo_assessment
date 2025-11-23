import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
import os

def export_onnx(model_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model.eval()
    
    dummy_input = tokenizer("This is a test", return_tensors="pt")
    
    torch.onnx.export(
        model, 
        (dummy_input["input_ids"], dummy_input["attention_mask"]), 
        f"{output_dir}/model.onnx",
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "sequence_length"}
        },
        opset_version=11
    )
    print(f"Exported to {output_dir}/model.onnx")

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    
    export_onnx(args.model_dir, args.output_dir)
