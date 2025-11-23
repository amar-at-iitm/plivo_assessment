# PII NER Model Assessment

## Overview
This repository contains a complete pipeline for building, training, and evaluating a token‑level Named Entity Recognition (NER) model that detects Personally Identifiable Information (PII) in noisy Speech‑to‑Text (STT) transcripts.

The solution uses a **DistilBERT** model optimized with **ONNX** to achieve high precision and low latency.

## Key Results
| Metric | Value | Target | Status |
| :--- | :--- | :--- | :--- |
| **PII Precision** | **0.9215** | ≥ 0.80 |  Passed |
| **Latency (p95)** | **11.12 ms** | ≤ 20 ms |  Passed |

## Prerequisites
- Linux environment
- Python 3.12
- Virtual Environment: `/home/amar-kumar/Desktop/MTech/plivo2/plivo/bin/activate`

## Setup & Usage

### 1. Activate Environment
```bash
source /home/amar-kumar/Desktop/MTech/plivo2/plivo/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
pip install onnx onnxruntime onnxscript
```

### 3. Generate Data
```bash
python data_gen.py
```
Generates `data/train.jsonl`, `data/dev.jsonl`, and `data/test.jsonl`.

### 4. Train Model
```bash
PYTHONPATH=. python src/train.py \
    --model_name distilbert-base-uncased \
    --train data/train.jsonl \
    --dev data/dev.jsonl \
    --out_dir out
```

### 5. Export to ONNX (Optimization)
```bash
PYTHONPATH=. python export_onnx.py --model_dir out --output_dir onnx
```

### 6. Evaluate & Measure Latency
**Precision/Recall (ONNX):**
```bash
PYTHONPATH=. python src/predict_onnx.py \
    --model_path onnx/model.onnx \
    --tokenizer_path out \
    --input data/dev.jsonl \
    --output out/dev_pred_onnx.json

PYTHONPATH=. python src/eval_span_f1.py \
    --gold data/dev.jsonl \
    --pred out/dev_pred_onnx.json
```

**Latency (ONNX):**
```bash
PYTHONPATH=. python src/measure_latency_onnx.py \
    --model_path onnx/model.onnx \
    --tokenizer_path out \
    --input data/dev.jsonl \
    --runs 50
```

## Folder Structure
```
plivo_assessment/
├── data/                     # Generated datasets
├── onnx/                     # Optimized ONNX model
├── out/                      # PyTorch model artifacts
├── src/                      # Source code
│   ├── train.py              # Training script
│   ├── predict_onnx.py       # ONNX inference script
│   ├── measure_latency.py    # PyTorch latency measurement
│   ├── measure_latency_onnx.py # ONNX latency measurement
│   └── ...
├── data_gen.py               # Data generation script
├── export_onnx.py            # ONNX export script
└── requirements.txt          # Dependencies
```