import argparse
import os
import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments, DataCollatorForTokenClassification
from src.dataset import PIIDataset
from src.model import PIIModel
from src.labels import LABELS, LABEL2ID, ID2LABEL
import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [ID2LABEL[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [ID2LABEL[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
    }
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--dev", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="out")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    train_dataset = PIIDataset(args.train, tokenizer)
    dev_dataset = PIIDataset(args.dev, tokenizer)
    
    model = PIIModel(args.model_name, num_labels=len(LABELS))
    
    data_collator = DataCollatorForTokenClassification(tokenizer)
    
    training_args = TrainingArguments(
        output_dir=args.out_dir,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=50,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        save_total_limit=2,
        logging_dir=f"{args.out_dir}/logs",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="f1"
    )
    
    trainer = Trainer(
        model=model.model, # Pass the inner HF model
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    trainer.train()
    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)

if __name__ == "__main__":
    main()
