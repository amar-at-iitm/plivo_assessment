import torch
import torch.nn as nn
from transformers import AutoModelForTokenClassification, AutoConfig

class PIIModel(nn.Module):
    def __init__(self, model_name, num_labels, dropout=0.1):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
        self.config.dropout = dropout
        self.model = AutoModelForTokenClassification.from_pretrained(model_name, config=self.config)

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
