import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import roc_auc_score
import pyarrow.parquet as pq
import numpy as np

# -------- Load Data --------
train_df = pq.read_table("data/processed/train.parquet").to_pandas()
val_df = pq.read_table("data/processed/val.parquet").to_pandas()

# Convert to HuggingFace Dataset
train_ds = Dataset.from_pandas(train_df)
val_ds = Dataset.from_pandas(val_df)

# -------- Tokenizer --------
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(example):
    return tokenizer(
        example["comment_text"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )

train_ds = train_ds.map(tokenize, batched=False)
val_ds = val_ds.map(tokenize, batched=False)

train_ds = train_ds.rename_column("y", "labels")
val_ds = val_ds.rename_column("y", "labels")

train_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
val_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# -------- Model --------
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)

# -------- Metrics --------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
    auc = roc_auc_score(labels, probs)
    return {"auc": auc}

# -------- Training Args --------
training_args = TrainingArguments(
    output_dir="./models/erm",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=2,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=200,
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.save_model("./models/erm")