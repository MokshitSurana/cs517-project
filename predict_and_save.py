import numpy as np
import pandas as pd
import torch
import pyarrow.parquet as pq
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer

MODEL_DIR = "models/erm"
OUT_DIR = Path("outputs/erm")
OUT_DIR.mkdir(parents=True, exist_ok=True)

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

def tokenize(example):
    return tokenizer(example["comment_text"], truncation=True, padding="max_length", max_length=128)

def run(split):
    df = pq.read_table(f"data/processed/{split}.parquet").to_pandas()
    ds = Dataset.from_pandas(df)
    ds = ds.map(tokenize, batched=False)
    ds.set_format("torch", columns=["input_ids", "attention_mask"])

    trainer = Trainer(model=model)

    preds = trainer.predict(ds)
    logits = preds.predictions
    probs = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()

    out = df.copy()
    out["logit0"] = logits[:, 0]
    out["logit1"] = logits[:, 1]
    out["prob"] = probs

    out_path = OUT_DIR / f"{split}_preds.parquet"
    out.to_parquet(out_path, index=False)
    print(f"Wrote {out_path} with cols: {list(out.columns)[:10]} ...")

for split in ["val", "test"]:
    run(split)