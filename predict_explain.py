"""
predict_hatexplain.py

Runs zero-shot inference using our trained ERM model on HateXplain test set.
Zero-shot means no fine-tuning on HateXplain — we just use the Civil Comments
trained model directly and see how it generalizes.
"""

import torch
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer

MODEL_DIR  = "models/erm"
DATA_PATH  = "data/hatexplain/test.parquet"
OUT_DIR    = Path("outputs/hatexplain")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "distilbert-base-uncased"

print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

def tokenize(example):
    return tokenizer(
        example["comment_text"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )

print("Loading HateXplain test set...")
df = pq.read_table(DATA_PATH).to_pandas()
print(f"  {len(df)} examples  |  toxic: {df['y'].sum()} ({100*df['y'].mean():.1f}%)")

ds = Dataset.from_pandas(df)
ds = ds.map(tokenize, batched=False)
ds.set_format("torch", columns=["input_ids", "attention_mask"])

trainer = Trainer(model=model)
print("Running inference...")
preds  = trainer.predict(ds)
logits = preds.predictions
probs  = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()

out = df.copy()
out["logit0"] = logits[:, 0]
out["logit1"] = logits[:, 1]
out["prob"]   = probs

out_path = OUT_DIR / "test_preds.parquet"
out.to_parquet(out_path, index=False)
print(f"Saved predictions to {out_path}")
print(f"Predicted toxic rate: {(probs >= 0.5).mean():.3f}  |  True toxic rate: {df['y'].mean():.3f}")