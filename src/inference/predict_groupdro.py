import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
import torch

MODEL_DIR = "models/groupdro"
OUT_DIR = Path("outputs/groupdro")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

def tokenize(example):
    return tokenizer(
        example["comment_text"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )

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
    out.to_parquet(OUT_DIR / f"{split}_preds.parquet", index=False)
    print(f"saved {split}")

for split in ["val", "test"]:
    run(split)