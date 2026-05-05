"""
train_reweighted_erm.py

Minimal modification of train_erm.py: adds sample-level class weights to the
cross-entropy loss so that the effective contribution of each (identity, label)
group is balanced. Everything else — tokenizer, model, args — matches ERM.

Weighting scheme: inverse-frequency on the group_id column that your data
pipeline already produces. Each example i gets weight w_i = N / (G * n_{g_i}),
where G is the number of distinct groups and n_g is the count of group g.
This makes the sum of weights equal to N (so the loss magnitude stays
comparable to unweighted ERM) and gives every group equal total mass.

This sits between ERM (flat weights) and Group DRO (adaptive minimax weights).
If it matches DRO on BPSN, the minimax framing isn't buying you anything.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pyarrow.parquet as pq
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import roc_auc_score

# ─── Load data ───────────────────────────────────────────────────────────────
train_df = pq.read_table("data/processed/train.parquet").to_pandas()
val_df = pq.read_table("data/processed/val.parquet").to_pandas()

# ─── Compute per-example weights from group_id ───────────────────────────────
N = len(train_df)
group_counts = train_df["group_id"].value_counts().to_dict()
G = len(group_counts)
print(f"Reweighting: N={N}, G={G} groups")
print(f"  smallest group: {min(group_counts.values())}, largest: {max(group_counts.values())}")

# w_i = N / (G * n_{g_i}) — sums to N, gives each group equal total mass
train_df["sample_weight"] = train_df["group_id"].map(
    lambda g: N / (G * group_counts[g])
).astype(np.float32)

# Clip extreme weights. Groups with n=1 would otherwise get weight ~4100, meaning
# a single training example contributes as much gradient as 4100 average examples
# — that's gradient poison. 50x cap keeps tail groups meaningfully upweighted
# without letting individual outliers dominate training.
WEIGHT_CAP = 50.0
n_clipped = (train_df["sample_weight"] > WEIGHT_CAP).sum()
print(f"  clipping {n_clipped} examples with weight > {WEIGHT_CAP}")
train_df["sample_weight"] = train_df["sample_weight"].clip(upper=WEIGHT_CAP).astype(np.float32)

# sanity check: after clipping, per-group totals will no longer be perfectly
# equal (tiny groups will have less mass than the cap allows), but the spread
# should be bounded.
total_mass_per_group = train_df.groupby("group_id")["sample_weight"].sum()
print(f"  per-group total weight after clipping (target was N/G = {N/G:.1f}):")
print(f"    min={total_mass_per_group.min():.1f}, max={total_mass_per_group.max():.1f}")
print(f"    number of groups below target: {(total_mass_per_group < N/G - 1).sum()}/{G}")

# val set doesn't need weights for eval AUC, but we carry group_id along
val_df["sample_weight"] = 1.0

# ─── HF Dataset ──────────────────────────────────────────────────────────────
train_ds = Dataset.from_pandas(train_df)
val_ds = Dataset.from_pandas(val_df)

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

# IMPORTANT: include sample_weight in the torch format so it reaches the Trainer
train_ds.set_format(
    "torch", columns=["input_ids", "attention_mask", "labels", "sample_weight"]
)
val_ds.set_format(
    "torch", columns=["input_ids", "attention_mask", "labels", "sample_weight"]
)

# ─── Model ───────────────────────────────────────────────────────────────────
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)


# ─── Custom Trainer that uses sample_weight in the loss ──────────────────────
class ReweightedTrainer(Trainer):
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        # pull off our extra column before forward
        weights = inputs.pop("sample_weight")
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        # per-example cross-entropy, then weighted mean
        loss_fct = nn.CrossEntropyLoss(reduction="none")
        per_example_loss = loss_fct(logits.view(-1, 2), labels.view(-1))
        loss = (weights * per_example_loss).sum() / weights.sum()
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
    auc = roc_auc_score(labels, probs)
    return {"auc": auc}


# ─── Training args (identical to ERM except output dir) ──────────────────────
training_args = TrainingArguments(
    output_dir="./models/reweighted_erm",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=2,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=200,
    load_best_model_at_end=True,
    # critical: don't let HF drop our sample_weight column
    remove_unused_columns=False,
)

trainer = ReweightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model("./models/reweighted_erm")
print("Done. Model saved to ./models/reweighted_erm")
print("Next: copy predict_and_save.py, change MODEL_DIR to 'models/reweighted_erm'")
print("      and OUT_DIR to 'outputs/reweighted_erm', then run it.")