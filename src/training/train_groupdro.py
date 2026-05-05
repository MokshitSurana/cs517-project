import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import roc_auc_score

MODEL_NAME = "distilbert-base-uncased"
TRAIN_PATH = "data/processed/train.parquet"
VAL_PATH   = "data/processed/val.parquet"
OUT_DIR = "./models/groupdro_v2"

# ---------- load ----------
train_df = pq.read_table(TRAIN_PATH).to_pandas()
val_df   = pq.read_table(VAL_PATH).to_pandas()

num_groups = int(train_df["group_id"].max()) + 1
print("num_groups:", num_groups)

train_ds = Dataset.from_pandas(train_df)
val_ds   = Dataset.from_pandas(val_df)

# ---------- tokenize ----------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(example):
    return tokenizer(
        example["comment_text"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )

train_ds = train_ds.map(tokenize, batched=False)
val_ds   = val_ds.map(tokenize, batched=False)

train_ds = train_ds.rename_column("y", "labels")
val_ds   = val_ds.rename_column("y", "labels")

# Keep group_id in the format — Trainer will pass it through to compute_loss
train_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels", "group_id"])
val_ds.set_format("torch",   columns=["input_ids", "attention_mask", "labels", "group_id"])

# ---------- model ----------
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# ---------- custom trainer ----------
class GroupDROTrainer(Trainer):
    def __init__(self, *args, num_groups=1, group_step_size=0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_groups = num_groups
        self.group_step_size = group_step_size
        self.registered_group_weights = torch.ones(num_groups) / num_groups

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels    = inputs.pop("labels")
        group_ids = inputs.pop("group_id")   # now safe — format keeps it

        outputs = model(**inputs)
        logits  = outputs.logits

        losses   = F.cross_entropy(logits, labels, reduction="none")
        device   = losses.device
        group_ids = group_ids.to(device)

        group_losses = []
        valid_groups = []
        for g in range(self.num_groups):
            mask = (group_ids == g)
            if mask.any():
                group_losses.append(losses[mask].mean())
                valid_groups.append(g)

        if len(group_losses) == 0:
            loss = losses.mean()
            return (loss, outputs) if return_outputs else loss

        group_losses_t = torch.stack(group_losses)

        group_weights = self.registered_group_weights.to(device).clone()
        for idx, g in enumerate(valid_groups):
            group_weights[g] = group_weights[g] * torch.exp(
                self.group_step_size * group_losses_t[idx].detach()
            )
        group_weights = group_weights / group_weights.sum()
        self.registered_group_weights = group_weights.detach().cpu()

        weighted_loss = sum(
            group_weights[g] * group_losses_t[idx]
            for idx, g in enumerate(valid_groups)
        )

        return (weighted_loss, outputs) if return_outputs else weighted_loss

# ---------- metrics ----------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
    return {"auc": roc_auc_score(labels, probs)}

# ---------- training args ----------
training_args = TrainingArguments(
    output_dir=OUT_DIR,
    max_grad_norm=1.0,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=2,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=200,
    load_best_model_at_end=True,
    report_to="none",
    remove_unused_columns=False,   # ← THE KEY FIX: don't strip group_id
)

trainer = GroupDROTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
    num_groups=num_groups,
    group_step_size=0.001,
)

trainer.train()
trainer.save_model(OUT_DIR)

print("\nFinal learned group weights:")
print(trainer.registered_group_weights)