import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

IN_CSV = Path("data/jigsaw/train.csv")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- settings ----------
LABEL_THRESHOLD = 0.5
IDENTITY_THRESHOLD = 0.5
DEV_N = 200_000         # start with 200k for first runs (change later)
RANDOM_STATE = 42

# Identities to use (you can expand later; these are common + interpretable)
IDENTITIES = [
    "asian","black","latino","white",
    "male","female","other_gender",
    "christian","muslim","jewish","hindu","buddhist","atheist","other_religion",
    "bisexual","heterosexual","homosexual_gay_or_lesbian","other_sexual_orientation",
    "psychiatric_or_mental_illness","intellectual_or_learning_disability",
    "physical_disability","other_disability",
    "other_race_or_ethnicity",
]

def main():
    df = pd.read_csv(IN_CSV, usecols=["comment_text","target"] + [c for c in IDENTITIES if c in pd.read_csv(IN_CSV, nrows=1).columns])
    # NOTE: above line reads header twice; simplest/robust. You can optimize later.

    # clean text
    df = df.dropna(subset=["comment_text"]).copy()
    df["comment_text"] = df["comment_text"].astype(str)

    # binary label
    df["y"] = (df["target"] >= LABEL_THRESHOLD).astype(int)

    # keep only identity columns that exist
    id_cols = [c for c in IDENTITIES if c in df.columns]
    print("Using identities:", id_cols)

    # identity mention flags
    for c in id_cols:
        df[f"{c}_flag"] = (df[c] >= IDENTITY_THRESHOLD).astype(int)

    # downsample for dev (stratified by y)
    if DEV_N is not None and DEV_N < len(df):
        df, _ = train_test_split(df, train_size=DEV_N, stratify=df["y"], random_state=RANDOM_STATE)

    # split train/val/test (80/10/10), stratified by y
    train_df, temp_df = train_test_split(df, train_size=0.8, stratify=df["y"], random_state=RANDOM_STATE)
    val_df, test_df = train_test_split(temp_df, train_size=0.5, stratify=temp_df["y"], random_state=RANDOM_STATE)

    # group_id for Group DRO: (identity, y) + ("none", y)
    # We assign each example to the FIRST identity mentioned (simple + stable). Later you can do multi-group.
    group_map = {}
    gid = 0
    for ident in id_cols:
        for y in [0, 1]:
            group_map[(ident, y)] = gid
            gid += 1
    for y in [0, 1]:
        group_map[("none", y)] = gid
        gid += 1

    def assign_group(row):
        for ident in id_cols:
            if row[f"{ident}_flag"] == 1:
                return group_map[(ident, row["y"])]
        return group_map[("none", row["y"])]

    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        split_df = split_df.copy()
        split_df["group_id"] = split_df.apply(assign_group, axis=1)

        keep_cols = ["comment_text","y","group_id"] + [f"{c}_flag" for c in id_cols]
        out_path = OUT_DIR / f"{split_name}.parquet"
        split_df[keep_cols].to_parquet(out_path, index=False)
        print(f"{split_name}: {len(split_df)} -> {out_path}")

    # save metadata for later reporting
    meta = {
        "label_threshold": LABEL_THRESHOLD,
        "identity_threshold": IDENTITY_THRESHOLD,
        "dev_n": DEV_N,
        "identities": id_cols,
        "num_groups": gid,
    }
    (OUT_DIR / "meta.json").write_text(pd.Series(meta).to_json())

if __name__ == "__main__":
    main()