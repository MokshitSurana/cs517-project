"""
preprocess_hatexplain.py - loads from GitHub JSON directly
"""

import json
import urllib.request
import pandas as pd
from pathlib import Path
from collections import Counter

OUT_DIR = Path("data/hatexplain")
OUT_DIR.mkdir(parents=True, exist_ok=True)

HATEXPLAIN_DATA_URL  = "https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data/dataset.json"
HATEXPLAIN_SPLIT_URL = "https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data/post_id_divisions.json"

COMMUNITY_MAP = {
    "African":    "black_flag",
    "Muslim":     "muslim_flag",
    "Women":      "female_flag",
    "Christian":  "christian_flag",
    "Jewish":     "jewish_flag",
    "Hispanic":   "latino_flag",
    "Asian":      "asian_flag",
    "homosexual": "homosexual_gay_or_lesbian_flag",
    "Homosexual": "homosexual_gay_or_lesbian_flag",
    "Indian":     "asian_flag",
    "Arab":       "muslim_flag",
}

ALL_FLAGS = [
    "asian_flag","black_flag","latino_flag","white_flag",
    "male_flag","female_flag","other_gender_flag",
    "christian_flag","muslim_flag","jewish_flag","hindu_flag",
    "buddhist_flag","atheist_flag","other_religion_flag",
    "bisexual_flag","heterosexual_flag","homosexual_gay_or_lesbian_flag",
    "other_sexual_orientation_flag","psychiatric_or_mental_illness_flag",
    "intellectual_or_learning_disability_flag","physical_disability_flag",
    "other_disability_flag","other_race_or_ethnicity_flag",
]

print("Downloading HateXplain from GitHub...")
with urllib.request.urlopen(HATEXPLAIN_DATA_URL) as r:
    dataset = json.loads(r.read().decode("utf-8"))
with urllib.request.urlopen(HATEXPLAIN_SPLIT_URL) as r:
    splits = json.loads(r.read().decode("utf-8"))
print(f"Loaded {len(dataset)} posts | train:{len(splits['train'])} val:{len(splits['val'])} test:{len(splits['test'])}")

def build_split(post_ids, dataset):
    rows = []
    for pid in post_ids:
        ex = dataset.get(pid)
        if ex is None:
            continue
        text = " ".join(ex.get("post_tokens", []))
        annotators = ex.get("annotators", [])
        if not annotators:
            continue
        label_strs = [a.get("label") for a in annotators if a.get("label")]
        if not label_strs:
            continue
        majority = Counter(label_strs).most_common(1)[0][0]
        y = 1 if majority == "hatespeech" else 0
        all_communities = set()
        for a in annotators:
            for t in (a.get("target") or []):
                all_communities.add(t)
        flags = {f: 0 for f in ALL_FLAGS}
        for c in all_communities:
            if COMMUNITY_MAP.get(c):
                flags[COMMUNITY_MAP[c]] = 1
        row = {"comment_text": text, "y": y, "group_id": 0}
        row.update(flags)
        rows.append(row)
    return pd.DataFrame(rows)

for gh_name, out_name in [("train","train"),("val","validation"),("test","test")]:
    df = build_split(splits[gh_name], dataset)
    out_path = OUT_DIR / f"{out_name}.parquet"
    df.to_parquet(out_path, index=False)
    print(f"{out_name}: {len(df)} examples | toxic: {df['y'].sum()} ({100*df['y'].mean():.1f}%) -> {out_path}")
    print(f"  flags: { {f: int(df[f].sum()) for f in ALL_FLAGS if df[f].sum() > 0} }")

print("\nDone. Next: uv run predict_hatexplain.py")