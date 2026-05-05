"""
eval_hatexplain.py

Runs the same fairness evaluation as eval_fairness.py but on HateXplain predictions.
Compares results to Civil Comments to check if bias patterns generalize.
"""

import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score

PRED_PATH = "outputs/hatexplain/test_preds.parquet"
THRESH    = 0.5

df = pq.read_table(PRED_PATH).to_pandas()

def safe_auc(y, p):
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, p))

def ece(y, p, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    idx  = np.digitize(p, bins) - 1
    out  = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        out += (m.sum() / len(p)) * abs(p[m].mean() - y[m].mean())
    return out

def bpsn_auc(df, flag_col):
    bg_pos = df[(df["y"] == 1) & (df[flag_col] == 0)]
    sg_neg = df[(df["y"] == 0) & (df[flag_col] == 1)]
    if len(bg_pos) < 20 or len(sg_neg) < 20:
        return None
    y = np.concatenate([bg_pos["y"].values, sg_neg["y"].values])
    p = np.concatenate([bg_pos["prob"].values, sg_neg["prob"].values])
    return safe_auc(y, p)

def bnsp_auc(df, flag_col):
    bg_neg = df[(df["y"] == 0) & (df[flag_col] == 0)]
    sg_pos = df[(df["y"] == 1) & (df[flag_col] == 1)]
    if len(bg_neg) < 20 or len(sg_pos) < 20:
        return None
    y = np.concatenate([bg_neg["y"].values, sg_pos["y"].values])
    p = np.concatenate([bg_neg["prob"].values, sg_pos["prob"].values])
    return safe_auc(y, p)

def subgroup_auc(df, flag_col):
    sub = df[df[flag_col] == 1]
    if len(sub) < 20 or len(np.unique(sub["y"])) < 2:
        return None, len(sub)
    return safe_auc(sub["y"].values, sub["prob"].values), len(sub)

def group_error_rate(df, flag_col):
    pred = (df["prob"].values >= THRESH).astype(int)
    err  = (pred != df["y"].values).astype(int)
    g1   = df[df[flag_col] == 1]
    g0   = df[df[flag_col] == 0]
    if len(g1) < 20 or len(g0) < 20:
        return None, None
    return float(err[g1.index].mean()), float(err[g0.index].mean())

# ── overall metrics ─────────────────────────────────────────────────────────
overall_auc = safe_auc(df["y"].values, df["prob"].values)
overall_ece = ece(df["y"].values, df["prob"].values)
pred        = (df["prob"].values >= THRESH).astype(int)
overall_err = float((pred != df["y"].values).mean())

print("=" * 65)
print("HATEXPLAIN ZERO-SHOT EVALUATION (ERM model trained on Civil Comments)")
print("=" * 65)
print(f"\nOverall AUC:        {overall_auc:.4f}")
print(f"Overall ECE:        {overall_ece:.4f}")
print(f"Overall error rate: {overall_err:.4f}")
print(f"N examples:         {len(df)}")
print(f"Toxic rate (true):  {df['y'].mean():.3f}")
print(f"Toxic rate (pred):  {(df['prob'].values >= 0.5).mean():.3f}")

# ── per-identity fairness ───────────────────────────────────────────────────
id_flags = [c for c in df.columns if c.endswith("_flag")]

print("\n" + "=" * 65)
print("SUBGROUP FAIRNESS (n >= 20 required)")
print("=" * 65)
print(f"\n{'flag':<45} {'n':>5} {'SubAUC':>8} {'BPSN':>8} {'BNSP':>8} {'err_sub':>8} {'err_bg':>8}")
print("-" * 95)

rows = []
for c in id_flags:
    s_auc, n = subgroup_auc(df, c)
    bpsn     = bpsn_auc(df, c)
    bnsp     = bnsp_auc(df, c)
    errs     = group_error_rate(df, c)
    err_sub, err_bg = errs if errs[0] is not None else (None, None)
    rows.append((c, n, s_auc, bpsn, bnsp, err_sub, err_bg))

# sort by subgroup AUC ascending (worst first), None last
rows.sort(key=lambda x: (x[2] is None, x[2] if x[2] is not None else 1))

for c, n, s_auc, bpsn, bnsp, err_sub, err_bg in rows:
    if n < 20:
        continue
    def fmt(v): return f"{v:.3f}" if v is not None else "  n/a"
    print(f"{c:<45} {n:>5} {fmt(s_auc):>8} {fmt(bpsn):>8} {fmt(bnsp):>8} {fmt(err_sub):>8} {fmt(err_bg):>8}")

# ── comparison summary ──────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("COMPARISON: Civil Comments vs HateXplain (ERM model, zero-shot)")
print("=" * 65)
print(f"\n{'Metric':<30} {'Civil Comments':>16} {'HateXplain':>12}")
print("-" * 60)
print(f"{'Overall AUC':<30} {'0.9433':>16} {overall_auc:>12.4f}")
print(f"{'Overall ECE':<30} {'0.0138':>16} {overall_ece:>12.4f}")
print(f"{'Overall error rate':<30} {'5.35%':>16} {overall_err*100:>11.2f}%")

# BPSN for groups with enough data
print("\nBPSN AUC comparison (Civil Comments → HateXplain):")
civil_bpsn = {
    "black_flag":                       0.784,
    "muslim_flag":                      0.833,
    "female_flag":                      0.877,
    "homosexual_gay_or_lesbian_flag":   0.789,
    "jewish_flag":                      0.876,
    "christian_flag":                   0.910,
}
for flag, civil_val in civil_bpsn.items():
    hx_val = bpsn_auc(df, flag) if flag in df.columns else None
    hx_str = f"{hx_val:.3f}" if hx_val is not None else "n/a (too few)"
    print(f"  {flag:<45} {civil_val:.3f}  ->  {hx_str}")