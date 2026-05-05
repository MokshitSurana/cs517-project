import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score

VAL_PATH  = "outputs/erm/val_preds.parquet"
TEST_PATH = "outputs/erm/test_preds.parquet"

def error_rate(y, prob, thresh):
    pred = (prob >= thresh).astype(int)
    return float((pred != y).mean())

def subgroup_error_at_thresh(df, flag_col, thresh):
    flags = df[flag_col].values
    y     = df["y"].values
    prob  = df["prob"].values
    pred  = (prob >= thresh).astype(int)
    err   = (pred != y).astype(int)
    mask_sub = flags == 1
    mask_bg  = flags == 0
    if mask_sub.sum() < 50 or mask_bg.sum() < 50:
        return None, None, None
    return float(err[mask_sub].mean()), float(err[mask_bg].mean()), float(err[mask_sub].mean() - err[mask_bg].mean())

def find_best_threshold(df, flag_col, thresholds=np.linspace(0.1, 0.9, 81)):
    best_thresh, best_gap = 0.5, 1e9
    for t in thresholds:
        result = subgroup_error_at_thresh(df, flag_col, t)
        if result[0] is None:
            continue
        if abs(result[2]) < best_gap:
            best_gap, best_thresh = abs(result[2]), t
    return best_thresh, best_gap

def safe_auc(y, p):
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, p))

def bpsn_auc(df, flag_col):
    bg_pos = df[(df["y"] == 1) & (df[flag_col] == 0)]
    sg_neg = df[(df["y"] == 0) & (df[flag_col] == 1)]
    if len(bg_pos) < 50 or len(sg_neg) < 50:
        return None
    y = np.concatenate([bg_pos["y"].values, sg_neg["y"].values])
    p = np.concatenate([bg_pos["prob"].values, sg_neg["prob"].values])
    return safe_auc(y, p)

def fmt_bpsn(v):
    return f"{v:.3f}" if v is not None else "n/a"

val_df  = pq.read_table(VAL_PATH).to_pandas()
test_df = pq.read_table(TEST_PATH).to_pandas()

FLAGS = ["white_flag", "muslim_flag", "female_flag", "male_flag", "christian_flag"]

print("=" * 70)
print("STEP 1: Find optimal thresholds on VAL set")
print("=" * 70)

best_thresholds = {}
for flag in FLAGS:
    if flag not in val_df.columns:
        continue
    thresh, _ = find_best_threshold(val_df, flag)
    best_thresholds[flag] = thresh
    e_sub, e_bg, g   = subgroup_error_at_thresh(val_df, flag, thresh)
    d_sub, d_bg, d_g = subgroup_error_at_thresh(val_df, flag, 0.5)
    if e_sub is None:
        continue
    print(f"\n{flag}")
    print(f"  Default thresh=0.50: err(sub)={d_sub:.3f}  err(bg)={d_bg:.3f}  gap={d_g:+.3f}")
    print(f"  Optimal thresh={thresh:.2f}: err(sub)={e_sub:.3f}  err(bg)={e_bg:.3f}  gap={g:+.3f}")

print("\n" + "=" * 70)
print("STEP 2: TEST set — ERM default (thresh=0.50)")
print("=" * 70)
print(f"Overall error rate: {error_rate(test_df['y'].values, test_df['prob'].values, 0.5):.4f}")
for flag in FLAGS:
    if flag not in test_df.columns:
        continue
    e_sub, e_bg, gap = subgroup_error_at_thresh(test_df, flag, 0.5)
    if e_sub is None:
        continue
    print(f"  {flag}: err(sub)={e_sub:.3f}  err(bg)={e_bg:.3f}  gap={gap:+.3f}  BPSN={fmt_bpsn(bpsn_auc(test_df, flag))}")

print("\n" + "=" * 70)
print("STEP 3: TEST set — Threshold Optimization")
print("=" * 70)

test_preds_opt = np.full(len(test_df), 0.5)
for flag in FLAGS:
    if flag not in test_df.columns or flag not in best_thresholds:
        continue
    test_preds_opt[test_df[flag].values == 1] = best_thresholds[flag]

pred_opt = (test_df["prob"].values >= test_preds_opt).astype(int)
print(f"Overall error rate: {float((pred_opt != test_df['y'].values).mean()):.4f}")

for flag in FLAGS:
    if flag not in test_df.columns or flag not in best_thresholds:
        continue
    thresh = best_thresholds[flag]
    e_sub, e_bg, gap = subgroup_error_at_thresh(test_df, flag, thresh)
    if e_sub is None:
        continue
    print(f"  {flag} (thresh={thresh:.2f}): err(sub)={e_sub:.3f}  err(bg)={e_bg:.3f}  gap={gap:+.3f}  BPSN={fmt_bpsn(bpsn_auc(test_df, flag))}")

print("\n" + "=" * 70)
print("STEP 4: Three-way comparison summary")
print("=" * 70)
print(f"\n{'Flag':<30} {'ERM gap':>10} {'ThreshOpt gap':>15} {'DRO gap':>10}")
print("-" * 70)

dro_gaps = {
    "white_flag":     +0.205,
    "muslim_flag":    +0.070,
    "female_flag":    +0.038,
    "male_flag":      +0.059,
    "christian_flag": None,
}

for flag in FLAGS:
    if flag not in test_df.columns:
        continue
    erm_sub, erm_bg, erm_gap = subgroup_error_at_thresh(test_df, flag, 0.5)
    if erm_sub is None:
        continue
    thresh = best_thresholds.get(flag, 0.5)
    opt_sub, opt_bg, opt_gap = subgroup_error_at_thresh(test_df, flag, thresh)
    dro_gap = dro_gaps.get(flag)
    print(f"{flag:<30} {erm_gap:>+10.3f} {opt_gap:>+15.3f} {(f'{dro_gap:+.3f}' if dro_gap is not None else 'n/a'):>10}")