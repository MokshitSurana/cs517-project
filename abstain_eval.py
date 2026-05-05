import numpy as np
import pyarrow.parquet as pq

TEST_PATH = "outputs/groupdro/test_preds.parquet"

def confidence(p):
    return np.maximum(p, 1 - p)

def eval_at_coverage(df, coverage):
    conf = confidence(df["prob"].values)
    k = int(len(df) * coverage)
    idx = np.argsort(-conf)[:k]
    sub = df.iloc[idx].reset_index(drop=True)  # reset so indices are 0..k-1
    pred = (sub["prob"].values >= 0.5).astype(int)
    risk = float((pred != sub["y"].values).mean())
    return risk, sub  # return the sliced df, not just n

def subgroup_error(sub_df, flag_col):
    # everything operates on positional numpy arrays — no index issues
    flags = sub_df[flag_col].values
    y     = sub_df["y"].values
    prob  = sub_df["prob"].values

    pred = (prob >= 0.5).astype(int)
    err  = (pred != y).astype(int)

    mask_sub = flags == 1
    mask_bg  = flags == 0

    if mask_sub.sum() < 50 or mask_bg.sum() < 50:
        return None

    return float(err[mask_sub].mean()), float(err[mask_bg].mean())

df = pq.read_table(TEST_PATH).to_pandas()

for cov in [1.0, 0.9, 0.8, 0.7]:
    risk, sub_df = eval_at_coverage(df, cov)
    print(f"\nCoverage={cov:.1f}  kept={len(sub_df)}  risk={risk:.4f}")

    for flag in ["white_flag", "muslim_flag", "female_flag", "male_flag"]:
        if flag not in sub_df.columns:
            continue
        result = subgroup_error(sub_df, flag)
        if result is None:
            print(f"  {flag}: too few subgroup examples")
            continue
        err_sub, err_bg = result
        print(f"  {flag}: err(sub)={err_sub:.3f}  err(bg)={err_bg:.3f}  gap={err_sub-err_bg:+.3f}")