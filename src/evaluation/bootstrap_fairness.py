"""
bootstrap_fairness.py

Computes bootstrap 95% confidence intervals for fairness metrics on
prediction parquets produced by predict_and_save.py.

Usage
-----
  python bootstrap_fairness.py \
      --erm outputs/erm/test_preds.parquet \
      --dro outputs/groupdro/test_preds.parquet \
      --out outputs/bootstrap/test_ci.csv \
      --n-boot 1000

Output
------
CSV with one row per (metric, identity, model), reporting point estimate and
(lo, hi) 95% CI. Also computes the paired bootstrap CI on the DRO - ERM
difference, which is what actually tells you whether DRO beats ERM.

Key design choice: we use the SAME resampled indices for both models on each
bootstrap iteration. That gives a valid paired test and much tighter CIs on
the difference than resampling each model independently.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score

RNG_SEED = 42
N_BOOT = 1000
MIN_N = 50   # minimum subgroup count to attempt a metric
THRESH = 0.5

# identities we actually care about in the report — skip the tiny ones
REPORT_IDENTITIES = [
    "white_flag", "black_flag", "muslim_flag", "christian_flag", "jewish_flag",
    "female_flag", "male_flag", "homosexual_gay_or_lesbian_flag",
]


# ─── metric primitives (operate on already-resampled dfs) ───────────────────

def safe_auc(y, p):
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, p))


def overall_auc(df):
    return safe_auc(df["y"].values, df["prob"].values)


def subgroup_auc(df, flag):
    sub = df[df[flag] == 1]
    if len(sub) < MIN_N:
        return np.nan
    return safe_auc(sub["y"].values, sub["prob"].values)


def bpsn_auc(df, flag):
    # Background Positive + Subgroup Negative
    bg_pos = df[(df["y"] == 1) & (df[flag] == 0)]
    sg_neg = df[(df["y"] == 0) & (df[flag] == 1)]
    if len(bg_pos) < MIN_N or len(sg_neg) < MIN_N:
        return np.nan
    y = np.concatenate([bg_pos["y"].values, sg_neg["y"].values])
    p = np.concatenate([bg_pos["prob"].values, sg_neg["prob"].values])
    return safe_auc(y, p)


def bnsp_auc(df, flag):
    # Background Negative + Subgroup Positive
    bg_neg = df[(df["y"] == 0) & (df[flag] == 0)]
    sg_pos = df[(df["y"] == 1) & (df[flag] == 1)]
    if len(bg_neg) < MIN_N or len(sg_pos) < MIN_N:
        return np.nan
    y = np.concatenate([bg_neg["y"].values, sg_pos["y"].values])
    p = np.concatenate([bg_neg["prob"].values, sg_pos["prob"].values])
    return safe_auc(y, p)


def error_gap(df, flag, thresh=THRESH):
    # Err(sub) - Err(bg)
    sub = df[df[flag] == 1]
    bg = df[df[flag] == 0]
    if len(sub) < MIN_N or len(bg) < MIN_N:
        return np.nan
    err_sub = ((sub["prob"] >= thresh).astype(int) != sub["y"]).mean()
    err_bg = ((bg["prob"] >= thresh).astype(int) != bg["y"]).mean()
    return float(err_sub - err_bg)


# ─── dispatch ────────────────────────────────────────────────────────────────

def compute_all_metrics(df):
    """Returns dict {metric_name: value}. Metric names encode the identity."""
    out = {"overall_auc": overall_auc(df)}
    for flag in REPORT_IDENTITIES:
        name = flag.replace("_flag", "")
        out[f"subgroup_auc/{name}"] = subgroup_auc(df, flag)
        out[f"bpsn_auc/{name}"] = bpsn_auc(df, flag)
        out[f"bnsp_auc/{name}"] = bnsp_auc(df, flag)
        out[f"error_gap/{name}"] = error_gap(df, flag)
    return out


# ─── bootstrap loop ──────────────────────────────────────────────────────────

def bootstrap(df_erm, df_dro, n_boot=N_BOOT, seed=RNG_SEED):
    """
    Paired bootstrap: same indices resampled for both models per iteration.
    Returns three dicts of length-n_boot arrays: erm, dro, diff (= dro - erm).
    """
    assert len(df_erm) == len(df_dro), "ERM and DRO preds must align row-by-row"
    rng = np.random.default_rng(seed)
    n = len(df_erm)

    metric_names = list(compute_all_metrics(df_erm).keys())
    boot_erm = {m: np.full(n_boot, np.nan) for m in metric_names}
    boot_dro = {m: np.full(n_boot, np.nan) for m in metric_names}

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)  # sample with replacement
        s_erm = df_erm.iloc[idx].reset_index(drop=True)
        s_dro = df_dro.iloc[idx].reset_index(drop=True)
        m_erm = compute_all_metrics(s_erm)
        m_dro = compute_all_metrics(s_dro)
        for m in metric_names:
            boot_erm[m][b] = m_erm[m]
            boot_dro[m][b] = m_dro[m]
        if (b + 1) % 100 == 0:
            print(f"  bootstrap iter {b+1}/{n_boot}")

    boot_diff = {m: boot_dro[m] - boot_erm[m] for m in metric_names}
    return boot_erm, boot_dro, boot_diff


def ci(arr, lo=2.5, hi=97.5):
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return np.nan, np.nan, np.nan
    return float(np.mean(arr)), float(np.percentile(arr, lo)), float(np.percentile(arr, hi))


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--erm", required=True, help="path to ERM predictions parquet")
    ap.add_argument("--dro", required=True, help="path to DRO predictions parquet")
    ap.add_argument("--out", required=True, help="output CSV path")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=RNG_SEED)
    args = ap.parse_args()

    print(f"Loading predictions...")
    df_erm = pd.read_parquet(args.erm)
    df_dro = pd.read_parquet(args.dro)
    print(f"  ERM: {len(df_erm)} rows")
    print(f"  DRO: {len(df_dro)} rows")

    # sanity: same labels, same order
    if not (df_erm["y"].values == df_dro["y"].values).all():
        raise ValueError("ERM and DRO predictions are not row-aligned (labels differ).")

    print("Point estimates...")
    point_erm = compute_all_metrics(df_erm)
    point_dro = compute_all_metrics(df_dro)

    print(f"Running {args.n_boot} bootstrap iterations...")
    boot_erm, boot_dro, boot_diff = bootstrap(df_erm, df_dro, args.n_boot, args.seed)

    rows = []
    for m in point_erm.keys():
        _, erm_lo, erm_hi = ci(boot_erm[m])
        _, dro_lo, dro_hi = ci(boot_dro[m])
        diff_mean, diff_lo, diff_hi = ci(boot_diff[m])
        # DRO "significantly better" if diff CI excludes 0 in the right direction.
        # For AUCs, higher is better. For error_gap, we want smaller |gap|,
        # so we report the raw diff and you interpret direction per metric.
        sig = "yes" if (diff_lo > 0 or diff_hi < 0) else "no"
        rows.append({
            "metric": m,
            "erm_point": point_erm[m],
            "erm_ci_lo": erm_lo,
            "erm_ci_hi": erm_hi,
            "dro_point": point_dro[m],
            "dro_ci_lo": dro_lo,
            "dro_ci_hi": dro_hi,
            "diff_mean": diff_mean,
            "diff_ci_lo": diff_lo,
            "diff_ci_hi": diff_hi,
            "diff_significant": sig,
        })

    out_df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"\nWrote {args.out}")
    print("\n=== Summary ===")
    # pretty-print for inspection
    with pd.option_context("display.max_rows", None, "display.width", 200, "display.float_format", "{:.4f}".format):
        print(out_df[["metric", "erm_point", "dro_point", "diff_mean", "diff_ci_lo", "diff_ci_hi", "diff_significant"]])


if __name__ == "__main__":
    main()