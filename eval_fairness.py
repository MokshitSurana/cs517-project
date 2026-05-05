import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score

PRED_PATH = "outputs/groupdro/val_preds.parquet"
THRESH = 0.5  # for error-rate parity style metrics

df = pq.read_table(PRED_PATH).to_pandas()

def safe_auc(y, p):
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, p))

def bpsn_auc(df, flag_col):
    # Background Positive (y=1, subgroup=0) + Subgroup Negative (y=0, subgroup=1)
    bg_pos = df[(df["y"] == 1) & (df[flag_col] == 0)]
    sg_neg = df[(df["y"] == 0) & (df[flag_col] == 1)]
    mix = np.concatenate([bg_pos, sg_neg])
    if len(bg_pos) < 50 or len(sg_neg) < 50:
        return None, len(bg_pos), len(sg_neg)
    y = np.concatenate([bg_pos["y"].values, sg_neg["y"].values])
    p = np.concatenate([bg_pos["prob"].values, sg_neg["prob"].values])
    return safe_auc(y, p), len(bg_pos), len(sg_neg)

def bnsp_auc(df, flag_col):
    # Background Negative (y=0, subgroup=0) + Subgroup Positive (y=1, subgroup=1)
    bg_neg = df[(df["y"] == 0) & (df[flag_col] == 0)]
    sg_pos = df[(df["y"] == 1) & (df[flag_col] == 1)]
    if len(bg_neg) < 50 or len(sg_pos) < 50:
        return None, len(bg_neg), len(sg_pos)
    y = np.concatenate([bg_neg["y"].values, sg_pos["y"].values])
    p = np.concatenate([bg_neg["prob"].values, sg_pos["prob"].values])
    return safe_auc(y, p), len(bg_neg), len(sg_pos)

def subgroup_auc(df, flag_col):
    sub = df[df[flag_col] == 1]
    if len(sub) < 200 or len(np.unique(sub["y"])) < 2:
        return None, len(sub)
    return safe_auc(sub["y"].values, sub["prob"].values), len(sub)

def group_error_rate(df, flag_col, thresh=0.5):
    # error rate among subgroup mentioned vs not mentioned (at fixed threshold)
    pred = (df["prob"] >= thresh).astype(int)
    err = (pred != df["y"]).astype(int)
    g1 = df[df[flag_col] == 1]
    g0 = df[df[flag_col] == 0]
    if len(g1) < 200 or len(g0) < 200:
        return None
    return float(err.loc[g1.index].mean()), float(err.loc[g0.index].mean())

id_flags = [c for c in df.columns if c.endswith("_flag")]
rows = []
for c in id_flags:
    s_auc, n_s = subgroup_auc(df, c)
    bpsn, n_bgpos, n_sgneg = bpsn_auc(df, c)
    bnsp, n_bgneg, n_sgpos = bnsp_auc(df, c)
    errs = group_error_rate(df, c, THRESH)
    if errs is None:
        err_s, err_bg = None, None
    else:
        err_s, err_bg = errs

    rows.append((c, n_s, s_auc, bpsn, bnsp, err_s, err_bg))

# sort by subgroup auc ascending (worst first)
rows.sort(key=lambda x: (x[2] is None, x[2]))

print("flag | n_sub | subgroupAUC | BPSN_AUC | BNSP_AUC | err(sub) | err(bg)")
for r in rows:
    print(r)