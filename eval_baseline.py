import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score

def ece(y, p, n_bins=15):
    bins = np.linspace(0, 1, n_bins+1)
    idx = np.digitize(p, bins) - 1
    out = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        out += (m.sum()/len(p)) * abs(p[m].mean() - y[m].mean())
    return out

df = pq.read_table("outputs/erm/val_preds.parquet").to_pandas()

# overall
overall_auc = roc_auc_score(df["y"], df["prob"])
overall_ece = ece(df["y"].values, df["prob"].values)

print("OVERALL")
print("AUC:", round(overall_auc, 4))
print("ECE:", round(overall_ece, 4))

# worst-group by group_id (your group_id is identity×label or none×label)
group_aucs = []
for gid, gdf in df.groupby("group_id"):
    if gdf["y"].nunique() < 2 or len(gdf) < 50:
        continue
    group_aucs.append((gid, roc_auc_score(gdf["y"], gdf["prob"]), len(gdf)))

group_aucs.sort(key=lambda x: x[1])
print("\nWORST GROUP AUC (by group_id):")
print(group_aucs[:10])

# subgroup AUC for each identity flag
id_flags = [c for c in df.columns if c.endswith("_flag")]
print("\nSUBGROUP AUC (identity mentioned):")
res = []
for c in id_flags:
    sdf = df[df[c] == 1]
    if sdf["y"].nunique() < 2 or len(sdf) < 200:
        continue
    res.append((c, roc_auc_score(sdf["y"], sdf["prob"]), len(sdf)))
res.sort(key=lambda x: x[1])
for r in res:
    print(r)