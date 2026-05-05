"""
make_figures.py

Generates the three figures for the CS 517 final report:
  1. BPSN/BNSP tradeoff (grouped bar chart)
  2. Tail distribution of benign identity-mentioning predictions (overlaid KDEs)
  3. Risk–coverage curves for abstention (ERM vs DRO, test set)

Run from the project root. Reads prediction parquets, computes everything
it needs, saves PNG files to figures/.

Usage:
    python make_figures.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score

# ─── config ──────────────────────────────────────────────────────────────────
PRED_PATHS = {
    "ERM":            "outputs/erm/test_preds.parquet",
    "Reweighted ERM": "outputs/reweighted_erm/test_preds.parquet",
    "Group DRO":      "outputs/groupdro/test_preds.parquet",
}
OUT_DIR = Path("figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# consistent color mapping across all figures
METHOD_COLORS = {
    "ERM":            "#4C72B0",   # blue
    "Reweighted ERM": "#DD8452",   # orange
    "Group DRO":      "#55A868",   # green
}

IDENTITIES = [
    ("white_flag",                       "white"),
    ("black_flag",                       "black"),
    ("muslim_flag",                      "muslim"),
    ("christian_flag",                   "christian"),
    ("jewish_flag",                      "jewish"),
    ("female_flag",                      "female"),
    ("male_flag",                        "male"),
    ("homosexual_gay_or_lesbian_flag",   "gay/lesbian"),
]

sns.set_style("whitegrid")
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        10,
    "axes.titlesize":   11,
    "axes.labelsize":   10,
    "legend.fontsize":  9,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "figure.dpi":       300,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
})


# ─── metric helpers ──────────────────────────────────────────────────────────
def bpsn_auc(df, flag):
    bg_pos = df[(df["y"] == 1) & (df[flag] == 0)]
    sg_neg = df[(df["y"] == 0) & (df[flag] == 1)]
    if len(bg_pos) < 50 or len(sg_neg) < 50:
        return np.nan
    y = np.concatenate([bg_pos["y"].values, sg_neg["y"].values])
    p = np.concatenate([bg_pos["prob"].values, sg_neg["prob"].values])
    return float(roc_auc_score(y, p))


def bnsp_auc(df, flag):
    bg_neg = df[(df["y"] == 0) & (df[flag] == 0)]
    sg_pos = df[(df["y"] == 1) & (df[flag] == 1)]
    if len(bg_neg) < 50 or len(sg_pos) < 50:
        return np.nan
    y = np.concatenate([bg_neg["y"].values, sg_pos["y"].values])
    p = np.concatenate([bg_neg["prob"].values, sg_pos["prob"].values])
    return float(roc_auc_score(y, p))


# ─── load ────────────────────────────────────────────────────────────────────
print("Loading prediction parquets...")
preds = {name: pd.read_parquet(path) for name, path in PRED_PATHS.items()}
for name, df in preds.items():
    print(f"  {name}: {len(df)} rows")


# ═══ FIGURE 1: BPSN/BNSP tradeoff ═══════════════════════════════════════════
print("\nBuilding Figure 1: BPSN/BNSP tradeoff...")

rows = []
for flag, label in IDENTITIES:
    for method, df in preds.items():
        rows.append({"identity": label, "method": method,
                     "metric": "BPSN AUC", "value": bpsn_auc(df, flag)})
        rows.append({"identity": label, "method": method,
                     "metric": "BNSP AUC", "value": bnsp_auc(df, flag)})
metrics_df = pd.DataFrame(rows)

fig, (ax_bpsn, ax_bnsp) = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)

for ax, metric in [(ax_bpsn, "BPSN AUC"), (ax_bnsp, "BNSP AUC")]:
    sub = metrics_df[metrics_df["metric"] == metric]
    sns.barplot(
        data=sub, x="identity", y="value", hue="method",
        palette=METHOD_COLORS, ax=ax, edgecolor="white", linewidth=0.5,
    )
    ax.set_title(metric + (" (higher = less false-positive bias)"
                           if metric == "BPSN AUC"
                           else " (higher = less false-negative bias)"))
    ax.set_ylabel("AUC")
    ax.set_xlabel("")
    ax.set_ylim(0.5, 1.0)
    ax.axhline(y=0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    for tick in ax.get_xticklabels():
        tick.set_rotation(30)
        tick.set_ha("right")
    if ax is ax_bnsp:
        ax.get_legend().remove()
    else:
        ax.legend(title=None, loc="lower right", frameon=True)

plt.tight_layout()
out1 = OUT_DIR / "fig1_bpsn_bnsp_tradeoff.png"
plt.savefig(out1)
plt.close()
print(f"  saved {out1}")


# ═══ FIGURE 2: Tail distribution on benign identity mentions ════════════════
print("\nBuilding Figure 2: tail distributions...")

focus_flags = [
    ("white_flag",                     "white"),
    ("muslim_flag",                    "muslim"),
    ("homosexual_gay_or_lesbian_flag", "gay/lesbian"),
]

fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)

for ax, (flag, label) in zip(axes, focus_flags):
    for method, df in preds.items():
        mask = (df[flag] == 1) & (df["y"] == 0)
        probs = df.loc[mask, "prob"].values
        # KDE with clip so we don't smear below 0 / above 1
        sns.kdeplot(
            probs, ax=ax,
            color=METHOD_COLORS[method],
            linewidth=2, label=f"{method} (n={mask.sum()})",
            clip=(0, 1), bw_adjust=0.75,
        )
    ax.axvline(x=0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.axvline(x=0.9, color="red",  linestyle=":", linewidth=0.8, alpha=0.6)
    ax.set_title(f"{label}-flagged benign comments")
    ax.set_xlabel("predicted toxicity $p$")
    ax.set_xlim(0, 1)
    if ax is axes[0]:
        ax.set_ylabel("density")
    ax.legend(loc="upper right", frameon=True, fontsize=8)

fig.suptitle(
    "Benign identity-mentioning predictions: ERM concentrates near 0; "
    "DRO uniformly right-shifts; Reweighted ERM bimodally sharpens",
    fontsize=10, y=1.03,
)
plt.tight_layout()
out2 = OUT_DIR / "fig2_tail_distributions.png"
plt.savefig(out2)
plt.close()
print(f"  saved {out2}")


# ═══ FIGURE 3: Risk–coverage curves (computed on test set) ══════════════════
print("\nBuilding Figure 3: risk–coverage curves...")


def risk_coverage_curve(df, coverages):
    """For each coverage c, retain top-c fraction by confidence, return error rate."""
    conf = np.maximum(df["prob"].values, 1 - df["prob"].values)
    pred = (df["prob"].values >= 0.5).astype(int)
    err  = (pred != df["y"].values).astype(int)
    order = np.argsort(-conf)  # descending confidence
    err_sorted = err[order]
    n = len(err_sorted)
    risks = []
    for c in coverages:
        k = max(1, int(round(c * n)))
        risks.append(float(err_sorted[:k].mean()))
    return np.array(risks)


def risk_coverage_curve_for_subgroup(df, flag, coverages):
    """Error rate on subgroup examples retained at top-c global confidence."""
    conf = np.maximum(df["prob"].values, 1 - df["prob"].values)
    pred = (df["prob"].values >= 0.5).astype(int)
    err  = (pred != df["y"].values).astype(int)
    is_sub = (df[flag].values == 1)
    order = np.argsort(-conf)
    n = len(conf)
    risks = []
    for c in coverages:
        k = max(1, int(round(c * n)))
        keep = order[:k]
        keep_sub = keep[is_sub[keep]]
        risks.append(float(err[keep_sub].mean()) if len(keep_sub) >= 20 else np.nan)
    return np.array(risks)


coverages = np.linspace(0.3, 1.0, 22)

# compute for ERM, Reweighted ERM, DRO — overall risk
curves_overall = {
    m: risk_coverage_curve(df, coverages) for m, df in preds.items()
}
# and subgroup risk for white_flag (the identity where abstention matters most)
curves_white = {
    m: risk_coverage_curve_for_subgroup(df, "white_flag", coverages)
    for m, df in preds.items()
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.0))

for method in preds.keys():
    ax1.plot(
        coverages, curves_overall[method],
        color=METHOD_COLORS[method], linewidth=2, marker="o", markersize=3,
        label=method,
    )
    ax2.plot(
        coverages, curves_white[method],
        color=METHOD_COLORS[method], linewidth=2, marker="o", markersize=3,
        label=method,
    )

for ax, title in [(ax1, "Overall"), (ax2, "White-flagged subgroup")]:
    ax.set_title(title)
    ax.set_xlabel("coverage (fraction retained; 1.0 = no deferral, 0.3 = defer 70%)")
    ax.set_xlim(1.02, 0.28)   # reversed: starts at 1.0 on left, decreases rightward
    ax.legend(loc="upper right", frameon=True)

ax1.set_ylabel("error rate (on retained examples)")
ax2.set_ylabel("error rate on retained white-flagged examples")

fig.suptitle(
    "Risk--coverage curves. Good abstention = error drops as coverage decreases (moving right).",
    fontsize=10, y=1.02,
)
plt.tight_layout()
out3 = OUT_DIR / "fig3_risk_coverage.png"
plt.savefig(out3)
plt.close()
print(f"  saved {out3}")


print("\nDone. All three figures written to figures/")
print("To include in your LaTeX:")
print(r"  \usepackage{graphicx}")
print(r"  \includegraphics[width=\textwidth]{figures/fig1_bpsn_bnsp_tradeoff.png}")