"""
Per-subgroup calibration analysis for FairGuard.

Produces:
- Per-subgroup ECE table (8 identities × 3 methods + background)
- Calibration-fairness gap (subgroup ECE − background ECE), with paired bootstrap CIs
- Reliability diagrams per (method, subgroup) pair
- Confidence-binned accuracy by subgroup
- Subgroup-conditional risk-coverage curves

Inputs:
  outputs/erm/test_preds.parquet
  outputs/reweighted_erm/test_preds.parquet
  outputs/groupdro/test_preds.parquet

Outputs:
  outputs/calibration_analysis/per_subgroup_ece.csv
  outputs/calibration_analysis/calibration_fairness_gap.csv
  outputs/calibration_analysis/conf_binned_accuracy.csv
  outputs/calibration_analysis/risk_coverage_by_subgroup.csv
  figures/reliability_diagrams.png
  figures/calibration_fairness_gap.png
  figures/risk_coverage_by_subgroup.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# ─── Config ─────────────────────────────────────────────────────────────────

PRED_PATHS = {
    "ERM":            "outputs/erm/test_preds.parquet",
    "Reweighted ERM": "outputs/reweighted_erm/test_preds.parquet",
    "Group DRO":      "outputs/groupdro/test_preds.parquet",
}

# 8 identities reported in the main results
SUBGROUPS = [
    "white_flag",
    "black_flag",
    "muslim_flag",
    "christian_flag",
    "jewish_flag",
    "female_flag",
    "male_flag",
    "homosexual_gay_or_lesbian_flag",
]

# Display names
SUBGROUP_LABELS = {
    "white_flag": "white",
    "black_flag": "black",
    "muslim_flag": "muslim",
    "christian_flag": "christian",
    "jewish_flag": "jewish",
    "female_flag": "female",
    "male_flag": "male",
    "homosexual_gay_or_lesbian_flag": "gay/lesbian",
}

# Method colors (match your figures)
METHOD_COLORS = {
    "ERM":            "#4C72B0",
    "Reweighted ERM": "#DD8452",
    "Group DRO":      "#55A868",
}

N_BINS = 15            # bins for ECE (matches your existing convention)
N_BOOTSTRAP = 1000     # paired bootstrap iterations
RNG_SEED = 42

OUTDIR_DATA = "outputs/calibration_analysis"
OUTDIR_FIGS = "figures"
os.makedirs(OUTDIR_DATA, exist_ok=True)
os.makedirs(OUTDIR_FIGS, exist_ok=True)


# ─── Style ──────────────────────────────────────────────────────────────────
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False


# ─── Calibration metrics ────────────────────────────────────────────────────

def compute_ece(probs, labels, n_bins=N_BINS):
    """
    Standard ECE with equal-width bins on max-class probability.
    For binary classification, max-class prob = max(p, 1-p), and 'correct'
    means the predicted class matches the true label.
    """
    probs = np.asarray(probs)
    labels = np.asarray(labels).astype(int)

    # confidence = max prob across classes; pred = argmax
    confidences = np.maximum(probs, 1 - probs)
    predictions = (probs >= 0.5).astype(int)
    correct = (predictions == labels).astype(int)

    bin_edges = np.linspace(0.5, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        bin_conf = confidences[mask].mean()
        bin_acc = correct[mask].mean()
        bin_weight = mask.sum() / n
        ece += bin_weight * abs(bin_conf - bin_acc)

    return ece


def reliability_curve(probs, labels, n_bins=N_BINS):
    """Returns (bin_centers, bin_accuracies, bin_counts) for plotting reliability diagrams."""
    probs = np.asarray(probs)
    labels = np.asarray(labels).astype(int)

    confidences = np.maximum(probs, 1 - probs)
    predictions = (probs >= 0.5).astype(int)
    correct = (predictions == labels).astype(int)

    bin_edges = np.linspace(0.5, 1.0, n_bins + 1)
    bin_centers, bin_accs, bin_confs, bin_counts = [], [], [], []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        bin_centers.append((lo + hi) / 2)
        if mask.sum() > 0:
            bin_accs.append(correct[mask].mean())
            bin_confs.append(confidences[mask].mean())
            bin_counts.append(mask.sum())
        else:
            bin_accs.append(np.nan)
            bin_confs.append(np.nan)
            bin_counts.append(0)

    return np.array(bin_centers), np.array(bin_accs), np.array(bin_confs), np.array(bin_counts)


# ─── Load all predictions ───────────────────────────────────────────────────

print("Loading predictions...")
preds = {}
for method, path in PRED_PATHS.items():
    df = pd.read_parquet(path)
    preds[method] = df
    print(f"  {method}: {df.shape[0]} rows, ECE check...")

# Sanity check: all three should have the same comments in the same order
for m1 in PRED_PATHS:
    for m2 in PRED_PATHS:
        if m1 < m2:
            assert (preds[m1]["y"].values == preds[m2]["y"].values).all(), \
                f"{m1} and {m2} have different labels — order mismatch?"

# Use the labels and identity flags from the first one (they're identical across methods)
base = preds["ERM"]
y = base["y"].values
n = len(y)
print(f"Total test examples: {n}\n")


# ─── 1. Per-subgroup ECE table ──────────────────────────────────────────────

print("=" * 60)
print("1. Per-subgroup ECE")
print("=" * 60)

ece_rows = []

# Background = no identity mentioned (all 8 main identity flags = 0)
# Note: we use the 8 reported identities, not all 23 — matches the paper's scope
bg_mask = (base[SUBGROUPS].sum(axis=1) == 0).values
print(f"Background (no main-identity mention): n = {bg_mask.sum()}")

for method, df in preds.items():
    p = df["prob"].values
    # Overall
    overall_ece = compute_ece(p, y)
    # Background
    bg_ece = compute_ece(p[bg_mask], y[bg_mask])

    row = {"method": method, "overall_ECE": overall_ece, "background_ECE": bg_ece}
    for sg_col in SUBGROUPS:
        mask = (base[sg_col] == 1).values
        nm = mask.sum()
        if nm < 30:  # too few examples for reliable ECE
            row[SUBGROUP_LABELS[sg_col] + "_ECE"] = np.nan
            row[SUBGROUP_LABELS[sg_col] + "_n"] = nm
        else:
            sg_ece = compute_ece(p[mask], y[mask])
            row[SUBGROUP_LABELS[sg_col] + "_ECE"] = sg_ece
            row[SUBGROUP_LABELS[sg_col] + "_n"] = nm
    ece_rows.append(row)

ece_df = pd.DataFrame(ece_rows).set_index("method")
ece_df.to_csv(f"{OUTDIR_DATA}/per_subgroup_ece.csv")
print("\nPer-subgroup ECE:")
# print just the ECE columns for readability
ece_cols = [c for c in ece_df.columns if c.endswith("_ECE")]
print(ece_df[ece_cols].round(4).to_string())
print(f"\nWrote: {OUTDIR_DATA}/per_subgroup_ece.csv\n")


# ─── 2. Calibration-fairness gap with paired bootstrap CIs ───────────────────

print("=" * 60)
print("2. Calibration-fairness gap (subgroup ECE − background ECE)")
print("=" * 60)

rng = np.random.default_rng(RNG_SEED)

gap_rows = []
for method, df in preds.items():
    p = df["prob"].values
    print(f"\n{method}: bootstrapping {N_BOOTSTRAP} iterations...")

    for sg_col in SUBGROUPS:
        sg_mask = (base[sg_col] == 1).values
        if sg_mask.sum() < 30:
            continue

        # Point estimates
        sg_ece = compute_ece(p[sg_mask], y[sg_mask])
        bg_ece = compute_ece(p[bg_mask], y[bg_mask])
        gap = sg_ece - bg_ece

        # Paired bootstrap on the gap
        gap_samples = np.zeros(N_BOOTSTRAP)
        idx_sg = np.where(sg_mask)[0]
        idx_bg = np.where(bg_mask)[0]

        for b in range(N_BOOTSTRAP):
            sg_resample = rng.choice(idx_sg, size=len(idx_sg), replace=True)
            bg_resample = rng.choice(idx_bg, size=len(idx_bg), replace=True)
            sg_ece_b = compute_ece(p[sg_resample], y[sg_resample])
            bg_ece_b = compute_ece(p[bg_resample], y[bg_resample])
            gap_samples[b] = sg_ece_b - bg_ece_b

        ci_lo, ci_hi = np.percentile(gap_samples, [2.5, 97.5])
        significant = (ci_lo > 0) or (ci_hi < 0)

        gap_rows.append({
            "method": method,
            "subgroup": SUBGROUP_LABELS[sg_col],
            "subgroup_ECE": sg_ece,
            "background_ECE": bg_ece,
            "gap": gap,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "significant": significant,
            "n_subgroup": sg_mask.sum(),
        })

gap_df = pd.DataFrame(gap_rows)
gap_df.to_csv(f"{OUTDIR_DATA}/calibration_fairness_gap.csv", index=False)
print("\nCalibration-fairness gaps (subgroup ECE − background ECE):")
print(gap_df[["method", "subgroup", "gap", "ci_lo", "ci_hi", "significant"]]
      .round(4).to_string(index=False))
print(f"\nWrote: {OUTDIR_DATA}/calibration_fairness_gap.csv\n")


# ─── 3. Confidence-binned accuracy by subgroup ──────────────────────────────

print("=" * 60)
print("3. Confidence-binned accuracy by subgroup")
print("=" * 60)

CONF_BINS = [(0.5, 0.7), (0.7, 0.9), (0.9, 0.99), (0.99, 1.0001)]
CONF_LABELS = ["[0.5, 0.7)", "[0.7, 0.9)", "[0.9, 0.99)", "[0.99, 1.0]"]

cba_rows = []
for method, df in preds.items():
    p = df["prob"].values
    conf = np.maximum(p, 1 - p)
    pred = (p >= 0.5).astype(int)
    correct = (pred == y).astype(int)

    for sg_col in ["__background__"] + SUBGROUPS:
        if sg_col == "__background__":
            mask = bg_mask
            sg_label = "background"
        else:
            mask = (base[sg_col] == 1).values
            sg_label = SUBGROUP_LABELS[sg_col]

        for (lo, hi), bin_label in zip(CONF_BINS, CONF_LABELS):
            in_bin = mask & (conf >= lo) & (conf < hi)
            n_in_bin = in_bin.sum()
            if n_in_bin == 0:
                acc = np.nan
            else:
                acc = correct[in_bin].mean()
            cba_rows.append({
                "method": method,
                "subgroup": sg_label,
                "conf_bin": bin_label,
                "accuracy": acc,
                "n": n_in_bin,
            })

cba_df = pd.DataFrame(cba_rows)
cba_df.to_csv(f"{OUTDIR_DATA}/conf_binned_accuracy.csv", index=False)

# Pivot for readability — accuracy at high confidence is the key cell
print("\nAccuracy at confidence ≥ 0.99 by (method, subgroup):")
high_conf = cba_df[cba_df["conf_bin"] == "[0.99, 1.0]"]
print(high_conf.pivot(index="subgroup", columns="method", values="accuracy")
      .round(3).to_string())
print(f"\nWrote: {OUTDIR_DATA}/conf_binned_accuracy.csv\n")


# ─── 4. Subgroup-conditional risk-coverage ──────────────────────────────────

print("=" * 60)
print("4. Subgroup-conditional risk-coverage")
print("=" * 60)

COVERAGES = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
KEY_SUBGROUPS = ["white_flag", "muslim_flag", "homosexual_gay_or_lesbian_flag"]

rc_rows = []
for method, df in preds.items():
    p = df["prob"].values
    conf = np.maximum(p, 1 - p)
    pred = (p >= 0.5).astype(int)
    correct = (pred == y).astype(int)

    for sg_col in ["__background__"] + KEY_SUBGROUPS:
        if sg_col == "__background__":
            mask = bg_mask
            sg_label = "background"
        else:
            mask = (base[sg_col] == 1).values
            sg_label = SUBGROUP_LABELS[sg_col]

        if mask.sum() < 30:
            continue

        # Restrict to subgroup, then compute risk-coverage within it
        sg_conf = conf[mask]
        sg_correct = correct[mask]
        order = np.argsort(-sg_conf)  # descending
        sg_correct_sorted = sg_correct[order]

        for cov in COVERAGES:
            k = int(round(cov * len(sg_conf)))
            if k == 0:
                continue
            kept = sg_correct_sorted[:k]
            risk = 1 - kept.mean()
            rc_rows.append({
                "method": method,
                "subgroup": sg_label,
                "coverage": cov,
                "risk": risk,
                "n_kept": k,
            })

rc_df = pd.DataFrame(rc_rows)
rc_df.to_csv(f"{OUTDIR_DATA}/risk_coverage_by_subgroup.csv", index=False)
print("\nRisk-coverage by subgroup (coverage 1.0 → 0.5):")
print(rc_df.pivot_table(index=["subgroup", "method"], columns="coverage", values="risk")
      .round(3).to_string())
print(f"\nWrote: {OUTDIR_DATA}/risk_coverage_by_subgroup.csv\n")


# ─── 5. Reliability diagrams (figure) ───────────────────────────────────────

print("=" * 60)
print("5. Generating reliability diagrams")
print("=" * 60)

PLOT_GROUPS = [
    ("background", bg_mask, "Background (no identity)"),
    ("white_flag", (base["white_flag"] == 1).values, "white-flagged"),
    ("muslim_flag", (base["muslim_flag"] == 1).values, "muslim-flagged"),
    ("homosexual_gay_or_lesbian_flag",
        (base["homosexual_gay_or_lesbian_flag"] == 1).values, "gay/lesbian-flagged"),
]

fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), dpi=120)

for ax, (key, mask, title) in zip(axes, PLOT_GROUPS):
    # Diagonal (perfect calibration)
    ax.plot([0.5, 1.0], [0.5, 1.0], color="gray", linestyle="--", alpha=0.5, lw=1)

    for method, df in preds.items():
        p = df["prob"].values
        if mask.sum() < 30:
            continue
        centers, accs, confs, counts = reliability_curve(p[mask], y[mask])
        # Drop empty bins
        valid = ~np.isnan(accs)
        ax.plot(centers[valid], accs[valid],
                marker="o", color=METHOD_COLORS[method], label=method, lw=1.8, ms=5)

    ax.set_xlim(0.5, 1.02)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Stated confidence")
    if ax is axes[0]:
        ax.set_ylabel("Actual accuracy")
        ax.legend(loc="lower right", fontsize=9, frameon=False)
    ax.set_title(f"{title} (n={mask.sum()})", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.4)

plt.suptitle("Reliability diagrams: calibration is worse on identity-mentioning subgroups",
             y=1.02, fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTDIR_FIGS}/reliability_diagrams.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"Wrote: {OUTDIR_FIGS}/reliability_diagrams.png\n")


# ─── 6. Calibration-fairness gap figure ─────────────────────────────────────

print("=" * 60)
print("6. Generating calibration-fairness gap figure")
print("=" * 60)

# Grouped bar chart: x = subgroups, y = gap, hue = method, with CI bars
fig, ax = plt.subplots(figsize=(10, 5), dpi=120)

methods = list(PRED_PATHS.keys())
subgroups_with_data = [s for s in [SUBGROUP_LABELS[c] for c in SUBGROUPS]
                      if s in gap_df["subgroup"].unique()]

x = np.arange(len(subgroups_with_data))
width = 0.27

for i, method in enumerate(methods):
    method_data = gap_df[gap_df["method"] == method].set_index("subgroup")
    gaps = [method_data.loc[sg, "gap"] if sg in method_data.index else 0
            for sg in subgroups_with_data]
    ci_los = [method_data.loc[sg, "ci_lo"] if sg in method_data.index else 0
              for sg in subgroups_with_data]
    ci_his = [method_data.loc[sg, "ci_hi"] if sg in method_data.index else 0
              for sg in subgroups_with_data]
    err_lo = [g - lo for g, lo in zip(gaps, ci_los)]
    err_hi = [hi - g for g, hi in zip(gaps, ci_his)]

    ax.bar(x + (i - 1) * width, gaps, width,
           yerr=[err_lo, err_hi],
           color=METHOD_COLORS[method], label=method, capsize=3,
           edgecolor="white", linewidth=1)

ax.axhline(y=0, color="black", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(subgroups_with_data, rotation=20, ha="right")
ax.set_ylabel("ECE gap (subgroup − background)")
ax.set_title("Calibration is a fairness property: subgroups have systematically worse calibration",
             fontsize=12, fontweight="bold")
ax.legend(loc="best", frameon=False)
ax.grid(True, axis="y", linestyle=":", alpha=0.4)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(f"{OUTDIR_FIGS}/calibration_fairness_gap.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"Wrote: {OUTDIR_FIGS}/calibration_fairness_gap.png\n")


# ─── 7. Subgroup-conditional risk-coverage figure ───────────────────────────

print("=" * 60)
print("7. Generating subgroup-conditional risk-coverage figure")
print("=" * 60)

key_subgroup_labels = [SUBGROUP_LABELS[c] for c in KEY_SUBGROUPS]
fig, axes = plt.subplots(1, 4, figsize=(16, 4), dpi=120, sharey=True)

panels = ["background"] + key_subgroup_labels
for ax, panel in zip(axes, panels):
    panel_data = rc_df[rc_df["subgroup"] == panel]
    for method in methods:
        md = panel_data[panel_data["method"] == method].sort_values("coverage", ascending=False)
        if len(md) == 0:
            continue
        ax.plot(md["coverage"], md["risk"],
                marker="o", color=METHOD_COLORS[method], label=method, lw=2, ms=6)
    ax.set_xlabel("Coverage (fraction retained)")
    ax.set_title(panel, fontsize=11)
    ax.invert_xaxis()
    ax.grid(True, linestyle=":", alpha=0.4)
    if ax is axes[0]:
        ax.set_ylabel("Error rate (risk)")
        ax.legend(loc="best", frameon=False, fontsize=9)

plt.suptitle("Risk-coverage by subgroup: abstention works for some groups, not others",
             y=1.02, fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTDIR_FIGS}/risk_coverage_by_subgroup.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"Wrote: {OUTDIR_FIGS}/risk_coverage_by_subgroup.png\n")


print("=" * 60)
print("All analyses complete.")
print("=" * 60)
print("\nKey takeaways to look for:")
print("  1. Per-subgroup ECE — does it vary across identities for each method?")
print("  2. Calibration-fairness gap — significant for which (method, subgroup) pairs?")
print("  3. Accuracy at conf ≥ 0.99 — how does it differ between background and identity subgroups?")
print("  4. Risk-coverage by subgroup — does abstention work equally well across groups?")
print("\nFiles written:")
print(f"  Tables: {OUTDIR_DATA}/*.csv")
print(f"  Figures: {OUTDIR_FIGS}/reliability_diagrams.png")
print(f"           {OUTDIR_FIGS}/calibration_fairness_gap.png")
print(f"           {OUTDIR_FIGS}/risk_coverage_by_subgroup.png")