# FairGuard: Fair and Calibrated Toxicity Detection

**CS 517: Socially Responsible AI — University of Illinois Chicago**  
_Mokshit Surana · Richa Rameshkrishna_

---

## Overview

This project studies fairness in toxicity classification on the Civil Comments dataset. We argue that fairness involves three integrated axes — **ranking**, **calibration**, and **abstention** — and that training-time interventions and post-hoc safety mechanisms cannot be evaluated independently.

### Training Methods

- **ERM** — standard cross-entropy baseline
- **Reweighted ERM** — inverse-frequency group reweighting
- **Group DRO** — adaptive minimax group weighting

### Post-hoc Interventions

- **Temperature scaling** — post-hoc calibration
- **Confidence-based abstention** — selective prediction via risk-coverage curves
- **Per-identity threshold optimization** — per-group decision boundary tuning

**Key finding:** Methods equivalent on standard ranking metrics (BPSN AUC) differ sharply in calibration disparity and abstention efficacy. No method dominates across all three axes.

---

## Repository Structure

```

cs-517-project/
│
├── src/
│   ├── training/
│   │   ├── train_erm.py
│   │   ├── train_reweighted_erm.py
│   │   └── train_groupdro.py
│   │
│   ├── inference/
│   │   ├── predict_and_save.py
│   │   ├── predict_groupdro.py
│   │   └── predict_explain.py
│   │
│   ├── data/
│   │   ├── prep_data.py
│   │   └── preprocess_hateexplain.py
│   │
│   ├── evaluation/
│   │   ├── eval_baseline.py
│   │   ├── eval_fairness.py
│   │   ├── eval_xplain.py
│   │   ├── abstain_eval.py
│   │   ├── bootstrap_fairness.py
│   │   └── temp_scale_and_eval.py
│   │
│   ├── calibration/
│   │   ├── calibration.py
│   │   └── threshold_opt.py
│   │
│   └── visualization/
│       └── make_figures.py
│
│
├── figures/
│   ├── coverage_vs_confidence.png
│   ├── setsize_vs_confidence.png
│   └── example_images_90pct.png
│
│
├── pyproject.toml
├── README.md
```

---

## Setup

### Install Dependencies

```bash
pip install torch transformers datasets pandas numpy scikit-learn matplotlib seaborn pyarrow
```

Or with `uv`:

```bash
uv pip install torch transformers datasets pandas numpy scikit-learn matplotlib seaborn pyarrow
```

---

## Datasets

### Civil Comments (Jigsaw)

Download from:
[https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data)

Place:

```
train.csv → data/jigsaw/
```

### HateXplain

```bash
git clone https://github.com/hate-alert/HateXplain data/hatexplain
```

---

## Training

All models use:

- `distilbert-base-uncased`
- 2 epochs
- Batch size: 16
- Learning rate: 5e-5 (linear decay)

```bash
# ERM baseline
python train_erm.py

# Reweighted ERM
python train_reweighted_erm.py

# Group DRO
python train_groupdro.py
```

### Output Format

Predictions are saved to:

```
outputs/{method}/test_preds.parquet
```

Columns:

- `comment_text`
- `y`
- `group_id`
- `[identity]_flag` (23 identities)
- `logit0`, `logit1`
- `prob`

---

## Evaluation

### Bootstrap Fairness Metrics

```bash
python bootstrap_fairness.py
```

Outputs to:

```
outputs/bootstrap/
```

Includes:

- BPSN / BNSP
- Error gaps
- Paired bootstrap confidence intervals (n = 1000)

---

### Calibration Analysis

```bash
python calibration_analysis.py
```

Computes:

- Per-subgroup ECE
- Calibration-fairness gap (with bootstrap CIs)
- Confidence-binned accuracy
- Risk-coverage curves

Outputs to:

```
outputs/calibration_analysis/
figures/
```

---

### Generate Figures

```bash
python make_figures.py
```

---

## Key Results

| Method         | Overall AUC | Overall ECE | Best BPSN (white) | Calib. Gap (white) | Abstention |
| -------------- | ----------- | ----------- | ----------------- | ------------------ | ---------- |
| ERM            | 0.940       | 0.013       | 0.780             | +0.077             | Works      |
| Reweighted ERM | 0.926       | 0.042       | **0.884**         | +0.232             | Partial    |
| Group DRO      | 0.928       | 0.134       | 0.866             | +0.017 (n.s.)      | Broken     |

---

## Key Findings

- ERM is well-calibrated globally but miscalibrated on identity subgroups
- Reweighted ERM improves BPSN AUC (+0.06 to +0.12) but worsens calibration gaps (up to 3×)
- Group DRO removes calibration disparity by becoming uniformly miscalibrated
- Temperature scaling yields **T\* = 1.0** → miscalibration is structural
- Fairness methods introduce **1.5–4.3% high-confidence benign errors (p > 0.99)**
- No post-hoc method fixes these failures

---

## Acknowledgments

- Borkan et al. (WWW 2019) — Civil Comments dataset, BPSN/BNSP metrics
- Sagawa et al. (ICLR 2020) — Group DRO
- Idrissi et al. (CLeaR 2022) — Simple balancing baseline
- Guo et al. (ICML 2017) — Temperature scaling
- Geifman & El-Yaniv (NeurIPS 2017) — Selective prediction

```

```
