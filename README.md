
# FairGuard: Fair and Calibrated Toxicity Detection

**CS 517: Socially Responsible AI - University of Illinois Chicago**  
*Mokshit Surana · Richa Rameshkrishna*

---

## Overview

This project studies fairness in toxicity classification on the Civil Comments dataset. We argue that fairness involves three integrated axes - **ranking**, **calibration**, and **abstention** - and that training-time interventions and post-hoc safety mechanisms cannot be evaluated independently.

### Training Methods

- **ERM** - standard cross-entropy baseline  
- **Reweighted ERM** - inverse-frequency group reweighting  
- **Group DRO** - adaptive minimax group weighting  

### Post-hoc Interventions

- **Temperature scaling** - post-hoc calibration  
- **Confidence-based abstention** - selective prediction via risk-coverage curves  
- **Per-identity threshold optimization** - per-group decision boundary tuning  

**Key finding:** Methods equivalent on standard ranking metrics (BPSN AUC) differ sharply in calibration disparity and abstention efficacy. No method dominates across all three axes.

---

## Repository Structure

```

cs-517-project/
│
├── src/
│   ├── training/
│   ├── inference/
│   ├── data/
│   ├── evaluation/
│   ├── calibration/
│   └── visualization/
│
├── figures/
│
├── pyproject.toml
└── README.md

````

---

## Setup

### Install Dependencies

```bash
pip install torch transformers datasets pandas numpy scikit-learn matplotlib seaborn pyarrow
````

Or with `uv`:

```bash
uv pip install torch transformers datasets pandas numpy scikit-learn matplotlib seaborn pyarrow
```

---

## Data & Reproducibility

⚠️ **Note:** Due to size constraints, datasets, trained models, and intermediate outputs are **not included** in this repository.

### Datasets

* **Civil Comments (Jigsaw Unintended Bias)**
  [https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data)

* **HateXplain**
  [https://github.com/hate-alert/HateXplain](https://github.com/hate-alert/HateXplain)

### Download & Setup

```bash
# Create directories
mkdir -p data/jigsaw data/hatexplain

# Clone HateXplain
git clone https://github.com/hate-alert/HateXplain data/hatexplain
```

Download Civil Comments from Kaggle and place:

```
train.csv → data/jigsaw/
```

---

## Data Preprocessing

```bash
python src/data/prep_data.py
python src/data/preprocess_hateexplain.py
```

---

## Training

All models use:

* `distilbert-base-uncased`
* 2 epochs
* Batch size: 16
* Learning rate: 5e-5 (linear decay)

```bash
# ERM
python src/training/train_erm.py

# Reweighted ERM
python src/training/train_reweighted_erm.py

# Group DRO
python src/training/train_groupdro.py
```

---

## Inference

```bash
python src/inference/predict_and_save.py
python src/inference/predict_groupdro.py
python src/inference/predict_explain.py
```

Outputs are saved to:

```
outputs/{method}/test_preds.parquet
```

---

## Evaluation

### Fairness Metrics

```bash
python src/evaluation/eval_fairness.py
```

### Bootstrap Confidence Intervals

```bash
python src/evaluation/bootstrap_fairness.py
```

### Calibration

```bash
python src/calibration/calibration.py
python src/calibration/threshold_opt.py
```

### Abstention (Risk-Coverage)

```bash
python src/evaluation/abstain_eval.py
```

---

## Visualization

```bash
python src/visualization/make_figures.py
```

Figures will be saved to:

```
figures/
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

* ERM is well-calibrated globally but miscalibrated on identity subgroups
* Reweighted ERM improves BPSN AUC (+0.06 to +0.12) but worsens calibration gaps (up to 3×)
* Group DRO removes calibration disparity by becoming uniformly miscalibrated
* Temperature scaling yields **T* = 1.0** → miscalibration is structural
* Fairness methods introduce **1.5–4.3% high-confidence benign errors (p > 0.99)**
* No post-hoc method fixes these failures

---

## Notes

* All intermediate outputs (predictions, calibration tables, bootstrap results) are written to `outputs/`
* Large files are excluded via `.gitignore`
* Results are fully reproducible using the provided scripts

---

## Acknowledgments

* Borkan et al. (WWW 2019) - Civil Comments dataset, BPSN/BNSP metrics
* Sagawa et al. (ICLR 2020) - Group DRO
* Idrissi et al. (CLeaR 2022) - Simple balancing baseline
* Guo et al. (ICML 2017) - Temperature scaling
* Geifman & El-Yaniv (NeurIPS 2017) - Selective prediction

---

```

---

If you want one last upgrade: add a **“Quick Start (3 commands)”** at the top - that’s what actually gets people to run your repo.
```
