import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import log_loss
import math

# to these
VAL_PATH  = "outputs/groupdro/val_preds.parquet"
TEST_PATH = "outputs/groupdro/test_preds.parquet"
def softmax(logits):
    x = logits - np.max(logits, axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)

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

# simple grid search for temperature (good enough + easy)
def fit_temperature(logits, y):
    temps = np.linspace(0.5, 5.0, 91)  # 0.5..5.0 step 0.05
    best_t, best_ll = None, 1e18
    for t in temps:
        probs = softmax(logits / t)[:, 1]
        ll = log_loss(y, probs, labels=[0,1])
        if ll < best_ll:
            best_ll, best_t = ll, t
    return float(best_t), float(best_ll)

val = pq.read_table(VAL_PATH).to_pandas()
test = pq.read_table(TEST_PATH).to_pandas()

val_logits = val[["logit0","logit1"]].values
val_y = val["y"].values

t, ll = fit_temperature(val_logits, val_y)
print("Best temperature:", t, "val logloss:", ll)

# before calibration
val_prob = val["prob"].values
print("VAL ECE before:", round(ece(val_y, val_prob), 4))

# after calibration
val_prob_cal = softmax(val_logits / t)[:, 1]
print("VAL ECE after :", round(ece(val_y, val_prob_cal), 4))

# apply to test
test_logits = test[["logit0","logit1"]].values
test_y = test["y"].values
test_prob = test["prob"].values
test_prob_cal = softmax(test_logits / t)[:, 1]
print("TEST ECE before:", round(ece(test_y, test_prob), 4))
print("TEST ECE after :", round(ece(test_y, test_prob_cal), 4))