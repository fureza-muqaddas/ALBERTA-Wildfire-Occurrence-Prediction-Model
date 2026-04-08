"""
train.py

Trains an XGBoost binary classifier to predict wildfire occurrence at
weather-station level (fire / no-fire on a given day at a given location).

Temporal split:
  Train  : 2006 – 2018
  Val    : 2019          (used for early stopping)
  Test   : 2020 – 2021

Outputs (saved to outputs/):
  wildfire_xgb.json  — trained model
  evaluation.png     — ROC curve + precision-recall + feature importance
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)

os.makedirs("outputs", exist_ok=True)

# ── 1. Load dataset ───────────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv("data/processed/model_dataset.csv", parse_dates=["rep_date"])
print(f"  Shape: {df.shape}  |  Fire rate: {df['fire'].mean():.3%}")

# ── 2. Define features ────────────────────────────────────────────────────────
BASE_FEATURES = [
    "temp", "rh", "ws", "precip",
    "ffmc", "dmc", "dc", "isi", "bui", "fwi",
    "month", "day_of_year", "lat", "lon",
]
feature_cols = BASE_FEATURES + [c for c in ["land_cover"] if c in df.columns]

df = df.dropna(subset=feature_cols + ["fire", "year"])
X = df[feature_cols]
y = df["fire"]
years = df["year"]

# ── 3. Temporal split ─────────────────────────────────────────────────────────
train_mask = years <= 2018
val_mask   = years == 2019
test_mask  = years >= 2020

X_train, y_train = X[train_mask], y[train_mask]
X_val,   y_val   = X[val_mask],   y[val_mask]
X_test,  y_test  = X[test_mask],  y[test_mask]

print(f"  Train: {train_mask.sum():,}  |  Val: {val_mask.sum():,}  |  Test: {test_mask.sum():,}")

if X_val.empty:
    raise ValueError("Validation set is empty — check that year 2019 exists in the dataset.")
if X_test.empty:
    raise ValueError("Test set is empty — check that years 2020-2021 exist in the dataset.")

# ── 4. Class-imbalance weight ─────────────────────────────────────────────────
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
scale_pos_weight = neg_count / pos_count
print(f"  scale_pos_weight: {scale_pos_weight:.1f}  ({neg_count:,} neg / {pos_count:,} pos)")

# ── 5. Train ──────────────────────────────────────────────────────────────────
model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=10,
    scale_pos_weight=scale_pos_weight,
    eval_metric="auc",
    early_stopping_rounds=20,
    random_state=42,
    n_jobs=-1,
)

print("\nTraining XGBoost...")
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=50,
)

model.save_model("outputs/wildfire_xgb.json")
print(f"Model saved → outputs/wildfire_xgb.json  (best iteration: {model.best_iteration})")

# ── 6. Evaluate ───────────────────────────────────────────────────────────────
print("\n" + "─" * 50)
for split_name, X_e, y_e in [("Val (2019)", X_val, y_val), ("Test (2020-21)", X_test, y_test)]:
    proba = model.predict_proba(X_e)[:, 1]
    pred  = (proba >= 0.5).astype(int)
    auc   = roc_auc_score(y_e, proba)
    ap    = average_precision_score(y_e, proba)
    print(f"\n{split_name}")
    print(f"  ROC-AUC: {auc:.4f}  |  Avg Precision: {ap:.4f}")
    print(classification_report(y_e, pred, target_names=["No Fire", "Fire"], digits=3))

# ── 7. Plots ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

test_proba = model.predict_proba(X_test)[:, 1]

RocCurveDisplay.from_predictions(y_test, test_proba, ax=axes[0])
axes[0].set_title("ROC Curve — Test (2020-21)")

PrecisionRecallDisplay.from_predictions(y_test, test_proba, ax=axes[1])
axes[1].set_title("Precision-Recall — Test (2020-21)")

xgb.plot_importance(model, ax=axes[2], max_num_features=14, importance_type="gain")
axes[2].set_title("Feature Importance (Gain)")

plt.tight_layout()
plt.savefig("outputs/evaluation.png", dpi=150)
print("\nPlot saved → outputs/evaluation.png")
