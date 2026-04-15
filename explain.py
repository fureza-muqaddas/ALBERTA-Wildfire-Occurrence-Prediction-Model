"""
explain.py

SHAP feature importance analysis for the trained XGBoost wildfire model.

Outputs (saved to outputs/):
  shap_summary.png      -- beeswarm plot (feature impact distribution)
  shap_bar.png          -- mean |SHAP| bar chart
  shap_dependence.png   -- dependence plots for top 4 features
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import shap

os.makedirs("outputs", exist_ok=True)

# ── 1. Load data and model ────────────────────────────────────────────────────
print("Loading data and model...")
df = pd.read_csv("data/processed/model_dataset.csv", parse_dates=["rep_date"])

BASE_FEATURES = [
    "lat", "lon",
    "temp", "rh", "ws", "precip",
    "ffmc", "dmc", "dc", "isi", "bui", "fwi",
    "month", "day_of_year",
]
feature_cols = BASE_FEATURES + [c for c in ["land_cover"] if c in df.columns]

if "elevation" in df.columns:
    feature_cols += ["elevation", "slope"]
if "aspect" in df.columns:
    df["aspect"] = df["aspect"].where(df["aspect"] >= 0, other=np.nan)
    df["aspect_sin"] = np.sin(np.radians(df["aspect"]))
    df["aspect_cos"] = np.cos(np.radians(df["aspect"]))
    feature_cols += ["aspect_sin", "aspect_cos"]

df = df.dropna(subset=BASE_FEATURES + ["fire", "year"])

X = df[feature_cols]
y = df["fire"]

model = xgb.XGBClassifier()
model.load_model("outputs/wildfire_xgb.json")

# ── 2. Sample for SHAP (full dataset is slow; stratified sample keeps balance)
print("Sampling data for SHAP...")
fire_rows    = df[df["fire"] == 1]
nofire_rows  = df[df["fire"] == 0].sample(n=min(5000, (df["fire"]==0).sum()), random_state=42)
sample       = pd.concat([fire_rows, nofire_rows]).sample(frac=1, random_state=42)
X_sample     = sample[feature_cols]
print(f"  Sample size: {len(X_sample):,}  (all {len(fire_rows):,} fires + 5,000 non-fires)")

# ── 3. Compute SHAP values ────────────────────────────────────────────────────
print("Computing SHAP values (TreeExplainer)...")
explainer   = shap.TreeExplainer(model)
shap_values = explainer(X_sample)
print("  Done.")

# ── 4. Summary beeswarm plot ──────────────────────────────────────────────────
print("Saving shap_summary.png...")
fig, ax = plt.subplots(figsize=(10, 7))
shap.plots.beeswarm(shap_values, max_display=19, show=False)
plt.title("SHAP Summary — Wildfire Occurrence Model", fontsize=13)
plt.tight_layout()
plt.savefig("outputs/shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()

# ── 5. Mean |SHAP| bar chart ──────────────────────────────────────────────────
print("Saving shap_bar.png...")
fig, ax = plt.subplots(figsize=(8, 6))
shap.plots.bar(shap_values, max_display=19, show=False)
plt.title("Mean |SHAP| Feature Importance", fontsize=13)
plt.tight_layout()
plt.savefig("outputs/shap_bar.png", dpi=150, bbox_inches="tight")
plt.close()

# ── 6. Dependence plots for top 4 features ────────────────────────────────────
print("Saving shap_dependence.png...")
mean_shap   = np.abs(shap_values.values).mean(axis=0)
top4        = [feature_cols[i] for i in np.argsort(mean_shap)[::-1][:4]]

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
for ax, feat in zip(axes.flatten(), top4):
    shap.plots.scatter(shap_values[:, feat], ax=ax, show=False)
    ax.set_title(f"SHAP dependence: {feat}", fontsize=11)

plt.suptitle("SHAP Dependence Plots — Top 4 Features", fontsize=13)
plt.tight_layout()
plt.savefig("outputs/shap_dependence.png", dpi=150, bbox_inches="tight")
plt.close()

print("\nDone. Outputs:")
print("  outputs/shap_summary.png")
print("  outputs/shap_bar.png")
print("  outputs/shap_dependence.png")
