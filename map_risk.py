"""
map_risk.py

Plots predicted fire-risk scores for all active weather stations on a chosen
high-risk day, overlaid on an SRTM hillshade of Alberta.

Output: outputs/risk_map_<date>.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import rasterio
import xgboost as xgb

os.makedirs("outputs", exist_ok=True)

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
TARGET_DATE = "2016-05-03"   # Fort McMurray fire start — 245 stations, FWI max 65
TOPO_PATH   = "data/raw/srtm_alberta_topo.tif"
MODEL_PATH  = "outputs/wildfire_xgb.json"
DATA_PATH   = "data/processed/model_dataset.csv"
BBOX        = (-120.0, 49.0, -110.0, 60.0)   # lon_min, lat_min, lon_max, lat_max

# Cities for orientation
CITIES = {
    "Edmonton":      (-113.49, 53.54),
    "Calgary":       (-114.07, 51.05),
    "Fort McMurray": (-111.38, 56.73),
    "Grande Prairie":(-118.80, 55.17),
    "Lethbridge":    (-112.83, 49.70),
    "Medicine Hat":  (-110.67, 50.04),
}

BASE_FEATURES = [
    "lat", "lon",
    "temp", "rh", "ws", "precip",
    "ffmc", "dmc", "dc", "isi", "bui", "fwi",
    "month", "day_of_year",
]

# --------------------------------------------------------------------------- #
# 1. Load dataset and filter to target date
# --------------------------------------------------------------------------- #
print(f"Loading dataset for {TARGET_DATE} ...")
df = pd.read_csv(DATA_PATH, parse_dates=["rep_date"])

# Build feature list in the same order as train.py
FEATURE_COLS = BASE_FEATURES + [c for c in ["land_cover"] if c in df.columns]

if "elevation" in df.columns:
    FEATURE_COLS += ["elevation", "slope"]
if "aspect" in df.columns:
    df["aspect"] = df["aspect"].where(df["aspect"] >= 0, other=np.nan)
    df["aspect_sin"] = np.sin(np.radians(df["aspect"]))
    df["aspect_cos"] = np.cos(np.radians(df["aspect"]))
    FEATURE_COLS += ["aspect_sin", "aspect_cos"]

day_df = df[df["rep_date"] == TARGET_DATE].copy()
print(f"  Stations on this day : {len(day_df)}")
print(f"  Confirmed fires      : {day_df['fire'].sum()}")
print(f"  FWI mean / max       : {day_df['fwi'].mean():.1f} / {day_df['fwi'].max():.1f}")

# --------------------------------------------------------------------------- #
# 2. Predict fire-risk probability for every station
# --------------------------------------------------------------------------- #
print("Predicting risk scores ...")
model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)

X_day = day_df[FEATURE_COLS]
day_df["risk"] = model.predict_proba(X_day)[:, 1]
print(f"  Risk range: {day_df['risk'].min():.3f} – {day_df['risk'].max():.3f}")

# --------------------------------------------------------------------------- #
# 3. Build hillshade from SRTM elevation band
# --------------------------------------------------------------------------- #
print("Rendering hillshade ...")
with rasterio.open(TOPO_PATH) as src:
    elev  = src.read(1).astype(np.float64)   # band 1 = elevation
    tf    = src.transform
    extent = [tf.c, tf.c + tf.a * src.width,
              tf.f + tf.e * src.height, tf.f]   # left, right, bottom, top

# Compute hillshade (Azimuth 315°, Altitude 45°)
az_rad  = np.radians(315)
alt_rad = np.radians(45)

dy_m = abs(tf.e) * 111_320.0
dx_m = abs(tf.a) * 111_320.0 * np.cos(np.radians(tf.f + tf.e * elev.shape[0] / 2))

dz_dy, dz_dx = np.gradient(elev, dy_m, dx_m)
# Normal vector components
nx = -dz_dx
ny =  dz_dy
nz = np.ones_like(elev)
norm = np.sqrt(nx**2 + ny**2 + nz**2)
nx, ny, nz = nx / norm, ny / norm, nz / norm

# Light direction
lx = np.cos(alt_rad) * np.cos(az_rad)
ly = np.cos(alt_rad) * np.sin(az_rad)
lz = np.sin(alt_rad)

hillshade = np.clip(nx * lx + ny * ly + nz * lz, 0, 1)

# Blend elevation tint with hillshade for terrain colour
elev_norm = (elev - np.nanmin(elev)) / (np.nanmax(elev) - np.nanmin(elev) + 1e-9)
# Greenish-brown terrain palette: low=green, mid=tan, high=white/grey
terrain_r = 0.45 + 0.40 * elev_norm
terrain_g = 0.55 + 0.25 * elev_norm
terrain_b = 0.35 + 0.45 * elev_norm
shade_factor = 0.6 + 0.4 * hillshade    # mix: 60% base colour, 40% shade
terrain_rgb = np.stack([
    np.clip(terrain_r * shade_factor, 0, 1),
    np.clip(terrain_g * shade_factor, 0, 1),
    np.clip(terrain_b * shade_factor, 0, 1),
], axis=-1)

# --------------------------------------------------------------------------- #
# 4. Plot
# --------------------------------------------------------------------------- #
fig, ax = plt.subplots(figsize=(10, 12))

# Terrain background
ax.imshow(terrain_rgb, extent=extent, origin="upper", aspect="auto", zorder=0)

# Station risk scores — size scales with FWI, colour with risk probability
risk     = day_df["risk"].values
fwi_size = np.clip(day_df["fwi"].values, 5, 80)
sizes    = 20 + 120 * (fwi_size / 80) ** 1.5

cmap = plt.cm.YlOrRd
norm = mcolors.Normalize(vmin=0, vmax=1)

sc = ax.scatter(
    day_df["lon"], day_df["lat"],
    c=risk, cmap=cmap, norm=norm,
    s=sizes, edgecolors="k", linewidths=0.4,
    zorder=3, label="_nolegend_",
)

# Confirmed fire stations — ring overlay
fire_mask = day_df["fire"] == 1
ax.scatter(
    day_df.loc[fire_mask, "lon"],
    day_df.loc[fire_mask, "lat"],
    s=sizes[fire_mask] + 60,
    facecolors="none", edgecolors="cyan",
    linewidths=1.8, zorder=4,
)
ax.scatter(
    day_df.loc[fire_mask, "lon"],
    day_df.loc[fire_mask, "lat"],
    marker="*", s=120,
    facecolors="cyan", edgecolors="k", linewidths=0.5,
    zorder=5,
)

# City labels
for city, (clon, clat) in CITIES.items():
    ax.plot(clon, clat, "o", ms=4, color="white", mec="k", mew=0.8, zorder=6)
    ax.text(clon + 0.15, clat, city, fontsize=7.5, color="white",
            fontweight="bold", va="center", zorder=6,
            path_effects=[pe.withStroke(linewidth=2, foreground="black")])

# Colourbar
cbar = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.02, aspect=30)
cbar.set_label("Predicted Fire Probability", fontsize=10)
cbar.ax.tick_params(labelsize=8)

# Legend patches
legend_elems = [
    mpatches.Patch(facecolor="none", edgecolor="none", label="Circle size ∝ FWI"),
    plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
               markeredgecolor="cyan", markersize=10, markeredgewidth=1.8,
               label="Confirmed fire"),
    plt.Line2D([0], [0], marker="*", color="none", markerfacecolor="cyan",
               markeredgecolor="k", markersize=9, label="Fire station"),
]
ax.legend(handles=legend_elems, loc="lower left", fontsize=8,
          framealpha=0.75, facecolor="#222222", labelcolor="white")

# Bounding box / axes
ax.set_xlim(BBOX[0], BBOX[2])
ax.set_ylim(BBOX[1], BBOX[3])
ax.set_xlabel("Longitude", fontsize=9)
ax.set_ylabel("Latitude", fontsize=9)
ax.tick_params(labelsize=8)

n_stations = len(day_df)
n_fires    = int(day_df["fire"].sum())
max_risk   = day_df["risk"].max()
max_fwi    = day_df["fwi"].max()

ax.set_title(
    f"Alberta Wildfire Risk — {TARGET_DATE}\n"
    f"{n_stations} active stations  |  {n_fires} confirmed fires  |"
    f"  Max FWI {max_fwi:.0f}  |  Max predicted risk {max_risk:.2f}",
    fontsize=11, pad=10,
)

plt.tight_layout()
out_path = f"outputs/risk_map_{TARGET_DATE}.png"
plt.savefig(out_path, dpi=180, bbox_inches="tight")
plt.close()
print(f"\nMap saved -> {out_path}")
