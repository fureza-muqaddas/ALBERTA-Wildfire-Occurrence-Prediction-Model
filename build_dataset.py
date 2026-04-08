"""
build_dataset.py

Joins FWI weather-station data with historical wildfire records to produce a
labeled, ML-ready CSV for binary fire-occurrence prediction.

Output: data/processed/model_dataset.csv
Columns: rep_date, name, lat, lon, temp, rh, ws, precip,
         ffmc, dmc, dc, isi, bui, fwi,
         month, day_of_year, year, [land_cover], fire (0/1)
"""

import os
import glob
import zipfile

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

os.makedirs("data/processed", exist_ok=True)

# -- 1. Load FWI data ----------------------------------------------------------
print("Loading FWI data...")
fwi = pd.read_csv("data/raw/fwi_alberta_2000_2025.csv", parse_dates=["rep_date"])
fwi["rep_date"] = fwi["rep_date"].dt.normalize()        # drop time component
fwi["name"] = fwi["name"].str.strip()

# Keep fire season (Apr-Oct) and rows where FWI indices were actually computed
fwi = fwi[fwi["rep_date"].dt.month.between(4, 10)]
fwi = fwi.dropna(subset=["fwi"])
print(f"  Fire-season, non-null FWI records: {len(fwi):,}")
print(f"  Date range: {fwi['rep_date'].min().date()} to {fwi['rep_date'].max().date()}")

# -- 2. Load wildfire data ----------------------------------------------------─
print("\nLoading wildfire data...")
fires = pd.read_csv("data/raw/fp-historical-wildfire-data-2006-2025 (1).csv")
fires["FIRE_START_DATE"] = pd.to_datetime(fires["FIRE_START_DATE"], errors="coerce")
fires = fires.dropna(subset=["LATITUDE", "LONGITUDE", "FIRE_START_DATE"])
fires["rep_date"] = fires["FIRE_START_DATE"].dt.normalize()

# Drop prescribed fires and rows outside the FWI temporal coverage
fires = fires[fires["GENERAL_CAUSE"] != "Prescribed Fire"]
fwi_start, fwi_end = fwi["rep_date"].min(), fwi["rep_date"].max()
fires = fires[(fires["rep_date"] >= fwi_start) & (fires["rep_date"] <= fwi_end)]
print(f"  Fire records within FWI coverage ({fwi_start.date()}-{fwi_end.date()}): {len(fires):,}")

# -- 3 & 4. Positive samples: nearest station WITH data on fire date (+/-1 day) --
# The problem with assigning nearest station first is that it often has no data
# on the fire date. Instead, for each fire date we search only among stations
# that actually reported on that date, then pick the closest one geographically.
print("\nSpatial join: fires -> nearest reporting station on fire date (+/-1 day)...")

FWI_COLS = ["rep_date", "name", "lat", "lon",
            "temp", "rh", "ws", "precip",
            "ffmc", "dmc", "dc", "isi", "bui", "fwi"]

one_day = pd.Timedelta(days=1)

# Group FWI records by date for fast lookup
fwi_by_date = {date: grp for date, grp in fwi.groupby("rep_date")}

matched = []
for fire_date, fire_group in fires.groupby("rep_date"):
    # Find FWI records for this date or adjacent days (prefer exact, then +/-1)
    fwi_day = None
    for offset in [pd.Timedelta(0), -one_day, one_day]:
        candidate_date = fire_date + offset
        if candidate_date in fwi_by_date:
            fwi_day = fwi_by_date[candidate_date]
            break
    if fwi_day is None:
        continue

    # For this date's active stations, find the nearest one to each fire
    station_rad = np.radians(fwi_day[["lat", "lon"]].values)
    fire_rad    = np.radians(fire_group[["LATITUDE", "LONGITUDE"]].values)
    tree = cKDTree(station_rad)
    dist_rad, idx = tree.query(fire_rad, k=1)
    dist_km = dist_rad * 6371.0

    # Only accept matches within 150 km
    mask = dist_km <= 150
    if not mask.any():
        continue

    rows = fwi_day.iloc[idx[mask]][FWI_COLS].copy()
    rows["rep_date"] = fire_date          # use fire date, not fwi date
    rows["station_dist_km"] = dist_km[mask]
    matched.append(rows)

positive = pd.concat(matched, ignore_index=True)[FWI_COLS].drop_duplicates()
positive["fire"] = 1
print(f"  Positive samples matched: {len(positive):,}")

# -- 5. Negative samples: station-days with no nearby fire --------------------─
pos_keys = positive[["name", "rep_date"]].drop_duplicates()
pos_keys["fire"] = 1

all_fwi = fwi[FWI_COLS].copy()
all_fwi = all_fwi.merge(pos_keys, on=["name", "rep_date"], how="left")
all_fwi["fire"] = all_fwi["fire"].fillna(0).astype(int)

negative = all_fwi[all_fwi["fire"] == 0].copy()
print(f"Negative samples (non-fire station-days): {len(negative):,}")
print(f"Class imbalance ratio: 1 : {len(negative) / max(len(positive), 1):.0f}")

# -- 6. Combine and add temporal features ------------------------------------─
dataset = pd.concat([positive, negative], ignore_index=True)
dataset["month"]      = dataset["rep_date"].dt.month
dataset["day_of_year"] = dataset["rep_date"].dt.dayofyear
dataset["year"]       = dataset["rep_date"].dt.year

# -- 7. Land cover (optional — requires a valid raster) ----------------------─
print("\nAttempting land cover sampling...")
lc_zip = "data/raw/AlbertaLandCover2020.zip"
lc_dir = "data/raw/AlbertaLandCover2020"

# Try to extract (skip silently if the zip is corrupt / not yet downloaded)
if not os.path.isdir(lc_dir):
    try:
        with zipfile.ZipFile(lc_zip, "r") as z:
            z.extractall(lc_dir)
        print("  Extracted land cover zip.")
    except zipfile.BadZipFile:
        print("  WARNING: AlbertaLandCover2020.zip is not a valid zip file.")
        print("  Re-run climate.py to download it, then re-run this script.")

tif_files = (
    glob.glob(f"{lc_dir}/**/*.tif", recursive=True)
    + glob.glob(f"{lc_dir}/**/*.img", recursive=True)
)

if tif_files:
    import rasterio
    from rasterio.warp import transform as warp_transform

    raster_path = tif_files[0]
    print(f"  Raster: {raster_path}")

    unique_st = dataset[["name", "lat", "lon"]].drop_duplicates("name")

    with rasterio.open(raster_path) as src:
        # Re-project coordinates to raster CRS if needed
        if src.crs and src.crs.to_epsg() != 4326:
            xs, ys = warp_transform(
                "EPSG:4326", src.crs,
                unique_st["lon"].tolist(),
                unique_st["lat"].tolist(),
            )
        else:
            xs = unique_st["lon"].tolist()
            ys = unique_st["lat"].tolist()

        lc_vals = [v[0] for v in src.sample(zip(xs, ys))]

    unique_st = unique_st.copy()
    unique_st["land_cover"] = lc_vals
    dataset = dataset.merge(unique_st[["name", "land_cover"]], on="name", how="left")
    print(f"  Land cover classes found: {dataset['land_cover'].nunique()}")
else:
    print("  No raster found — land_cover column will not be added.")

# -- 8. Save ------------------------------------------------------------------─
out_path = "data/processed/model_dataset.csv"
dataset.to_csv(out_path, index=False)

print(f"\n{'-'*50}")
print(f"Dataset saved to: {out_path}")
print(f"Shape:            {dataset.shape}")
print(f"Fire / No-fire:   {dataset['fire'].sum():,} / {(dataset['fire']==0).sum():,}")
print(f"Year range:       {dataset['year'].min()} - {dataset['year'].max()}")
print(f"Columns:          {list(dataset.columns)}")
