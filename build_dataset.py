"""
build_dataset.py

Joins FWI weather-station data with historical wildfire records to produce a
labeled, ML-ready CSV for binary fire-occurrence prediction.

Output: data/processed/model_dataset.csv
Columns: rep_date, name, lat, lon, temp, rh, ws, precip,
         ffmc, dmc, dc, isi, bui, fwi,
         month, day_of_year, year, [land_cover],
         [elevation, slope, aspect], fire (0/1)
"""

import os

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

# -- 7. Land cover sampling (AB_LULC_2020, EPSG:3400) ------------------------
# 14-class Sentinel-2 land cover: 1=Water 2=Bryoids 3=Wetland-Treed 4=Herbs
# 5=Exposed/Barren 6=Shrubland 7=Wetland 8=Grassland 9=Coniferous
# 10=Broadleaf 11=Mixedwood 12=Agriculture 13=Developed 14=Burned Areas
print("\nSampling land cover (AB_LULC_2020_10tm.tif)...")
import rasterio
from rasterio.warp import transform as warp_transform

raster_path = "data/raw/AB_LULC_2020_10tm.tif"

# Sample at each unique station location (stations are fixed points)
unique_st = dataset[["name", "lat", "lon"]].drop_duplicates("name").copy()

with rasterio.open(raster_path) as src:
    # Reproject station coords from WGS84 (EPSG:4326) to Alberta 10TM (EPSG:3400)
    xs, ys = warp_transform(
        "EPSG:4326", src.crs,
        unique_st["lon"].tolist(),
        unique_st["lat"].tolist(),
    )
    lc_vals = [v[0] for v in src.sample(zip(xs, ys))]

unique_st["land_cover"] = lc_vals
# 0 = outside raster extent — treat as missing
unique_st.loc[unique_st["land_cover"] == 0, "land_cover"] = None

dataset = dataset.merge(unique_st[["name", "land_cover"]], on="name", how="left")
# Fill stations outside raster extent with 0 (unknown) so no rows are dropped
dataset["land_cover"] = dataset["land_cover"].fillna(0).astype(int)
print(f"  Stations sampled: {unique_st['name'].nunique()}")
print(f"  Land cover classes found: {sorted(dataset['land_cover'].unique().tolist())}")
print(f"  Rows with land_cover=0 (outside extent): {(dataset['land_cover']==0).sum():,}")

# -- 8. Topography sampling (SRTM 90m, WGS84) ---------------------------------
# Elevation (m), slope (deg), aspect (deg 0=N clockwise; -1=flat)
# The topo raster is already in WGS84 so we sample with raw lon/lat.
topo_path = "data/raw/srtm_alberta_topo.tif"
if os.path.exists(topo_path):
    print("\nSampling topography (elevation / slope / aspect)...")
    unique_topo = dataset[["name", "lat", "lon"]].drop_duplicates("name").copy()

    with rasterio.open(topo_path) as src:
        coords = list(zip(unique_topo["lon"], unique_topo["lat"]))
        # src.sample yields (b1, b2, b3) per coordinate
        vals = np.array(list(src.sample(coords)), dtype=np.float32)

    unique_topo["elevation"] = vals[:, 0]
    unique_topo["slope"]     = vals[:, 1]
    unique_topo["aspect"]    = vals[:, 2]

    # Replace NaN-encoded nodata (the raster nodata is np.nan for float32)
    for col in ["elevation", "slope", "aspect"]:
        unique_topo.loc[~np.isfinite(unique_topo[col]), col] = np.nan

    dataset = dataset.merge(unique_topo[["name", "elevation", "slope", "aspect"]],
                            on="name", how="left")

    # Fill nulls (stations outside raster extent) with column median so no
    # rows are dropped; XGBoost will also handle these gracefully at inference.
    for col in ["elevation", "slope", "aspect"]:
        n_null = dataset[col].isna().sum()
        if n_null > 0:
            med = dataset[col].median()
            dataset[col] = dataset[col].fillna(med)
            print(f"  {col}: filled {n_null:,} nulls with median ({med:.1f})")

    print(f"  Elevation range : {dataset['elevation'].min():.0f} – "
          f"{dataset['elevation'].max():.0f} m")
    print(f"  Slope range     : {dataset['slope'].min():.1f} – "
          f"{dataset['slope'].max():.1f} deg")
    print(f"  Stations with topo data: "
          f"{(unique_topo['elevation'].notna()).sum()} / {len(unique_topo)}")
else:
    print(f"\n[WARN] {topo_path} not found — skipping topo features.")
    print("       Run: python download_dem.py  to generate it.")

# -- 9. Save ------------------------------------------------------------------─
out_path = "data/processed/model_dataset.csv"
dataset.to_csv(out_path, index=False)

print(f"\n{'-'*50}")
print(f"Dataset saved to: {out_path}")
print(f"Shape:            {dataset.shape}")
print(f"Fire / No-fire:   {dataset['fire'].sum():,} / {(dataset['fire']==0).sum():,}")
print(f"Year range:       {dataset['year'].min()} - {dataset['year'].max()}")
print(f"Columns:          {list(dataset.columns)}")
