# Data Directory

All raw files are gitignored and must be downloaded manually. This document provides exact sources, filenames, and reproduction steps for every file in `data/raw/`.

---

## Directory layout

```
data/
├── raw/
│   ├── cwfis_fwi2000sv3.0_ll.csv          # CWFIS FWI decade 2000–2009 (521 MB)
│   ├── cwfis_fwi2010sv3.0_ll.csv          # CWFIS FWI decade 2010–2019 (738 MB)
│   ├── cwfis_fwi2020sv3.0_ll.csv          # CWFIS FWI decade 2020–    (143 MB, growing)
│   ├── fp-historical-wildfire-data-2006-2025.csv   # AB fire perimeters (9 MB)
│   ├── AB_LULC_2020_10tm.tif              # Alberta land cover raster (8.7 GB)
│   ├── AB_LULC_2020_10tm.hdr              # Accompanying header
│   ├── fwi_alberta_2000_2025.csv          # Output of preprocess.py — do not download
│   ├── srtm_alberta_topo.tif              # Output of download_dem.py — do not download
│   └── srtm_tiles/                        # Intermediate SRTM tiles — do not download
│       └── srtm_{13,14}_{01..04}.tif
└── processed/
    └── model_dataset.csv                  # Output of build_dataset.py — do not download
```

---

## Source 1 — CWFIS Fire Weather Index (FWI) Data

**What it is:** Daily fire weather observations (temperature, RH, wind, precipitation, and the six FWI indices) from ~5,000 Canadian weather stations, distributed as decade-chunked CSVs. Alberta stations fall within lat 49–60°N, lon −120–−110°W.

**Format:** CSV, 23 columns, one row per station per day.

| Column | Description |
|---|---|
| `rep_date` | Observation date (YYYY-MM-DD) |
| `name` | Station name |
| `lat`, `lon` | WGS84 coordinates |
| `temp` | Temperature (°C) |
| `rh` | Relative humidity (%) |
| `ws` | Wind speed (km/h) |
| `precip` | 24-hour precipitation (mm) |
| `ffmc` | Fine Fuel Moisture Code |
| `dmc` | Duff Moisture Code |
| `dc` | Drought Code |
| `isi` | Initial Spread Index |
| `bui` | Buildup Index |
| `fwi` | Fire Weather Index |

**Download:**
1. Go to [https://cwfis.cfs.nrcan.gc.ca/datamart/download/fwi](https://cwfis.cfs.nrcan.gc.ca/datamart/download/fwi)
2. Select product **"FWI Observations (CSV)"**, version **3.0**, format **"lon/lat"**
3. Download each decade separately and save as:

```
data/raw/cwfis_fwi2000sv3.0_ll.csv   # 2000–2009
data/raw/cwfis_fwi2010sv3.0_ll.csv   # 2010–2019
data/raw/cwfis_fwi2020sv3.0_ll.csv   # 2020–present
```

**Approximate sizes:** 521 MB / 738 MB / 143 MB (2020s file grows as new seasons are added).

---

## Source 2 — Alberta Historical Wildfire Data

**What it is:** Fire perimeter and occurrence records for all wildfires in Alberta from 2006 onward, maintained by Alberta Wildfire. Includes ignition coordinates, fire size, cause, and suppression cost.

**Format:** CSV, 20+ columns. Key columns used by this project:

| Column | Description |
|---|---|
| `FIRE_START_DATE` | Ignition date (ISO 8601) |
| `LATITUDE`, `LONGITUDE` | Ignition point (WGS84) |
| `GENERAL_CAUSE` | e.g., Lightning, Human, Prescribed Fire |
| `CURRENT_SIZE` | Final fire size (ha) |

Prescribed fires are excluded during preprocessing.

**Download:**
1. Go to the [Alberta Open Government Portal](https://open.alberta.ca/opendata/wildfire-historical-data)
2. Download **"Historical wildfire data 2006 to present"** (CSV format)
3. Save as:

```
data/raw/fp-historical-wildfire-data-2006-2025.csv
```

> **Note:** The file downloaded from the portal may include a copy number in the filename (e.g., `fp-historical-wildfire-data-2006-2025 (1).csv`). `build_dataset.py` expects the name with ` (1)` suffix as downloaded — rename or update the path in the script if needed.

**Size:** ~9 MB.

---

## Source 3 — Alberta Land Use/Land Cover 2020 (LULC)

**What it is:** A 10-metre resolution, 14-class land cover raster derived from Sentinel-2 imagery covering the entire province. Published by Alberta Environment and Protected Areas (AEP) as dataset DIG_2021_0019.

**Classes used:**

| Value | Class |
|---|---|
| 1 | Water |
| 2 | Bryoids |
| 3 | Wetland-Treed |
| 4 | Herbs |
| 5 | Exposed/Barren |
| 6 | Shrubland |
| 7 | Wetland |
| 8 | Grassland |
| 9 | Coniferous |
| 10 | Broadleaf |
| 11 | Mixedwood |
| 12 | Agriculture |
| 13 | Developed |
| 14 | Burned Areas |

Value 0 indicates pixels outside the raster extent; these are filled with 0 (unknown) and do not cause row drops.

**Projection:** Alberta 10TM (EPSG:3400). `build_dataset.py` reprojects station coordinates into this CRS before sampling.

**Download:**
1. Go to [https://geodiscover.alberta.ca](https://geodiscover.alberta.ca) and search for **"DIG_2021_0019"**
2. Or direct link: [Alberta Land Cover 2020 — AEP Open Data](https://open.alberta.ca/opendata/alberta-land-cover-2020)
3. Download the GeoTIFF package and extract:

```
data/raw/AB_LULC_2020_10tm.tif    # 8.7 GB — the raster used by build_dataset.py
data/raw/AB_LULC_2020_10tm.hdr    # accompanying header (included in the zip)
```

**Size:** 8.7 GB uncompressed. Expect a long download; the zip from AEP is ~1.5 GB.

---

## Source 4 — SRTM3 Topography (elevation / slope / aspect)

**What it is:** NASA Shuttle Radar Topography Mission 3 arc-second (~90 m) DEM tiles for Alberta, downloaded from the CGIAR-CSI tile server and processed into a 3-band raster.

**This file is generated automatically — do not download manually.**

Run:

```bash
python download_dem.py
```

The script fetches 8 tiles covering Alberta (`srtm_13_01` through `srtm_14_04`), merges and clips them to the Alberta bounding box, then derives slope and aspect using numpy gradients with geodetic pixel-spacing corrections.

**Output:** `data/raw/srtm_alberta_topo.tif` (~895 MB, float32, 3 bands, EPSG:4326, 13200×12000 px)

| Band | Variable | Units | Notes |
|---|---|---|---|
| 1 | Elevation | metres | 169–3932 m across Alberta |
| 2 | Slope | degrees | 0 = flat, 0–82° range |
| 3 | Aspect | degrees | 0 = North, clockwise; −1 = flat sentinel |

Flat-cell aspect (sentinel −1) is converted to NaN in `build_dataset.py` and `train.py` before sin/cos encoding; XGBoost handles these NaNs natively.

Tile cache is saved to `data/raw/srtm_tiles/` so re-runs skip the download.

---

## Generated files (do not download)

| File | Generated by | Description |
|---|---|---|
| `data/raw/fwi_alberta_2000_2025.csv` | `preprocess.py` | CWFIS CSVs filtered to Alberta bbox and merged |
| `data/raw/srtm_alberta_topo.tif` | `download_dem.py` | 3-band topo raster (elevation/slope/aspect) |
| `data/raw/srtm_tiles/*.tif` | `download_dem.py` | Intermediate per-tile DEMs (cached) |
| `data/processed/model_dataset.csv` | `build_dataset.py` | Final labeled ML dataset (762,442 rows × 22 cols) |

---

## Reproduction checklist

```
[ ] Download cwfis_fwi2000sv3.0_ll.csv  (~521 MB)
[ ] Download cwfis_fwi2010sv3.0_ll.csv  (~738 MB)
[ ] Download cwfis_fwi2020sv3.0_ll.csv  (~143 MB)
[ ] Download fp-historical-wildfire-data-2006-2025 (1).csv  (~9 MB)
[ ] Download AB_LULC_2020_10tm.tif  (~8.7 GB)
[ ] Run: python preprocess.py
[ ] Run: python download_dem.py
[ ] Run: python build_dataset.py
```

Total raw data footprint: ~11 GB (dominated by the LULC raster).
