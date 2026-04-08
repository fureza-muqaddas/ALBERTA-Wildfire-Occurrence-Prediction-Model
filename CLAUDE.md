
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Alberta wildfire prediction model using Fire Weather Index (FWI) data, historical wildfire records, and land cover data. The pipeline covers data acquisition → preprocessing → feature engineering → XGBoost model training.

## Setup

```bash
pip install -r requirements.txt
```

A `.env` file is required at the project root with Copernicus Climate Data Store credentials:
```
url=https://cds.climate.copernicus.eu/api
CDS_API_KEY=<your-key>
```

## Running Scripts

```bash
python climate.py      # Downloads Alberta land cover zip from AER
python preprocess.py   # Filters CWFIS FWI CSVs to Alberta bounding box and merges them
```

## Data Architecture

All raw data lives in `data/raw/`. CSV files and `.nc` (NetCDF) files are gitignored.

**Data sources:**
- `cwfis_fwi{2000,2010,2020}sv3.0_ll.csv` — Canadian Wildland Fire Information System FWI data by decade (point observations with lat/lon)
- `fp-historical-wildfire-data-2006-2025.csv` — Alberta historical fire perimeter/occurrence records
- `AlbertaLandCover2020.zip` — Alberta land cover raster from AER (DIG_2021_0019)
- `fwi_alberta_2000_2025.csv` — Output of `preprocess.py`: Alberta-clipped, date-sorted FWI records

**Alberta spatial bounding box:** lat 49–60°N, lon -120–-110°W

**FWI columns retained:** `rep_date`, `name`, `lat`, `lon`, `temp`, `rh`, `ws`, `precip`, `ffmc`, `dmc`, `dc`, `isi`, `bui`, `fwi`

## Stack

Python · pandas · numpy · scikit-learn · XGBoost · rasterio · matplotlib/seaborn · cdsapi (Copernicus)
