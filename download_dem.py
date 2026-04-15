"""
download_dem.py

Downloads SRTM3 (90m) tiles for Alberta directly from the CGIAR-CSI server
via HTTPS, merges them, clips to the Alberta bounding box, then derives slope
and aspect rasters — all without requiring `make` or any Unix tooling.

Output: data/raw/srtm_alberta_topo.tif  (3 bands, WGS84, float32)
  Band 1: elevation  (metres above sea level)
  Band 2: slope      (degrees, 0 = flat)
  Band 3: aspect     (degrees, 0 = North clockwise; -1 = flat)

Usage:
    python download_dem.py
"""

import io
import os
import zipfile

import numpy as np
import requests
import rasterio
from rasterio.merge import merge as rio_merge
from rasterio.windows import from_bounds

os.makedirs("data/raw/srtm_tiles", exist_ok=True)

TOPO_OUT = "data/raw/srtm_alberta_topo.tif"

# Alberta bounding box
BBOX = (-120.0, 49.0, -110.0, 60.0)

# CGIAR SRTM3 tile index (5-degree tiles):
#   XX = floor((lon + 180) / 5) + 1   (1-72, west to east)
#   YY = floor((60 - lat) / 5) + 1    (1-24, north to south)
# Alberta lon -120..-110 → XX 13, 14  (tile 15 covers -110..-105, not needed)
# Alberta lat  49..60    → YY  1 (60-65 edge), 2 (55-60), 3 (50-55), 4 (45-50)
# Tile YY=1 only touches lat=60 at its very bottom edge — include it for safety.
TILES = [
    (13, 1), (14, 1),
    (13, 2), (14, 2),
    (13, 3), (14, 3),
    (13, 4), (14, 4),
]

BASE_URL = (
    "https://srtm.csi.cgiar.org/wp-content/uploads/files/srtm_5x5/TIFF/"
    "srtm_{xx:02d}_{yy:02d}.zip"
)

# -- 1. Download tiles ---------------------------------------------------------
tile_paths = []
for xx, yy in TILES:
    fname = f"srtm_{xx:02d}_{yy:02d}.tif"
    fpath = os.path.join("data/raw/srtm_tiles", fname)
    if os.path.exists(fpath):
        print(f"  Already cached: {fname}")
        tile_paths.append(fpath)
        continue

    url = BASE_URL.format(xx=xx, yy=yy)
    print(f"  Downloading {fname} from CGIAR ...")
    r = requests.get(url, timeout=120, stream=True)
    if r.status_code == 404:
        print(f"    [SKIP] tile not found (ocean / no data): {fname}")
        continue
    r.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        tif_names = [n for n in zf.namelist() if n.lower().endswith(".tif")]
        if not tif_names:
            print(f"    [SKIP] no .tif inside zip for {fname}")
            continue
        zf.extract(tif_names[0], path="data/raw/srtm_tiles")
        extracted = os.path.join("data/raw/srtm_tiles", tif_names[0])
        if extracted != fpath:
            os.rename(extracted, fpath)
    tile_paths.append(fpath)
    print(f"    Saved -> {fpath}")

if not tile_paths:
    raise RuntimeError("No SRTM tiles downloaded — check your internet connection.")

# -- 2. Merge tiles and clip to Alberta bounding box --------------------------
print("\nMerging tiles and clipping to Alberta bounding box...")
datasets = [rasterio.open(p) for p in tile_paths]
merged, merged_transform = rio_merge(datasets, bounds=BBOX, nodata=-32768)
for ds in datasets:
    ds.close()

profile = rasterio.open(tile_paths[0]).profile.copy()
profile.update(
    driver="GTiff",
    height=merged.shape[1],
    width=merged.shape[2],
    transform=merged_transform,
    count=1,
    dtype="float64",
    nodata=np.nan,
    compress="lzw",
    crs="EPSG:4326",
)

elev = merged[0].astype(np.float64)
elev[elev == -32768] = np.nan

print(f"  Merged shape: {elev.shape}  |  "
      f"Elevation: {np.nanmin(elev):.0f} – {np.nanmax(elev):.0f} m")

# -- 3. Compute slope and aspect -----------------------------------------------
print("Computing slope and aspect...")
nrows, ncols  = elev.shape
pixel_x_deg   = abs(merged_transform.a)
pixel_y_deg   = abs(merged_transform.e)
top_lat       = merged_transform.f

row_lats = top_lat - (np.arange(nrows) + 0.5) * pixel_y_deg
dy_m     = pixel_y_deg * 111_320.0
dx_m     = pixel_x_deg * 111_320.0 * np.cos(np.radians(row_lats))

dz_drow, dz_dcol = np.gradient(elev)
p = dz_dcol / dx_m[:, np.newaxis]
q = -dz_drow / dy_m

slope      = np.degrees(np.arctan(np.sqrt(p**2 + q**2))).astype(np.float32)
aspect_rad = np.arctan2(-p, q)
aspect     = ((np.degrees(aspect_rad) + 360) % 360).astype(np.float32)
aspect[slope < 0.01] = -1.0  # flat sentinel

elev = elev.astype(np.float32)

# -- 4. Write 3-band topo raster -----------------------------------------------
profile.update(count=3, dtype="float32")

with rasterio.open(TOPO_OUT, "w", **profile) as dst:
    dst.write(elev,   1)
    dst.write(slope,  2)
    dst.write(aspect, 3)

print(f"\nTopo raster saved -> {TOPO_OUT}")
print(f"  Shape       : {nrows} rows x {ncols} cols")
print(f"  Elevation   : {np.nanmin(elev):.0f} – {np.nanmax(elev):.0f} m")
print(f"  Slope       : {np.nanmin(slope):.1f} – {np.nanmax(slope):.1f} deg")
print(f"  Aspect      : 0-360 deg  (-1 = flat)")
print("\nDone. Run build_dataset.py next to rebuild the ML dataset.")
