"""E2a: pseudo-region bags (level L3') from field centroids, without storing coordinates.

A pseudo-region is a square grid cell of side R km (R = 10, 20, 50) on a local equirectangular
projection per country, crossed with crop and year. This mimics commune/district statistics
(fixed administrative cells that cut across farms). Only the bag IDs are written; centroids
stay in memory (coordinates are sensitive under the YieldSAT datasheet).

Usage: python -I e2_bags.py RAW_DIR RECORDS_CSV OUT_CSV
"""
import glob, os, sys

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform

MASK, REC, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
RADII = (10, 20, 50)
MEAN = "yield_masks/mean_scaled_yield_masked_regional_statistical_outlier.tif"

rec = pd.read_csv(REC)
lon, lat = [], []
for f, c in zip(rec.field, rec.country):
    p = glob.glob(os.path.join(MASK, c, f, MEAN))[0]
    with rasterio.open(p) as r:
        a = r.read(1); v = (a != r.nodata) & np.isfinite(a)
        rows, cols = np.nonzero(v)
        x, y = r.xy(rows.mean(), cols.mean())
        (lo,), (la,) = transform(r.crs, "EPSG:4326", [x], [y])
    lon.append(lo); lat.append(la)
lon, lat = np.array(lon), np.array(lat)
out = rec[["field", "country", "crop", "year"]].copy()
for c in rec.country.unique():
    m = (rec.country == c).values
    lat0 = np.deg2rad(lat[m].mean())
    x = (lon[m] - lon[m].min()) * 111.32 * np.cos(lat0)
    y = (lat[m] - lat[m].min()) * 110.57
    for R in RADII:
        out.loc[m, f"cell{R}"] = [f"{int(a)}_{int(b)}" for a, b in zip(x // R, y // R)]
for R in RADII:
    out[f"L3_{R}"] = out.country + "|" + out.crop + "|" + out.year.astype(str) + "|" + out[f"cell{R}"]
out.drop(columns=[f"cell{R}" for R in RADII]).to_csv(OUT, index=False)
for R in RADII:
    g = out.groupby(["country", "crop"])[f"L3_{R}"]
    print(f"R={R} km: bags per country x crop\n", g.nunique().to_string())
