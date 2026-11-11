"""E0b: per-pixel feature table from the raw YieldSAT folders (one Parquet file per record).

Only labelled pixels are kept (mean_ raster != nodata). Columns:
  ids        field, row, col
  labels     y (mean_ t/ha, dry), y_n (number_ = harvester points in the pixel), y_sd (std_)
  s2         10 bands + NDVI, GCVI, NDRE, NDWI, interpolated to K points of relative season
             time t = (date - seeding) / (harvest - seeding), from clear observations only
             (SCL 4 vegetation or 5 bare soil); plus per-pixel n_clear, ndvi_max, ndvi_tmax, ndvi_auc
  soil       8 properties x 12 bands, as stored (band meaning: see DATA_CARD, not needed by trees)
  dem        aspect, curvature, dem, slope (twi dropped: NaN for many fields)
  weather    field-level season aggregates in 4 equal phases: tmean, tmax, tmin (degC), prec (mm), gdd10
Coordinates are not stored as features (sensitive under the datasheet; also a leakage shortcut).

Usage: python -I e0_features.py RAW_DIR OUT_DIR [n_workers]
"""
import glob, json, os, sys, traceback
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import numpy as np
import pandas as pd
import rasterio

RAW, OUT = sys.argv[1], sys.argv[2]
NW = int(sys.argv[3]) if len(sys.argv) > 3 else 16
K = 10
TGRID = (np.arange(K) + 0.5) / K
BANDS = {"B02": 2, "B03": 3, "B04": 4, "B05": 5, "B06": 6, "B07": 7, "B08": 8, "B8A": 9, "B11": 11, "B12": 12}
YM = "yield_masks/{}_scaled_yield_masked_regional_statistical_outlier.tif"


def date_of(p):
    return datetime.strptime(p[-12:-4], "%Y%m%d")


def interp_rows(t_obs, V, valid):
    """Linear interpolation per pixel of V[n_pix, n_obs] onto TGRID, using only valid obs.
    Outside the observed range the nearest valid value is used; pixels with no valid obs get NaN."""
    n_pix = V.shape[0]
    out = np.full((n_pix, K), np.nan, np.float32)
    for i in range(n_pix):
        m = valid[i]
        if m.any():
            out[i] = np.interp(TGRID, t_obs[m], V[i, m])
    return out


def one(d):
    field = os.path.basename(d.rstrip("/"))
    country = d.rstrip("/").split("/")[-2]
    dst = os.path.join(OUT, country, field + ".parquet")
    if os.path.exists(dst):
        return field, "skip"
    meta = json.load(open(glob.glob(os.path.join(d, "metadata-*.json"))[0]))
    sow = datetime.strptime(meta["seeding_date"], "%d.%m.%Y")
    har = datetime.strptime(meta["harvesting_date"], "%d.%m.%Y")
    L = (har - sow).days

    with rasterio.open(os.path.join(d, YM.format("mean"))) as r:
        y = r.read(1); lab = (y != r.nodata) & np.isfinite(y)
    rows, cols = np.nonzero(lab)
    cols_out = {"row": rows.astype(np.int16), "col": cols.astype(np.int16), "y": y[lab].astype(np.float32)}
    for k, name in (("number", "y_n"), ("std", "y_sd")):
        with rasterio.open(os.path.join(d, YM.format(k))) as r:
            cols_out[name] = r.read(1)[lab].astype(np.float32)

    # S2 in season
    imgs = [p for p in sorted(glob.glob(os.path.join(d, "s2_images", "*.tif"))) if sow <= date_of(p) <= har]
    t_obs, stack, clear = [], [], []
    for p in imgs:
        scl = rasterio.open(p.replace("s2_images/S2_L2A_", "scl_masks/S2_L2A_SCL_")).read(1)[lab]
        c = np.isin(scl, (4, 5))
        if not c.any():
            continue
        a = rasterio.open(p).read(list(BANDS.values())).astype(np.float32)[:, lab] * 1e-4
        t_obs.append((date_of(p) - sow).days / L); stack.append(a); clear.append(c)
    n_pix = len(rows)
    if stack:
        S = np.stack(stack, 2)            # band, pix, obs
        C = np.stack(clear, 1)            # pix, obs
        t_obs = np.array(t_obs)
        b = dict(zip(BANDS, S))
        eps = 1e-6
        idx = {"NDVI": (b["B08"] - b["B04"]) / (b["B08"] + b["B04"] + eps),
               "GCVI": b["B08"] / (b["B03"] + eps) - 1,
               "NDRE": (b["B8A"] - b["B05"]) / (b["B8A"] + b["B05"] + eps),
               "NDWI": (b["B08"] - b["B11"]) / (b["B08"] + b["B11"] + eps)}
        for name, V in list(b.items()) + list(idx.items()):
            G = interp_rows(t_obs, V, C)
            for k in range(K):
                cols_out[f"{name}_t{k}"] = G[:, k]
        nd = np.where(C, idx["NDVI"], np.nan)
        cols_out["n_clear"] = C.sum(1).astype(np.int16)
        with np.errstate(all="ignore"):
            cols_out["ndvi_max"] = np.nanmax(nd, 1).astype(np.float32)
            j = np.nanargmax(np.where(np.isnan(nd), -9, nd), 1)
        cols_out["ndvi_tmax"] = t_obs[j].astype(np.float32)
        G = np.stack([cols_out[f"NDVI_t{k}"] for k in range(K)], 1)
        cols_out["ndvi_auc"] = np.nanmean(G, 1).astype(np.float32)
    else:
        cols_out["n_clear"] = np.zeros(n_pix, np.int16)

    for p in sorted(glob.glob(os.path.join(d, "soil", "*.tif"))):
        prop = os.path.basename(p).split("_")[0]
        a = rasterio.open(p).read().astype(np.float32)[:, lab]
        for i in range(a.shape[0]):
            cols_out[f"soil_{prop}_{i}"] = a[i]
    for name in ("aspect", "curvature", "dem", "slope"):
        p = glob.glob(os.path.join(d, "dem", f"{name}-*.tif"))
        if p:
            cols_out[f"topo_{name}"] = rasterio.open(p[0]).read(1).astype(np.float32)[lab]

    w = pd.read_csv(glob.glob(os.path.join(d, "weather", "*.csv"))[0], parse_dates=["Date"])
    w = w[(w.Date >= sow) & (w.Date <= har)].copy()
    w["ph"] = np.minimum(((w.Date - sow).dt.days / max(L, 1) * 4).astype(int), 3)
    for c in ("Temp_mean", "Temp_max", "Temp_min"):
        w[c] = w[c] - 273.15
    w["prec"] = w.Total_prec * 1000
    w["gdd"] = np.clip(w.Temp_mean - 10, 0, None)
    agg = w.groupby("ph").agg(tmean=("Temp_mean", "mean"), tmax=("Temp_max", "mean"),
                              tmin=("Temp_min", "mean"), prec=("prec", "sum"), gdd=("gdd", "sum"))
    for ph in range(4):
        for c in agg.columns:
            cols_out[f"w_{c}_p{ph}"] = np.full(n_pix, agg.loc[ph, c] if ph in agg.index else np.nan, np.float32)
    cols_out["season_days"] = np.full(n_pix, L, np.int16)

    df = pd.DataFrame(cols_out)
    df.insert(0, "field", field)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    df.to_parquet(dst + ".tmp", index=False)
    os.replace(dst + ".tmp", dst)
    return field, n_pix


def safe(d):
    try:
        return one(d)
    except Exception:
        return d, "ERROR " + traceback.format_exc(limit=2).replace("\n", " | ")


if __name__ == "__main__":
    dirs = sorted(glob.glob(os.path.join(RAW, "*", "*", "")))
    n_err = 0
    with ProcessPoolExecutor(NW) as ex:
        for i, (f, r) in enumerate(ex.map(safe, dirs, chunksize=2)):
            if isinstance(r, str) and r.startswith("ERROR"):
                n_err += 1; print(f, r, flush=True)
            if i % 100 == 0:
                print(i, f, r, flush=True)
    print("done, errors:", n_err)
