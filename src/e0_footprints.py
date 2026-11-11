"""E0a: confirm repeated physical fields from label-mask overlap and build leak-free split units.

For every record, the valid label pixels (mean_ raster != nodata) are turned into keys
(epsg, col, row) on the 10 m UTM grid. All rasters of a record share one grid aligned to 10 m.
Two records are the same physical field when overlap = |A ∩ B| / min(|A|, |B|) >= 0.5.

Split units ("blocks") are the connected components of the graph whose nodes are farms and
whose edges join two farms that share a physical field. A split by block therefore never puts
the same land, or the same farm, on both sides.

Usage:
  python -I e0_footprints.py RAW_DIR OUT_DIR
Writes OUT_DIR/footprint_pairs.csv, OUT_DIR/records.csv and prints a summary.
"""
import glob, json, os, sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import rasterio

RAW, OUT = sys.argv[1], sys.argv[2]
MEAN = "yield_masks/mean_scaled_yield_masked_regional_statistical_outlier.tif"
THR = 0.5


def footprint(d):
    meta = json.load(open(glob.glob(os.path.join(d, "metadata-*.json"))[0]))
    with rasterio.open(os.path.join(d, MEAN)) as r:
        a = r.read(1)
        valid = (a != r.nodata) & np.isfinite(a)
        rows, cols = np.nonzero(valid)
        x0, y0 = r.transform.c, r.transform.f
    gx = np.round((x0 + 10 * cols) / 10).astype(np.int64)
    gy = np.round((y0 - 10 * rows) / 10).astype(np.int64)
    keys = np.unique(gx * 10_000_000 + gy)
    return dict(field=os.path.basename(d.rstrip("/")), country=meta["adm_units"]["country"],
                farm=meta["farm_identifier"], provider=meta["data_provider"], crop=meta["crop"],
                year=meta["year"], epsg=meta["projected_crs_epsg"], n_px=len(keys),
                xmin=gx.min(), xmax=gx.max(), ymin=gy.min(), ymax=gy.max()), keys


def main():
    dirs = sorted(glob.glob(os.path.join(RAW, "*", "*", "")))
    with ProcessPoolExecutor(24) as ex:
        res = list(ex.map(footprint, dirs, chunksize=8))
    rec = pd.DataFrame([r[0] for r in res])
    keys = [r[1] for r in res]
    rec["farm_id"] = rec.country + "_" + rec.provider + "_" + rec.farm

    pairs = []
    for (c, e), g in rec.groupby(["country", "epsg"]):
        idx = g.index.values
        for a, i in enumerate(idx):
            j = idx[a + 1:]
            j = j[(rec.xmin[j] <= rec.xmax[i]) & (rec.xmax[j] >= rec.xmin[i]) &
                  (rec.ymin[j] <= rec.ymax[i]) & (rec.ymax[j] >= rec.ymin[i])]
            for k in j:
                inter = np.intersect1d(keys[i], keys[k], assume_unique=True).size
                if inter:
                    pairs.append((rec.field[i], rec.field[k], inter,
                                  inter / min(len(keys[i]), len(keys[k])),
                                  inter / (len(keys[i]) + len(keys[k]) - inter)))
    pairs = pd.DataFrame(pairs, columns=["a", "b", "n_common", "overlap", "iou"])

    def components(nodes, edges):
        parent = {n: n for n in nodes}
        def find(n):
            while parent[n] != n:
                parent[n] = parent[parent[n]]; n = parent[n]
            return n
        for u, v in edges:
            parent[find(u)] = find(v)
        return {n: find(n) for n in nodes}

    same = pairs[pairs.overlap >= THR]
    phys = components(rec.field, zip(same.a, same.b))
    rec["physical_field"] = rec.field.map(phys)
    f2farm = dict(zip(rec.field, rec.farm_id))
    blk = components(rec.farm_id.unique(), [(f2farm[a], f2farm[b]) for a, b in zip(same.a, same.b)])
    rec["block"] = rec.farm_id.map(blk)
    for col in ["physical_field", "block"]:
        rec[col] = rec[col].astype("category").cat.codes

    os.makedirs(OUT, exist_ok=True)
    pairs.to_csv(f"{OUT}/footprint_pairs.csv", index=False)
    rec.drop(columns=["xmin", "xmax", "ymin", "ymax"]).to_csv(f"{OUT}/records.csv", index=False)

    multi = rec.groupby("physical_field").size()
    print("records", len(rec), "| physical fields", rec.physical_field.nunique(),
          "| groups with >1 record", (multi > 1).sum(), "holding", multi[multi > 1].sum(), "records")
    print("pairs with any overlap", len(pairs), "| overlap>=0.5", len(same),
          "| partial overlap 0.05-0.5", ((pairs.overlap > 0.05) & (pairs.overlap < THR)).sum())
    print("farms", rec.farm_id.nunique(), "| blocks", rec.block.nunique())
    print(rec.groupby(["country", "crop"]).agg(records=("field", "size"), phys=("physical_field", "nunique"),
                                                farms=("farm_id", "nunique"), blocks=("block", "nunique")))
    big = rec.groupby("block").field.size().sort_values(ascending=False).head(5)
    print("largest blocks (records):", big.to_dict())


if __name__ == "__main__":
    main()
