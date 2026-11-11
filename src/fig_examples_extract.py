"""Example fields for the qualitative map figure (no coordinates: pixel row/col inside each record's raster).

Selection rule, fixed in advance: one country x crop and seed; among test fields with at least MIN_PX
pixels, the fields whose oracle within-field R2 is closest to the 25th, 50th and 75th percentile.
Writes OUT_NPZ with, per selected field, the grids of truth and of the predictions (NaN outside the field).

Usage: python -I fig_examples_extract.py E1_BLOCK_DIR E2_DIR TAG SEED OUT_NPZ
"""
import sys

import numpy as np
import pandas as pd

E1, E2, TAG, SEED, OUT = sys.argv[1:6]
MIN_PX = 2000
COLS = ["B0", "L1_B11", "L1_B2pv", "L1_B1"]
d = pd.read_parquet(f"{E1}/{TAG}_seed{SEED}_pred.parquet").rename(columns={"pred": "B0"})
e2 = pd.read_parquet(f"{E2}/{TAG}_seed{SEED}_pred.parquet")
d = pd.concat([d, e2[COLS[1:]]], axis=1)


def r2w(g, c):
    y = g.y - g.y.mean(); p = g[c] - g[c].mean()
    return 1 - ((y - p) ** 2).sum() / (y ** 2).sum()


stats = d.groupby("field").filter(lambda g: len(g) >= MIN_PX).groupby("field").apply(lambda g: r2w(g, "B0"))
sel = [(stats - stats.quantile(q)).abs().idxmin() for q in (0.25, 0.5, 0.75)]
out = {"fields": np.array(["field %d" % (i + 1) for i in range(len(sel))]), "cols": np.array(["y"] + COLS),
       "n_candidates": len(stats)}
for i, f in enumerate(sel):
    g = d[d.field == f]
    r0, c0 = g.row.min(), g.col.min()
    for c in ["y"] + COLS:
        a = np.full((g.row.max() - r0 + 1, g.col.max() - c0 + 1), np.nan, np.float32)
        a[g.row - r0, g.col - c0] = g[c].values
        out[f"{i}_{c}"] = a
    out[f"{i}_r2w"] = np.array([r2w(g, c) for c in COLS])
np.savez_compressed(OUT, **out)
print(len(stats), "candidate fields; selected r2w(B0):", [round(float(out[f'{i}_r2w'][0]), 2) for i in range(len(sel))])
