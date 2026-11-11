"""Per-block sufficient statistics of within-field R2 for the block bootstrap (E1 oracle and E2 learners).

For every country x crop, seed, prediction column and block, writes the within-field sums
  sse = sum_i ((y_i - ybar_f) - (p_i - pbar_f))^2      sst = sum_i (y_i - ybar_f)^2
over the out-of-fold test pixels of the block, so that pooled R2_w = 1 - sum(sse) / sum(sst) for any
multiset of blocks. Mode-P columns only: R2_w is invariant to the mode-D shift at field level or coarser.

Usage: python -I e2_blockstats.py E1_BLOCK_DIR E2_DIR RECORDS_CSV OUT_CSV
"""
import glob, os, sys

import numpy as np
import pandas as pd

E1, E2, REC, OUT = sys.argv[1:5]
rec = pd.read_csv(REC).set_index("field")
rows = []
for fp in sorted(glob.glob(f"{E1}/*_seed*_pred.parquet")):
    tag, seed = os.path.basename(fp).rsplit("_seed", 1); seed = int(seed.split("_")[0])
    d = pd.read_parquet(fp, columns=["field", "y", "pred"]).rename(columns={"pred": "B0"})
    f2 = f"{E2}/{tag}_seed{seed}_pred.parquet"
    if os.path.exists(f2):
        e2 = pd.read_parquet(f2)
        assert len(e2) == len(d), (tag, seed)
        d = pd.concat([d, e2.drop(columns=["fold"])], axis=1)
    d["block"] = d.field.map(rec.block).values
    g = d.groupby("field")
    yc = d.y - g.y.transform("mean")
    d["_sst"] = yc ** 2
    for col in [c for c in d.columns if c not in ("field", "y", "block", "_sst")]:
        e = yc - (d[col] - g[col].transform("mean"))
        s = pd.DataFrame({"block": d.block, "sse": e ** 2, "sst": d._sst}).groupby("block").sum()
        for b, r in s.iterrows():
            rows.append((tag, seed, col, int(b), float(r.sse), float(r.sst)))
    print(tag, seed, flush=True)
pd.DataFrame(rows, columns=["tag", "seed", "col", "block", "sse", "sst"]).to_csv(OUT, index=False)
