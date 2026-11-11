"""E6 input: farmer-reported field yield and yield-map quality flag per record, from the YieldSAT metadata.

Writes a CSV with columns dir (<country>/<record>), yield_gt (reported yield, t/ha, NaN if absent)
and quality (yieldmap_quality). No coordinates are read or written.

Usage: python -I e6_fields.py RAW_DIR OUT_CSV
"""
import glob, json, os, sys

import numpy as np
import pandas as pd

RAW, OUT = sys.argv[1], sys.argv[2]
rows = []
for p in sorted(glob.glob(os.path.join(RAW, "*", "*", "metadata-*.json"))):
    d = os.path.dirname(p)
    m = json.load(open(p))
    try:
        rep = float(m.get("yield_ground_truth"))
    except (TypeError, ValueError):
        rep = np.nan
    rows.append(dict(dir=os.path.relpath(d, RAW), yield_gt=rep, quality=m.get("yieldmap_quality")))
pd.DataFrame(rows).to_csv(OUT, index=False)
print(len(rows), "records,", int(np.isfinite([r["yield_gt"] for r in rows]).sum()), "with reported yield")
