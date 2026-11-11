#!/usr/bin/env python3
"""Block-bootstrap confidence intervals for within-field R2 (E1 oracle and E2 learners, mode P).

Input: $RUNS/e2_blockstats.csv (src/e2_blockstats.py): within-field sse and sst per
(pair, seed, column, block). One bootstrap replicate resamples the blocks of every pair with
replacement (the same draw for the three seeds, since seeds only reassign blocks to folds), recomputes
pooled R2_w = 1 - sum(sse) / sum(sst) per seed, averages over seeds and then over pairs.
Output: results/e2ci/seed_0.json with, per scope (pair, MEAN5, MEAN9),
  <scope>_<col>             r2w, lo, hi                (95 % percentile interval)
  <scope>_<a>__vs__<b>      d, lo, hi, p_le0           (difference a - b; p_le0 = share of replicates <= 0)
"""
import json, os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ISO = {"Argentina": "AR", "Brazil": "BR", "Germany": "DE", "Uruguay": "UY"}
MEAN5 = {"AR-corn", "AR-soybean", "AR-wheat", "BR-corn", "BR-soybean"}
NB = 2000

COMP = [("L1_B2pv", "L1_B11"), ("L1_B8", "L1_B11"), ("L1_B2pv", "L1_B8"), ("L1_B2pv", "L1_B2"),
        ("L1_B2", "L1_B1"), ("B0", "L1_B2pv"), ("B0", "L1_B11"), ("L1_B8", "L1_B8pv"),
        ("L1_B2pv", "L2_B2pv"), ("L1_B8", "L2_B8"), ("L1_B11", "L2_B11"),
        ("L2_B2pv", "Lrand_B2pv"), ("L2_B8", "Lrand_B8"), ("L2_B11", "Lrand_B11"),
        ("L3_20_B2pv", "Lrand_B2pv"), ("L3_20_B8", "Lrand_B8")]


def main():
    s = pd.read_csv(os.path.join(os.environ.get("RUNS", os.path.join(ROOT, "runs")), "e2_blockstats.csv"))
    s["pair"] = [ISO[t.split("_", 1)[0]] + "-" + t.split("_", 1)[1] for t in s.tag]
    cols = sorted(s.col.unique())
    rng = np.random.default_rng(0)
    point, boot = {}, {}  # pair -> {col: r2w}, pair -> {col: array[NB]}
    for pr, d in s.groupby("pair"):
        blocks = np.sort(d.block.unique()); seeds = np.sort(d.seed.unique())
        W = rng.multinomial(len(blocks), np.full(len(blocks), 1 / len(blocks)), size=NB).astype(float)
        point[pr], boot[pr] = {}, {}
        for c in cols:
            dc = d[d.col == c]
            if dc.seed.nunique() != len(seeds):
                continue
            r_pt, r_bt = [], []
            for sd in seeds:
                x = dc[dc.seed == sd].set_index("block").reindex(blocks).fillna(0.0)
                r_pt.append(1 - x.sse.sum() / x.sst.sum())
                r_bt.append(1 - (W @ x.sse.values) / (W @ x.sst.values))
            point[pr][c] = float(np.mean(r_pt)); boot[pr][c] = np.mean(r_bt, axis=0)
    out = {}

    def put(scope, P, B):
        for c in cols:
            if all(c in P[p] for p in P):
                v = np.mean([B[p][c] for p in B], axis=0)
                out[f"{scope}_{c}"] = dict(r2w=float(np.mean([P[p][c] for p in P])),
                                           lo=float(np.percentile(v, 2.5)), hi=float(np.percentile(v, 97.5)))
        for a, b in COMP:
            if all(a in P[p] and b in P[p] for p in P):
                v = np.mean([B[p][a] - B[p][b] for p in B], axis=0)
                out[f"{scope}_{a}__vs__{b}"] = dict(
                    d=float(np.mean([P[p][a] - P[p][b] for p in P])), lo=float(np.percentile(v, 2.5)),
                    hi=float(np.percentile(v, 97.5)), p_le0=float(np.mean(v <= 0)))

    for pr in point:
        put(pr, {pr: point[pr]}, {pr: boot[pr]})
    put("MEAN5", {p: point[p] for p in MEAN5}, {p: boot[p] for p in MEAN5})
    put("MEAN9", point, boot)
    d = os.path.join(ROOT, "results", "e2ci"); os.makedirs(d, exist_ok=True)
    json.dump(dict(seed=0, n_test=0, n_boot=NB, metrics=out), open(f"{d}/seed_0.json", "w"), indent=1, sort_keys=True)
    for k in ("MEAN5_L1_B2pv__vs__L1_B11", "MEAN5_L1_B8__vs__L1_B11", "MEAN5_L1_B2pv__vs__L1_B8",
              "MEAN5_B0__vs__L1_B2pv", "MEAN5_L2_B2pv__vs__Lrand_B2pv", "MEAN5_L2_B8__vs__Lrand_B8",
              "MEAN5_L1_B2pv__vs__L2_B2pv", "MEAN5_L2_B11__vs__Lrand_B11", "MEAN5_L1_B2__vs__L1_B1",
              "MEAN5_L1_B2pv__vs__L1_B2", "MEAN9_L1_B2pv__vs__L1_B11", "MEAN9_L1_B8__vs__L1_B11"):
        print(k, {a: round(v, 3) for a, v in out[k].items()})
    for k in ("MEAN5_B0", "MEAN5_L1_B2pv", "MEAN5_L1_B8", "MEAN5_L1_B11", "MEAN9_L1_B11"):
        print(k, {a: round(v, 3) for a, v in out[k].items()})


if __name__ == "__main__":
    main()
