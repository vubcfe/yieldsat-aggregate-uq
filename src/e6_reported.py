"""E6 (RQ4): farmer-reported field yields as the aggregate in mode D, instead of the clean pixel mean.

For each country x crop with reported yields (not Uruguay) and each seed, using block-split OOF
predictions (E1 oracle by default, or an E2 method column):
  D1_clean    shift each field's prediction so its mean equals the clean pixel mean (upper bound)
  D1_rep      shift to the farmer-reported yield (what a user actually has)
  D1_rep_cal  reported yield x c, with c = median(clean / reported) over calibration fields
              (other folds); needs dense data on the calibration side
  P           no aggregate
Pixel intervals (90 %, 80 %): conformal quantile from calibration fields of the other folds,
computed either under the clean shift ("naive": the user believes reports are exact) or under the
same reported shift ("matched": needs dense calibration data). Fields: only those with reported
yield > 0. Values are compared as provided (no moisture conversion: the reported basis is not documented).

Usage: python -I e6_reported.py PRED_DIR RECORDS_CSV FIELDS_CSV OUT_JSON [column=pred]
  FIELDS_CSV: output of e6_fields.py (columns dir, yield_gt, quality).
"""
import glob, json, os, sys

import numpy as np
import pandas as pd

PRED, REC, FIELDS, OUT = sys.argv[1:5]
COL = sys.argv[5] if len(sys.argv) > 5 else "pred"
E1DIR = os.environ.get("E1_DIR")
ALPHAS = (0.1, 0.2)


def cp_q(r, alpha, w):
    r = np.abs(r); o = np.argsort(r); cw = np.cumsum(w[o]) / (w.sum() + w.max())
    i = np.searchsorted(cw, 1 - alpha)
    return np.inf if i >= len(r) else float(r[o][i])


def r2(y, p):
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


rec = pd.read_csv(REC)
rec["fy"] = rec.farm_id + "|" + rec.year.astype(str)
fl = pd.read_csv(FIELDS, usecols=["dir", "yield_gt", "quality"])
fl["field"] = fl.dir.str.split("/").str[-1]
fl["rep"] = pd.to_numeric(fl.yield_gt, errors="coerce")
rep = fl.set_index("field").rep
qual = fl.set_index("field").quality
rows = []
for fp in sorted(glob.glob(f"{PRED}/*_seed*_pred.parquet")):
    tag, seed = os.path.basename(fp).rsplit("_seed", 1); seed = int(seed.split("_")[0])
    df = pd.read_parquet(fp)
    if "field" not in df:
        keys = pd.read_parquet(f"{E1DIR}/{tag}_seed{seed}_pred.parquet", columns=["field", "row", "col", "y"])
        df = pd.concat([keys, df], axis=1)
    df = df[["field", "y", COL, "fold"]].rename(columns={COL: "p"})
    df["rep"] = df.field.map(rep)
    df = df[df.rep > 0].reset_index(drop=True)
    if df.field.nunique() < 20:
        continue
    g = df.groupby("field")
    df["ybar"] = g.y.transform("mean"); df["pbar"] = g.p.transform("mean")
    fm = g.agg(ybar=("y", "mean"), rep=("rep", "first"), fold=("fold", "first"))
    out = dict(tag=tag, seed=seed, column=COL, n_fields=int(len(fm)),
               rel_bias_median=float(np.median(fm.ybar / fm.rep - 1)),
               rel_abs_err_median=float(np.median(np.abs(fm.ybar / fm.rep - 1))),
               r2_field_reported_vs_clean=r2(fm.ybar.values, fm.rep.values))
    preds = {m: np.empty(len(df)) for m in ("P", "D1_clean", "D1_rep", "D1_rep_cal")}
    cov = {f"{m}_{c}_{al}": [] for m in ("D1_rep", "D1_rep_cal") for c in ("naive", "matched") for al in ALPHAS}
    cov.update({f"D1_clean_matched_{al}": [] for al in ALPHAS}); cov.update({f"P_matched_{al}": [] for al in ALPHAS})
    wts = []
    for k in sorted(df.fold.unique()):
        te, ca = (df.fold == k).values, (df.fold != k).values
        c = float(np.median(fm.ybar[fm.fold != k] / fm.rep[fm.fold != k]))
        base = df.p - df.pbar
        P = {"P": df.p.values, "D1_clean": (base + df.ybar).values, "D1_rep": (base + df.rep).values,
             "D1_rep_cal": (base + c * df.rep).values}
        for m, v in P.items():
            preds[m][te] = v[te]
        w = 1.0 / df[ca].groupby("field").y.transform("size").values
        for al in ALPHAS:
            qclean = cp_q((df.y - P["D1_clean"]).values[ca], al, w)
            for m in ("D1_rep", "D1_rep_cal", "D1_clean", "P"):
                e_te = (df.y.values - P[m])[te]
                qm = cp_q((df.y.values - P[m])[ca], al, w)
                cov[f"{m}_matched_{al}"].append(float(np.mean(np.abs(e_te) <= qm)))
                if m in ("D1_rep", "D1_rep_cal"):
                    cov[f"{m}_naive_{al}"].append(float(np.mean(np.abs(e_te) <= qclean)))
        wts.append(int(te.sum()))
    for m, v in preds.items():
        out[f"r2_pixel_{m}"] = r2(df.y.values, v)
        out[f"rmse_{m}"] = float(np.sqrt(np.mean((df.y.values - v) ** 2)))
    for k, v in cov.items():
        out[f"cov_{k}"] = float(np.average(v, weights=wts))
    rows.append(out)
    print(tag, seed, {k: round(v, 3) for k, v in out.items() if isinstance(v, float)}, flush=True)
json.dump(rows, open(OUT, "w"), indent=1)
