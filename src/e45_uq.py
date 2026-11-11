"""E4/E5: calibration of pixel intervals and aggregation of uncertainty to field and farm-year scales.

Input: out-of-fold predictions on the block split (E1 oracle B0, or an E2 method column), with
pixel grid positions (row, col; 10 m, relative to each record's own raster, never georeferenced).
For test fold k, everything is calibrated on the OOF residuals of the other folds (other blocks):
a cross-conformal scheme; no test-fold label is used.

Scales: pixel, field (record mean), farm-year (area-weighted mean of its fields).
Modes:
  P    point prediction as is
  D1   field aggregate known: prediction shifted per field so its mean equals the field mean
  D2   farm-year aggregate known: shift per farm-year

Interval constructions for a coarse unit U (field or farm-year) from pixel-level information:
  a_indep    pixel errors independent: hw = q_px / sqrt(n_U)
  b_full     fully correlated: hw = q_px
  c_vario    Wadoux-style: Gaussian with Var(mean error) from an exponential covariance of the
             within-field residuals (fit on calibration fields), no between-field term
  c_vario_re c_vario plus a field random effect sigma_f^2 (variance of calibration field-mean errors
             not explained by the within-field covariance); for farm-years the field effects are
             treated as independent between fields
  d_direct   split-conformal directly on |unit-mean residual| of calibration units
Pixel intervals: conformal quantile of |pixel residual| over calibration pixels (each calibration
field weighted equally, up to 1000 pixels per field).

Aggregate-only calibration (RQ2): pixel half-width = kappa x q_field, where q_field comes from the
target's own calibration field aggregates (no dense label) and kappa = q_px / q_field is
transferred from other country x crop pairs (written out per pair so the transfer is done in
analysis/export_results.py). Also: pixel CP from only k dense calibration fields (k = 1, 2, 5, 10, 20),
200 random draws each.

Usage: python -I e45_uq.py PRED_DIR RECORDS_CSV OUT_DIR [column=pred] [seeds]
  PRED_DIR holds <tag>_seed<s>_pred.parquet; for E2 files the row keys come from E1_DIR (env E1_DIR).
"""
import glob, json, os, sys

import numpy as np
import pandas as pd

PRED, REC, OUT = sys.argv[1:4]
COL = sys.argv[4] if len(sys.argv) > 4 else "pred"
SEEDS = [int(s) for s in sys.argv[5].split(",")] if len(sys.argv) > 5 else [0, 1, 2]
E1DIR = os.environ.get("E1_DIR")
ALPHAS = (0.1, 0.2)
Z = {0.1: 1.6449, 0.2: 1.2816}
KS = (1, 2, 5, 10, 20)
NDRAW = 200
KGRID = np.geomspace(0.1, 30, 120)  # pixel half-width as a multiple of q_field (for kappa transfer)


def cp_q(r, alpha, w=None):
    """Conformal (1-alpha) quantile of |r| with optional weights (finite-sample corrected)."""
    r = np.abs(np.asarray(r, float))
    if w is None:
        n = len(r); k = int(np.ceil((n + 1) * (1 - alpha)))
        return np.inf if k > n else float(np.sort(r)[k - 1])
    o = np.argsort(r); cw = np.cumsum(w[o]) / (w.sum() + w.max())
    i = np.searchsorted(cw, 1 - alpha)
    return np.inf if i >= len(r) else float(r[o][i])


def expcov_fit(cal, rng, n_per=300, maxlag=400.0):
    """Fit C(h) = s * exp(-h / a) (h > 0) plus nugget on within-field demeaned residuals.
    Returns (nugget, s, a) with nugget + s = within-field residual variance."""
    hs, ps = [], []
    for _, d in cal.groupby("field"):
        if len(d) < 30:
            continue
        d = d.sample(min(len(d), n_per), random_state=int(rng.integers(1 << 30)))
        e = (d.e - d.e.mean()).values; x = d.col.values * 10.0; y = d.row.values * 10.0
        i, j = np.triu_indices(len(e), 1)
        h = np.hypot(x[i] - x[j], y[i] - y[j]); m = h <= maxlag
        hs.append(h[m]); ps.append(e[i[m]] * e[j[m]])
    h, p = np.concatenate(hs), np.concatenate(ps)
    var = float((cal.e - cal.groupby("field").e.transform("mean")).var())
    bins = np.arange(0, maxlag + 20, 20.0); b = np.digitize(h, bins) - 1
    hc = np.array([h[b == k].mean() for k in range(len(bins) - 1) if (b == k).sum() > 50])
    cc = np.array([p[b == k].mean() for k in range(len(bins) - 1) if (b == k).sum() > 50])
    best = None
    for a in np.geomspace(5, 2000, 60):
        g = np.exp(-hc / a); s = max(float(g @ cc / (g @ g)), 0.0)
        s = min(s, var); sse = np.sum((cc - s * g) ** 2)
        if best is None or sse < best[0]:
            best = (sse, s, a)
    _, s, a = best
    return var - s, s, a


def mean_cov(d, nug, s, a, rng, n=400):
    """Variance of the mean of the within-field error over the pixels of one field."""
    N = len(d)
    if N > n:
        d = d.sample(n, random_state=int(rng.integers(1 << 30)))
    x = d.col.values * 10.0; y = d.row.values * 10.0
    H = np.hypot(x[:, None] - x[None], y[:, None] - y[None])
    C = s * np.exp(-H / a); C[np.diag_indices_from(C)] += nug
    # sampling n of N pixels: off-diagonal mean estimates the pair mean, diagonal shrinks as 1/N
    m = len(d)
    off = (C.sum() - np.trace(C)) / max(m * (m - 1), 1)
    return float(off + (nug + s - off) / N)


def main():
    rec = pd.read_csv(REC)
    rec["fy"] = rec.farm_id + "|" + rec.year.astype(str)
    os.makedirs(OUT, exist_ok=True)
    files = sorted(glob.glob(f"{PRED}/*_seed*_pred.parquet"))
    for fp in files:
        tag, seed = os.path.basename(fp).rsplit("_seed", 1); seed = int(seed.split("_")[0])
        if seed not in SEEDS:
            continue
        dst = f"{OUT}/{tag}_seed{seed}_{COL}.json"
        if os.path.exists(dst):
            continue
        rng = np.random.default_rng(seed)
        df = pd.read_parquet(fp)
        if "field" not in df:  # E2 file: keys from E1
            keys = pd.read_parquet(f"{E1DIR}/{tag}_seed{seed}_pred.parquet", columns=["field", "row", "col", "y"])
            df = pd.concat([keys, df], axis=1)
        df = df[["field", "row", "col", "y", COL, "fold"]].rename(columns={COL: "p"})
        df = df.merge(rec[["field", "fy"]], on="field")
        df["pD1"] = df.p - df.groupby("field").p.transform("mean") + df.groupby("field").y.transform("mean")
        df["pD2"] = df.p - df.groupby("fy").p.transform("mean") + df.groupby("fy").y.transform("mean")
        res = dict(tag=tag, seed=seed, column=COL, n_fields=int(df.field.nunique()),
                   n_fy=int(df.fy.nunique()), folds=[])
        for k in sorted(df.fold.unique()):
            te, ca = df[df.fold == k], df[df.fold != k]
            qfP = {}
            for mode, pc in (("P", "p"), ("D1", "pD1"), ("D2", "pD2")):
                cal = ca.assign(e=ca.y - ca[pc]); tst = te.assign(e=te.y - te[pc])
                # pixel calibration: each calibration field weighted equally (<= 1000 px)
                sh = cal.sample(frac=1.0, random_state=seed)
                cs = sh[sh.groupby("field").cumcount().values < 1000]
                w = 1.0 / cs.groupby("field").e.transform("size").values
                nug, s, a = expcov_fit(cal, rng)
                # field-level quantities on calibration and test
                fcal = cal.groupby("field").agg(e=("e", "mean"), n=("e", "size"), fy=("fy", "first"))
                ftst = tst.groupby("field").agg(e=("e", "mean"), n=("e", "size"), fy=("fy", "first"))
                vw_cal = np.array([mean_cov(d, nug, s, a, rng) for _, d in cal.groupby("field")])
                sig_f2 = max(float(np.mean(fcal.e.values ** 2) - np.mean(vw_cal)), 0.0)
                vw_tst = pd.Series([mean_cov(d, nug, s, a, rng) for _, d in tst.groupby("field")],
                                   index=ftst.index)
                yc = cal.groupby("fy").agg(e=("e", "mean"), n=("e", "size"))
                yt = tst.groupby("fy").agg(e=("e", "mean"), n=("e", "size"))
                # farm-year variance under c_vario / c_vario_re: area-weighted field means, fields independent
                ftst["wt"] = ftst.n / ftst.groupby("fy").n.transform("sum")
                v_fy_vario = (ftst.wt ** 2 * vw_tst).groupby(ftst.fy).sum()
                v_fy_re = (ftst.wt ** 2 * (vw_tst + sig_f2)).groupby(ftst.fy).sum()
                fo = dict(fold=int(k), mode=mode, n_test_fields=int(len(ftst)), n_test_fy=int(len(yt)),
                          vario=dict(nugget=nug, sill=s, range_m=a), sigma_field2=sig_f2,
                          rmse_px=float(np.sqrt(np.mean(tst.e ** 2))),
                          rmse_field=float(np.sqrt(np.mean(ftst.e ** 2))),
                          rmse_fy=float(np.sqrt(np.mean(yt.e ** 2))),
                          mean_var_field_vario=float(vw_tst.mean()),
                          mean_var_field_re=float((vw_tst + sig_f2).mean()))
                for al in ALPHAS:
                    q_px = cp_q(cs.e.values, al, w)
                    q_f = cp_q(fcal.e.values, al)
                    q_y = cp_q(yc.e.values, al)
                    r = dict(q_px=q_px, q_field=q_f, q_fy=q_y,
                             cov_px=float(np.mean(np.abs(tst.e) <= q_px)),
                             cov_px_fieldmin=float(tst.assign(c=np.abs(tst.e) <= q_px).groupby("field").c.mean().quantile(0.1)),
                             width_px=2 * q_px)
                    for unit, T, V1, V2, qd in (("field", ftst, vw_tst, vw_tst + sig_f2, q_f),
                                                ("fy", yt, v_fy_vario.reindex(yt.index), v_fy_re.reindex(yt.index), q_y)):
                        hw = dict(a_indep=q_px / np.sqrt(T.n.values), b_full=np.full(len(T), q_px),
                                  c_vario=Z[al] * np.sqrt(V1.values), c_vario_re=Z[al] * np.sqrt(V2.values),
                                  d_direct=np.full(len(T), qd))
                        for name, h in hw.items():
                            r[f"cov_{unit}_{name}"] = float(np.mean(np.abs(T.e.values) <= h))
                            r[f"width_{unit}_{name}"] = float(np.mean(2 * h))
                    # aggregate-only pixel calibration needs kappa from another pair: store own kappa
                    # kappa is always relative to the mode-P field quantile: that is what field aggregates
                    # alone can calibrate (mode-P field-mean errors are observable from aggregates)
                    if mode == "P":
                        qfP[al] = q_f
                    r["q_field_P"] = qfP[al]
                    r["kappa"] = q_px / qfP[al]
                    r["cov_px_vs_kappa"] = [float(np.mean(np.abs(tst.e) <= g * qfP[al])) for g in KGRID]
                    # k dense calibration fields: coverage spread over random draws
                    flds = cs.field.unique(); kk = {}
                    for kf in KS:
                        if kf > len(flds):
                            continue
                        covs = []
                        for _ in range(NDRAW):
                            pick = rng.choice(flds, kf, replace=False)
                            sub = cs[cs.field.isin(pick)]
                            qq = cp_q(sub.e.values, al, 1.0 / sub.groupby("field").e.transform("size").values)
                            covs.append(float(np.mean(np.abs(tst.e) <= qq)))
                        kk[str(kf)] = [float(np.mean(covs)), float(np.quantile(covs, 0.05)),
                                       float(np.quantile(covs, 0.95))]
                    r["k_dense"] = kk
                    fo[f"a{al}"] = r
                res["folds"].append(fo)
        json.dump(res, open(dst, "w"), indent=1, default=float)
        print(dst, flush=True)


if __name__ == "__main__":
    main()
