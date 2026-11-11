"""E2: learning pixel yield maps from aggregate (bag) labels, per country x crop, leak-free block CV.

Same data, clipping, folds (block split) and per-field training subsample as E1, so the dense-label
oracle B0 (E1 predictions) is directly comparable; the script asserts that the folds match E1.

Bag levels (training side; bag label = mean of all clipped labelled pixels of the bag's training
records, i.e. the area-weighted "clean" aggregate):
  L1      field (record)
  L2      farm-year (farm_id x year)
  L3_20   pseudo-region: 20 km grid cell x crop x year (src/e2_bags.py)
  L3_50   pseudo-region: 50 km grid cell x crop x year
  Lrand   random bags of training records with the same size distribution as L2 (control for H1)

Methods (LightGBM M1, E1 hyper-parameters):
  B1  instance-level: every pixel gets its bag label
  B2  bag-mean squared loss (LLP): L = sum_b n_b (mean_{i in b} p_i - y_b)^2 / 2, custom objective with
      grad_i = pbar_b - y_b and hess_i = 1 (Newton step of a leaf holding whole bags = exact bag correction)
  B8  ecological (Goodman) ridge: bag label on bag-mean features, GCV-chosen penalty, applied to pixels;
      inputs winsorised to the training 0.5-99.5 % range per feature
      (= linear LLP: the bag mean of a linear model is the model at the bag-mean features)
  B2pv, B8pv  as B2 / B8 but on pixel-varying features only (S2 series and summaries, slope, curvature,
      aspect). Field-constant covariates (weather, season length) and identifying ones (elevation, 250 m
      soil) let a flexible learner fit bag labels with bag identifiers ("bag shortcut") and predict flat maps.
  B11 NDVI pseudo-labels: pixel label = bag label x NDVI_max / bag-mean NDVI_max (dasymetric), then
      LightGBM on all features (mode P model carrying an NDVI-shaped within-bag pattern)

Modes:
  P   prediction, no aggregate known at test time
  D   disaggregation: the test bag aggregate at the same level is known; prediction shifted additively
      so the bag mean equals the aggregate (Gaussian posterior with equal pixel variances).
      Extra mode-D baselines: FLAT (every pixel = bag aggregate) and B6 (NDVI-max proportional allocation).
      Lrand has no mode D (random bags are not an observable unit).
Metrics as in E1 (r2_within is field-centred and unchanged by mode D at any level >= field).

Usage: python -I e2_aggregate.py FEAT_DIR RECORDS_CSV BAGS_L3_CSV E1_BLOCK_DIR OUT_DIR [seeds]
"""
import json, os, sys, time

sys.modules["sklearn"] = None  # see e1_oracle.py: avoids a second libgomp
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

FEAT, REC, BAGS, E1DIR, OUT = sys.argv[1:6]
SEEDS = [int(s) for s in sys.argv[6].split(",")] if len(sys.argv) > 6 else [0, 1, 2]
ONLY = os.environ.get("ONLY")  # optional "Argentina_soybean,Brazil_corn"
N_TRAIN = 1000
DROP = {"field", "row", "col", "y", "y_n", "y_sd"}
PARAMS = dict(objective="regression", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
              feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              num_threads=int(os.environ.get("NTHREADS", 16)), verbose=-1)
N_ROUNDS = 600
LEVELS = ["L1", "L2", "L3_20", "L3_50", "Lrand"]
METHODS = ["B1", "B2", "B2pv", "B11", "B8", "B8pv"]
PV_PREFIX = ("B0", "B1", "B8A", "NDVI", "GCVI", "NDRE", "NDWI", "n_clear", "ndvi_", "topo_slope",
             "topo_curvature", "topo_aspect")  # S2 band columns are B02_t0 ... B12_t9, B8A_t*


DROP_COLS = {"field", "row", "col", "y", "y_n", "y_sd"}


def lean_load(paths, cap=None):
    """paths: list of parquet files in record order. cap: optional max pixels per field (random,
    seed 0, as DataFrame.sample(cap, random_state=0)). Returns (ids DataFrame [field,row,col,y], X, feats, lo, hi)."""
    cols, seen, ys, keep = [], set(), [], []
    for p in paths:
        names = pq.read_schema(p).names
        for c in names:
            if c not in seen:
                seen.add(c); cols.append(c)
        y = pd.read_parquet(p, columns=["y"]).y.to_numpy()
        idx = np.arange(len(y))
        if cap and len(y) > cap:
            idx = pd.Series(idx).sample(cap, random_state=0).to_numpy()
        keep.append(idx); ys.append(y[idx])
    yall = np.concatenate(ys)
    lo, hi = np.percentile(yall, [0.5, 99.5])
    feats = [c for c in cols if c not in DROP_COLS]
    fpos = {c: j for j, c in enumerate(feats)}
    masks = [(y > max(lo, 0)) & (y <= hi) for y in ys]
    n = int(sum(m.sum() for m in masks))
    X = np.full((n, len(feats)), np.nan, np.float32)
    ids, a = [], 0
    for p, idx, m in zip(paths, keep, masks):
        d = pd.read_parquet(p).iloc[idx][m]
        k = len(d)
        for c in d.columns:
            if c in fpos:
                v = d[c].to_numpy(np.float32)
                if c.startswith("soil_"):
                    v = np.where(v < 32000, v, np.nan).astype(np.float32)
                X[a:a + k, fpos[c]] = v
        ids.append(d[["field", "row", "col", "y"]]); a += k
    return pd.concat(ids, ignore_index=True), X, feats, lo, hi


def group_kfold(groups, n_splits):  # identical to e1_oracle.py
    uniq, inv, counts = np.unique(groups, return_inverse=True, return_counts=True)
    fold_of_group = np.empty(len(uniq), int); load = np.zeros(n_splits)
    for g in range(len(uniq)):
        f = int(np.argmin(load)); fold_of_group[g] = f; load[f] += counts[g]
    fold = fold_of_group[inv]
    for k in range(n_splits):
        yield np.nonzero(fold != k)[0], np.nonzero(fold == k)[0]


def spearman(a, b):
    return float(pd.Series(a).rank().corr(pd.Series(b).rank()))


def metrics(y, p, field):
    df = pd.DataFrame(dict(y=y, pred=p, field=field))
    g = df.groupby("field")
    yb, pb = g.y.transform("mean").values, g.pred.transform("mean").values
    sst_w = np.sum((y - yb) ** 2)
    fm = g.agg(y=("y", "mean"), p=("pred", "mean"))
    rhos = [spearman(d.y.values, d.pred.values) for _, d in g if len(d) >= 20 and d.y.std() > 0
            and d.pred.std() > 0]
    return dict(r2_pixel=float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)),
                rmse=float(np.sqrt(np.mean((y - p) ** 2))),
                r2_within=float(1 - np.sum(((y - yb) - (p - pb)) ** 2) / sst_w),
                r2_field=float(1 - np.sum((fm.y - fm.p) ** 2) / np.sum((fm.y - fm.y.mean()) ** 2)),
                rmse_field=float(np.sqrt(np.mean((fm.y - fm.p) ** 2))),
                rho_within=float(np.nanmedian(rhos)) if rhos else float("nan"))


def bag_mean(values, bag, n_bags):
    return np.bincount(bag, values, n_bags) / np.maximum(np.bincount(bag, None, n_bags), 1)


def llp_objective(bag, ybag, n_bags):
    def f(preds, data):
        pbar = bag_mean(preds, bag, n_bags)
        return (pbar - ybag)[bag], np.ones_like(preds)
    return f


RETRIES = []  # (level, method, fold, fallback) for the JSON record


def train_safe(params, ds, tag):
    """LightGBM with a custom objective can abort with 'best_split_info.left_count > 0' depending on the
    thread count (gradients are constant within bags). Retry with row-wise histograms, then with a tiny
    min_gain_to_split; every retry is recorded."""
    for extra in ({}, {"force_row_wise": True}, {"force_row_wise": True, "min_gain_to_split": 1e-6}):
        try:
            m = lgb.train({**params, **extra}, ds, N_ROUNDS)
            if extra:
                RETRIES.append(dict(where=tag, fallback=extra))
                print(f"  retry ok {tag} with {extra}", flush=True)
            return m
        except lgb.basic.LightGBMError as e:
            print(f"  LightGBMError {tag} {extra}: {str(e)[:80]}", flush=True)
            ds = lgb.Dataset(ds.data, ds.label, init_score=ds.init_score, free_raw_data=False)
    raise RuntimeError(f"{tag}: all LightGBM fallbacks failed")


def ridge_gcv(Xb, yb):
    """Ridge with intercept, penalty chosen by generalised cross-validation. Returns (coef, intercept)."""
    mu, ym = Xb.mean(0), yb.mean()
    U, s, Vt = np.linalg.svd(Xb - mu, full_matrices=False)
    uy = U.T @ (yb - ym)
    n = len(yb); best = None
    for a in np.logspace(-3, 5, 33) * max(s[0] ** 2, 1e-12):
        d = s / (s ** 2 + a)
        fit = U @ (s * d * uy)
        gcv = n * np.sum((yb - ym - fit) ** 2) / max(n - np.sum(s * d), 1e-3) ** 2
        if best is None or gcv < best[0]:
            best = (gcv, a, Vt.T @ (d * uy))
    coef = best[2]
    return coef, ym - mu @ coef


def make_bags(rc_train, level, rng, l2_sizes):
    """Map each training record to a bag id at the given level."""
    if level == "L1":
        key = rc_train.field
    elif level == "L2":
        key = rc_train.farm_id + "|" + rc_train.year.astype(str)
    elif level.startswith("L3_"):
        key = rc_train[level]
    elif level == "Lrand":
        f = rng.permutation(rc_train.field.values)
        sizes = rng.permutation(l2_sizes)
        key, i, b = {}, 0, 0
        while i < len(f):
            s = int(sizes[b % len(sizes)])
            for x in f[i:i + s]:
                key[x] = f"r{b}"
            i += s; b += 1
        key = rc_train.field.map(key)
    return dict(zip(rc_train.field, key))


def main():
    rec = pd.read_csv(REC).merge(pd.read_csv(BAGS)[["field", "L3_20", "L3_50"]], on="field")
    os.makedirs(OUT, exist_ok=True)
    for (country, crop), rc in rec.groupby(["country", "crop"]):
        tag = f"{country}_{crop}"
        if ONLY and tag not in ONLY.split(","):
            continue
        if all(os.path.exists(f"{OUT}/{tag}_seed{s}.json") for s in SEEDS):
            continue
        t0 = time.time()
        # lean loading: same X, ids and feature order as e1_oracle.py (checked bit-identical)
        ids, X, feats, lo, hi = lean_load([f"{FEAT}/{country}/{f}.parquet" for f in rc.field])
        df = ids.merge(rc[["field", "block", "physical_field", "year"]], on="field")
        ndvi = X[:, feats.index("ndvi_max")].copy() if "ndvi_max" in feats else np.full(len(df), np.nan, np.float32)
        pv = np.array([f.startswith(PV_PREFIX) for f in feats])
        assert not any(f.startswith(("w_", "soil_", "season", "topo_dem")) for f in np.array(feats)[pv])
        y, field = df.y.values, df.field.values
        rcf = rc.set_index("field")
        print(f"{tag}: {len(df):,} px, {df.field.nunique()} fields, {df.block.nunique()} blocks", flush=True)
        for seed in SEEDS:
            dst = f"{OUT}/{tag}_seed{seed}.json"
            if os.path.exists(dst):
                continue
            rng = np.random.default_rng(seed)
            unit = df["block"]
            blocks = unit.unique(); perm = dict(zip(blocks, rng.permutation(len(blocks))))
            rng_bag = np.random.default_rng(1000 + seed)
            e1 = pd.read_parquet(f"{E1DIR}/{tag}_seed{seed}_pred.parquet")
            assert len(e1) == len(df) and (e1.field.values == field).all()
            pred = {"B0": e1.pred.values.astype(np.float64)}
            for lv in LEVELS:
                for m in METHODS:
                    pred[f"{lv}_{m}"] = np.full(len(df), np.nan)
            agg_d = {lv: np.full(len(df), np.nan) for lv in LEVELS if lv != "Lrand"}  # test aggregate per pixel
            bid_d = {lv: np.full(len(df), -1, np.int64) for lv in agg_d}  # test bag id per pixel (global)
            info = []
            for k, (tr, te) in enumerate(group_kfold(unit.map(perm).values, min(5, len(blocks)))):
                assert (e1.fold.values[te] == k).all(), "folds differ from E1"
                trd = df.iloc[tr]
                sub = trd.groupby("field", group_keys=False).apply(
                    lambda d: d.sample(min(len(d), N_TRAIN), random_state=seed)).index.values
                f_tr = np.unique(field[tr]); f_te = np.unique(field[te])
                rc_tr, rc_te = rcf.loc[f_tr].reset_index(), rcf.loc[f_te].reset_index()
                l2 = (rc_tr.farm_id + "|" + rc_tr.year.astype(str)).value_counts().values
                # column means of the training subsample for NaN filling (B8 only)
                # B8 inputs are winsorised to the training 0.5-99.5 % range per feature: ratio indices
                # (e.g. GCVI with B03 near 0) have extreme pixel values that a linear model extrapolates
                wlo, whi = np.nanpercentile(X[sub], [0.5, 99.5], axis=0)
                wlo = np.where(np.isfinite(wlo), wlo, -np.inf); whi = np.where(np.isfinite(whi), whi, np.inf)
                Xs = np.clip(X[sub], wlo, whi)
                cm = np.nanmean(Xs, 0); cm = np.where(np.isfinite(cm), cm, 0)
                sd = np.nanstd(Xs, 0); sd = np.where(sd > 0, sd, 1)
                for lv in LEVELS:
                    bmap = make_bags(rc_tr, lv, rng_bag, l2)
                    # bag ids over ALL training pixels (labels from all pixels, not the subsample)
                    bnames, bag_all = np.unique(pd.Series(field[tr]).map(bmap).values, return_inverse=True)
                    nb = len(bnames)
                    ybag = bag_mean(y[tr], bag_all, nb)
                    pos = np.searchsorted(tr, sub)  # sub indices are a subset of tr (sorted)
                    bag_sub = bag_all[pos]
                    info.append(dict(fold=k, level=lv, n_bags=int(nb),
                                     median_fields_per_bag=float(np.median(pd.Series(bmap).value_counts()))))
                    # B1 instance-level
                    m1 = lgb.train({**PARAMS, "seed": seed}, lgb.Dataset(X[sub], ybag[bag_sub]), N_ROUNDS)
                    pred[f"{lv}_B1"][te] = m1.predict(X[te])
                    # B2 LLP bag-mean loss, initialised at the mean bag label
                    base = float(ybag[bag_sub].mean())
                    ds = lgb.Dataset(X[sub], ybag[bag_sub], init_score=np.full(len(sub), base), free_raw_data=False)
                    m2 = train_safe({**PARAMS, "seed": seed, "objective": llp_objective(bag_sub, ybag, nb)},
                                    ds, f"seed{seed} fold{k} {lv} B2")
                    pred[f"{lv}_B2"][te] = m2.predict(X[te]) + base
                    Xpv = X[:, pv]
                    ds = lgb.Dataset(Xpv[sub], ybag[bag_sub], init_score=np.full(len(sub), base), free_raw_data=False)
                    m2 = train_safe({**PARAMS, "seed": seed, "objective": llp_objective(bag_sub, ybag, nb)},
                                    ds, f"seed{seed} fold{k} {lv} B2pv")
                    pred[f"{lv}_B2pv"][te] = m2.predict(Xpv[te]) + base
                    # B11 NDVI pseudo-labels
                    nv = ndvi[sub].astype(np.float64)
                    nb_ = bag_mean(np.nan_to_num(nv), bag_sub, nb) / np.maximum(
                        bag_mean(np.isfinite(nv).astype(float), bag_sub, nb), 1e-9)
                    nv = np.where(np.isfinite(nv), nv, nb_[bag_sub])
                    m11 = lgb.train({**PARAMS, "seed": seed},
                                    lgb.Dataset(X[sub], ybag[bag_sub] * nv / nb_[bag_sub]), N_ROUNDS)
                    pred[f"{lv}_B11"][te] = m11.predict(X[te])
                    # B8 Goodman ecological ridge on standardised bag-mean features
                    Z = ((np.where(np.isnan(Xs), cm, Xs) - cm) / sd).astype(np.float64)
                    Zb = np.stack([np.bincount(bag_sub, Z[:, j], nb) for j in range(Z.shape[1])], 1) \
                        / np.bincount(bag_sub, None, nb)[:, None]
                    del Z
                    for name, cols in (("B8", np.ones(len(feats), bool)), ("B8pv", pv)):
                        coef, b0 = ridge_gcv(Zb[:, cols], ybag)
                        w = np.zeros(len(feats)); w[cols] = coef / sd[cols]; c0 = b0 - cm @ w
                        for a in range(0, len(te), 200_000):  # chunked: test sets reach millions of pixels
                            Xc = np.clip(X[te[a:a + 200_000]], wlo, whi)
                            pred[f"{lv}_{name}"][te[a:a + 200_000]] = np.where(np.isnan(Xc), cm, Xc) @ w + c0
                    for mname in METHODS:
                        if not np.isfinite(pred[f"{lv}_{mname}"][te]).all():
                            raise RuntimeError(f"{tag} seed {seed} fold {k} {lv} {mname}: non-finite")
                    # mode-D aggregate on the test side at the same level (all test pixels)
                    if lv != "Lrand":
                        tmap = make_bags(rc_te, lv, None, None)
                        _, bt = np.unique(pd.Series(field[te]).map(tmap).values, return_inverse=True)
                        agg_d[lv][te] = bag_mean(y[te], bt, bt.max() + 1)[bt]
                        bid_d[lv][te] = bt + k * 1_000_000
                print(f"  seed {seed} fold {k} done ({time.time() - t0:.0f} s)", flush=True)

            res = dict(country=country, crop=crop, seed=seed, split="block", levels=LEVELS, methods=METHODS,
                       n_train_per_field=N_TRAIN, params=PARAMS, n_rounds=N_ROUNDS, bags=info,
                       P={}, D={}, seconds=None, lightgbm_retries=[r for r in RETRIES if r["where"].startswith(f"seed{seed} ")])
            res["P"]["B0"] = metrics(y, pred["B0"], field)
            for lv in LEVELS:
                for m in METHODS:
                    res["P"][f"{lv}_{m}"] = metrics(y, pred[f"{lv}_{m}"], field)
            # mode D: shift each test bag's mean prediction onto its known aggregate
            for lv in agg_d:
                _, bid = np.unique(bid_d[lv], return_inverse=True)
                for m in ["B0"] + METHODS:
                    p = pred["B0"] if m == "B0" else pred[f"{lv}_{m}"]
                    pd_ = p - bag_mean(p, bid, bid.max() + 1)[bid] + agg_d[lv]
                    res["D"][f"{lv}_{m}"] = metrics(y, pd_, field)
                res["D"][f"{lv}_FLAT"] = metrics(y, agg_d[lv], field)
                nd = np.where(np.isfinite(ndvi), ndvi, np.nan)
                ndb = bag_mean(np.nan_to_num(nd, nan=0), bid, bid.max() + 1) / np.maximum(
                    bag_mean(np.isfinite(nd).astype(float), bid, bid.max() + 1), 1e-9)
                nd = np.where(np.isfinite(nd), nd, ndb[bid])
                res["D"][f"{lv}_B6"] = metrics(y, agg_d[lv] * nd / ndb[bid], field)
            res["seconds"] = round(time.time() - t0, 1)
            json.dump(res, open(dst, "w"), indent=1)
            pd.DataFrame({k: v.astype(np.float32) for k, v in pred.items() if k != "B0"}).assign(
                fold=e1.fold.values).to_parquet(f"{OUT}/{tag}_seed{seed}_pred.parquet", index=False)
            P = res["P"]
            print(f"  seed {seed}: r2_within B0 {P['B0']['r2_within']:.3f} | " + " ".join(
                f"{lv}:" + "/".join(f"{P[f'{lv}_{m}']['r2_within']:.2f}" for m in METHODS) for lv in LEVELS)
                + f" ({res['seconds']} s)", flush=True)


if __name__ == "__main__":
    main()
