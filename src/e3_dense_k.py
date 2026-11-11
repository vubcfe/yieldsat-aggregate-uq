"""E3: label-value curve. How many densely labelled fields buy the within-field skill, and what do
aggregates add on top (mode D)?

Same data, clipping and block folds as E1/E2. In each training fold, k fields are drawn at random
(k = 1, 2, 5, 10, 20, 50; R = 3 draws per seed) and LightGBM is trained on their dense pixels
(<= 1000 per field, E1 hyper-parameters; min_data_in_leaf scaled down for tiny training sets).
Test predictions are scored in mode P, D1 (field aggregate known) and D2 (farm-year aggregate known).
k = all is the E1 oracle.

Second part (TRANSFER): train on all fields of the same crop in the OTHER countries and predict the
target country (no target label at all; mode D1 adds the target field aggregates). This is the
Vietnam-like case: no local dense labels, only aggregates.

Usage: python -I e3_dense_k.py FEAT_DIR RECORDS_CSV OUT_DIR [seeds]
"""
import json, os, sys, time

sys.modules["sklearn"] = None
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

FEAT, REC, OUT = sys.argv[1:4]
SEEDS = [int(s) for s in sys.argv[4].split(",")] if len(sys.argv) > 4 else [0, 1, 2]
N_TRAIN = 1000
KS = (1, 2, 5, 10, 20, 50)
R = 3
DROP = {"field", "row", "col", "y", "y_n", "y_sd"}
PARAMS = dict(objective="regression", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
              feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              num_threads=int(os.environ.get("NTHREADS", 16)), verbose=-1)
N_ROUNDS = 600


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
    fm = g.agg(y=("y", "mean"), p=("pred", "mean"))
    rhos = [spearman(d.y.values, d.pred.values) for _, d in g if len(d) >= 20 and d.y.std() > 0 and d.pred.std() > 0]
    return dict(r2_pixel=float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)),
                r2_within=float(1 - np.sum(((y - yb) - (p - pb)) ** 2) / np.sum((y - yb) ** 2)),
                r2_field=float(1 - np.sum((fm.y - fm.p) ** 2) / np.sum((fm.y - fm.y.mean()) ** 2)),
                rho_within=float(np.nanmedian(rhos)) if rhos else float("nan"))


def shift(p, y, key):
    s = pd.DataFrame(dict(p=p, y=y, k=key)).groupby("k")
    return p - s.p.transform("mean").values + s.y.transform("mean").values


def all_modes(y, p, field, fy):
    return dict(P=metrics(y, p, field), D1=metrics(y, shift(p, y, field), field),
                D2=metrics(y, shift(p, y, fy), field))


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


def load(rc, cap=None):
    """Lean loading (see src/lean_load.py). cap: max pixels per field (transfer part only)."""
    ids, X, feats, lo, hi = lean_load([f"{FEAT}/{c}/{f}.parquet" for c, f in zip(rc.country, rc.field)], cap)
    df = ids.merge(rc[["field", "block", "physical_field", "year", "fy"]], on="field")
    return df[["field", "y", "block", "fy"]].copy(), X, feats


def subsample(df, idx, seed):
    d = df.iloc[idx]
    sh = d.sample(frac=1.0, random_state=seed)
    return sh.index.values[sh.groupby("field").cumcount().values < N_TRAIN]


def train(X, y, seed, n):
    p = {**PARAMS, "seed": seed, "min_data_in_leaf": int(min(200, max(20, n // 50)))}
    return lgb.train(p, lgb.Dataset(X, y), N_ROUNDS)


def main():
    rec = pd.read_csv(REC)
    rec["fy"] = rec.farm_id + "|" + rec.year.astype(str)
    os.makedirs(OUT, exist_ok=True)
    part = os.environ.get("PART", "curve,transfer").split(",")
    if "curve" in part:
        for (country, crop), rc in rec.groupby(["country", "crop"]):
            tag = f"{country}_{crop}"; dst = f"{OUT}/curve_{tag}.json"
            if os.environ.get("ONLY") and tag not in os.environ["ONLY"].split(","):
                continue
            if os.path.exists(dst):
                continue
            t0 = time.time()
            df, X, _ = load(rc)
            y, field, fy = df.y.values, df.field.values, df.fy.values
            res = dict(country=country, crop=crop, ks=KS, R=R, runs=[])
            part_f = dst + ".partial"  # per-seed checkpoint (long jobs can be resumed after a crash); seeds are independent
            if os.path.exists(part_f):
                res = json.load(open(part_f))
            for seed in SEEDS:
                if any(r["seed"] == seed for r in res["runs"]):
                    continue
                rng = np.random.default_rng(seed)
                blocks = df.block.unique(); perm = dict(zip(blocks, rng.permutation(len(blocks))))
                rng_k = np.random.default_rng(2000 + seed)
                folds = list(group_kfold(df.block.map(perm).values, min(5, len(blocks))))
                for kf in KS:
                    for r in range(R):
                        pred = np.full(len(df), np.nan)
                        ok = True
                        for k, (tr, te) in enumerate(folds):
                            ftr = np.unique(field[tr])
                            if kf > len(ftr):
                                ok = False; break
                            pick = rng_k.choice(ftr, kf, replace=False)
                            sub = subsample(df, tr[np.isin(field[tr], pick)], seed)
                            m = train(X[sub], y[sub], seed, len(sub))
                            pred[te] = m.predict(X[te])
                        if ok:
                            res["runs"].append(dict(seed=seed, k=kf, rep=r, **all_modes(y, pred, field, fy)))
                print(f"{tag} seed {seed} ({time.time() - t0:.0f} s)", flush=True)
                json.dump(res, open(part_f + ".tmp", "w"), indent=1); os.replace(part_f + ".tmp", part_f)
            json.dump(res, open(dst + ".tmp", "w"), indent=1); os.replace(dst + ".tmp", dst)
            os.remove(part_f)
    if "transfer" in part:
        for crop, rcc in rec.groupby("crop"):
            countries = sorted(rcc.country.unique())
            if len(countries) < 2:
                continue
            dst = f"{OUT}/transfer_{crop}.json"
            if os.environ.get("ONLY_CROP") and crop not in os.environ["ONLY_CROP"].split(","):
                continue
            if os.path.exists(dst):
                continue
            df, X, feats = load(rcc, cap=2000)
            df["country"] = df.field.map(rcc.set_index("field").country)
            res = dict(crop=crop, runs=[])
            for tgt in countries:
                te = np.nonzero(df.country.values == tgt)[0]
                for src in [c for c in countries if c != tgt] + (["ALL_OTHER"] if len(countries) > 2 else []):
                    srcs = [c for c in countries if c != tgt] if src == "ALL_OTHER" else [src]
                    for seed in SEEDS:
                        sub = subsample(df, np.nonzero(df.country.isin(srcs).values)[0], seed)
                        m = train(X[sub], df.y.values[sub], seed, len(sub))
                        p = m.predict(X[te])
                        res["runs"].append(dict(target=tgt, source=src, seed=seed,
                                                **all_modes(df.y.values[te], p, df.field.values[te],
                                                            df.fy.values[te])))
                        print(crop, tgt, "<-", src, seed, round(res["runs"][-1]["P"]["r2_within"], 3), flush=True)
            json.dump(res, open(dst, "w"), indent=1)


if __name__ == "__main__":
    main()
