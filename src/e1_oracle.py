"""E1: dense-label oracle (upper bound for RQ1) per country x crop, leak-free block CV.

Model: LightGBM on the E0 pixel features, trained on dense pixel labels.
Split: GroupKFold(5) over blocks (connected farms that share land, from e0/records.csv).
Labels: pixels outside the [0.5, 99.5] percentile of their country x crop are dropped
(YieldSAT's own caps are loose, e.g. corn 45 t/ha). Up to N_TRAIN pixels per field for training;
all test pixels are scored.

Metrics (per fold and pooled out-of-fold):
  r2_pixel      R2 over all test pixels (mixes between- and within-field variance)
  r2_within     within-field R2: truth and prediction each centred on their own field mean,
                1 - sum((y - ybar_f) - (p - pbar_f))^2 / sum((y - ybar_f)^2). Measures the
                within-field pattern only (the part aggregate labels do not contain).
  r2_within_puffin  Puffin Eq. 6: both centred on the TRUE field mean (penalises field bias)
  r2_field      R2 of field means
  rho_within    median over fields of the Spearman correlation inside the field
Predictions are saved for E4/E5.

Usage: python -I e1_oracle.py FEAT_DIR RECORDS_CSV OUT_DIR [seeds]
"""
import glob, json, os, sys, time

sys.modules["sklearn"] = None  # lightgbm imports sklearn if present, which loads a second libgomp
import lightgbm as lgb
import numpy as np
import pandas as pd
# sklearn and scipy are NOT imported: their pip wheels bundle a second libgomp, and two OpenMP
# runtimes in one process with LightGBM caused heap corruption and non-finite predictions on our machine.


def group_kfold(groups, n_splits):
    """Assign whole groups to folds, visiting groups in label order (labels are a per-seed random
    permutation, so folds differ between seeds) and putting each into the currently smallest fold."""
    uniq, inv, counts = np.unique(groups, return_inverse=True, return_counts=True)
    fold_of_group = np.empty(len(uniq), int); load = np.zeros(n_splits)
    for g in range(len(uniq)):
        f = int(np.argmin(load)); fold_of_group[g] = f; load[f] += counts[g]
    fold = fold_of_group[inv]
    for k in range(n_splits):
        yield np.nonzero(fold != k)[0], np.nonzero(fold == k)[0]


def spearman(a, b):
    return float(pd.Series(a).rank().corr(pd.Series(b).rank()))

FEAT, REC, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
SEEDS = [int(s) for s in sys.argv[4].split(",")] if len(sys.argv) > 4 else [0, 1, 2]
# SPLIT=block (default, leak-free), physical_field (no shared land, farms may be shared)
# or field (field-ID grouping, as in the YieldSAT benchmark)
SPLIT = os.environ.get("SPLIT", "block")
N_TRAIN = 1000
DROP = {"field", "row", "col", "y", "y_n", "y_sd"}
PARAMS = dict(objective="regression", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
              feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              num_threads=int(os.environ.get("NTHREADS", 16)), verbose=-1)
N_ROUNDS = 600


def metrics(df):
    y, p = df.y.values, df.pred.values
    g = df.groupby("field")
    yb, pb = g.y.transform("mean").values, g.pred.transform("mean").values
    sst_w = np.sum((y - yb) ** 2)
    fm = g.agg(y=("y", "mean"), p=("pred", "mean"))
    rhos = [spearman(d.y.values, d.pred.values) for _, d in g if len(d) >= 20 and d.y.std() > 0]
    return dict(n_pix=int(len(df)), n_fields=int(fm.shape[0]),
                r2_pixel=float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)),
                rmse=float(np.sqrt(np.mean((y - p) ** 2))), mae=float(np.mean(np.abs(y - p))),
                r2_within=float(1 - np.sum(((y - yb) - (p - pb)) ** 2) / sst_w),
                # Puffin Eq. 6 with the true field mean as centre for both terms reduces to this
                r2_within_puffin=float(1 - np.sum((y - p) ** 2) / sst_w),
                r2_field=float(1 - np.sum((fm.y - fm.p) ** 2) / np.sum((fm.y - fm.y.mean()) ** 2)),
                rho_within=float(np.nanmedian(rhos)) if rhos else float("nan"))


def main():
    rec = pd.read_csv(REC)
    os.makedirs(OUT, exist_ok=True)
    for (country, crop), rc in rec.groupby(["country", "crop"]):
        tag = f"{country}_{crop}"
        if all(os.path.exists(f"{OUT}/{tag}_seed{s}.json") for s in SEEDS):
            continue
        t0 = time.time()
        df = pd.concat([pd.read_parquet(f"{FEAT}/{country}/{f}.parquet") for f in rc.field], ignore_index=True)
        soil = [c for c in df.columns if c.startswith("soil_")]
        df[soil] = df[soil].where(df[soil] < 32000)  # SoilGrids int16 nodata (32767 and neighbours)
        lo, hi = np.percentile(df.y, [0.5, 99.5])
        df = df[(df.y > max(lo, 0)) & (df.y <= hi)].reset_index(drop=True)
        df = df.merge(rc[["field", "block", "physical_field", "year"]], on="field")
        feats = [c for c in df.columns if c not in DROP | {"block", "physical_field", "year"}]
        X = df[feats].to_numpy(np.float32)
        df = df[["field", "row", "col", "y", "block", "physical_field", "year"]].copy()  # free the feature columns
        print(f"{tag}: {len(df):,} px, {df.field.nunique()} fields, {df.block.nunique()} blocks, "
              f"{len(feats)} feats, clip [{lo:.2f}, {hi:.2f}]", flush=True)
        for seed in SEEDS:
            dst = f"{OUT}/{tag}_seed{seed}.json"
            if os.path.exists(dst):
                continue
            rng = np.random.default_rng(seed)
            pred = np.full(len(df), np.nan, np.float32)
            fold_of = np.full(len(df), -1, np.int8)
            folds = []
            # shuffle block order per seed so the 5 folds differ between seeds
            unit = df[SPLIT]
            blocks = unit.unique(); perm = dict(zip(blocks, rng.permutation(len(blocks))))
            for k, (tr, te) in enumerate(group_kfold(unit.map(perm).values, min(5, len(blocks)))):
                trd = df.iloc[tr]
                sub = trd.groupby("field", group_keys=False).apply(
                    lambda d: d.sample(min(len(d), N_TRAIN), random_state=seed)).index.values
                m = lgb.train({**PARAMS, "seed": seed}, lgb.Dataset(X[sub], df.y.values[sub]), N_ROUNDS)
                pt = m.predict(X[te])
                n_bad = int(np.sum(~np.isfinite(pt)))
                if n_bad:  # rare, not reproducible in isolation: multi-threaded predict on large arrays
                    print(f"  WARNING {tag} seed {seed} fold {k}: {n_bad} non-finite predictions, re-predicting single-threaded", flush=True)
                    pt = m.predict(X[te].astype(np.float64), num_threads=1)
                    if not np.isfinite(pt).all():
                        raise RuntimeError(f"{tag} seed {seed} fold {k}: still non-finite after re-predict")
                pred[te] = pt; fold_of[te] = k
                fm = metrics(df.iloc[te].assign(pred=pred[te]))
                fm.update(fold=k, n_blocks_test=int(df.block.iloc[te].nunique()))
                fm["n_repredicted"] = n_bad
                folds.append(fm)
            res = dict(country=country, crop=crop, seed=seed, split=SPLIT, n_train_per_field=N_TRAIN, params=PARAMS,
                       n_rounds=N_ROUNDS, clip=[float(lo), float(hi)], pooled=metrics(df.assign(pred=pred)),
                       folds=folds, seconds=round(time.time() - t0, 1))
            json.dump(res, open(dst, "w"), indent=1)
            df[["field", "row", "col", "y"]].assign(pred=pred, fold=fold_of).to_parquet(
                f"{OUT}/{tag}_seed{seed}_pred.parquet", index=False)
            p = res["pooled"]
            print(f"  [{SPLIT}] seed {seed}: r2_pixel {p['r2_pixel']:.3f} r2_within {p['r2_within']:.3f} "
                  f"r2_field {p['r2_field']:.3f} rho_within {p['rho_within']:.3f} ({res['seconds']} s)", flush=True)


if __name__ == "__main__":
    main()
