"""Diagnostic for B2 (reviewer check): does the bag-level objective fit the training bags?

Re-trains B1, B2 and B2pv exactly as e2_aggregate.py (same functions, parameters, folds, subsample) on
fold 0 of seed 0 at level L1, and reports on the TRAINING subsample:
  bag_r2     R2 of bag means of the predictions against the bag labels (fit to the aggregates)
  within_r2  within-field R2 against the dense truth (pattern learnt inside training fields)
and on the test fold the within-field R2. A high bag_r2 with a near-zero within_r2 means the
optimiser works and the aggregates are fitted through covariates that are constant within bags.

Usage: python e2_trainfit_check.py FEAT_DIR RECORDS_CSV BAGS_L3_CSV E1_BLOCK_DIR OUT_JSON TAG[,TAG...]
"""
import json, os, sys

FEAT, REC, BAGS, E1DIR, OUT, TAGS = sys.argv[1:7]
sys.argv = [sys.argv[0], FEAT, REC, BAGS, E1DIR, "/nonexistent", "0"]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e2_aggregate as E  # noqa: E402  (module only defines functions; main() is guarded)

np, pd, lgb = E.np, E.pd, E.lgb


def r2(a, b):
    return float(1 - np.sum((a - b) ** 2) / np.sum((a - a.mean()) ** 2))


def within(y, p, f):
    s = pd.DataFrame({"y": y, "p": p, "f": f})
    g = s.groupby("f")
    return r2((s.y - g.y.transform("mean")).values, (s.p - g.p.transform("mean")).values)


rec = pd.read_csv(REC).merge(pd.read_csv(BAGS)[["field", "L3_20", "L3_50"]], on="field")
res = []
for tag in TAGS.split(","):
    country, crop = tag.split("_", 1)
    rc = rec[(rec.country == country) & (rec.crop == crop)]
    ids, X, feats, lo, hi = E.lean_load([f"{FEAT}/{country}/{f}.parquet" for f in rc.field])
    df = ids.merge(rc[["field", "block"]], on="field")
    y, field = df.y.values, df.field.values
    pv = np.array([f.startswith(E.PV_PREFIX) for f in feats])
    seed = 0
    rng = np.random.default_rng(seed)
    blocks = df.block.unique(); perm = dict(zip(blocks, rng.permutation(len(blocks))))
    e1 = pd.read_parquet(f"{E1DIR}/{tag}_seed{seed}_pred.parquet")
    tr, te = next(iter(E.group_kfold(df.block.map(perm).values, min(5, len(blocks)))))
    assert (e1.fold.values[te] == 0).all(), "folds differ from E1"
    sub = df.iloc[tr].groupby("field", group_keys=False).apply(
        lambda d: d.sample(min(len(d), E.N_TRAIN), random_state=seed)).index.values
    bnames, bag_all = np.unique(field[tr], return_inverse=True); nb = len(bnames)
    ybag = E.bag_mean(y[tr], bag_all, nb)
    bag_sub = bag_all[np.searchsorted(tr, sub)]
    base = float(ybag[bag_sub].mean())
    out = dict(tag=tag, seed=seed, fold=0, n_train_fields=int(nb))
    for name, Xm, kind in (("B1", X, "inst"), ("B2", X, "llp"), ("B2pv", X[:, pv], "llp")):
        if kind == "inst":
            m = lgb.train({**E.PARAMS, "seed": seed}, lgb.Dataset(Xm[sub], ybag[bag_sub]), E.N_ROUNDS)
            ptr, pte = m.predict(Xm[sub]), m.predict(Xm[te])
        else:
            ds = lgb.Dataset(Xm[sub], ybag[bag_sub], init_score=np.full(len(sub), base), free_raw_data=False)
            m = E.train_safe({**E.PARAMS, "seed": seed, "objective": E.llp_objective(bag_sub, ybag, nb)}, ds, name)
            ptr, pte = m.predict(Xm[sub]) + base, m.predict(Xm[te]) + base
        out[name] = dict(bag_r2=r2(ybag, E.bag_mean(ptr, bag_sub, nb)), within_r2_train=within(y[sub], ptr, field[sub]),
                         within_r2_test=within(y[te], pte, field[te]))
        print(tag, name, {k: round(v, 3) for k, v in out[name].items()}, flush=True)
    res.append(out)
json.dump(res, open(OUT, "w"), indent=1)
