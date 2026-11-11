#!/usr/bin/env python3
"""Collect the raw outputs of E1-E6 (directory $RUNS, default runs/, as written by run_all.sh) into
per-seed summaries results/<exp>/seed_<s>.json, the input of make_tables.py and make_figures.py.

Method keys:
  e1    <PAIR>_<split>                          split in field, physical_field, block
  e2    <PAIR>_<mode>_<level>_<method>          mode P or D; also MEAN9_* and MEAN5_* (pair averages)
  e3    <PAIR>_<mode>_k<k>                       mean over the R draws; also MEAN9_/MEAN5_
  e3t   <crop>_<target>_<source>_<mode>
  e45   <PAIR>_<mode>_<column>_a<90|80>          coverage and width metrics; also MEAN9_
  e6    <PAIR>
PAIR = two-letter country + crop, e.g. AR-soybean. MEAN5 = the five pairs whose E1 block oracle has
within-field R2 >= 0.20 (AR corn, soybean, wheat; BR corn, soybean).
"""
import glob, json, os
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A = os.environ.get("RUNS", os.path.join(ROOT, "runs"))
MEAN5 = {"AR-corn", "AR-soybean", "AR-wheat", "BR-corn", "BR-soybean"}



ISO = {"Argentina": "AR", "Brazil": "BR", "Germany": "DE", "Uruguay": "UY"}


def pair(country, crop):
    return ISO[country] + "-" + crop


def clean(d):
    return {k: float(v) for k, v in d.items() if isinstance(v, (int, float)) and np.isfinite(v)}


def add_means(res):
    """res[seed][method] -> add MEAN9_/MEAN5_ for keys that start with a pair."""
    for seed, ms in res.items():
        groups = defaultdict(list)
        for m, v in ms.items():
            p, rest = m.split("_", 1)
            if "-" in p:
                groups[rest].append((p, v))
        for rest, lst in groups.items():
            for name, sel in (("MEAN9", None), ("MEAN5", MEAN5)):
                vs = [v for p, v in lst if sel is None or p in sel]
                if len(vs) < (9 if sel is None else 5):
                    continue
                keys = set.intersection(*[set(v) for v in vs])
                ms_new = {k: float(np.mean([v[k] for v in vs])) for k in keys}
                ms[f"{name}_{rest}"] = ms_new


def write(exp, res, n_test):
    d = os.path.join(ROOT, "results", exp); os.makedirs(d, exist_ok=True)
    for f in glob.glob(f"{d}/seed_*.json"):
        os.remove(f)
    for seed, ms in sorted(res.items()):
        json.dump(dict(seed=seed, n_test=n_test, metrics=ms),
                  open(f"{d}/seed_{seed}.json", "w"), indent=1, sort_keys=True)
    print(exp, len(res), "seeds,", len(next(iter(res.values()))), "methods")


def main():
    n_pix = 0
    # E1
    res = defaultdict(dict)
    for p in glob.glob(f"{A}/e1/*/*_seed*.json"):
        d = json.load(open(p))
        res[d["seed"]][f"{pair(d['country'], d['crop'])}_{d['split']}"] = clean(d["pooled"])
        if d["split"] == "block" and d["seed"] == 0:
            n_pix += d["pooled"]["n_pix"]
    if res:
        add_means(res); write("e1", res, n_pix)
    # E2
    res = defaultdict(dict)
    for p in glob.glob(f"{A}/e2/*_seed*.json"):
        d = json.load(open(p)); pr = pair(d["country"], d["crop"])
        for mode in ("P", "D"):
            for k, v in d[mode].items():
                res[d["seed"]][f"{pr}_{mode}_{'L0_B0' if k == 'B0' else k}"] = clean(v)
    if res:
        add_means(res); write("e2", res, n_pix)
    # E3 curve and transfer
    res = defaultdict(lambda: defaultdict(list))
    for p in glob.glob(f"{A}/e3/curve_*.json"):
        d = json.load(open(p)); pr = pair(d["country"], d["crop"])
        for r in d["runs"]:
            for mode in ("P", "D1", "D2"):
                res[r["seed"]][f"{pr}_{mode}_k{r['k']}"].append(clean(r[mode]))
    if res:
        out = {s: {m: {k: float(np.mean([x[k] for x in v])) for k in v[0]} for m, v in ms.items()}
               for s, ms in res.items()}
        # k-equivalent of aggregate learning: the number of densely labelled training fields (per fold) whose
        # curve reaches the same R2_w as an E2 method with field aggregates (L1), interpolated in log k,
        # same seed; clipped to [1, 50] with flags below/above.
        e2s = defaultdict(dict)
        for p in glob.glob(f"{A}/e2/*_seed*.json"):
            d = json.load(open(p))
            for k, v in d["P"].items():
                e2s[d["seed"]][f"{pair(d['country'], d['crop'])}_{k}"] = v
        for s, ms in out.items():
            for pr in {m.split("_")[0] for m in ms if "-" in m.split("_")[0]}:
                ks = sorted(int(m.split("_k")[1]) for m in ms if m.startswith(f"{pr}_P_k"))
                curve = np.maximum.accumulate([ms[f"{pr}_P_k{k}"]["r2_within"] for k in ks])
                acc = {}
                for meth in ("B11", "B2pv", "B8"):
                    v = e2s.get(s, {}).get(f"{pr}_L1_{meth}", {}).get("r2_within")
                    if v is None:
                        continue
                    acc[f"keq_{meth}"] = float(np.exp(np.interp(v, curve, np.log(ks))))
                    acc[f"keq_{meth}_below"] = float(v < curve[0])
                    acc[f"keq_{meth}_above"] = float(v > curve[-1])
                if acc:
                    ms[f"{pr}_keq"] = acc
            # median over pairs of the clipped k-equivalent (robust to the open ends <1 and >50)
            for name, sel in (("MED9", None), ("MED5", MEAN5)):
                kq = [v for m, v in ms.items() if m.endswith("_keq") and "-" in m.split("_")[0]
                      and (sel is None or m.split("_")[0] in sel)]
                if len(kq) == (9 if sel is None else 5):
                    ms[f"{name}_keq"] = {f"keq_{meth}": float(np.median([v[f"keq_{meth}"] for v in kq]))
                                         for meth in ("B11", "B2pv", "B8") if all(f"keq_{meth}" in v for v in kq)}
        add_means(out); write("e3", out, n_pix)
    res = defaultdict(dict)
    for p in glob.glob(f"{A}/e3/transfer_*.json"):
        d = json.load(open(p))
        for r in d["runs"]:
            for mode in ("P", "D1", "D2"):
                res[r["seed"]][f"{d['crop']}_{ISO[r['target']]}_{ISO.get(r['source'], 'ALL')}_{mode}"] = clean(r[mode])
    if res:
        write("e3t", res, n_pix)
    # E4/E5
    res = defaultdict(dict)
    # kappa transfer (aggregate-only pixel calibration): kappa of a target is replaced by the median kappa
    # (over all folds and seeds) of the other pairs (rule all), the same crop elsewhere (crop) or the same
    # country, other crops (country); target coverage is read off its own coverage-vs-kappa curve per fold.
    KGRID = np.geomspace(0.1, 30, 120)
    E45 = [json.load(open(p)) for p in glob.glob(f"{A}/e45/*/*.json")]
    kap = defaultdict(list)
    for d in E45:
        for f in d["folds"]:
            for al in ("0.1", "0.2"):
                kap[(d["column"], d["tag"], f["mode"], al)].append(f[f"a{al}"]["kappa"])
    kmed = {k: float(np.median(v)) for k, v in kap.items()}

    def kt_cov(d, F, al):
        country, crop = d["tag"].split("_", 1)
        out = {}
        for rule in ("all", "crop", "country", "own"):
            if rule == "own":
                ks = [kmed[(d["column"], d["tag"], F[0]["mode"], al)]]
            else:
                ks = [v for (c, t, m, a), v in kmed.items() if c == d["column"] and m == F[0]["mode"] and a == al
                      and t != d["tag"] and (rule == "all" or (rule == "crop" and t.split("_", 1)[1] == crop)
                                             or (rule == "country" and t.split("_", 1)[0] == country))]
            if not ks:
                continue
            k = float(np.median(ks))
            cov = [np.interp(np.log(k), np.log(KGRID), f[f"a{al}"]["cov_px_vs_kappa"]) for f in F]
            out[f"cov_px_kt_{rule}"] = float(np.average(cov, weights=[f["n_test_fields"] for f in F]))
            out[f"kappa_kt_{rule}"] = k
        return out

    for d in E45:
        country, crop = d["tag"].split("_", 1); pr = pair(country, crop)
        for mode in ("P", "D1", "D2"):
            F = [f for f in d["folds"] if f["mode"] == mode]
            for al, nm in (("0.1", "a90"), ("0.2", "a80")):
                acc = {}
                for key in F[0][f"a{al}"]:
                    if not isinstance(F[0][f"a{al}"][key], (int, float)):
                        continue
                    w = [f["n_test_fy"] if "_fy_" in key else f["n_test_fields"] for f in F]
                    vals = [f[f"a{al}"][key] for f in F]
                    if all(np.isfinite(vals)):
                        acc[key] = float(np.average(vals, weights=w))
                for key in ("rmse_field", "rmse_px", "rmse_fy", "sigma_field2"):
                    acc[key] = float(np.average([f[key] for f in F], weights=[f["n_test_fields"] for f in F]))
                acc["range_m"] = float(np.median([f["vario"]["range_m"] for f in F]))
                nf = np.array([f["n_test_fields"] for f in F]); mse = np.array([f["rmse_field"] ** 2 for f in F])
                for nm2, key in (("varratio_c", "mean_var_field_vario"), ("varratio_d", "mean_var_field_re")):
                    acc[nm2] = float(np.sum(nf * mse) / np.sum(nf * np.array([f[key] for f in F])))
                for kf in F[0][f"a{al}"]["k_dense"]:
                    acc[f"kdense{kf}_mean"] = float(np.average([f[f"a{al}"]["k_dense"][kf][0] for f in F],
                                                               weights=[f["n_test_fields"] for f in F]))
                    acc[f"kdense{kf}_p05"] = float(np.average([f[f"a{al}"]["k_dense"][kf][1] for f in F],
                                                              weights=[f["n_test_fields"] for f in F]))
                acc.update(kt_cov(d, F, al))
                res[d["seed"]][f"{pr}_{mode}_{d['column'].replace('_', '')}_{nm}"] = acc
    if res:
        add_means(res); write("e45", res, n_pix)
    # E6
    res = defaultdict(dict)
    for p in glob.glob(f"{A}/e6/*.json"):
        col = os.path.basename(p)[:-5]
        for r in json.load(open(p)):
            country, crop = r["tag"].split("_", 1)
            res[r["seed"]][f"{pair(country, crop)}_{col}"] = clean(r)
    if res:
        write("e6", res, n_pix)


if __name__ == "__main__":
    main()
