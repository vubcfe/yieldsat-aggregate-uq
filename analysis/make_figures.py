#!/usr/bin/env python3
"""Paper figures from results/ (mean over seeds). Writes outputs/figures/*.pdf. Colours: first three
categorical slots of the validated reference palette (blue, orange, aqua) plus neutral grey."""
import glob, json, os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "outputs", "figures")
BLUE, ORANGE, AQUA, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b", "#52514e"
PAIRS = ["AR-corn", "AR-soybean", "AR-wheat", "BR-corn", "BR-soybean", "BR-wheat", "DE-rapeseed", "DE-wheat",
         "UY-soybean"]
NAME = {"AR": "Argentina", "BR": "Brazil", "DE": "Germany", "UY": "Uruguay"}
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.5, "lines.linewidth": 1.5,
                     "legend.frameon": False, "pdf.fonttype": 42})


def load(exp):
    out = defaultdict(lambda: defaultdict(list))
    for f in glob.glob(os.path.join(ROOT, "results", exp, "seed_*.json")):
        for m, ms in json.load(open(f))["metrics"].items():
            for k, v in ms.items():
                out[m][k].append(v)
    return {m: {k: float(np.mean(v)) for k, v in ms.items()} for m, ms in out.items()}


def title(p):
    c, crop = p.split("-")
    return f"{NAME[c]} {crop}"


def fig_curve():
    E3, E2, E1 = load("e3"), load("e2"), load("e1")
    if not E3:
        return
    ks = [1, 2, 5, 10, 20, 50]
    fig, axs = plt.subplots(3, 3, figsize=(7.0, 5.6), sharex=True)
    for ax, p in zip(axs.flat, PAIRS):
        x = [k for k in ks if f"{p}_P_k{k}" in E3]
        ax.plot(x, [E3[f"{p}_P_k{k}"]["r2_within"] for k in x], color=BLUE, marker="o", ms=4,
                label="Dense labels on $k$ fields")
        o = E1.get(f"{p}_block", {}).get("r2_within")
        if o is not None:
            ax.axhline(o, color=GREY, ls="--", lw=1, label="Dense labels, all training fields")
        for key, col, lab in ((f"{p}_P_L1_B11", ORANGE, "Field aggregates (B11)"),
                              (f"{p}_P_L1_B2pv", AQUA, "Field aggregates (B2pv)")):
            if key in E2:
                ax.axhline(E2[key]["r2_within"], color=col, lw=1.5, label=lab)
        ax.set_xscale("log"); ax.set_xticks(ks); ax.set_xticklabels([str(k) for k in ks])
        ax.set_title(title(p), fontsize=8, color=INK); ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))
    for ax in axs[-1]:
        ax.set_xlabel("Densely labelled training fields $k$")
    for ax in axs[:, 0]:
        ax.set_ylabel("Within-field $R^2_w$")
    h, l = axs.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=7)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(os.path.join(FIG, "label_value_curve.pdf")); plt.close(fig)


def fig_cov():
    E = load("e45")
    if not E:
        return
    cons = [("a_indep", "(a) independent"), ("b_full", "(b) fully correlated"), ("c_vario", "(c) within-field covariance"),
            ("c_vario_re", "(d) (c) + field effect"), ("d_direct", "(e) direct conformal")]
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
    for ax, unit, lab in ((axs[0], "field", "Field mean"), (axs[1], "fy", "Farm-year mean")):
        for j, (c, _) in enumerate(cons):
            vals = [E[f"{p}_P_pred_a90"][f"cov_{unit}_{c}"] for p in PAIRS if f"{p}_P_pred_a90" in E]
            ax.scatter(vals, np.full(len(vals), j) + np.linspace(-0.15, 0.15, len(vals)), s=10, color=GREY,
                       label="Country, crop" if j == 0 else None, zorder=2)
            ax.scatter([np.mean(vals)], [j], s=40, color=BLUE, edgecolor="white", lw=1, zorder=3,
                       label="Mean of 9 pairs" if j == 0 else None)
        ax.axvline(0.9, color=INK, lw=0.8, ls=":", label="Nominal 90 %")
        ax.set_xlim(0, 1.02); ax.set_title(lab, fontsize=8, color=INK); ax.set_xlabel("Coverage")
        ax.set_yticks(range(len(cons))); ax.set_yticklabels([n for _, n in cons]); ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
    h, l = axs[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=7)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(os.path.join(FIG, "coverage_by_scale.pdf")); plt.close(fig)


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    fig_curve(); fig_cov()
    print(sorted(os.listdir(FIG)))
