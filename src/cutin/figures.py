"""Figures from the CSVs in results/."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import ROOT

BLUE, ORANGE, GREEN, GREY, INK = "#0072B2", "#D55E00", "#009E73", "#8a8f98", "#222222"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(color="#e6e8eb", lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK, labelsize=9)


def fig_param(res, out):
    d = pd.read_csv(res / "parameterization.csv")
    fam = lambda m: BLUE if "Bezier" in m else (ORANGE if "jerk" in m else (GREEN if "tanh" in m else GREY))
    fig, axs = plt.subplots(1, 2, figsize=(10.4, 4.2))
    for ax, col, lab, log in ((axs[0], "rmse_cm_median", "median lateral error over the 6 s window (cm)", True),
                              (axs[1], "critical_ttc3_missed", "critical cut-ins (TTC < 3 s) lost in simulation", False)):
        for r in d.itertuples():
            v = getattr(r, col)
            ax.plot(r.parameters, v if not (log is False) else v * 100, "o", ms=8, color=fam(r.method), mec="white", mew=1.2)
            ax.annotate(r.method.replace(" (window)", "*").replace("polynomial degree", "poly").replace("Bezier degree", "Bezier"),
                        (r.parameters, v if not (log is False) else v * 100), textcoords="offset points", xytext=(6, 4), fontsize=7.5, color=INK)
        ax.set_xlabel("parameters per cut-in (all shape + timing + offsets)", fontsize=9)
        ax.set_ylabel(lab + ("" if log else " (%)"), fontsize=9)
        if log:
            ax.set_yscale("log")
        style(ax)
    axs[0].set_title("Fit error", loc="left", fontsize=10.5, color=INK)
    axs[1].set_title("What the test would conclude", loc="left", fontsize=10.5, color=INK)
    fig.text(0.01, 0.01, "* fitted to the whole 6 s window; the others fit a detected start-to-end maneuver", fontsize=7.5, color=GREY)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out / "parameterization.png", dpi=170)


def fig_strata(res, out):
    g = pd.read_csv(res / "strata.csv")
    g = g[g.budget == 100]
    kpis = list(g.kpi.unique())
    fig, axs = plt.subplots(1, len(kpis), figsize=(11, 3.8), sharey=False)
    spec = {"k-means on descriptors": BLUE, "k-means on curve parameters": GREEN, "k-means on PCA of raw trajectories": GREY,
            "regression tree on descriptors, fitted to pilot outcomes": ORANGE}
    for ax, kpi in zip(np.atleast_1d(axs), kpis):
        for name, c in spec.items():
            s = g[(g.kpi == kpi) & (g.strata == name)].sort_values("k")
            if len(s):
                ax.plot(s.k, s.eff_neyman, color=c, lw=2, marker="o", ms=5)
        s = g[(g.kpi == kpi)]
        ctrl = s.groupby("k").eff_control.mean()
        ax.plot(ctrl.index, ctrl.values, color="#555", lw=1.2, ls="--")
        ax.axhline(1.0, color=INK, lw=0.8)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("number of strata", fontsize=9)
        ax.set_title(kpi, loc="left", fontsize=9.5, color=INK)
        style(ax)
    np.atleast_1d(axs)[0].set_ylabel("simulation saving at equal precision (x)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "strata.png", dpi=170)


def fig_atlas(res, out, k=6):
    """k-means atlas of all real cut-ins on scenario descriptors: where they sit and how critical each group is."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    from . import features as F
    from .config import load
    from .replay import run_all
    from .study_param import load_data

    cfg = load()
    sig, meta, fps = load_data(cfg)
    centre = int(round(cfg["data"]["before_s"] * fps))
    X = F.scalar_features(sig, centre, fps, meta.lane_width.to_numpy())
    lab = KMeans(k, n_init=10, random_state=0).fit_predict(StandardScaler().fit_transform(X))
    o = run_all(sig, meta, cfg, repeats=5)
    crit, hard = (o["min_ttc"] < 3.0).mean((1, 2)), (o["peak_decel"] >= 2.0).mean((1, 2))
    order = np.argsort([X[lab == c, 4].mean() for c in range(k)])         # by mean TTC at the lane-change frame
    cols = [BLUE, ORANGE, GREEN, "#CC79A7", "#E69F00", "#56B4E9"]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axs[0]
    rows = []
    for rank, c in enumerate(order):
        m = lab == c
        ax.scatter(X[m, 0], X[m, 1], s=9, color=cols[rank], alpha=0.55, lw=0, rasterized=True)
        rows.append((rank, m.sum(), np.median(X[m, 0]), np.median(X[m, 1]), crit[m].mean() * 100, hard[m].mean() * 100))
        ax.text(np.median(X[m, 0]), np.median(X[m, 1]), str(rank + 1), fontsize=11, weight="bold", color="white",
                ha="center", va="center", bbox=dict(boxstyle="circle,pad=0.25", fc=cols[rank], ec="white", lw=1.2))
    ax.axhline(0, color=INK, lw=0.7)
    ax.set_xlim(0, 80)
    ax.set_xlabel("bumper gap to the follower at the lane-change frame (m)", fontsize=9)
    ax.set_ylabel("follower speed minus cutter speed (m/s)", fontsize=9)
    ax.set_title(f"{len(X):,} real cut-ins in {k} groups", loc="left", fontsize=10.5, color=INK)
    style(ax)
    ax = axs[1]
    ax.axis("off")
    ax.text(0.0, 1.0, "group   cut-ins   gap (m)   closing (m/s)   critical   hard brake", fontsize=8.8, weight="bold", color=INK, va="top", transform=ax.transAxes)
    for i, r in enumerate(rows):
        y = 0.9 - i * 0.105
        ax.scatter([0.01], [y], s=80, color=cols[r[0]], transform=ax.transAxes)
        ax.text(0.05, y, f"{r[0] + 1}        {r[1]:5d}     {r[2]:5.0f}        {r[3]:+5.1f}         {r[4]:4.1f}%       {r[5]:4.0f}%",
                fontsize=8.8, color=INK, va="center", transform=ax.transAxes)
    ax.text(0, 0.18, "critical = minimum time to collision below 3 s in the replay;\nhard brake = the cruise control demands at least 2 m/s2.\n"
            "Each cut-in is replayed against an ACC + AEB system.", fontsize=8, color=GREY, va="top", transform=ax.transAxes)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.14, top=0.92, wspace=0.05)
    fig.savefig(out / "atlas.png", dpi=170)


def run():
    res = ROOT / "results"
    out = res / "figures"
    out.mkdir(exist_ok=True)
    if (res / "parameterization.csv").exists():
        fig_param(res, out)
    if (res / "strata.csv").exists():
        fig_strata(res, out)
    fig_atlas(res, out)


if __name__ == "__main__":
    run()
