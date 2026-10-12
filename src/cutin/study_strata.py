"""C6 study, part 2: how many strata, and does knowing the pilot outcomes help?

Strata are built on one half of the recordings (a "pilot campaign") and used to plan sampling on the other half.
 - unsupervised: k-means on scenario descriptors / curve parameters / PCA of raw trajectories, k = 2..32
 - pilot-supervised: a regression tree on the descriptors fitted to the pilot half's simulated outcome, L leaves
 - control: the same strata with the cut-ins shuffled between strata (no information)
Efficiency = variance of the plain random-sample mean / variance of the stratified estimate, at the same budget,
so 2.0 means the same precision from half the simulations.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeRegressor

from . import features as F
from .config import Config, load
from .replay import run_all
from .study_cluster import stratified_variances, two_phase_mc
from .study_param import load_data


def run(cfg: Config | None = None, splits: int = 20) -> pd.DataFrame:
    cfg = cfg or load()
    sig, meta, fps = load_data(cfg)
    centre = int(round(cfg["data"]["before_s"] * fps))
    lw = meta.lane_width.to_numpy()
    out = run_all(sig, meta, cfg, repeats=5)
    mt = np.clip(out["min_ttc"], 0.1, 60.0)
    kpis = {"hard ACC response (>=2 m/s2)": (out["peak_decel"] >= 2.0).mean((1, 2)),
            "critical (min TTC < 3 s)": (out["min_ttc"] < 3.0).mean((1, 2)),
            "closeness (-log min TTC)": -np.log(mt).mean((1, 2))}
    d = {"scalar": F.scalar_features(sig, centre, fps, lw), "tanh": F.tanh_params(sig["off"], centre, fps), "raw": F.raw_matrix(sig)}
    recs = meta.rec.to_numpy()
    ur = np.unique(recs)
    rng = np.random.default_rng(cfg["cluster"]["seed"] + 5)
    rows = []
    for sp in range(splits):
        tr_recs = rng.choice(ur, len(ur) // 2, replace=False)
        tr = np.flatnonzero(np.isin(recs, tr_recs))
        te = np.flatnonzero(~np.isin(recs, tr_recs))
        reps = {}
        for rep in ("descriptors", "curve parameters", "PCA of raw trajectories"):
            R = F.Representation(rep, seed=sp).fit(F.subset(d, tr))
            reps[rep] = (R.transform(F.subset(d, tr)), R.transform(F.subset(d, te)))
        for kpi, y_all in kpis.items():
            y_te = y_all[te]
            if y_te.std() == 0:
                continue
            plans = []
            for rep, (ztr, zte) in reps.items():
                for k in (2, 4, 8, 16, 32):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        km = KMeans(k, n_init=3, random_state=sp).fit(ztr)
                    plans.append((f"k-means on {rep}", k, km.predict(zte)))
            Xtr, Xte = F.subset(d, tr)["scalar"], F.subset(d, te)["scalar"]
            for L in (4, 8, 16, 32):
                tree = DecisionTreeRegressor(max_leaf_nodes=L, min_samples_leaf=15, random_state=sp).fit(Xtr, y_all[tr])
                plans.append(("regression tree on descriptors, fitted to pilot outcomes", L, tree.apply(Xte)))
            for name, k, strata in plans:
                shuf = rng.permutation(strata)
                for m in (50, 100, 200):
                    v_srs, v_prop, v_ney, eta = stratified_variances(y_te, strata, m)
                    _, _, v_ney_c, _ = stratified_variances(y_te, shuf, m)
                    row = {"split": sp, "kpi": kpi, "strata": name, "k": k, "budget": m, "eta2": eta,
                           "eff_proportional": v_srs / v_prop, "eff_neyman": v_srs / v_ney, "eff_control": v_srs / v_ney_c}
                    if m == 100 and sp < 8:
                        row["eff_two_phase"] = v_srs / two_phase_mc(y_te, strata, m, rng, reps=300)
                        row["eff_two_phase_control"] = v_srs / two_phase_mc(y_te, shuf, m, rng, reps=300)
                    rows.append(row)
        print(f"split {sp + 1}/{splits}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(cfg.out_dir.parent / "results" / "strata_raw.csv", index=False)
    g = df.groupby(["kpi", "strata", "k", "budget"]).agg(
        eta2=("eta2", "mean"), eff_proportional=("eff_proportional", "mean"), eff_neyman=("eff_neyman", "mean"),
        eff_control=("eff_control", "mean"), eff_two_phase=("eff_two_phase", "mean"), eff_two_phase_sd=("eff_two_phase", "std"),
        eff_two_phase_control=("eff_two_phase_control", "mean")).reset_index()
    g.to_csv(cfg.out_dir.parent / "results" / "strata.csv", index=False)
    return g


if __name__ == "__main__":
    g = run()
    pd.set_option("display.width", 250)
    print(g[g.budget == 100].round(2).to_string())
