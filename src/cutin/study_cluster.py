"""C6 study: cluster real cut-ins, then test whether clusters let a simulation campaign reach the same
precision with fewer runs (stratified sampling), with the clustering fitted on one half of the recordings and
evaluated on the other half so nothing about the outcome or the test cut-ins leaks into the strata."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture

from . import features as F
from .config import Config, load
from .replay import run_all
from .study_param import load_data

ALGOS = ["k-means", "Gaussian mixture", "Ward hierarchical", "HDBSCAN"]
REPS = ["descriptors", "curve parameters", "PCA of raw trajectories", "autoencoder of raw trajectories"]


def fit_cluster(algo: str, z: np.ndarray, k: int, seed: int):
    """Returns (train labels, assign function for new points)."""
    if algo == "k-means":
        m = KMeans(k, n_init=10, random_state=seed).fit(z)
        return m.labels_, m.predict
    if algo == "Gaussian mixture":
        m = GaussianMixture(k, covariance_type="diag", n_init=3, random_state=seed).fit(z)
        return m.predict(z), m.predict
    lab = AgglomerativeClustering(k, linkage="ward").fit_predict(z)
    cen = np.stack([z[lab == c].mean(0) for c in range(k)])
    return lab, lambda x: ((x[:, None, :] - cen[None]) ** 2).sum(-1).argmin(1)


def choose_k(algo: str, z: np.ndarray, kr, seed: int):
    if algo == "HDBSCAN":
        m = HDBSCAN(min_cluster_size=max(15, len(z) // 40)).fit(z)
        lab = m.labels_
        ids = [c for c in np.unique(lab) if c >= 0]
        if len(ids) < 2:
            return None
        cen = np.stack([z[lab == c].mean(0) for c in ids])
        keep = lab >= 0
        sil = silhouette_score(z[keep], lab[keep]) if keep.sum() > len(ids) else -1
        return len(ids), sil, (lambda x: ((x[:, None, :] - cen[None]) ** 2).sum(-1).argmin(1)), np.where(lab >= 0, lab, -1)
    best = None
    for k in range(kr[0], kr[1] + 1):
        lab, assign = fit_cluster(algo, z, k, seed)
        if len(np.unique(lab)) < 2:
            continue
        sil = silhouette_score(z, lab)
        if best is None or sil > best[1]:
            best = (k, sil, assign, lab)
    return best


def stratified_variances(y: np.ndarray, strata: np.ndarray, m: int):
    """Finite-population variance of the sample-mean estimator: SRS, proportional and Neyman stratified."""
    N = len(y)
    f = m / N
    S2 = y.var(ddof=1)
    ids = np.unique(strata)
    W = np.array([(strata == h).mean() for h in ids])
    Sh = np.array([y[strata == h].std(ddof=1) if (strata == h).sum() > 1 else 0.0 for h in ids])
    v_srs = (1 - f) * S2 / m
    v_prop = (1 - f) * np.sum(W * Sh ** 2) / m
    v_ney = (np.sum(W * Sh)) ** 2 / m - np.sum(W * Sh ** 2) / N
    return v_srs, v_prop, max(v_ney, 1e-18), float(1 - np.sum(W * Sh ** 2) / max(S2, 1e-18))


def two_phase_mc(y, strata, m, rng, reps=2000, pilot_frac=0.3):
    """Realistic variant of Neyman: spend a pilot on proportional sampling, estimate each stratum's spread from it,
    then allocate the rest with the plug-in. Returns the mean squared error of the final stratified estimate against the true
    population mean (compare with the variance of a plain random sample)."""
    ids = np.unique(strata)
    idx = {h: np.flatnonzero(strata == h) for h in ids}
    W = np.array([len(idx[h]) / len(y) for h in ids])
    est = []
    for _ in range(reps):
        pilot = np.maximum(1, np.round(W * m * pilot_frac).astype(int))
        samples = {h: list(rng.choice(idx[h], min(p, len(idx[h])), replace=False)) for h, p in zip(ids, pilot)}
        sd = np.array([max(np.std(y[samples[h]], ddof=1), 0.05 * y.std()) if len(samples[h]) > 1 else y.std() for h in ids])
        rest = m - sum(len(v) for v in samples.values())
        alloc = W * sd
        extra = np.floor(rest * alloc / alloc.sum()).astype(int)
        for h, e in zip(ids, extra):
            pool = np.setdiff1d(idx[h], samples[h])
            if e > 0 and len(pool):
                samples[h] += list(rng.choice(pool, min(e, len(pool)), replace=False))
        est.append(sum(W[j] * y[samples[h]].mean() for j, h in enumerate(ids)))
    return float(np.mean((np.asarray(est) - y.mean()) ** 2))      # mean squared error, so any bias from adaptive allocation counts


def run(cfg: Config | None = None, splits: int = 10) -> None:
    cfg = cfg or load()
    cc = cfg["cluster"]
    sig, meta, fps = load_data(cfg)
    centre = int(round(cfg["data"]["before_s"] * fps))
    lw = meta.lane_width.to_numpy()
    out = run_all(sig, meta, cfg, repeats=5)
    kpis = {"hard ACC response (>=2 m/s2)": (out["peak_decel"] >= 2.0).mean((1, 2)),
            "critical (min TTC < 3 s)": (out["min_ttc"] < 3.0).mean((1, 2)),
            "contact": out["crash"].mean((1, 2))}
    d = {"scalar": F.scalar_features(sig, centre, fps, lw), "tanh": F.tanh_params(sig["off"], centre, fps), "raw": F.raw_matrix(sig)}
    recs = meta.rec.to_numpy()
    ur = np.unique(recs)
    rows = []
    rng = np.random.default_rng(cc["seed"])
    for sp in range(splits):
        tr_recs = rng.choice(ur, len(ur) // 2, replace=False)
        tr = np.flatnonzero(np.isin(recs, tr_recs))
        te = np.flatnonzero(~np.isin(recs, tr_recs))
        for rep in REPS:
            R = F.Representation(rep, seed=sp).fit(F.subset(d, tr))
            ztr, zte = R.transform(F.subset(d, tr)), R.transform(F.subset(d, te))
            for algo in ALGOS:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    best = choose_k(algo, ztr, cc["k_range"], sp)
                    if best is None:
                        continue
                    k, sil, assign, lab = best
                    # stability: re-cluster a bootstrap resample, compare labels on the training points
                    aris = []
                    if algo != "HDBSCAN":
                        for b in range(10):
                            ib = rng.choice(len(ztr), len(ztr), replace=True)
                            lb, asg = fit_cluster(algo, ztr[ib], k, b)
                            aris.append(adjusted_rand_score(lab, asg(ztr)))
                strata = assign(zte)
                for name, y_all in kpis.items():
                    y = y_all[te]
                    if y.std() == 0:
                        continue
                    shuffled = rng.permutation(strata)                       # control: same strata sizes, no information
                    for m in (50, 100, 200):
                        v_srs, v_prop, v_ney, eta = stratified_variances(y, strata, m)
                        _, v_prop_ctrl, v_ney_ctrl, _ = stratified_variances(y, shuffled, m)
                        row = {"split": sp, "representation": rep, "algorithm": algo, "k": k, "silhouette_train": sil,
                               "stability_ari": float(np.mean(aris)) if aris else np.nan, "kpi": name, "budget": m,
                               "eta2": eta, "eff_proportional": v_srs / v_prop, "eff_neyman_oracle": v_srs / v_ney,
                               "eff_control_shuffled": v_srs / v_ney_ctrl}
                        if m == 100 and name != "contact":
                            row["eff_two_phase"] = v_srs / two_phase_mc(y, strata, m, rng, reps=1000)
                        rows.append(row)
        print(f"split {sp + 1}/{splits} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(cfg.out_dir.parent / "results" / "sampling_raw.csv", index=False)
    g = df.groupby(["representation", "algorithm", "kpi", "budget"]).agg(
        k=("k", "median"), silhouette=("silhouette_train", "mean"), stability_ari=("stability_ari", "mean"), eta2=("eta2", "mean"),
        eff_proportional=("eff_proportional", "mean"), eff_neyman_oracle=("eff_neyman_oracle", "mean"),
        eff_control_shuffled=("eff_control_shuffled", "mean"), eff_two_phase=("eff_two_phase", "mean"),
        eff_two_phase_sd=("eff_two_phase", "std")).reset_index()
    g.to_csv(cfg.out_dir.parent / "results" / "sampling.csv", index=False)
    print(g[g.budget == 100].round(3).to_string())


if __name__ == "__main__":
    run()
