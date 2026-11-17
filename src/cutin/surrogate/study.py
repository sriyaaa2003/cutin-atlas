"""The surrogate study: how many finite element runs does a graph network need?

Run it with:  python -m cutin.surrogate.study

Every number is produced here: random plate cases are solved with the FEM solver in `fem.py`, graph
networks are trained on the first n of them, and each model is scored on 40 held-out cases that were
never used for training or for choosing the design. The closed-form beam formula (which does not
know about the hole) is the baseline everything has to beat.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import fem, gnn
from . import graph as G

MODELS = ("gnn_res", "gnn", "mlp")
TEST_SEED = 2000   # a different seed from anything used while developing the models


def fem_validation() -> dict:
    """Exact-answer checks of the solver, recorded next to the study so the data can be trusted."""
    plate = fem.Plate(nx=12, ny=6)
    mesh = fem.make_mesh(plate)
    left = np.where(np.abs(mesh.nodes[:, 0]) < 1e-9)[0]
    corner = np.where((np.abs(mesh.nodes[:, 0]) < 1e-9) & (np.abs(mesh.nodes[:, 1]) < 1e-9))[0][0]
    s0 = 1e6
    f = fem.edge_load(mesh, plate.length, (s0 * plate.height * plate.thickness, 0.0))
    _, _, stress = fem.solve_linear(mesh, plate, np.concatenate([2 * left, [2 * corner + 1]]), f)
    patch = float(np.abs(stress[:, 0] / s0 - 1.0).max())

    ratios = {}
    for nx, ny in [(30, 6), (60, 12), (120, 24)]:
        p = fem.Plate(nx=nx, ny=ny)
        sol = fem.solve_cantilever(p, (0.0, 100.0))
        tip = sol.u[np.abs(sol.mesh.nodes[:, 0] - p.length) < 1e-9, 1].mean()
        ratios[f"{nx}x{ny}"] = float(tip / fem.tip_deflection_timoshenko(p, 100.0))

    p = fem.Plate(hole=(0.5, 0.12, 0.05))
    mesh = fem.make_mesh(p)
    ids = np.where(np.abs(mesh.nodes[:, 0]) < 1e-9)[0]
    force = (30.0, -80.0)
    _, reactions, _ = fem.solve_linear(mesh, p, np.concatenate([2 * ids, 2 * ids + 1]), fem.edge_load(mesh, p.length, force))
    equilibrium = float(np.abs(reactions.sum(axis=0) + np.array(force)).max())
    return {"patch_test_max_error": patch, "tip_deflection_over_timoshenko": ratios,
            "equilibrium_max_residual_newtons": equilibrium}


def score(pred_fn, graphs) -> dict:
    errs = [gnn.relative_errors(pred_fn(g), g.y.numpy()) for g in graphs]
    out = {}
    for k in ("disp", "stress", "peak_stress"):
        v = np.array([e[k] for e in errs])
        out[f"{k}_mean"] = float(v.mean())
        out[f"{k}_p90"] = float(np.percentile(v, 90))
    return out


def run(sizes=(8, 16, 32, 64, 128), seeds=(0, 1), steps=500, n_test=40, models=MODELS,
        out=Path("results/surrogate"), quiet=False) -> dict:
    torch.set_num_threads(4)
    out.mkdir(parents=True, exist_ok=True)
    pool = G.make_dataset(max(sizes), seed=0)
    test = G.make_dataset(n_test, seed=TEST_SEED)
    rows = [{"model": "beam_formula", "n_train": 0, "seed": 0, "train_seconds": 0.0, **score(gnn.beam_prediction, test)}]
    last = {}
    for n in sizes:
        for seed in seeds:
            for kind in models:
                model = gnn.make_model(kind)
                t0 = time.time()
                gnn.train(model, pool[:n], steps=steps, seed=seed)
                dt = time.time() - t0
                row = {"model": kind, "n_train": n, "seed": seed, "train_seconds": round(dt, 1),
                       **score(lambda g, m=model: gnn.predict(m, g), test)}
                rows.append(row)
                last[kind] = model
                if not quiet:
                    print(f"{kind:8s} n={n:<4d} seed={seed} disp {row['disp_mean']:.3f} stress {row['stress_mean']:.3f} "
                          f"peak {row['peak_stress_mean']:.3f} ({dt:.0f}s)", flush=True)

    import csv
    with open(out / "learning_curve.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    model = last.get("gnn_res") or next(iter(last.values()))
    g0 = test[0]
    t0 = time.perf_counter()
    for _ in range(20):
        gnn.predict(model, g0)
    gnn_ms = (time.perf_counter() - t0) / 20 * 1e3
    summary = {
        "n_test": n_test, "sizes": list(sizes), "seeds": list(seeds), "steps": steps,
        "nodes_per_case": float(np.mean([g.x.shape[0] for g in test])),
        "fem_ms_per_case": float(np.mean([g.fem_seconds for g in pool]) * 1e3),
        "gnn_inference_ms_per_case": gnn_ms,
        "fem_validation": fem_validation(),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    plot(rows, out / "learning_curve.png")
    return summary


def plot(rows, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, key, title in zip(axes, ("disp_mean", "stress_mean"), ("Displacement error", "Von Mises stress error")):
        base = next(r[key] for r in rows if r["model"] == "beam_formula")
        ax.axhline(base, color="grey", ls="--", label="beam formula (no hole)")
        for kind, colour in (("gnn_res", "tab:green"), ("gnn", "tab:blue"), ("mlp", "tab:red")):
            sel = [r for r in rows if r["model"] == kind]
            if not sel:
                continue
            ns = sorted({r["n_train"] for r in sel})
            mean = [np.mean([r[key] for r in sel if r["n_train"] == n]) for n in ns]
            lo = [min(r[key] for r in sel if r["n_train"] == n) for n in ns]
            hi = [max(r[key] for r in sel if r["n_train"] == n) for n in ns]
            ax.plot(ns, mean, "o-", color=colour, label={"gnn_res": "graph net + beam formula", "gnn": "graph net", "mlp": "no message passing"}[kind])
            ax.fill_between(ns, lo, hi, color=colour, alpha=0.15)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("FEM runs used for training")
        ax.set_ylabel("relative error on 40 unseen plates")
        ax.set_title(title)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--out", type=Path, default=Path("results/surrogate"))
    a = ap.parse_args()
    print(json.dumps(run(tuple(a.sizes), tuple(a.seeds), a.steps, out=a.out), indent=2))


if __name__ == "__main__":
    main()
