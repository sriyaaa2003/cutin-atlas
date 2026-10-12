"""C1 study: how well do competing trajectory parameterizations preserve what a simulation test would conclude?"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import param
from .config import Config, load
from .replay import run_all


def load_data(cfg: Config):
    z = np.load(cfg.out_dir / "signals.npz")
    sig = {k: z[k].astype(np.float64) for k in ("off", "gap", "vc", "vf")}
    return sig, pd.read_parquet(cfg.out_dir / "cutins.parquet"), float(z["fps"])


def run(cfg: Config | None = None, repeats: int = 5) -> pd.DataFrame:
    cfg = cfg or load()
    sig, meta, fps = load_data(cfg)
    centre = int(round(cfg["data"]["before_s"] * fps))
    lw = meta.lane_width.to_numpy()
    recon, seg, valid = param.fit_all(sig["off"], lw, fps, cfg, centre)
    pd.DataFrame(seg).assign(valid=valid).to_parquet(cfg.out_dir / "segments.parquet", index=False)
    base = run_all(sig, meta, cfg, repeats=repeats)
    rows = []
    for name in param.METHODS:
        off_hat = np.nan_to_num(recon[name], nan=1.0)
        out = run_all(sig, meta, cfg, off=off_hat, repeats=repeats)
        r = param.rmse_cm(sig["off"], recon[name], lw)
        both = np.isfinite(base["min_ttc"]) & np.isfinite(out["min_ttc"])
        dlog = np.abs(np.log(np.clip(out["min_ttc"], 0.1, 60)[both]) - np.log(np.clip(base["min_ttc"], 0.1, 60)[both]))
        dseen = np.abs(out["t_seen"] - base["t_seen"])
        hard_b, hard_o = base["peak_decel"] >= 2.0, out["peak_decel"] >= 2.0
        crit_b, crit_o = base["min_ttc"] < 3.0, out["min_ttc"] < 3.0
        rows.append({
            "method": name, "parameters": param.n_params(name),
            "rmse_cm_median": float(np.nanmedian(r)), "rmse_cm_p95": float(np.nanpercentile(r, 95)),
            "rmse_cm_median_complete": float(np.nanmedian(r[valid])),
            "t_seen_error_s_median": float(np.nanmedian(dseen)), "t_seen_error_s_p95": float(np.nanpercentile(dseen[np.isfinite(dseen)], 95)),
            "min_ttc_log_error_median": float(np.median(dlog)),
            "hard_intervention_agreement": float((hard_b == hard_o).mean()),
            "critical_ttc3_agreement": float((crit_b == crit_o).mean()),
            "critical_ttc3_missed": float((crit_b & ~crit_o).sum() / max(crit_b.sum(), 1)),
            "contact_cases_real": int(base["crash"].any(axis=(1, 2)).sum()),
            "contact_cases_reproduced": int((base["crash"].any(axis=(1, 2)) & out["crash"].any(axis=(1, 2))).sum()),
        })
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(cfg.out_dir.parent / "results" / "parameterization.csv", index=False)
    return df


if __name__ == "__main__":
    run()
