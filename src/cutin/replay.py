"""Closed-loop replay of cut-ins against an ACC + AEB system, vectorised over cut-ins.

The cutter C follows its recorded (or reconstructed) trajectory. The follower F is replaced by the
system under test (SUT), which starts at the follower's recorded speed and holds it as its set speed.
The SUT only reacts once C is in its path (lateral offset below `path_offset`), through delayed, noisy
measurements: a constant time-gap ACC plus an autonomous emergency brake that triggers on measured TTC.
"""
from __future__ import annotations

import itertools

import numpy as np

from .config import Config, load


def configs(p) -> list[dict]:
    names = ["aeb_ttc", "sensor_delay", "max_brake"]
    keys = ["aeb_ttc_s", "sensor_delay_s", "max_brake_ms2"]
    return [dict(zip(names, c)) for c in itertools.product(*[p[k] for k in keys])]


def replay(off, gap0, vc, vf0, len_sum, *, aeb_ttc, sensor_delay, max_brake, p, rng):
    """off, vc: (n, T) cutter lateral offset (lane widths) and speed; gap0: (n,) bumper gap at the first sample;
    vf0: (n,) SUT start speed; len_sum: (n,) cutter + SUT length.
    Returns dict of (n,) arrays: crash, min_ttc, min_gap, t_seen (s, nan if never), impact_dv, aeb, peak_decel."""
    n, T = off.shape
    dt = p["dt_s"]
    D = max(1, int(round(sensor_delay / dt)))
    gap = gap0.astype(np.float64).copy()
    v = vf0.astype(np.float64).copy()
    v_set = v.copy()
    a = np.zeros(n)
    trig = np.zeros(n, bool)
    crashed = np.zeros(n, bool)
    impact = np.zeros(n)
    min_ttc = np.full(n, np.inf)
    min_gap = np.full(n, np.inf)
    t_seen = np.full(n, np.nan)
    peak = np.zeros(n)
    h_gap, h_v = np.tile(gap, (D + 1, 1)), np.tile(v, (D + 1, 1))
    h_vc = np.tile(vc[:, 0].astype(np.float64), (D + 1, 1))
    h_in = np.tile(np.abs(off[:, 0]) < p["path_offset"], (D + 1, 1))
    k1, k2, th, d0 = p["acc_k1"], p["acc_k2"], p["acc_time_gap_s"], p["acc_standstill_m"]
    for k in range(T):
        idx = k % (D + 1)
        in_path_m = h_in[idx] & (h_gap[idx] > 0)
        g_m = h_gap[idx] + rng.normal(0, p["sigma_gap_m"], n)
        v_m = h_v[idx] + rng.normal(0, p["sigma_speed_ms"], n)
        vc_m = h_vc[idx] + rng.normal(0, p["sigma_speed_ms"], n)
        rel = v_m - vc_m
        ttc_m = np.where(in_path_m & (rel > 0.1), g_m / np.maximum(rel, 0.1), np.inf)
        trig |= ttc_m < aeb_ttc
        a_follow = np.where(in_path_m, k1 * (g_m - d0 - th * v_m) + k2 * (vc_m - v_m), np.inf)
        a_acc = np.clip(np.minimum(a_follow, p["acc_speed_gain"] * (v_set - v_m)), -p["acc_max_decel"], p["acc_max_accel"])
        a_cmd = np.where(trig, -max_brake, a_acc)
        a = np.clip(a + (a_cmd - a) * dt / p["actuator_tau_s"], -max_brake, p["acc_max_accel"])
        peak = np.maximum(peak, -a)
        # bookkeeping on the TRUE state
        in_path = np.abs(off[:, k]) < p["path_offset"]
        rel_true = v - vc[:, k]
        ttc_true = np.where(in_path & (gap > 0) & (rel_true > 0.05), gap / np.maximum(rel_true, 0.05), np.inf)
        min_ttc = np.minimum(min_ttc, ttc_true)
        min_gap = np.where(in_path & (gap > 0), np.minimum(min_gap, gap), min_gap)
        first = in_path & np.isnan(t_seen)
        t_seen[first] = k * dt
        overlap = (np.abs(off[:, k]) < p["overlap_offset"]) & (gap <= 0) & (gap > -len_sum)
        hit = overlap & ~crashed
        impact[hit] = np.maximum(rel_true, 0.0)[hit]
        crashed |= hit
        h_gap[idx], h_v[idx], h_vc[idx], h_in[idx] = gap, v, vc[:, k], in_path
        v_n = np.maximum(v + a * dt, 0.0)
        gap = gap + 0.5 * (vc[:, k] + vc[:, min(k + 1, T - 1)]) * dt - 0.5 * (v + v_n) * dt
        v = v_n
    return {"crash": crashed, "min_ttc": min_ttc, "min_gap": min_gap, "t_seen": t_seen, "impact_dv": impact,
            "aeb": trig, "peak_decel": peak}


def run_all(sig: dict, meta, cfg: Config | None = None, off=None, repeats: int = 5):
    """Replay every cut-in under every SUT configuration and `repeats` noise seeds.
    Returns a dict of (n, n_cfg, repeats) arrays: crash, min_ttc, aeb, peak_decel."""
    cfg = cfg or load()
    p = cfg["replay"]
    off = sig["off"] if off is None else off
    cs = configs(p)
    n = len(off)
    out = {"crash": np.zeros((n, len(cs), repeats), bool), "aeb": np.zeros((n, len(cs), repeats), bool),
           "min_ttc": np.full((n, len(cs), repeats), np.inf), "peak_decel": np.zeros((n, len(cs), repeats)),
           "t_seen": np.full((n, len(cs), repeats), np.nan)}
    len_sum = (meta.len_c + meta.len_f).to_numpy()
    for j, c in enumerate(cs):
        for r in range(repeats):
            rng = np.random.default_rng(p["seed"] + 1000 * j + r)         # same noise for every variant of the cut-in
            o = replay(off, sig["gap"][:, 0], sig["vc"], sig["vf"][:, 0], len_sum, p=p, rng=rng, **c)
            for key in out:
                out[key][:, j, r] = o[key]
    return out
