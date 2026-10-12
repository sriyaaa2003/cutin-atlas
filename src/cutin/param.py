"""Parameterizing the lateral cut-in trajectory: segmentation of the maneuver plus competing curve families.

Segmented families describe the maneuver by (t_start, duration, off_start, off_end) plus shape parameters.
Window families fit one polynomial to the whole 6 s window and need no segmentation.
"""
from __future__ import annotations

import numpy as np
from math import comb
from scipy.optimize import curve_fit

from .config import Config

CENTRE = 75   # index of the lane-change frame in a 6 s window at 25 Hz (set from the data at run time)


def lateral_speed(off: np.ndarray, lane_width: np.ndarray, fps: float, smooth: int = 5) -> np.ndarray:
    """Speed toward the target lane (m/s, positive = moving in), (n, T)."""
    k = np.ones(smooth) / smooth
    sm = np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), 1, off)
    return -np.gradient(sm, axis=1) * fps * lane_width[:, None]


def segment(off: np.ndarray, lane_width: np.ndarray, fps: float, cfg: Config, centre: int):
    """Start and end index of the lateral maneuver around the lane-change frame."""
    m = cfg["maneuver"]
    v = lateral_speed(off, lane_width, fps)
    n, T = off.shape
    s, e = np.zeros(n, int), np.full(n, T - 1)
    for i in range(n):
        thr = max(m["start_frac"] * v[i].max(), m["min_speed_ms"])
        back = np.flatnonzero(v[i, :centre + 1] < thr)
        s[i] = back[-1] + 1 if len(back) else 0
        fwd = np.flatnonzero(v[i, centre:] < thr)
        e[i] = centre + fwd[0] if len(fwd) else T - 1
    return s, e, v


# ---- shape functions on normalised time s in [0, 1], going from 1 (start) to 0 (end) ----------------------------
def minjerk(s):
    s = np.clip(s, 0, 1)
    return 1 - (10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5)


def bernstein(s, n):
    s = np.clip(s, 0, 1)
    return np.stack([comb(n, i) * s ** i * (1 - s) ** (n - i) for i in range(n + 1)], -1)


def fit_bezier(s, y, n):
    """Bezier of degree n whose end control points are fixed at 1 and 0, interior points free (n-1 parameters)."""
    B = bernstein(s, n)
    rhs = y - B[:, 0] * 1.0
    P, *_ = np.linalg.lstsq(B[:, 1:n], rhs, rcond=None)
    return np.concatenate([[1.0], P, [0.0]])


def eval_bezier(P, s):
    return bernstein(s, len(P) - 1) @ P


def fit_warped_minjerk(s, y):
    """minjerk(s ** gamma): one extra parameter that makes the lateral motion front- or back-loaded."""
    best, bg = np.inf, 1.0
    for g in np.linspace(0.5, 2.0, 31):
        err = np.sum((minjerk(s ** g) - y) ** 2)
        if err < best:
            best, bg = err, g
    return bg


def tanh_curve(t, tm, w, a, b):
    return b + (a - b) * 0.5 * (1 - np.tanh((t - tm) / w))


METHODS = {   # name -> (shape parameters, family)
    "min-jerk quintic": (0, "seg"), "warped min-jerk": (1, "seg"),
    "Bezier degree 3": (2, "seg"), "Bezier degree 5": (4, "seg"), "Bezier degree 7": (6, "seg"),
    "tanh sigmoid": (0, "tanh"),
    "polynomial degree 3 (window)": (4, "poly"), "polynomial degree 5 (window)": (6, "poly"),
    "polynomial degree 7 (window)": (8, "poly"),
}
SEG_COMMON = 4          # t_start, duration, off_start, off_end


def n_params(name: str) -> int:
    k, fam = METHODS[name]
    return k if fam == "poly" else (k + SEG_COMMON if fam == "seg" else 4)


def fit_all(off: np.ndarray, lane_width: np.ndarray, fps: float, cfg: Config, centre: int):
    """Reconstruct every cut-in with every method. Returns (recon dict name -> (n, T), seg table, valid mask)."""
    n, T = off.shape
    t = np.arange(T) / fps
    s_idx, e_idx, _ = segment(off, lane_width, fps, cfg, centre)
    dur = (e_idx - s_idx) / fps
    valid = (dur >= 1.0) & (dur <= 6.0) & (s_idx > 0) & (e_idx < T - 1)
    recon = {m: np.zeros_like(off, dtype=np.float64) for m in METHODS}
    for i in range(n):
        a, b = s_idx[i], e_idx[i]
        o_s, o_e = float(off[i, a]), float(off[i, b])
        span = o_s - o_e if abs(o_s - o_e) > 1e-3 else 1e-3
        s_all = (np.arange(T) - a) / max(b - a, 1)
        m = slice(a, b + 1)
        y = (off[i, m] - o_e) / span
        s_seg = s_all[m]

        def place(shape_vals):             # constant outside the maneuver
            out = np.where(s_all <= 0, 1.0, np.where(s_all >= 1, 0.0, shape_vals))
            return o_e + span * out

        recon["min-jerk quintic"][i] = place(minjerk(s_all))
        g = fit_warped_minjerk(s_seg, y)
        recon["warped min-jerk"][i] = place(minjerk(np.clip(s_all, 0, 1) ** g))
        for deg in (3, 5, 7):
            P = fit_bezier(s_seg, y, deg)
            recon[f"Bezier degree {deg}"][i] = place(eval_bezier(P, np.clip(s_all, 0, 1)))
        try:
            popt, _ = curve_fit(tanh_curve, t, off[i], p0=[t[centre], 0.5, o_s, o_e],
                                bounds=([0, 0.05, -1, -1], [t[-1], 5.0, 2, 2]), maxfev=4000)
            recon["tanh sigmoid"][i] = tanh_curve(t, *popt)
        except RuntimeError:
            recon["tanh sigmoid"][i] = np.nan
        for deg in (3, 5, 7):
            x = (t - t[centre]) / t[-1]
            c = np.polyfit(x, off[i], deg)
            recon[f"polynomial degree {deg} (window)"][i] = np.polyval(c, x)
    seg = {"start_s": s_idx / fps, "end_s": e_idx / fps, "duration_s": dur, "off_start": off[np.arange(n), s_idx],
           "off_end": off[np.arange(n), e_idx]}
    return recon, seg, valid


def rmse_cm(off, recon, lane_width):
    return np.sqrt(np.nanmean((recon - off) ** 2, axis=1)) * lane_width * 100.0
