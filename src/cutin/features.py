"""Scenario representations used for clustering: hand-built descriptors, curve parameters, PCA, autoencoder."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .param import tanh_curve


def scalar_features(sig: dict, centre: int, fps: float, lane_width: np.ndarray) -> np.ndarray:
    """Descriptors an engineer would write down: state at the lane-change frame plus how the cutter moves."""
    c = centre
    gap, vc, vf, off = sig["gap"], sig["vc"], sig["vf"], sig["off"]
    rel = vf[:, c] - vc[:, c]
    ttc = np.where(rel > 0.1, gap[:, c] / np.maximum(rel, 0.1), 30.0).clip(max=30.0)
    lat = -np.gradient(off, axis=1) * fps * lane_width[:, None]
    k = int(fps * 0.5)
    ac = (vc[:, c + k] - vc[:, c - k]) / 1.0
    af = (vf[:, c] - vf[:, c - int(fps)]) / 1.0
    return np.column_stack([gap[:, c], rel, vf[:, c], vc[:, c], ttc, lat.max(1), ac, af])


def tanh_params(off: np.ndarray, centre: int, fps: float) -> np.ndarray:
    n, T = off.shape
    t = np.arange(T) / fps
    out = np.full((n, 4), np.nan)
    for i in range(n):
        try:
            p, _ = curve_fit(tanh_curve, t, off[i], p0=[t[centre], 0.5, off[i, 0], off[i, -1]],
                             bounds=([0, 0.05, -1, -1], [t[-1], 5.0, 2, 2]), maxfev=4000)
            out[i] = p
        except RuntimeError:
            pass
    for j in range(4):
        out[np.isnan(out[:, j]), j] = np.nanmedian(out[:, j])
    return out


def raw_matrix(sig: dict, points: int = 31) -> np.ndarray:
    """Channels resampled to `points` samples: lateral offset, gap, relative speed."""
    T = sig["off"].shape[1]
    idx = np.linspace(0, T - 1, points).round().astype(int)
    return np.concatenate([sig["off"][:, idx], sig["gap"][:, idx], (sig["vf"] - sig["vc"])[:, idx]], axis=1)


class Autoencoder(torch.nn.Module):
    def __init__(self, d_in: int, d_lat: int = 8):
        super().__init__()
        self.enc = torch.nn.Sequential(torch.nn.Linear(d_in, 64), torch.nn.ReLU(), torch.nn.Linear(64, d_lat))
        self.dec = torch.nn.Sequential(torch.nn.Linear(d_lat, 64), torch.nn.ReLU(), torch.nn.Linear(64, d_in))

    def forward(self, x):
        return self.dec(self.enc(x))


class Representation:
    """fit on a training set, transform any set: so the clustering of one half can be applied to the other."""

    def __init__(self, kind: str, seed: int = 0):
        self.kind, self.seed = kind, seed

    def _x(self, d):
        return {"descriptors": d["scalar"], "curve parameters": np.column_stack([d["tanh"], d["scalar"][:, :4]]),
                "PCA of raw trajectories": d["raw"], "autoencoder of raw trajectories": d["raw"]}[self.kind]

    def _embed(self, z):
        if self.kind == "PCA of raw trajectories":
            return self.pca.transform(z)
        if self.kind == "autoencoder of raw trajectories":
            with torch.no_grad():
                return self.ae.enc(torch.tensor(z, dtype=torch.float32)).numpy()
        return z

    def fit(self, d):
        x = self._x(d)
        self.sc = StandardScaler().fit(x)
        z = self.sc.transform(x)
        if self.kind == "PCA of raw trajectories":
            self.pca = PCA(8, random_state=self.seed).fit(z)
        elif self.kind == "autoencoder of raw trajectories":
            torch.manual_seed(self.seed)
            self.ae = Autoencoder(z.shape[1], 8)
            opt = torch.optim.Adam(self.ae.parameters(), 1e-3, weight_decay=1e-5)
            xt = torch.tensor(z, dtype=torch.float32)
            for _ in range(300):
                for b in torch.randperm(len(xt)).split(128):
                    opt.zero_grad()
                    torch.nn.functional.mse_loss(self.ae(xt[b]), xt[b]).backward()
                    opt.step()
        self.sc2 = StandardScaler().fit(self._embed(z))     # scale the embedding with TRAIN statistics only
        return self

    def transform(self, d):
        return self.sc2.transform(self._embed(self.sc.transform(self._x(d))))


def subset(d: dict, idx: np.ndarray) -> dict:
    return {k: v[idx] for k, v in d.items()}
