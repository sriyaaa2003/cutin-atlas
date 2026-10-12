"""highD -> real cut-in events, each as a fixed time grid around the lane-change frame.

A cut-in is a vehicle C that changes lane and lands in front of a follower F in the target lane.
Signals (all per frame, 25 Hz):
  off   lateral offset of C from the target-lane centre, in lane widths: +1 at the source-lane centre, 0 at the target centre
  gap   signed bumper-to-bumper gap C rear to F front along the driving direction (m)
  vc    speed of C (m/s),  vf  speed of F (m/s)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config, load

COLS = ["frame", "id", "x", "y", "width", "height", "xVelocity", "yVelocity", "followingId", "laneId"]


def lane_centres(rm: pd.Series) -> dict[int, tuple[float, float]]:
    """laneId -> (centre y, lane width) from the lane markings in the recording metadata."""
    up = [float(v) for v in str(rm.upperLaneMarkings).split(";") if v]
    lo = [float(v) for v in str(rm.lowerLaneMarkings).split(";") if v]
    out = {}
    for i in range(len(up) - 1):
        out[2 + i] = ((up[i] + up[i + 1]) / 2, up[i + 1] - up[i])
    for i in range(len(lo) - 1):
        out[len(up) + 2 + i] = ((lo[i] + lo[i + 1]) / 2, lo[i + 1] - lo[i])
    return out


def recordings(root) -> list[int]:
    return sorted(int(p.name[:2]) for p in root.glob("[0-9][0-9]_tracks.csv"))


def extract_recording(root, rid: int, cfg: Config):
    d = cfg["data"]
    pre = root / f"{rid:02d}"
    rm = pd.read_csv(f"{pre}_recordingMeta.csv").iloc[0]
    fps = float(rm.frameRate)
    tm = pd.read_csv(f"{pre}_tracksMeta.csv")[["id", "class", "drivingDirection"]]
    t = pd.read_csv(f"{pre}_tracks.csv", usecols=COLS).merge(tm, on="id", how="left")
    t = t.sort_values(["id", "frame"], kind="stable").reset_index(drop=True)
    lanes = lane_centres(rm)
    nb, na = int(round(d["before_s"] * fps)), int(round(d["after_s"] * fps))
    veh = {i: g for i, g in t.groupby("id")}
    rows, sig = [], []
    for cid, g in veh.items():
        lane = g.laneId.to_numpy()
        for k in np.flatnonzero(np.diff(lane) != 0) + 1:           # first frame in the new lane
            src_lane, tgt_lane = int(lane[k - 1]), int(lane[k])
            if src_lane not in lanes or tgt_lane not in lanes:
                continue
            fid = int(g.followingId.iloc[k])
            f_c = int(g.frame.iloc[k])
            if fid <= 0 or fid not in veh:
                continue
            f = veh[fid]
            f0, f1 = f_c - nb, f_c + na
            gc = g[(g.frame >= f0) & (g.frame <= f1)]
            gf = f[(f.frame >= f0) & (f.frame <= f1)]
            if len(gc) != nb + na + 1 or len(gf) != nb + na + 1:        # both vehicles tracked through the window
                continue
            if (gf.laneId.to_numpy() != tgt_lane).any() or gf["class"].iloc[0] != d["follower_class"]:
                continue
            if (gc.laneId.to_numpy()[:nb] == tgt_lane).any():           # C must come from another lane
                continue
            ts, ss = lanes[tgt_lane][0], lanes[src_lane][0]
            yc = gc.y.to_numpy() + gc.height.to_numpy() / 2
            off = (yc - ts) / (ss - ts)
            sgn = 1.0 if g.drivingDirection.iloc[0] == 2 else -1.0
            if sgn > 0:
                gap = gc.x.to_numpy() - (gf.x.to_numpy() + gf.width.to_numpy())
            else:
                gap = gf.x.to_numpy() - (gc.x.to_numpy() + gc.width.to_numpy())
            vc, vf = sgn * gc.xVelocity.to_numpy(), sgn * gf.xVelocity.to_numpy()
            at = nb
            if not (0.0 < gap[at] <= d["max_gap_m"]) or vf[at] < d["min_speed_ms"] or vc[at] < d["min_speed_ms"]:
                continue
            rows.append({"rec": rid, "loc": int(rm.locationId), "cutter": int(cid), "follower": fid, "frame_c": f_c,
                         "lane_width": float(lanes[tgt_lane][1]), "len_c": float(gc.width.iloc[0]), "len_f": float(gf.width.iloc[0]),
                         "dir": int(g.drivingDirection.iloc[0]), "cutter_class": str(g["class"].iloc[0])})
            sig.append(np.stack([off, gap, vc, vf]))
    return rows, sig, fps


def run(cfg: Config | None = None) -> pd.DataFrame:
    cfg = cfg or load()
    root = cfg.raw_dir
    rows, sig = [], []
    fps = 25.0
    for rid in recordings(root):
        r, s, fps = extract_recording(root, rid, cfg)
        rows += r
        sig += s
        print(f"recording {rid:02d}: {len(r)} cut-ins", flush=True)
    df = pd.DataFrame(rows)
    arr = np.stack(sig).astype(np.float32)                       # (n, 4, T)
    np.savez_compressed(cfg.out_dir / "signals.npz", off=arr[:, 0], gap=arr[:, 1], vc=arr[:, 2], vf=arr[:, 3], fps=fps)
    df.to_parquet(cfg.out_dir / "cutins.parquet", index=False)
    print(f"{len(df)} cut-ins from {df.rec.nunique()} recordings at {df['loc'].nunique()} locations")
    return df


if __name__ == "__main__":
    run()
