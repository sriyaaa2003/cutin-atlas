"""Random plate cases, their FEM solutions, and the mesh-as-graph form the network reads.

A case is a plate (height, optional hole) plus a load on its free edge. The mesh nodes become graph
nodes and the triangle edges become graph edges, so the network sees the same discretisation the
FEM solver used. The targets are the displacement (ux, uy) and the von Mises stress at every node.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

from . import fem

# The FEM is linear, so every output is exactly proportional to the size of the load. The network
# therefore predicts the result *per newton* and the load magnitude is multiplied back afterwards.
# These fixed scales bring the per-newton outputs to order one (metres per N, pascals per N).
UX_SCALE = 2e-8
UY_SCALE = 1e-7
S_SCALE = 2e3
H_MAX = 0.33     # largest plate height, used to scale lengths
EDGE_SCALE = 0.02
NODE_FEATURES = 12
EDGE_FEATURES = 3
OUT = 3


@dataclass
class Case:
    plate: fem.Plate
    force: tuple


@dataclass
class Graph:
    x: torch.Tensor           # (N, NODE_FEATURES)
    edge_index: torch.Tensor  # (2, E) source, destination
    edge_attr: torch.Tensor   # (E, EDGE_FEATURES)
    y: torch.Tensor           # (N, 3) FEM results: ux (m), uy (m), von Mises (Pa)
    y_unit: torch.Tensor      # (N, 3) the same per newton of load, scaled (what the network is trained on)
    base: torch.Tensor        # (N, 3) the closed-form beam prediction per newton, scaled the same way
    magnitude: float          # size of the load in newtons
    nodes: np.ndarray         # (N, 2) coordinates, for the closed-form baselines
    case: Case
    fem_seconds: float = 0.0


def sample_case(rng: np.random.Generator, nx: int = 60, ny: int = 12) -> Case:
    """A random plate with a hole (85% of cases) and a random load on the free edge."""
    height = rng.uniform(0.20, H_MAX)
    hole = None
    if rng.random() < 0.85:
        r = rng.uniform(0.15, 0.25) * height
        hole = (rng.uniform(0.25, 0.75), rng.uniform(0.40, 0.60) * height, r)
    magnitude = rng.uniform(50.0, 200.0) * rng.choice([-1.0, 1.0])
    angle = np.deg2rad(rng.uniform(-60.0, 60.0))
    force = (magnitude * np.sin(angle), magnitude * np.cos(angle))
    return Case(fem.Plate(height=height, nx=nx, ny=ny, hole=hole), force)


def mesh_edges(tris: np.ndarray) -> np.ndarray:
    """Every triangle edge once in each direction, as a (2, E) array."""
    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    e = np.unique(np.sort(e, axis=1), axis=0)
    return np.concatenate([e, e[:, ::-1]]).T


def node_features(case: Case, sol: fem.Solution) -> np.ndarray:
    nodes = sol.mesh.nodes
    n = len(nodes)
    hole = case.plate.hole
    feats = np.zeros((n, NODE_FEATURES))
    feats[:, 0] = nodes[:, 0] / case.plate.length
    feats[:, 1] = nodes[:, 1] / H_MAX
    feats[:, 2] = sol.clamped
    mag = float(np.hypot(*case.force))
    feats[:, 3:5] = 10.0 * sol.load / mag          # where the load is applied (direction times weight)
    feats[:, 5] = case.plate.height / H_MAX
    angle = np.arctan2(case.force[0], case.force[1])
    feats[:, 10] = np.sin(angle)                   # load direction, known to every node
    feats[:, 11] = np.cos(angle)
    if hole is not None:
        cx, cy, r = hole
        feats[:, 6] = 1.0
        feats[:, 7] = (nodes[:, 0] - cx) / case.plate.length
        feats[:, 8] = (nodes[:, 1] - cy) / H_MAX
        feats[:, 9] = r / H_MAX
    return feats


def build_graph(case: Case, sol: fem.Solution, fem_seconds: float = 0.0) -> Graph:
    nodes = sol.mesh.nodes
    edge_index = mesh_edges(sol.mesh.tris)
    rel = nodes[edge_index[1]] - nodes[edge_index[0]]
    edge_attr = np.column_stack([rel / EDGE_SCALE, np.linalg.norm(rel, axis=1) / EDGE_SCALE])
    y = np.column_stack([sol.u, sol.vm_node])
    mag = float(np.hypot(*case.force))
    unit = (case.force[0] / mag, case.force[1] / mag)
    scale = target_scale().numpy()
    base = np.column_stack([fem.beam_field(nodes, case.plate, unit), fem.beam_von_mises(nodes, case.plate, unit)]) / scale
    return Graph(
        x=torch.tensor(node_features(case, sol), dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
        y=torch.tensor(y, dtype=torch.float32),
        y_unit=torch.tensor(y / mag / scale, dtype=torch.float32),
        base=torch.tensor(base, dtype=torch.float32),
        magnitude=mag,
        nodes=nodes,
        case=case,
        fem_seconds=fem_seconds,
    )


def make_dataset(n: int, seed: int, nx: int = 60, ny: int = 12) -> list[Graph]:
    """Solve n random cases with the FEM solver and turn each into a graph."""
    rng = np.random.default_rng(seed)
    graphs = []
    for _ in range(n):
        case = sample_case(rng, nx, ny)
        t0 = time.perf_counter()
        sol = fem.solve_cantilever(case.plate, case.force)
        graphs.append(build_graph(case, sol, time.perf_counter() - t0))
    return graphs


def collate(graphs: list[Graph]):
    """Join several graphs into one big disconnected graph (the usual batching trick)."""
    xs, eis, eas, ys, bs, offset = [], [], [], [], [], 0
    for g in graphs:
        xs.append(g.x)
        eis.append(g.edge_index + offset)
        eas.append(g.edge_attr)
        ys.append(g.y_unit)
        bs.append(g.base)
        offset += g.x.shape[0]
    return torch.cat(xs), torch.cat(eis, dim=1), torch.cat(eas), torch.cat(ys), torch.cat(bs)


def target_scale() -> torch.Tensor:
    return torch.tensor([UX_SCALE, UY_SCALE, S_SCALE])
