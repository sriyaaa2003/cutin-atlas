"""A small 2D linear-elastic finite element solver (plane stress, constant-strain triangles).

This is the "expensive simulation" that the graph neural network in `gnn.py` learns to imitate.
It is kept deliberately small and is checked against exact results (patch test, equilibrium,
beam theory) so the data the network learns from can be trusted.

Geometry: a rectangular plate, clamped on the left edge, loaded on the right edge, with an
optional circular hole. The hole is cut out of a structured triangle mesh by removing every
triangle whose centre lies inside the circle (a staircase hole, so the mesh stays simple).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import connected_components


@dataclass(frozen=True)
class Plate:
    length: float = 1.0
    height: float = 0.25
    thickness: float = 0.02
    youngs: float = 70e9      # Pa (aluminium-like)
    poisson: float = 0.33
    nx: int = 60              # quads along the length
    ny: int = 12              # quads along the height
    hole: tuple | None = None  # (cx, cy, r) in metres


@dataclass
class Mesh:
    nodes: np.ndarray  # (N, 2) coordinates in metres
    tris: np.ndarray   # (M, 3) node indices, counter-clockwise


@dataclass
class Solution:
    mesh: Mesh
    u: np.ndarray         # (N, 2) displacement in metres
    load: np.ndarray      # (N, 2) applied nodal forces in newtons
    clamped: np.ndarray   # (N,) bool, nodes held fixed
    vm_node: np.ndarray   # (N,) von Mises stress averaged to the nodes, in pascals
    vm_elem: np.ndarray   # (M,) von Mises stress per element


def make_mesh(plate: Plate) -> Mesh:
    xs = np.linspace(0.0, plate.length, plate.nx + 1)
    ys = np.linspace(0.0, plate.height, plate.ny + 1)
    X, Y = np.meshgrid(xs, ys)
    nodes = np.column_stack([X.ravel(), Y.ravel()])
    idx = np.arange((plate.nx + 1) * (plate.ny + 1)).reshape(plate.ny + 1, plate.nx + 1)
    n00, n10 = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
    n01, n11 = idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    # two counter-clockwise triangles per quad
    tris = np.concatenate([np.column_stack([n00, n10, n11]), np.column_stack([n00, n11, n01])])
    if plate.hole is not None:
        cx, cy, r = plate.hole
        centre = nodes[tris].mean(axis=1)
        tris = tris[(centre[:, 0] - cx) ** 2 + (centre[:, 1] - cy) ** 2 > r ** 2]
        tris = _largest_component(tris, len(nodes))
    used = np.unique(tris)
    remap = -np.ones(len(nodes), dtype=int)
    remap[used] = np.arange(len(used))
    return Mesh(nodes[used], remap[tris])


def _largest_component(tris: np.ndarray, n_nodes: int) -> np.ndarray:
    """Keep only the triangles of the largest connected piece, so a hole can never cut off an island."""
    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    g = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n_nodes, n_nodes))
    _, label = connected_components(g, directed=False)
    big = np.bincount(label[np.unique(tris)]).argmax()
    return tris[label[tris[:, 0]] == big]


def _strain_matrices(nodes: np.ndarray, tris: np.ndarray):
    p = nodes[tris]
    x, y = p[:, :, 0], p[:, :, 1]
    b = np.stack([y[:, 1] - y[:, 2], y[:, 2] - y[:, 0], y[:, 0] - y[:, 1]], axis=1)
    c = np.stack([x[:, 2] - x[:, 1], x[:, 0] - x[:, 2], x[:, 1] - x[:, 0]], axis=1)
    area = 0.5 * (x[:, 0] * (y[:, 1] - y[:, 2]) + x[:, 1] * (y[:, 2] - y[:, 0]) + x[:, 2] * (y[:, 0] - y[:, 1]))
    B = np.zeros((len(tris), 3, 6))
    B[:, 0, 0::2] = b
    B[:, 1, 1::2] = c
    B[:, 2, 0::2] = c
    B[:, 2, 1::2] = b
    B /= (2.0 * area)[:, None, None]
    return B, area


def _elasticity(plate: Plate) -> np.ndarray:
    E, nu = plate.youngs, plate.poisson
    return E / (1.0 - nu ** 2) * np.array([[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, (1.0 - nu) / 2.0]])


def assemble(mesh: Mesh, plate: Plate) -> sp.csr_matrix:
    B, area = _strain_matrices(mesh.nodes, mesh.tris)
    D = _elasticity(plate)
    Ke = np.einsum("mji,jk,mkl->mil", B, D, B) * (plate.thickness * area)[:, None, None]
    dofs = np.stack([2 * mesh.tris[:, 0], 2 * mesh.tris[:, 0] + 1,
                     2 * mesh.tris[:, 1], 2 * mesh.tris[:, 1] + 1,
                     2 * mesh.tris[:, 2], 2 * mesh.tris[:, 2] + 1], axis=1)
    rows = np.broadcast_to(dofs[:, :, None], Ke.shape).ravel()
    cols = np.broadcast_to(dofs[:, None, :], Ke.shape).ravel()
    n = 2 * len(mesh.nodes)
    return sp.coo_matrix((Ke.ravel(), (rows, cols)), shape=(n, n)).tocsr()


def edge_load(mesh: Mesh, x_edge: float, total_force, tol: float = 1e-9) -> np.ndarray:
    """Spread a total force (Fx, Fy) over the nodes on the vertical edge x = x_edge.

    The weights are the consistent nodal weights of a uniform traction: each node gets half of
    the segment below plus half of the segment above, so the end nodes get half as much.
    """
    ids = np.where(np.abs(mesh.nodes[:, 0] - x_edge) < tol)[0]
    ids = ids[np.argsort(mesh.nodes[ids, 1])]
    seg = np.diff(mesh.nodes[ids, 1])
    w = np.zeros(len(ids))
    w[:-1] += seg / 2.0
    w[1:] += seg / 2.0
    w /= w.sum()
    f = np.zeros((len(mesh.nodes), 2))
    f[ids] = w[:, None] * np.asarray(total_force, dtype=float)[None, :]
    return f.ravel()


def solve_linear(mesh: Mesh, plate: Plate, fixed_dofs: np.ndarray, force: np.ndarray):
    """Solve K u = f with the given fixed dofs. Returns (u (N, 2), reactions (N, 2), element stresses (M, 3))."""
    K = assemble(mesh, plate)
    n = 2 * len(mesh.nodes)
    free = np.setdiff1d(np.arange(n), fixed_dofs)
    u = np.zeros(n)
    u[free] = spla.spsolve(K[free][:, free].tocsc(), force[free])
    reactions = K @ u - force
    B, _ = _strain_matrices(mesh.nodes, mesh.tris)
    ue = u[np.stack([2 * mesh.tris[:, 0], 2 * mesh.tris[:, 0] + 1,
                     2 * mesh.tris[:, 1], 2 * mesh.tris[:, 1] + 1,
                     2 * mesh.tris[:, 2], 2 * mesh.tris[:, 2] + 1], axis=1)]
    stress = np.einsum("mij,mj->mi", B, ue) @ _elasticity(plate).T
    return u.reshape(-1, 2), reactions.reshape(-1, 2), stress


def von_mises(stress: np.ndarray) -> np.ndarray:
    sx, sy, txy = stress[:, 0], stress[:, 1], stress[:, 2]
    return np.sqrt(np.maximum(sx ** 2 - sx * sy + sy ** 2 + 3.0 * txy ** 2, 0.0))


def solve_cantilever(plate: Plate, force) -> Solution:
    """Clamp the left edge and load the right edge with a total force (Fx, Fy) in newtons."""
    mesh = make_mesh(plate)
    clamped = np.abs(mesh.nodes[:, 0]) < 1e-9
    ids = np.where(clamped)[0]
    fixed = np.concatenate([2 * ids, 2 * ids + 1])
    f = edge_load(mesh, plate.length, force)
    u, _, stress = solve_linear(mesh, plate, fixed, f)
    vm_elem = von_mises(stress)
    acc = np.zeros(len(mesh.nodes))
    cnt = np.zeros(len(mesh.nodes))
    for k in range(3):
        np.add.at(acc, mesh.tris[:, k], vm_elem)
        np.add.at(cnt, mesh.tris[:, k], 1.0)
    return Solution(mesh, u, f.reshape(-1, 2), clamped, acc / np.maximum(cnt, 1.0), vm_elem)


def beam_field(nodes: np.ndarray, plate: Plate, force) -> np.ndarray:
    """Euler-Bernoulli cantilever displacement field (axial load plus tip shear), ignoring any hole.

    A cheap closed-form baseline: it knows the load and the plate size but not the hole.
    """
    L, H, t, E = plate.length, plate.height, plate.thickness, plate.youngs
    I = t * H ** 3 / 12.0
    A = t * H
    Fx, Fy = float(force[0]), float(force[1])
    x = nodes[:, 0]
    yc = nodes[:, 1] - H / 2.0
    v = Fy * x ** 2 * (3.0 * L - x) / (6.0 * E * I)
    theta = Fy * x * (2.0 * L - x) / (2.0 * E * I)
    ux = Fx * x / (E * A) - yc * theta
    return np.column_stack([ux, v])


def beam_von_mises(nodes: np.ndarray, plate: Plate, force) -> np.ndarray:
    """Von Mises stress from simple beam theory (bending, axial load and parabolic shear), ignoring any hole.

    The closed-form baseline for the stress field: it cannot know about the stress concentration at a hole.
    """
    L, H, t = plate.length, plate.height, plate.thickness
    I = t * H ** 3 / 12.0
    A = t * H
    Fx, Fy = float(force[0]), float(force[1])
    x = nodes[:, 0]
    yc = nodes[:, 1] - H / 2.0
    sx = Fx / A + Fy * (L - x) * yc / I
    tau = 1.5 * Fy / A * (1.0 - (yc / (H / 2.0)) ** 2)
    return np.sqrt(sx ** 2 + 3.0 * tau ** 2)


def tip_deflection_timoshenko(plate: Plate, fy: float) -> float:
    """Tip deflection of a cantilever with a tip shear force, including the shear term."""
    L, H, t, E, nu = plate.length, plate.height, plate.thickness, plate.youngs, plate.poisson
    I = t * H ** 3 / 12.0
    G = E / (2.0 * (1.0 + nu))
    return fy * L ** 3 / (3.0 * E * I) + fy * L / ((5.0 / 6.0) * G * t * H)
