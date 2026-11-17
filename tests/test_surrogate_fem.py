"""Checks of the finite element solver against exact results."""
import numpy as np

from cutin.surrogate import fem


def _right_edge(mesh, plate):
    return np.where(np.abs(mesh.nodes[:, 0] - plate.length) < 1e-9)[0]


def test_patch_test_uniform_tension_is_exact():
    # With the left edge free to contract sideways, uniform tension must give a uniform stress field.
    plate = fem.Plate(nx=12, ny=6)
    mesh = fem.make_mesh(plate)
    left = np.where(np.abs(mesh.nodes[:, 0]) < 1e-9)[0]
    corner = np.where((np.abs(mesh.nodes[:, 0]) < 1e-9) & (np.abs(mesh.nodes[:, 1]) < 1e-9))[0][0]
    fixed = np.concatenate([2 * left, [2 * corner + 1]])
    s0 = 1e6
    f = fem.edge_load(mesh, plate.length, (s0 * plate.height * plate.thickness, 0.0))
    u, _, stress = fem.solve_linear(mesh, plate, fixed, f)
    assert np.allclose(stress[:, 0], s0, rtol=1e-9)
    assert np.abs(stress[:, 1]).max() < 1e-6 * s0
    assert np.abs(stress[:, 2]).max() < 1e-6 * s0
    ux_tip = u[_right_edge(mesh, plate), 0].mean()
    assert np.isclose(ux_tip, s0 * plate.length / plate.youngs, rtol=1e-9)


def test_reactions_balance_the_load_with_a_hole():
    plate = fem.Plate(hole=(0.5, 0.12, 0.05))
    mesh = fem.make_mesh(plate)
    ids = np.where(np.abs(mesh.nodes[:, 0]) < 1e-9)[0]
    force = (30.0, -80.0)
    f = fem.edge_load(mesh, plate.length, force)
    _, reactions, _ = fem.solve_linear(mesh, plate, np.concatenate([2 * ids, 2 * ids + 1]), f)
    assert np.allclose(reactions.sum(axis=0), -np.array(force), atol=1e-6)


def test_cantilever_converges_to_beam_theory():
    ratios = []
    for nx, ny in [(30, 6), (60, 12), (120, 24)]:
        plate = fem.Plate(nx=nx, ny=ny)
        sol = fem.solve_cantilever(plate, (0.0, 100.0))
        tip = sol.u[_right_edge(sol.mesh, plate), 1].mean()
        ratios.append(tip / fem.tip_deflection_timoshenko(plate, 100.0))
    assert ratios[0] < ratios[1] < ratios[2] < 1.0   # the triangles are a little stiff, and the error shrinks
    assert ratios[2] > 0.98


def test_beam_field_matches_the_solver_without_a_hole():
    plate = fem.Plate()
    force = (20.0, 80.0)
    sol = fem.solve_cantilever(plate, force)
    beam = fem.beam_field(sol.mesh.nodes, plate, force)
    assert np.linalg.norm(beam - sol.u) / np.linalg.norm(sol.u) < 0.05


def test_hole_makes_the_plate_more_flexible():
    plain = fem.solve_cantilever(fem.Plate(), (0.0, 100.0))
    holed = fem.solve_cantilever(fem.Plate(hole=(0.5, 0.125, 0.06)), (0.0, 100.0))
    tip = lambda s, p: s.u[_right_edge(s.mesh, p), 1].mean()
    assert tip(holed, fem.Plate(hole=(0.5, 0.125, 0.06))) > tip(plain, fem.Plate())


def test_hole_raises_the_peak_stress_near_it():
    plate = fem.Plate(hole=(0.5, 0.125, 0.06))
    sol = fem.solve_cantilever(plate, (0.0, 100.0))
    near = np.hypot(sol.mesh.nodes[:, 0] - 0.5, sol.mesh.nodes[:, 1] - 0.125) < 0.09
    beam = fem.beam_von_mises(sol.mesh.nodes, plate, (0.0, 100.0))
    assert (sol.vm_node[near] / np.maximum(beam[near], 1.0)).max() > 1.5   # stress concentration the beam formula misses


def test_results_scale_linearly_with_the_load():
    plate = fem.Plate(nx=30, ny=6, hole=(0.5, 0.125, 0.05))
    a = fem.solve_cantilever(plate, (10.0, 40.0))
    b = fem.solve_cantilever(plate, (20.0, 80.0))
    assert np.allclose(b.u, 2 * a.u, rtol=1e-9)
    assert np.allclose(b.vm_node, 2 * a.vm_node, rtol=1e-9)


def test_mesh_is_one_connected_piece_with_no_unused_nodes():
    mesh = fem.make_mesh(fem.Plate(hole=(0.4, 0.12, 0.06)))
    assert set(np.unique(mesh.tris)) == set(range(len(mesh.nodes)))
    centre = mesh.nodes[mesh.tris].mean(axis=1)
    assert ((centre[:, 0] - 0.4) ** 2 + (centre[:, 1] - 0.12) ** 2 > 0.06 ** 2).all()   # no triangle inside the hole
