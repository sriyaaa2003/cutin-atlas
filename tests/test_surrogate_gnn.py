"""Checks of the graph construction and the graph network."""
import numpy as np
import torch

from cutin.surrogate import fem, gnn
from cutin.surrogate import graph as G


def small_graphs(n=2, seed=0):
    return G.make_dataset(n, seed=seed, nx=20, ny=6)


def test_cases_are_reproducible_and_vary():
    a = G.sample_case(np.random.default_rng(3))
    b = G.sample_case(np.random.default_rng(3))
    c = G.sample_case(np.random.default_rng(4))
    assert a.force == b.force and a.plate.hole == b.plate.hole
    assert a.force != c.force


def test_graph_edges_are_symmetric_unique_and_have_no_loops():
    g = small_graphs(1)[0]
    ei = g.edge_index.numpy()
    pairs = set(map(tuple, ei.T))
    assert len(pairs) == ei.shape[1]
    assert all((b, a) in pairs for a, b in pairs)
    assert (ei[0] != ei[1]).all()
    assert ei.max() == g.x.shape[0] - 1


def test_targets_are_per_newton_and_the_load_is_multiplied_back():
    g = small_graphs(1)[0]
    scale = G.target_scale()
    back = (g.y_unit * scale).numpy() * g.magnitude
    assert np.allclose(back, g.y.numpy(), rtol=1e-4, atol=1e-12)


def test_beam_baseline_is_stored_per_node():
    g = small_graphs(1)[0]
    pred = gnn.beam_prediction(g)
    assert pred.shape == (g.x.shape[0], 3)
    direct = fem.beam_field(g.nodes, g.case.plate, g.case.force)
    assert np.allclose(pred[:, :2], direct, rtol=1e-3, atol=1e-12)


def test_network_is_permutation_equivariant():
    g = small_graphs(1)[0]
    torch.manual_seed(0)
    model = gnn.make_model("gnn_res").eval()
    perm = torch.randperm(g.x.shape[0])
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(len(perm))
    with torch.no_grad():
        out = model(g.x, g.edge_index, g.edge_attr, g.base)
        out_p = model(g.x[perm], inv[g.edge_index], g.edge_attr, g.base[perm])
    assert torch.allclose(out[perm], out_p, atol=1e-5)


def test_network_learns_a_tiny_dataset():
    graphs = small_graphs(2)
    torch.manual_seed(0)
    model = gnn.make_model("gnn", hidden=16, steps=3)
    losses = gnn.train(model, graphs, steps=150, batch_size=2)
    assert np.mean(losses[-10:]) < 0.5 * np.mean(losses[:10])


def test_residual_network_starts_from_the_beam_formula():
    g = small_graphs(1)[0]
    torch.manual_seed(0)
    model = gnn.make_model("gnn_res", hidden=16, steps=2)
    start = gnn.predict(model, g)
    beam = gnn.beam_prediction(g)
    # an untrained residual network adds only a small correction to the physics prediction
    assert np.linalg.norm(start - beam) < 5 * np.linalg.norm(beam)


def test_relative_errors_are_zero_for_a_perfect_prediction():
    g = small_graphs(1)[0]
    e = gnn.relative_errors(g.y.numpy(), g.y.numpy())
    assert e["disp"] == 0 and e["stress"] == 0 and e["peak_stress"] == 0
