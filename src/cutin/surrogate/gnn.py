"""A small mesh graph network and a no-message-passing baseline, in plain PyTorch.

The network follows the encode, process, decode pattern used for mesh simulations: node and edge
features are encoded, then several rounds of message passing let each node learn from its neighbours
(edges update from their two end nodes, nodes update from the sum of their edges), and a decoder
turns the final node state into the three outputs. No graph library is needed.

Three models are compared:
  gnn      the mesh graph network, learning the FEM result directly
  gnn_res  the same network, but it only learns the correction to the closed-form beam formula
           (a physics prediction plus a learned residual, the same design as evprior)
  mlp      the same inputs with no message passing, to show what the graph structure adds
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from . import graph as G


def mlp(n_in: int, hidden: int, n_out: int, norm: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, n_out)]
    if norm:
        layers.append(nn.LayerNorm(n_out))
    return nn.Sequential(*layers)


class MeshGNN(nn.Module):
    def __init__(self, hidden: int = 32, steps: int = 6, residual: bool = False):
        super().__init__()
        self.residual = residual
        self.node_enc = mlp(G.NODE_FEATURES + (G.OUT if residual else 0), hidden, hidden)
        self.edge_enc = mlp(G.EDGE_FEATURES, hidden, hidden)
        self.edge_blocks = nn.ModuleList([mlp(3 * hidden, hidden, hidden) for _ in range(steps)])
        self.node_blocks = nn.ModuleList([mlp(2 * hidden, hidden, hidden) for _ in range(steps)])
        self.decoder = mlp(hidden, hidden, G.OUT, norm=False)

    def forward(self, x, edge_index, edge_attr, base=None):
        h_in = torch.cat([x, base], dim=-1) if self.residual else x
        h = self.node_enc(h_in)
        e = self.edge_enc(edge_attr)
        src, dst = edge_index
        for edge_block, node_block in zip(self.edge_blocks, self.node_blocks):
            e = e + edge_block(torch.cat([e, h[src], h[dst]], dim=-1))
            agg = torch.zeros_like(h).index_add_(0, dst, e)
            h = h + node_block(torch.cat([h, agg], dim=-1))
        out = self.decoder(h)
        return out + base if self.residual else out


class PointwiseMLP(nn.Module):
    """The same inputs and outputs but no message passing: every node is predicted on its own."""

    residual = False

    def __init__(self, hidden: int = 32, steps: int = 6):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(G.NODE_FEATURES, hidden), nn.ReLU()]
        for _ in range(steps - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers.append(nn.Linear(hidden, G.OUT))
        self.net = nn.Sequential(*layers)

    def forward(self, x, edge_index=None, edge_attr=None, base=None):
        return self.net(x)


def make_model(kind: str, hidden: int = 32, steps: int = 6) -> nn.Module:
    if kind == "gnn":
        return MeshGNN(hidden, steps)
    if kind == "gnn_res":
        return MeshGNN(hidden, steps, residual=True)
    if kind == "mlp":
        return PointwiseMLP(hidden, steps)
    raise ValueError(kind)


def train(model: nn.Module, graphs: list[G.Graph], steps: int = 500, batch_size: int = 4,
          lr: float = 3e-3, seed: int = 0) -> list[float]:
    """Fit the model to FEM results with Adam and a cosine learning-rate decay. Returns the loss curve."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr / 10)
    losses = []
    model.train()
    for _ in range(steps):
        pick = rng.choice(len(graphs), size=min(batch_size, len(graphs)), replace=False)
        x, ei, ea, y, base = G.collate([graphs[i] for i in pick])
        loss = ((model(x, ei, ea, base) - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(float(loss.detach()))
    return losses


@torch.no_grad()
def predict(model: nn.Module, g: G.Graph) -> np.ndarray:
    """Predicted (ux, uy, von Mises) at every node, in physical units (per-newton output times the load)."""
    model.eval()
    out = model(g.x, g.edge_index, g.edge_attr, g.base) * G.target_scale()
    return out.numpy() * g.magnitude


def relative_errors(pred: np.ndarray, y: np.ndarray) -> dict:
    """Relative L2 error of the displacement field and of the stress field, and the error of the peak stress."""
    d = np.linalg.norm(pred[:, :2] - y[:, :2]) / np.linalg.norm(y[:, :2])
    s = np.linalg.norm(pred[:, 2] - y[:, 2]) / np.linalg.norm(y[:, 2])
    peak = abs(pred[:, 2].max() - y[:, 2].max()) / y[:, 2].max()
    return {"disp": float(d), "stress": float(s), "peak_stress": float(peak)}


def beam_prediction(g: G.Graph) -> np.ndarray:
    """The closed-form beam formula as a prediction, in physical units (ignores the hole)."""
    return (g.base * G.target_scale()).numpy() * g.magnitude
