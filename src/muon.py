#!/usr/bin/env python
"""Muon optimizer (Jordan et al., 2024): MomentUm Orthogonalized by Newton-Schulz.

For 2D *hidden* weight matrices only (attention, MLP, the classifier's hidden layer);
embeddings, the tied LM head, LayerNorms and biases stay on AdamW. Orthogonalizing the
momentum update spreads the step across singular directions, giving more effective
learning per step -- aimed squarely at our underfitting (10-epoch-capped) regime, without
adding capacity or hardening the task (the two Phase-2 traps).

Newton-Schulz runs in fp32 for MPS safety (our matrices are tiny, <=2560x384).
"""
import torch


@torch.no_grad()
def newtonschulz5(G, steps=5, eps=1e-7):
    """Quintic Newton-Schulz iteration -> approximate orthogonalization of a 2D matrix."""
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    X = X / (X.norm() + eps)
    transpose = X.size(0) > X.size(1)
    if transpose:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transpose:
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, mom, nesterov, ns = group["lr"], group["momentum"], group["nesterov"], group["ns_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if "moment" not in st:
                    st["moment"] = torch.zeros_like(g)
                buf = st["moment"]
                buf.mul_(mom).add_(g)
                u = g.add(buf, alpha=mom) if nesterov else buf
                u = newtonschulz5(u, ns)
                # shape-aware scale so the update RMS is comparable across matrix shapes
                p.add_(u.to(p.dtype), alpha=-lr * (max(1.0, p.size(0) / p.size(1)) ** 0.5))
