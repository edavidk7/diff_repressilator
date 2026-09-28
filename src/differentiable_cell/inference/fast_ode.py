"""A fast, vectorised RK4 for the repressilator, for inner loops of inference.

:class:`differentiable_cell.model.Repressilator` writes each species on its
own line, which is the readable form and the reference.  Inference calls the
forward model 10^3-10^5 times, and there the per-species Python overhead
dominates: this module computes the same right-hand side on four gene
"lanes" at once and ``torch.compile``s the RK4 step, which is ~20x faster
for a forward+backward pass on CPU.  ``tests/test_estimators.py`` checks it
against the reference model.

Lane layout (``FIELDS`` order is ``m_lacI, m_tetR, m_cI, p_LacI, p_TetR,
p_CI, m_gfp, p_GFP``)::

    lane      mRNA    protein   repressor
    lacI      0       3         5 (CI)
    tetR      1       4         3 (LacI)
    cI        2       5         4 (TetR)
    gfp       6       7         4 (TetR)
"""

import os

import torch

_M = torch.tensor([0, 1, 2, 6])
_P = torch.tensor([3, 4, 5, 7])
_REP = torch.tensor([5, 3, 4, 4])
# Inverse permutation: lanes (m0..m3, p0..p3) back to FIELDS order.
_UNLANE = torch.argsort(torch.cat([_M, _P]))


def _rhs(y, alpha, alpha_0, n, beta):
    m, p, rep = y[..., _M], y[..., _P], y[..., _REP]
    dm = -m + alpha / (1.0 + rep.clamp_min(0.0) ** n) + alpha_0
    dp = beta * (m - p)
    return torch.cat([dm, dp], -1)[..., _UNLANE]


def _step(y, alpha, alpha_0, n, beta, h):
    k1 = _rhs(y, alpha, alpha_0, n, beta)
    k2 = _rhs(y + 0.5 * h * k1, alpha, alpha_0, n, beta)
    k3 = _rhs(y + 0.5 * h * k2, alpha, alpha_0, n, beta)
    k4 = _rhs(y + h * k3, alpha, alpha_0, n, beta)
    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


_COMPILE = os.environ.get("DC_NO_COMPILE", "") == ""
_compiled_step = torch.compile(_step, dynamic=True) if _COMPILE else _step


def rk4(
    y0: torch.Tensor,
    alpha: torch.Tensor,
    alpha_0: torch.Tensor,
    n: torch.Tensor,
    beta: torch.Tensor,
    alpha_gfp: torch.Tensor,
    beta_gfp: torch.Tensor,
    h: float,
    n_steps: int,
    stride: int = 1,
    compiled: bool = True,
) -> torch.Tensor:
    """Integrate from ``y0`` (``(..., 8)``), returning every ``stride``-th state.

    Parameters broadcast over the batch; the output has shape
    ``(..., n_steps // stride + 1, 8)``.  ``compiled=False`` (or the
    environment variable ``DC_NO_COMPILE=1``) runs eagerly, which forward-mode
    AD and double backward need.
    """
    step = _compiled_step if compiled else _step
    lane = lambda shared, gfp: torch.stack(torch.broadcast_tensors(shared, shared, shared, gfp), -1)
    a = lane(alpha, alpha_gfp)
    b = lane(beta, beta_gfp)
    a0 = alpha_0.unsqueeze(-1)
    nn = n.unsqueeze(-1)
    y = y0
    out = [y]
    for k in range(n_steps):
        y = step(y, a, a0, nn, b, h)
        if (k + 1) % stride == 0:
            out.append(y)
    return torch.stack(out, dim=-2)
