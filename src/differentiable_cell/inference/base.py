"""What every estimator returns, and small shared utilities."""

import time
from dataclasses import dataclass, field
from typing import Any

import torch

from differentiable_cell.inference.problem import Problem


@dataclass
class Result:
    """Output of one estimator on one problem.

    Attributes:
        method: estimator name.
        theta: point estimate in ``theta`` space (log values), ``(dim,)``.
        samples: posterior (or approximate-posterior) draws in ``theta``
            space, ``(N, dim)``, equally weighted; ``None`` for pure point
            estimators.
        n_sims: forward trajectories simulated (a batch of B counts B; a
            gradient evaluation counts its forward pass once).
        wall: wall-clock seconds.
        info: method-specific diagnostics (JSON-serialisable).
    """

    method: str
    theta: torch.Tensor
    samples: torch.Tensor | None
    n_sims: int
    wall: float
    info: dict[str, Any] = field(default_factory=dict)


class Timer:
    """Wall clock plus the problem's simulation counter, for a :class:`Result`."""

    def __init__(self, problem: Problem):
        self.problem = problem

    def __enter__(self) -> "Timer":
        self.t0 = time.perf_counter()
        self.sims0 = self.problem.n_sims
        return self

    def __exit__(self, *exc) -> None:
        self.wall = time.perf_counter() - self.t0
        self.n_sims = self.problem.n_sims - self.sims0

    def elapsed(self) -> float:
        return time.perf_counter() - self.t0


def clip_to_box(problem: Problem, theta: torch.Tensor) -> torch.Tensor:
    lo, hi = problem.bounds.unbind(-1)
    return torch.maximum(torch.minimum(theta, hi), lo)


def gauss_newton_covariance(problem: Problem, theta: torch.Tensor) -> torch.Tensor:
    """Laplace covariance of ``theta`` from the Gauss-Newton Hessian.

    ``H = J^T diag(1/sigma^2) J + P`` with ``J`` the Jacobian of the
    predicted observations and ``P`` the precision of the uniform prior box
    (variance ``width^2 / 12`` per coordinate), which keeps unidentifiable
    directions at prior width rather than infinite.  Forward-mode AD on the
    eager integrator: one pass per coordinate.
    """
    theta = theta.detach()
    f = lambda th: problem.simulate(th, compiled=False).reshape(-1)
    jac = torch.autograd.functional.jacobian(f, theta, vectorize=True, strategy="forward-mode")
    var = problem.obs_variance().reshape(-1)
    lo, hi = problem.bounds.unbind(-1)
    h = jac.T @ (jac / var.unsqueeze(-1)) + torch.diag(12.0 / (hi - lo) ** 2)
    h = 0.5 * (h + h.T)
    evals, evecs = torch.linalg.eigh(h)
    return (evecs / evals.clamp_min(1e-12)) @ evecs.T


def laplace_samples(problem: Problem, theta: torch.Tensor, cov: torch.Tensor,
                    n: int = 4000, generator: torch.Generator | None = None) -> torch.Tensor:
    """Draws from ``N(theta, cov)`` truncated to the prior box (by rejection)."""
    chol = torch.linalg.cholesky(cov + 1e-12 * torch.eye(cov.shape[0], dtype=cov.dtype))
    lo, hi = problem.bounds.unbind(-1)
    keep: list[torch.Tensor] = []
    total = 0
    for _ in range(50):
        eps = torch.randn(4 * n, cov.shape[0], generator=generator, dtype=cov.dtype)
        draw = theta + eps @ chol.T
        ok = ((draw >= lo) & (draw <= hi)).all(-1)
        keep.append(draw[ok])
        total += int(ok.sum())
        if total >= n:
            break
    draws = torch.cat(keep)[:n]
    if draws.shape[0] < n:  # box truncation too severe: fall back to clipping
        eps = torch.randn(n - draws.shape[0], cov.shape[0], generator=generator, dtype=cov.dtype)
        draws = torch.cat([draws, clip_to_box(problem, theta + eps @ chol.T)])
    return draws
