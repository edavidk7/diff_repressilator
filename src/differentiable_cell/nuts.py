"""NUTS through the ODE: the Bayesian version of backprop.

Same gradients as :func:`differentiable_cell.fit.fit_gradient`, spent on
sampling the posterior instead of finding its mode:

* likelihood: Gaussian on the range-normalised residuals, with each channel's
  noise variance estimated from the residuals of the gradient fit (a plug-in;
  no truth used); prior: uniform on the boxes;
* start and metric: the gradient fit (MAP), preconditioned by the exact
  Hessian of the log posterior there. Eigenvalues are floored at the prior's
  own curvature (0.25 in z), so directions the data do not determine get
  prior-width steps - a tiny floor made the step size collapse to ~1e-7;
* 4 chains from jittered starts, split-R-hat reported (> 1.1: not converged).

The sampler (Hoffman & Gelman 2014, Algorithm 6 with dual averaging) is
written out below.
"""

import math
import time
from typing import Callable

import torch

from differentiable_cell.fit import Problem

LogProb = Callable[[torch.Tensor], tuple[float, torch.Tensor]]


def _leapfrog(x, r, grad, eps, logp_grad: LogProb):
    r = r + 0.5 * eps * grad
    x = x + eps * r
    lp, grad = logp_grad(x)
    r = r + 0.5 * eps * grad
    return x, r, lp, grad


def _build_tree(x, r, grad, log_u, v, j, eps, joint0, logp_grad, rng):
    """Recursive doubling; returns the tree's edges, proposal and statistics."""
    if j == 0:
        x1, r1, lp1, g1 = _leapfrog(x, r, grad, v * eps, logp_grad)
        joint = lp1 - 0.5 * float(r1 @ r1)
        if not math.isfinite(joint):
            joint = -math.inf
        n1 = int(log_u <= joint)
        s1 = int(log_u < joint + 1000.0)
        accept = min(1.0, math.exp(min(0.0, joint - joint0))) if math.isfinite(joint) else 0.0
        return x1, r1, g1, x1, r1, g1, x1, lp1, g1, n1, s1, accept, 1
    (xm, rm, gm, xp, rp, gp, x1, lp1, g1, n1, s1, a1, na1) = _build_tree(
        x, r, grad, log_u, v, j - 1, eps, joint0, logp_grad, rng)
    if s1 == 1:
        if v == -1:
            xm, rm, gm, _, _, _, x2, lp2, g2, n2, s2, a2, na2 = _build_tree(
                xm, rm, gm, log_u, v, j - 1, eps, joint0, logp_grad, rng)
        else:
            _, _, _, xp, rp, gp, x2, lp2, g2, n2, s2, a2, na2 = _build_tree(
                xp, rp, gp, log_u, v, j - 1, eps, joint0, logp_grad, rng)
        if n1 + n2 > 0 and float(torch.rand((), generator=rng)) < n2 / (n1 + n2):
            x1, lp1, g1 = x2, lp2, g2
        a1 += a2
        na1 += na2
        dx = xp - xm
        s1 = s2 * int(float(dx @ rm) >= 0) * int(float(dx @ rp) >= 0)
        n1 += n2
    return xm, rm, gm, xp, rp, gp, x1, lp1, g1, n1, s1, a1, na1


def nuts(
    logp_grad: LogProb,
    x0: torch.Tensor,
    n_warmup: int = 300,
    n_samples: int = 500,
    target_accept: float = 0.8,
    max_depth: int = 6,
    seed: int = 0,
    max_seconds: float | None = None,
) -> tuple[torch.Tensor, dict]:
    """Sample with NUTS (identity metric) from ``x0``.

    Args:
        logp_grad: returns ``(log density, gradient)`` at a point.
        x0: starting point.
        n_warmup: dual-averaging adaptation iterations (discarded).
        n_samples: kept iterations.
        target_accept: dual-averaging target.
        max_depth: maximum tree depth (``2**max_depth - 1`` leapfrogs).
        seed: random seed.
        max_seconds: stop sampling (keeping what was drawn) after this.

    Returns:
        ``(samples (N, d), diagnostics)``.
    """
    rng = torch.Generator().manual_seed(seed)
    d = x0.numel()
    x = x0.clone()
    lp, grad = logp_grad(x)

    # Heuristic initial step size (Algorithm 4).
    eps = 0.5
    r = torch.randn(d, generator=rng, dtype=x.dtype)
    _, r1, lp1, _ = _leapfrog(x, r, grad, eps, logp_grad)
    ratio = (lp1 - 0.5 * float(r1 @ r1)) - (lp - 0.5 * float(r @ r))
    a = 1 if (math.isfinite(ratio) and ratio > math.log(0.5)) else -1
    for _ in range(50):
        _, r1, lp1, _ = _leapfrog(x, r, grad, eps, logp_grad)
        ratio = (lp1 - 0.5 * float(r1 @ r1)) - (lp - 0.5 * float(r @ r))
        if not math.isfinite(ratio):
            ratio = -math.inf
        if a * ratio <= -a * math.log(2):
            break
        eps *= 2.0**a

    mu, log_eps_bar, h_bar = math.log(10 * eps), 0.0, 0.0
    gamma, t0, kappa = 0.05, 10.0, 0.75
    samples, depths, accepts = [], [], []
    start = time.perf_counter()
    for m in range(1, n_warmup + n_samples + 1):
        r0 = torch.randn(d, generator=rng, dtype=x.dtype)
        joint0 = lp - 0.5 * float(r0 @ r0)
        log_u = joint0 + math.log(float(torch.rand((), generator=rng)) + 1e-300)
        xm = xp = x
        rm = rp = r0
        gm = gp = grad
        j, n, s = 0, 1, 1
        acc, n_acc = 0.0, 1
        while s == 1 and j < max_depth:
            v = 1 if float(torch.rand((), generator=rng)) < 0.5 else -1
            if v == -1:
                xm, rm, gm, _, _, _, x1, lp1, g1, n1, s1, acc, n_acc = _build_tree(
                    xm, rm, gm, log_u, v, j, eps, joint0, logp_grad, rng)
            else:
                _, _, _, xp, rp, gp, x1, lp1, g1, n1, s1, acc, n_acc = _build_tree(
                    xp, rp, gp, log_u, v, j, eps, joint0, logp_grad, rng)
            if s1 == 1 and float(torch.rand((), generator=rng)) < min(1.0, n1 / n):
                x, lp, grad = x1, lp1, g1
            n += n1
            dx = xp - xm
            s = s1 * int(float(dx @ rm) >= 0) * int(float(dx @ rp) >= 0)
            j += 1
        stat = acc / max(n_acc, 1)
        if m <= n_warmup:
            h_bar = (1 - 1 / (m + t0)) * h_bar + (target_accept - stat) / (m + t0)
            log_eps = mu - math.sqrt(m) / gamma * h_bar
            eta = m**-kappa
            log_eps_bar = eta * log_eps + (1 - eta) * log_eps_bar
            eps = math.exp(log_eps)
            if m == n_warmup:
                eps = math.exp(log_eps_bar)
        else:
            samples.append(x.clone())
            depths.append(j)
            accepts.append(stat)
        if max_seconds is not None and time.perf_counter() - start > max_seconds:
            break
    out = torch.stack(samples) if samples else x.unsqueeze(0)
    return out, {
        "step_size": eps,
        "mean_tree_depth": sum(depths) / max(len(depths), 1),
        "mean_accept": sum(accepts) / max(len(accepts), 1),
        "n_kept": len(samples),
    }


def split_rhat(chains: torch.Tensor) -> torch.Tensor:
    """Split-R-hat per coordinate for ``(n_chains, n_draws, d)`` draws."""
    n = chains.shape[1] // 2
    if n < 2:
        return torch.full((chains.shape[-1],), float("nan"), dtype=chains.dtype)
    halves = torch.cat([chains[:, :n], chains[:, n:2 * n]], dim=0)  # (2C, n, d)
    within = halves.var(1).mean(0)
    between = n * halves.mean(1).var(0)
    var_plus = (n - 1) / n * within + between / n
    return (var_plus / within.clamp_min(1e-300)).sqrt()




def fit_nuts(problem: Problem, theta_map: torch.Tensor, chains: int = 4, warmup: int = 150,
             samples: int = 250, max_depth: int = 6, max_seconds: float = 1200.0, seed: int = 0):
    """Sample the posterior from the MAP. Returns posterior draws ``(N, dim)`` and diagnostics."""
    with torch.no_grad():
        resid = (problem.predict(theta_map) - problem.y) / problem.range
        noise_var = resid.pow(2).mean(0)  # per channel, from the MAP fit
    z_map = problem.to_z(theta_map).detach()

    def log_post(z):
        r = (problem.predict(problem.from_z(z)) - problem.y) / problem.range
        log_lik = -0.5 * (r**2 / noise_var).sum()
        log_prior = (torch.nn.functional.logsigmoid(z) + torch.nn.functional.logsigmoid(-z)).sum()
        return log_lik + log_prior

    hess = torch.autograd.functional.hessian(lambda z: -log_post(z), z_map)
    evals, evecs = torch.linalg.eigh(0.5 * (hess + hess.T))
    chol = evecs / evals.clamp_min(0.25).sqrt()  # chol @ chol.T = floored inverse Hessian

    def logp_grad(x):
        x = x.detach().requires_grad_(True)
        lp = log_post(z_map + chol @ x)
        if not torch.isfinite(lp):
            return -math.inf, torch.zeros_like(x)
        (g,) = torch.autograd.grad(lp, x)
        return float(lp), torch.nan_to_num(g)

    g = torch.Generator().manual_seed(seed)
    draws, diags = [], []
    for c in range(chains):
        x0 = torch.randn(len(z_map), generator=g, dtype=z_map.dtype)
        xs, diag = nuts(logp_grad, x0, n_warmup=warmup, n_samples=samples, max_depth=max_depth,
                        seed=seed * 100 + c, max_seconds=max_seconds / chains)
        draws.append(xs)
        diags.append(diag)
    kept = min(len(d) for d in draws)
    theta_chains = problem.from_z(z_map + torch.stack([d[-kept:] for d in draws]) @ chol.T)
    return theta_chains.reshape(-1, len(z_map)), {
        "draws_per_chain": kept, "rhat_max": float(split_rhat(theta_chains).max()),
        "step_size": [d["step_size"] for d in diags], "mean_tree_depth": [d["mean_tree_depth"] for d in diags]}


def fit_laplace(problem: Problem, theta_map: torch.Tensor, n_draws: int = 4000, seed: int = 0):
    """Laplace approximation at the MAP: a Gaussian with the (floored) inverse Hessian.

    The same log posterior and Hessian as :func:`fit_nuts`, but no sampling of
    the posterior itself: the "free" uncertainty that comes with backprop.
    Returns draws ``(n_draws, dim)`` (in the box) and diagnostics.
    """
    with torch.no_grad():
        resid = (problem.predict(theta_map) - problem.y) / problem.range
        noise_var = resid.pow(2).mean(0)
    z_map = problem.to_z(theta_map).detach()

    def neg_log_post(z):
        r = (problem.predict(problem.from_z(z)) - problem.y) / problem.range
        log_prior = (torch.nn.functional.logsigmoid(z) + torch.nn.functional.logsigmoid(-z)).sum()
        return 0.5 * (r**2 / noise_var).sum() - log_prior

    hess = torch.autograd.functional.hessian(neg_log_post, z_map)
    evals, evecs = torch.linalg.eigh(0.5 * (hess + hess.T))
    chol = evecs / evals.clamp_min(0.25).sqrt()
    g = torch.Generator().manual_seed(seed)
    eps = torch.randn(n_draws, len(z_map), generator=g, dtype=z_map.dtype)
    return problem.from_z(z_map + eps @ chol.T), {"negative_hessian_eigs": int((evals < 0).sum())}
