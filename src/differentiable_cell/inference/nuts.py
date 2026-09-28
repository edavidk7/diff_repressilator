"""No-U-Turn Sampler through the differentiable ODE.

The Bayesian counterpart of :mod:`.grad_ode`: the same gradients, spent on
sampling the posterior instead of finding its mode.  Hoffman & Gelman (2014),
Algorithm 6 (slice NUTS with dual-averaging step size), written out here.

It runs in whitened coordinates ``x = L^{-1} (z - z_map)`` where ``L L^T`` is
the inverse of the exact Hessian of the log posterior at the MAP (in ``z``
space), i.e. with that Laplace covariance as a dense mass matrix.  Starting at
the MAP with that metric is what makes NUTS affordable here: the
repressilator posterior is strongly correlated (``alpha`` against
``alpha_GFP``, ``n`` against ``alpha``), and an identity metric needs trees
tens of times deeper.
"""

import math
import time
from typing import Callable

import torch

from differentiable_cell.inference import grad_ode
from differentiable_cell.inference.base import Result, Timer
from differentiable_cell.inference.problem import Problem

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


def fit(
    problem: Problem,
    map_result: Result | None = None,
    chains: int = 4,
    n_warmup: int = 150,
    n_samples: int = 250,
    max_depth: int = 6,
    seed: int = 0,
    max_seconds: float | None = None,
    **map_kwargs,
) -> Result:
    """NUTS on the ODE posterior: several chains started at, and preconditioned
    by, the MAP.

    The metric is the inverse of the *exact* Hessian of the z-space
    log-posterior at the MAP (eigenvalues floored), not the Gauss-Newton
    one.  Gauss-Newton drops the residual curvature, which is large exactly
    where the model is misspecified or the posterior curved: with it, the
    step size collapsed to ~5e-7 on stochastic data and to ~0.01 on partial
    observations, and the chains never left the MAP.

    Chains start from the MAP plus a unit-scale jitter in whitened
    coordinates, with independent seeds; split-R-hat and draws per chain are
    reported so a run that did not mix is visible.

    Args:
        problem: the inverse problem.
        map_result: an existing :func:`grad_ode.fit` result to start from; one
            is computed if omitted (its cost is included either way).
        chains: independent chains (run one after another).
        n_warmup, n_samples, max_depth: per chain, see :func:`nuts`.
        seed: random seed.
        max_seconds: total sampling budget, split evenly over the chains.
        map_kwargs: passed to :func:`grad_ode.fit` when it is run here.
    """
    given = map_result is not None
    with Timer(problem) as timer:
        if map_result is None:
            map_result = grad_ode.fit(problem, seed=seed, **map_kwargs)
        z_map = problem.to_z(map_result.theta).detach()

        def neg_logp(z: torch.Tensor) -> torch.Tensor:
            return -(problem.log_likelihood(problem.from_z(z), compiled=False)
                     + problem.log_prior_z(z))

        hess = torch.autograd.functional.hessian(neg_logp, z_map)
        hess = 0.5 * (hess + hess.T)
        evals, evecs = torch.linalg.eigh(hess)
        # Directions the data do not determine (near-zero or negative
        # curvature: the MAP is a saddle along a ridge) get the prior's own
        # z-space curvature, 0.25 at z = 0, i.e. prior-width steps.  A tiny
        # floor (1e-6) made them ~1000 z-units long, every step along them was
        # rejected, and dual averaging shrank the step size to ~1e-7.
        floor = 0.25
        chol = evecs / evals.clamp_min(floor).sqrt()  # chol @ chol.T = H^-1

        def logp_grad(x: torch.Tensor):
            x = x.detach().requires_grad_(True)
            z = z_map + chol @ x
            lp = problem.log_likelihood(problem.from_z(z)) + problem.log_prior_z(z)
            if not torch.isfinite(lp):
                return -math.inf, torch.zeros_like(x)
            (g,) = torch.autograd.grad(lp, x)
            return float(lp), torch.nan_to_num(g.detach())

        g = torch.Generator().manual_seed(seed)
        per_chain = None if max_seconds is None else max_seconds / chains
        draws, diags = [], []
        for c in range(chains):
            x0 = torch.randn(problem.dim, generator=g, dtype=z_map.dtype)
            xs, diag = nuts(logp_grad, x0, n_warmup=n_warmup, n_samples=n_samples,
                            max_depth=max_depth, seed=seed * 100 + c, max_seconds=per_chain)
            draws.append(xs)
            diags.append(diag)
        kept = min(d.shape[0] for d in draws)
        stacked = torch.stack([d[-kept:] for d in draws])  # (C, n, d) whitened
        z_chains = z_map + stacked @ chol.T
        theta_chains = problem.from_z(z_chains)
        rhat = split_rhat(theta_chains)
        samples = theta_chains.reshape(-1, problem.dim)
    k = len(problem.free_params)
    return Result(
        method="nuts",
        theta=samples.median(0).values,
        samples=samples,
        # The MAP is part of this method's cost whether it was run here or not.
        n_sims=timer.n_sims + (map_result.n_sims if given else 0),
        wall=timer.wall + (map_result.wall if given else 0.0),
        info={"chains": chains, "draws_per_chain": kept,
              "step_size": [d["step_size"] for d in diags],
              "mean_tree_depth": [d["mean_tree_depth"] for d in diags],
              "mean_accept": [d["mean_accept"] for d in diags],
              "rhat_params": rhat[:k].tolist(), "max_rhat_params": float(rhat[:k].max()),
              "negative_hessian_eigs": int((evals < 0).sum()),
              "map_wall": map_result.wall, "map_sims": map_result.n_sims},
    )
