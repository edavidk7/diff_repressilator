"""Baseline: multistart trust-region least squares with sensitivities (the pyPESTO / Fides recipe; Raue et al. 2013).

Same objective family as ``fit.fit_gradient`` (single shooting from the same prior draws, same box), but each start
is optimised on its own by Fides (Froehlich & Sorger 2022) with the Gauss-Newton Hessian J^T J, where J holds the
sensitivities of every predicted frame to every unknown. No curriculum and no learning rate.
"""

import logging
import math
import time

import numpy as np
import torch

from differentiable_cell import reporters as R
from differentiable_cell.fit import RK4_STEP, Problem


def trajectory(problem, theta: torch.Tensor) -> torch.Tensor:
    """Observed channels at the frames ``(..., G, k)`` for ``theta (..., dim)``; classic RK4."""
    if isinstance(problem, Problem):
        p, x = problem.params_and_start(theta)
        f = lambda t, y: R.rhs(y, p, problem.mask)
    else:  # a circuits.CircuitProblem
        n_p = len(problem.c.params)
        p, x = {k: theta[..., i].exp() for i, k in enumerate(problem.c.params)}, theta[..., n_p:].exp()
        f = lambda t, y: problem.c.rhs(t, y, p)
    out = [x]
    for g in range(problem.t.numel() - 1):
        dt = float(problem.t[g + 1] - problem.t[g])
        substeps = math.ceil(dt / RK4_STEP - 1e-9)  # at most fit.RK4_STEP per step: 5 per 5-min frame
        t, h = problem.t[g], dt / substeps
        for _ in range(substeps):
            k1 = f(t, x)
            k2 = f(t + h / 2, x + h / 2 * k1)
            k3 = f(t + h / 2, x + h / 2 * k2)
            k4 = f(t + h, x + h * k3)
            x, t = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4), t + h
        out.append(x)
    return torch.stack(out, -2)[..., problem.observed_state]


FD_STEP = 1e-6  # in log-parameter units


def sensitivities(problem, theta: torch.Tensor):
    """Residuals ``(points,)`` and their Jacobian ``(points, dim)`` by central differences, from one batched solve
    of the 2 dim + 1 perturbed problems (agrees with forward-mode autodiff to 1e-9, at a twentieth of its cost)."""
    d = theta.numel()
    eye = FD_STEP * torch.eye(d, dtype=theta.dtype)
    with torch.no_grad():
        r = ((trajectory(problem, torch.cat([theta[None], theta + eye, theta - eye])) - problem.y) / problem.range).flatten(-2)
    return r[0], ((r[1:d + 1] - r[d + 1:]) / (2 * FD_STEP)).T


def fit_fides(problem, restarts: int = 64, maxiter: int = 200, seed: int = 0):
    """Returns the best start's theta and per-start diagnostics (final MSE, iterations, exit flag)."""
    import fides

    n = problem.y.numel()
    evals = 0

    def objective(x):  # MSE (paper Eq. (5) with rho = square), its gradient and Gauss-Newton Hessian
        nonlocal evals
        evals += 1
        r, J = sensitivities(problem, torch.from_numpy(x))
        if not (torch.isfinite(J).all() and torch.isfinite(r).all()):
            return math.inf, np.zeros_like(x), np.eye(len(x))
        return float(r @ r) / n, (2 / n * (J.T @ r)).numpy(), (2 / n * (J.T @ J)).numpy()

    # the first ``restarts`` of the 64 draws fit_gradient starts from
    starts = problem.sample_prior(64, torch.Generator().manual_seed(seed))[:restarts]
    lo, hi = problem.lo.numpy(), problem.hi.numpy()
    thetas, info = [], {"mse": [], "iterations": [], "exit": [], "seconds": []}
    for x0 in starts.numpy():
        opt = fides.Optimizer(objective, ub=hi, lb=lo, verbose=logging.ERROR,
                              options={fides.Options.MAXITER: maxiter})
        t0 = time.perf_counter()
        try:
            opt.minimize(x0)
            x, fval = opt.x_min, float(opt.fval_min)
        except Exception:  # a start whose first evaluation is not finite
            x, fval = x0, math.inf
        thetas.append(torch.from_numpy(np.clip(x, lo, hi)))
        info["mse"].append(fval); info["iterations"].append(int(opt.iteration)); info["exit"].append(int(opt.exitflag))
        info["seconds"].append(time.perf_counter() - t0)
    info["evaluations"] = evals
    # choose the start as fit_gradient does, by the full-record MSE under the solver that scores every method:
    # a start can fit well only through this module's own RK4 error
    with torch.no_grad():
        info["mse_scored"] = torch.nan_to_num(problem.loss(torch.stack(thetas)), nan=math.inf).tolist()
    return thetas[int(np.argmin(info["mse_scored"]))], info
