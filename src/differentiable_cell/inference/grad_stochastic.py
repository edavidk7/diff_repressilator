"""Backpropagation through a stochastic simulator (CLE or DGA).

The stochastic counterpart of :mod:`.grad_ode`.  For stochastic data the ODE
is misspecified - it has no intrinsic noise, and a single cell's phase
wanders away from any deterministic trajectory - so here the loss compares
*distributions*: each step simulates a small population of cells per
restart and descends the regime's distance of :func:`.stochastic.distance`
(summary statistics for a single-cell trace, mean and spread for an
ensemble).  Fresh noise every step makes this stochastic gradient descent
with pathwise (reparameterised) gradients.

The differentiable Gillespie (``kind="dga"``) keeps every reaction event on
the autograd tape, so it is only affordable at a small system size and/or
on a leading window of the record (``n_obs_times``).
"""

import math
import time

import torch

from differentiable_cell.inference import stochastic
from differentiable_cell.inference.base import Result, Timer
from differentiable_cell.inference.problem import Problem


def fit(
    problem: Problem,
    kind: str = "cle",
    restarts: int = 8,
    epochs: int = 600,
    n_cells: int = 32,
    lr: float = 0.05,
    lr_final: float = 0.0025,
    n_obs_times: int | None = None,
    eval_cells: int = 128,
    seed: int = 0,
    max_seconds: float | None = None,
    dga_kwargs: dict | None = None,
) -> Result:
    """Adam on the stochastic-simulator distance, from prior restarts.

    Args:
        problem: inverse problem with ``regime`` ``"single_cell"`` or
            ``"ensemble"`` (``"ode"`` data works too: per-cell residuals).
        kind: ``"cle"`` or ``"dga"``.
        restarts: independent starts (batched for the CLE, sequential for
            the DGA).
        epochs: Adam steps.
        n_cells: simulated cells per restart per step.
        lr, lr_final: learning rate, decayed geometrically.
        n_obs_times: fit only this many leading frames (DGA cost control).
        eval_cells: cells used to score restarts at the end.
        seed: random seed.
        max_seconds: optional wall-clock cap on the optimisation.
        dga_kwargs: forwarded to :func:`differentiable_cell.gillespie.run_dga`.
    """
    g = torch.Generator().manual_seed(seed)
    batched = kind == "cle"
    z0 = problem.to_z(problem.sample_prior(restarts, g))
    groups = [z0] if batched else [z0[i:i + 1] for i in range(restarts)]
    finals, histories = [], []

    def loss_of(z: torch.Tensor, cells: int) -> torch.Tensor:
        sim = stochastic.simulate(problem, problem.from_z(z), cells, kind=kind, generator=g,
                                  n_obs_times=n_obs_times, dga_kwargs=dga_kwargs)
        if n_obs_times is not None:
            sub = _window(problem, n_obs_times)
            return stochastic.distance(sub, sim) ** 2
        return stochastic.distance(problem, sim) ** 2

    with Timer(problem) as timer:
        budget = None if max_seconds is None else max_seconds / len(groups)
        for z_init in groups:
            z = z_init.clone().requires_grad_(True)
            opt = torch.optim.Adam([z], lr=lr)
            sched = torch.optim.lr_scheduler.LambdaLR(
                opt, lambda e: (lr_final / lr) ** (min(e, epochs - 1) / max(epochs - 1, 1)))
            hist = []
            t0 = time.perf_counter()
            for _ in range(epochs):
                opt.zero_grad(set_to_none=True)
                per = loss_of(z, n_cells)
                ok = torch.isfinite(per)
                torch.where(ok, per, torch.zeros_like(per)).sum().backward()
                with torch.no_grad():
                    z.grad = torch.nan_to_num(z.grad, nan=0.0, posinf=0.0, neginf=0.0)
                    z.grad[~ok] = 0.0
                opt.step()
                sched.step()
                hist.append(float(per.detach()[ok].min()) if ok.any() else math.nan)
                if budget is not None and time.perf_counter() - t0 > budget:
                    break
            finals.append(z.detach())
            histories.append(hist[::25])
        z_all = torch.cat(finals)
        with torch.no_grad():
            if batched:
                score = loss_of(z_all, eval_cells)
            else:
                score = torch.cat([loss_of(z_all[i:i + 1], min(eval_cells, n_cells))
                                   for i in range(z_all.shape[0])])
    score = torch.nan_to_num(score, nan=math.inf)
    best = int(score.argmin())
    theta_all = problem.from_z(z_all)
    return Result(
        method=f"grad_{kind}",
        theta=theta_all[best],
        samples=None,
        n_sims=timer.n_sims,
        wall=timer.wall,
        info={"restart_distance2": score.tolist(),
              "restart_theta_std": theta_all.std(0).tolist() if restarts > 1 else None,
              "loss_history_every_25": histories},
    )


def _window(problem: Problem, n: int) -> Problem:
    """The same problem restricted to its first ``n`` frames."""
    from dataclasses import replace
    return replace(problem, t=problem.t[:n], y=problem.y[:n],
                   y_var=None if problem.y_var is None else problem.y_var[:n])
