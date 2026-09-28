"""Backpropagation through the ODE: the "free" baseline.

Maximum likelihood by shooting, as in ``experiments/run_partial.py``, with
its load-bearing choices kept:

* log-space parameters (here inside the prior box via ``theta = box(sigmoid(z))``);
* a growing horizon - the loss of a period-setting parameter saturates over
  long windows, so the fit sees the first quarter of the record, then half,
  then all of it.  The schedule is in fractions of the record, not periods:
  the period is unknown to the estimator;
* per-species normalisation - the Gaussian likelihood with ``sigma``
  proportional to each species' range does exactly this;
* independent restarts from the prior, here run as one batch;
* a step that makes a restart unintegrable is rejected for that restart only.

On top, :func:`fit` returns a Laplace approximation (Gauss-Newton Hessian at
the best restart) as its uncertainty.
"""

import math
import time

import torch

from differentiable_cell.inference.base import (
    Result, Timer, gauss_newton_covariance, laplace_samples,
)
from differentiable_cell.inference.problem import Problem


def optimise(
    problem: Problem,
    z0: torch.Tensor,
    epochs: int = 3000,
    lr: float = 0.05,
    lr_final: float = 2.5e-4,
    stages: tuple[float, ...] = (0.25, 0.5, 1.0),
    max_seconds: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
    """Adam on a batch of unconstrained starts ``z0`` ``(R, dim)``.

    Returns the final ``z``, the full-record log-likelihood of each restart,
    and the per-epoch best loss (for diagnostics).
    """
    g_total = problem.t.numel()
    windows = [max(4, math.ceil(f * g_total)) for f in stages]
    per_stage = math.ceil(epochs / len(windows))
    z = z0.clone().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda e: (lr_final / lr) ** (min(e, epochs - 1) / max(epochs - 1, 1))
    )
    history: list[float] = []
    t0 = time.perf_counter()
    for epoch in range(epochs):
        window = windows[min(epoch // per_stage, len(windows) - 1)]
        opt.zero_grad(set_to_none=True)
        theta = problem.from_z(z)
        pred = problem.simulate(theta, n_obs_times=window)
        r2 = ((pred - problem.y[:window]) / problem.sigma) ** 2
        per_restart = 0.5 * r2.mean((-1, -2))  # (R,)
        ok = torch.isfinite(per_restart)
        loss = torch.where(ok, per_restart, torch.zeros_like(per_restart)).sum()
        loss.backward()
        with torch.no_grad():
            z.grad[~ok] = 0.0
            z.grad = torch.nan_to_num(z.grad, nan=0.0, posinf=0.0, neginf=0.0)
            before = z.detach().clone()
        opt.step()
        sched.step()
        with torch.no_grad():
            # A restart that just failed is returned to where it last worked,
            # and its momentum cleared so it does not walk straight back.
            z[~ok] = before[~ok]
            state = opt.state[z]
            if "exp_avg" in state:
                state["exp_avg"][~ok] = 0.0
        history.append(float(per_restart.detach()[ok].min()) if ok.any() else float("nan"))
        if max_seconds is not None and time.perf_counter() - t0 > max_seconds:
            break
    with torch.no_grad():
        ll = problem.log_likelihood(problem.from_z(z))
    return z.detach(), ll, history


def polish(problem: Problem, z: torch.Tensor, iterations: int = 200) -> torch.Tensor:
    """L-BFGS (strong Wolfe) on the full-record likelihood from one ``z``.

    Adam with a decaying rate gets close but stalls short of the optimum -
    measured on the fully observed Box 1 case, 3 nats below the truth's own
    likelihood.  A quasi-Newton finish closes that gap cheaply.
    """
    z = z.detach().clone().requires_grad_(True)
    opt = torch.optim.LBFGS([z], lr=1.0, max_iter=iterations, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-9, tolerance_change=1e-12, history_size=20)

    def closure():
        opt.zero_grad()
        loss = -problem.log_likelihood(problem.from_z(z))
        if not torch.isfinite(loss):
            return torch.tensor(1e30, dtype=loss.dtype, requires_grad=True)
        loss.backward()
        return loss

    start = z.detach().clone()
    try:
        opt.step(closure)
    except RuntimeError:
        return start
    return z.detach() if torch.isfinite(z).all() else start


def fit(
    problem: Problem,
    restarts: int = 32,
    epochs: int = 3000,
    lr: float = 0.05,
    laplace: bool = True,
    seed: int = 0,
    max_seconds: float | None = None,
    polish_top: int = 3,
) -> Result:
    """Multi-restart maximum likelihood through the ODE, plus Laplace.

    Args:
        problem: the inverse problem.
        restarts: independent starts drawn from the prior (batched, so 32
            costs ~1.7x of 8).  8 missed the basin on a `rand0` case that
            32 found; the likelihood is multimodal.
        epochs: Adam steps.
        lr: initial learning rate in ``z`` space (decays 200x; 1500 epochs with
            a 20x decay stopped ~3 nats short of the optimum on the Box 1 case).
        laplace: add a Gauss-Newton Laplace approximation as ``samples``.
        seed: seeds the starts and the Laplace draws.
        max_seconds: optional wall-clock cap on the optimisation.
        polish_top: restarts finished with L-BFGS (:func:`polish`).
    """
    g = torch.Generator().manual_seed(seed)
    with Timer(problem) as timer:
        z0 = problem.to_z(problem.sample_prior(restarts, g))
        z, ll, history = optimise(problem, z0, epochs=epochs, lr=lr, max_seconds=max_seconds)
        # Polish the few most promising restarts, then pick the best.
        order = torch.nan_to_num(ll, nan=-torch.inf).argsort(descending=True)
        for i in order[:polish_top].tolist():
            z[i] = polish(problem, z[i])
        with torch.no_grad():
            ll = problem.log_likelihood(problem.from_z(z))
        best = int(torch.nan_to_num(ll, nan=-torch.inf).argmax())
        theta = problem.from_z(z[best])
        samples, cov = None, None
        if laplace:
            cov = gauss_newton_covariance(problem, theta)
            samples = laplace_samples(problem, theta, cov, generator=g)
    all_theta = problem.from_z(z)
    finite = torch.isfinite(ll)
    return Result(
        method="grad_ode+laplace" if laplace else "grad_ode",
        theta=theta,
        samples=samples,
        n_sims=timer.n_sims,
        wall=timer.wall,
        info={
            "restart_loglik": ll.tolist(),
            "best_loglik": float(ll[best]),
            "restart_theta_std": all_theta[finite].std(0).tolist() if finite.sum() > 1 else None,
            "loss_history_every_50": history[::50],
            "laplace_sd": cov.diagonal().sqrt().tolist() if cov is not None else None,
        },
    )
