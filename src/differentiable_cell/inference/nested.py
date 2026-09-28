"""Nested sampling (Skilling; as used by Pullen & Morris, 2014), via dynesty.

Pullen & Morris ran MultiNest on the repressilator: nested sampling with
multi-ellipsoid bounds, a Gaussian likelihood, and the initial conditions of
the unobserved species as sampled parameters.  This is the same recipe with
dynesty (multi-ellipsoid bounds, random-walk proposals inside them), on the
shared log-uniform prior box.  Besides the posterior it returns the log
evidence, which is what Pullen & Morris use for model comparison.

dynesty evaluates the likelihood one point at a time, so the batched
simulator does not help; instead the likelihood runs in a pool of worker
processes, each with its own single-threaded copy of the problem.
"""

import multiprocessing as mp

import numpy as np
import torch

from differentiable_cell.inference.base import Result, Timer
from differentiable_cell.inference.problem import Problem

_PROBLEM: Problem | None = None


def _init_worker(problem: Problem) -> None:
    global _PROBLEM
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    _PROBLEM = problem


def _loglike(theta: np.ndarray) -> float:
    with torch.no_grad():
        ll = float(_PROBLEM.log_likelihood(torch.as_tensor(theta, dtype=torch.float64)))
    return ll if np.isfinite(ll) else -1e300


class _Transform:
    """Unit cube -> theta, picklable."""

    def __init__(self, lo: np.ndarray, hi: np.ndarray):
        self.lo, self.hi = lo, hi

    def __call__(self, u: np.ndarray) -> np.ndarray:
        return self.lo + u * (self.hi - self.lo)


def fit(
    problem: Problem,
    nlive: int = 300,
    dlogz: float = 0.5,
    max_calls: int = 2_000_000,
    workers: int = 8,
    sample: str = "rwalk",
    seed: int = 0,
) -> Result:
    """Run static nested sampling to ``dlogz`` or ``max_calls``.

    Args:
        problem: the inverse problem.
        nlive: live points.
        dlogz: stop when the remaining evidence estimate is below this.
        max_calls: likelihood-call budget.
        workers: likelihood processes.
        sample: dynesty proposal inside the bounds.
        seed: random seed.
    """
    import dynesty

    lo, hi = (b.numpy() for b in problem.bounds.unbind(-1))
    transform = _Transform(lo, hi)
    with Timer(problem) as timer:
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers, initializer=_init_worker, initargs=(problem,)) as pool:
            sampler = dynesty.NestedSampler(
                _loglike, transform, problem.dim, nlive=nlive, bound="multi",
                sample=sample, pool=pool, queue_size=workers,
                rstate=np.random.default_rng(seed),
            )
            sampler.run_nested(dlogz=dlogz, maxcall=max_calls, print_progress=False)
        res = sampler.results
    samples = torch.as_tensor(res.samples_equal(rstate=np.random.default_rng(seed)))
    n_calls = int(np.sum(res.ncall))
    return Result(
        method="nested",
        theta=samples.median(0).values,
        samples=samples,
        n_sims=n_calls,  # evaluated in the workers, so not on problem.n_sims
        wall=timer.wall,
        info={"logz": float(res.logz[-1]), "logzerr": float(res.logzerr[-1]),
              "n_calls": n_calls, "iterations": int(res.niter),
              "information_nats": float(res.information[-1]),
              "hit_call_budget": n_calls >= max_calls},
    )
