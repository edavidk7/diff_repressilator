"""MAGI: manifold-constrained Gaussian process inference (Yang, Wong & Kou,
2021; Wong 2024's MAGI-DDE is its delay extension and reduces to it at zero
delay, which is the repressilator's case).

No numerical solver.  Each species ``d`` gets a Gaussian-process prior at a
set of discretisation times ``I``; the ODE enters by requiring that the
GP's derivative match the vector field.  With ``C_d`` the GP covariance and
``m_d, K_d`` the conditional mean and covariance of the derivative given
``x_d``, the (tempered) log posterior over trajectories and parameters is::

    -1/2 sum_d [ (x_d - mu_d)^T C_d^-1 (x_d - mu_d)
               + (f_d(x, theta) - m_d)^T K_d^-1 (f_d(x, theta) - m_d) ] / T
    -1/2 sum_(observed d, t) (x_d(t) - y_d(t))^2 / sigma_d^2  + log prior(theta)

``T`` is the prior temperature, (discretisation points x species) /
observations as in the MAGI software.  Hyperparameters, as in MAGI:
observed species by GP marginal likelihood on their data; unobserved ones
cannot be fitted that way, so they take the median observed length scale
and the mean observed variance and level.

Inference: a MAP over (trajectories, theta) by Adam from several starts,
then NUTS (:func:`.nuts.nuts`) with the MAP's Laplace covariance as metric.  Initial values are
read off the trajectories at ``t = 0``.
"""

import math

import torch

from differentiable_cell.inference.base import Result, Timer, clip_to_box
from differentiable_cell.inference.nuts import nuts
from differentiable_cell.inference.problem import Problem
from differentiable_cell.run import FIELDS

_M = [0, 1, 2, 6]
_P = [3, 4, 5, 7]
_REP = [5, 3, 4, 4]


def _rhs(x: torch.Tensor, v: dict[str, torch.Tensor]) -> torch.Tensor:
    """Vector field on ``(8, N)`` trajectories -> ``(8, N)``."""
    m, p, rep = x[_M], x[_P], x[_REP].clamp_min(0.0)
    alpha = torch.stack([v["alpha"]] * 3 + [v["alpha_GFP"]]).unsqueeze(-1)
    beta = torch.stack([v["beta"]] * 3 + [v["beta_GFP"]]).unsqueeze(-1)
    dm = -m + alpha / (1 + rep ** v["n"]) + v["alpha_0"]
    dp = beta * (m - p)
    out = torch.empty_like(x)
    out[_M] = dm
    out[_P] = dp
    return out


def matern52(t: torch.Tensor, var: float, ls: float, jitter: float = 1e-6):
    """``C``, ``dC/ds`` (cov of x'(s) with x(t)) and ``d2C/ds dt`` on grid ``t``."""
    r = t.unsqueeze(1) - t.unsqueeze(0)
    a = math.sqrt(5.0) / ls
    ar = a * r.abs()
    e = torch.exp(-ar)
    c = var * (1 + ar + ar**2 / 3) * e
    dc = -var * (a**2 * r / 3) * (1 + ar) * e
    ddc = var * (a**2 / 3) * (1 + ar - ar**2) * e
    eye = torch.eye(t.numel(), dtype=t.dtype)
    return c + jitter * var * eye, dc, ddc + jitter * var * a**2 * eye


def _gp_hyper(t: torch.Tensor, y: torch.Tensor, sigma: float) -> tuple[float, float, float]:
    """Mean, variance and length scale of one observed series (marginal likelihood)."""
    mu = float(y.mean())
    yc = y - mu
    span = float(t[-1] - t[0])
    best = (-math.inf, float(yc.var()), span / 10)
    for ls in torch.logspace(math.log10(span / 100), math.log10(span / 2), 40).tolist():
        for vf in (0.25, 0.5, 1.0, 2.0, 4.0):
            var = vf * float(yc.var()) + 1e-12
            c, _, _ = matern52(t, var, ls)
            k = c + sigma**2 * torch.eye(t.numel(), dtype=t.dtype)
            chol = torch.linalg.cholesky(k)
            alpha = torch.cholesky_solve(yc.unsqueeze(-1), chol)
            ll = float(-0.5 * (yc @ alpha.squeeze(-1)) - torch.log(chol.diagonal()).sum())
            if ll > best[0]:
                best = (ll, var, ls)
    return mu, best[1], best[2]


class MagiPosterior:
    """The MAGI log posterior as a function of a flat vector ``[x/scale, z]``."""

    def __init__(self, problem: Problem, refine: int = 2, temperature: float | None = None):
        self.problem = problem
        t_obs = problem.t
        n_obs = t_obs.numel()
        self.t = torch.linspace(float(t_obs[0]), float(t_obs[-1]), (n_obs - 1) * refine + 1,
                                dtype=t_obs.dtype)
        self.obs_rows = torch.arange(0, self.t.numel(), refine)
        self.k = len(problem.free_params)
        hyper = {}
        for j, name in enumerate(problem.observed):
            hyper[name] = _gp_hyper(t_obs, problem.y[:, j], float(problem.sigma[j]))
        if hyper:
            mus, vars_, lss = zip(*hyper.values())
            fallback = (sum(mus) / len(mus), sum(vars_) / len(vars_),
                        sorted(lss)[len(lss) // 2])
        else:
            fallback = (1.0, 1.0, 10.0)
        self.hyper = {f: hyper.get(f, fallback) for f in FIELDS}
        self.mu = torch.tensor([self.hyper[f][0] for f in FIELDS], dtype=t_obs.dtype)
        self.scale = torch.tensor([math.sqrt(self.hyper[f][1]) for f in FIELDS], dtype=t_obs.dtype)
        cinv, mproj, kdinv = [], [], []
        for f in FIELDS:
            _, var, ls = self.hyper[f]
            c, dc, ddc = matern52(self.t, var, ls)
            ci = torch.linalg.inv(c)
            m = dc @ ci
            kd = ddc - m @ dc.T
            cinv.append(ci)
            mproj.append(m)
            kdinv.append(torch.linalg.inv(0.5 * (kd + kd.T)))
        # Trajectories are optimised as x / scale.  (Whitening them by the GP
        # Cholesky factor was tried and made Adam slower: the ill-conditioning
        # sits in the manifold term K_d^-1, which whitening amplifies.)
        self.cinv, self.mproj, self.kdinv = torch.stack(cinv), torch.stack(mproj), torch.stack(kdinv)
        n_data = problem.y.numel()
        self.temperature = temperature or (self.t.numel() * len(FIELDS)) / n_data
        self.lo, self.hi = problem.bounds[: self.k].unbind(-1)

    @property
    def dim(self) -> int:
        return len(FIELDS) * self.t.numel() + self.k

    def unpack(self, w: torch.Tensor):
        """``w = [x / scale (flattened), z]`` -> ``(x, z)``."""
        n = self.t.numel()
        x = w[: len(FIELDS) * n].reshape(len(FIELDS), n) * self.scale.unsqueeze(-1)
        return x, w[len(FIELDS) * n:]

    def params(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        vals = (self.lo + torch.sigmoid(z) * (self.hi - self.lo)).exp()
        v = {name: vals[i] for i, name in enumerate(self.problem.free_params)}
        for name, val in self.problem.fixed_params.items():
            v[name] = torch.tensor(val, dtype=torch.float64)
        return v

    def log_prob(self, w: torch.Tensor) -> torch.Tensor:
        x, z = self.unpack(w)
        xc = x - self.mu.unsqueeze(-1)
        prior = torch.einsum("dn,dnm,dm->", xc, self.cinv, xc)
        f = _rhs(x, self.params(z))
        dev = f - torch.einsum("dnm,dm->dn", self.mproj, xc)
        manifold = torch.einsum("dn,dnm,dm->", dev, self.kdinv, dev)
        x_obs = x[self.problem.obs_idx][:, self.obs_rows].T  # (G, k)
        data = (((x_obs - self.problem.y) / self.problem.sigma) ** 2).sum()
        log_prior_z = (torch.nn.functional.logsigmoid(z) + torch.nn.functional.logsigmoid(-z)).sum()
        return -0.5 * (prior + manifold) / self.temperature - 0.5 * data + log_prior_z

    def initial(self, generator: torch.Generator) -> torch.Tensor:
        """Observed species: data interpolated onto ``I``; hidden: their GP mean."""
        n = self.t.numel()
        x = self.mu.unsqueeze(-1).expand(len(FIELDS), n).clone()
        for j, idx in enumerate(self.problem.obs_idx):
            x[idx] = torch.from_numpy(
                __import__("numpy").interp(self.t.numpy(), self.problem.t.numpy(),
                                           self.problem.y[:, j].numpy()))
        z = self.problem.to_z(self.problem.sample_prior(1, generator)[0])[: self.k]
        return torch.cat([(x / self.scale.unsqueeze(-1)).reshape(-1), z])

    def theta(self, w: torch.Tensor) -> torch.Tensor:
        x, z = self.unpack(w)
        ic_idx = [FIELDS.index(f) for f in self.problem.ic_fields]
        theta = torch.cat([self.lo + torch.sigmoid(z) * (self.hi - self.lo),
                           x[ic_idx, 0].clamp_min(1e-12).log()])
        return clip_to_box(self.problem, theta)


def fit(
    problem: Problem,
    refine: int = 2,
    restarts: int = 3,
    map_iters: int = 40000,
    lr: float = 0.01,
    n_warmup: int = 300,
    n_samples: int = 500,
    max_depth: int = 8,
    seed: int = 0,
    max_seconds: float | None = None,
) -> Result:
    """MAP then NUTS on the MAGI posterior.

    Args:
        problem: the inverse problem.
        refine: discretisation points per observation interval.
        restarts: MAP starts (parameters from the prior; trajectories from
            the data / GP mean).
        map_iters: Adam steps per start.  The posterior is badly conditioned:
            on the Box 1 case 3000 steps end ~10^4 nats short, 40000 reach
            the truth's log posterior (~1 ms per step).
        lr: Adam learning rate.
        n_warmup, n_samples, max_depth: NUTS settings (``n_samples = 0``
            skips sampling and returns the MAP only).
        seed: random seed.
        max_seconds: optional cap on the NUTS stage.
    """
    g = torch.Generator().manual_seed(seed)
    with Timer(problem) as timer:
        post = MagiPosterior(problem, refine=refine)
        best_w, best_lp, lps = None, -math.inf, []
        for _ in range(restarts):
            w = post.initial(g).requires_grad_(True)
            opt = torch.optim.Adam([w], lr=lr)
            sched = torch.optim.lr_scheduler.LambdaLR(
                opt, lambda e: 0.05 ** (min(e, map_iters) / map_iters))
            for _ in range(map_iters):
                opt.zero_grad()
                loss = -post.log_prob(w)
                if not torch.isfinite(loss):
                    break
                loss.backward()
                opt.step()
                sched.step()
            with torch.no_grad():
                lp = float(post.log_prob(w))
            lps.append(lp)
            if math.isfinite(lp) and lp > best_lp:
                best_lp, best_w = lp, w.detach().clone()
        theta_map = post.theta(best_w)
        samples, diag = None, {}
        if n_samples > 0:
            # Precondition with the Laplace covariance at the MAP (inverse of
            # the full Hessian, eigenvalues floored), exactly as NUTS through
            # the ODE is preconditioned.  With an identity metric the sampler
            # adapted to a 3e-5 step and saturated trees: it did not move.
            hess = torch.autograd.functional.hessian(lambda w: -post.log_prob(w), best_w)
            evals, evecs = torch.linalg.eigh(0.5 * (hess + hess.T))
            floor = 1e-6 * float(evals.abs().max())
            chol = evecs / evals.clamp_min(floor).sqrt()  # chol @ chol.T = H^-1

            def logp_grad(v: torch.Tensor):
                v = v.detach().requires_grad_(True)
                lp = post.log_prob(best_w + chol @ v)
                if not torch.isfinite(lp):
                    return -math.inf, torch.zeros_like(v)
                (gr,) = torch.autograd.grad(lp, v)
                return float(lp), gr
            vs, diag = nuts(logp_grad, torch.zeros_like(best_w), n_warmup=n_warmup,
                            n_samples=n_samples, max_depth=max_depth, seed=seed,
                            max_seconds=max_seconds)
            diag["negative_hessian_eigs"] = int((evals < 0).sum())
            samples = torch.stack([post.theta(best_w + chol @ v) for v in vs])
    return Result(
        method="magi",
        theta=samples.median(0).values if samples is not None else theta_map,
        samples=samples,
        n_sims=0,
        wall=timer.wall,
        info={"map_logpost": lps, "temperature": post.temperature,
              "theta_map": theta_map.tolist(), **diag,
              "hyper": {f: list(v) for f, v in post.hyper.items()}},
    )
