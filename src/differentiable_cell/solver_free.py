"""Solver-free estimators: MAGI and an inverse PINN.

Neither integrates the ODE. Both represent each modelled species as a smooth
function of time and enforce the dynamics as a residual:

* MAGI (Yang, Wong & Kou 2021; Wong 2024 at zero delay): a Matern-5/2
  Gaussian process per species on a grid ``I`` (the frames, refined), tied to
  the ODE by requiring the GP's derivative to match the vector field.
  Hyperparameters of observed species by GP marginal likelihood (noise
  included); hidden species borrow a scale (see :func:`species_scales`). MAP by
  Adam, then NUTS preconditioned by the MAP's Hessian.
* PINN (Casajuana et al. 2026): a 5 x 100 sine network ``t -> species``,
  softplus output, loss = ODE residual + data misfit, Adam.

Modelled species: the six circuit species plus each present reporter's r, D, F.
Unknowns and their boxes are those of :class:`differentiable_cell.fit.Problem`
(kinetic parameters); starting values are read off the fitted curves at t = 0.
Both return a full ``theta`` in the layout of ``Problem.names``.
"""

import math

import torch
from torch import nn

from differentiable_cell import reporters as R
from differentiable_cell.fit import Problem
from differentiable_cell.nuts import nuts


def modelled_species(problem: Problem) -> list[int]:
    out = list(range(6))
    for j in problem.present:
        out += [R.R_IDX[j], R.D_IDX[j], R.F_IDX[j]]
    return out


def species_scales(problem: Problem, species: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and standard deviation for each modelled species (normalisation, GP fallback).

    Observed: from the data. Hidden reporter mRNA / dark FP: that reporter's
    channel (dark FP at ~5%). Hidden circuit species: the mean over observed
    fusions, or over all observed channels if there are none.
    """
    y = problem.y
    obs = {s: c for c, s in enumerate(problem.observed_state)}
    fusion_cols = [c for s, c in obs.items() if s in R.P_IDX] or list(range(y.shape[1]))
    circ_mean, circ_sd = float(y[:, fusion_cols].mean()), float(y[:, fusion_cols].std())
    means, sds = [], []
    for s in species:
        if s in obs:
            means.append(float(y[:, obs[s]].mean())); sds.append(float(y[:, obs[s]].std()))
        elif s >= 6:
            j = next(j for j in problem.present if s in (R.R_IDX[j], R.D_IDX[j], R.F_IDX[j]))
            col = obs.get(R.F_IDX[j])
            m, sd = (float(y[:, col].mean()), float(y[:, col].std())) if col is not None else (circ_mean, circ_sd)
            factor = 0.05 if s == R.D_IDX[j] else 1.0
            means.append(m * factor); sds.append(sd * factor)
        else:
            means.append(circ_mean); sds.append(circ_sd)
    return torch.tensor(means, dtype=torch.float64), torch.tensor(sds, dtype=torch.float64).clamp_min(1e-6)


def kinetic_index(problem: Problem) -> list[int]:
    return [i for i, n in enumerate(problem.names) if not n.endswith("(0)")]


def params_from_z(problem: Problem, z_kin: torch.Tensor) -> dict:
    """Model parameters from the unconstrained kinetic coordinates."""
    kin = kinetic_index(problem)
    theta = problem.lo.clone()
    theta[kin] = problem.lo[kin] + torch.sigmoid(z_kin) * (problem.hi[kin] - problem.lo[kin])
    p, _ = problem.params_and_start(theta)
    return p


def rhs_on(problem: Problem, species: list[int], x: torch.Tensor, p: dict) -> torch.Tensor:
    """Vector field for the modelled species, ``x`` of shape ``(N, S)``."""
    full = torch.zeros(x.shape[0], R.N_STATE, dtype=x.dtype)
    full[:, species] = x
    return R.rhs(full, p, problem.mask)[:, species]


def full_theta(problem: Problem, z_kin: torch.Tensor, x0: torch.Tensor, species: list[int]) -> torch.Tensor:
    """``theta`` in the layout of ``problem.names``: kinetic parameters + starting values."""
    kin = kinetic_index(problem)
    theta = torch.zeros(len(problem.names), dtype=torch.float64)
    theta[kin] = problem.lo[kin] + torch.sigmoid(z_kin) * (problem.hi[kin] - problem.lo[kin])
    starts = {f"{n}(0)": i for i, n in enumerate(["m_lacI", "m_tetR", "m_cI", "p_LacI", "p_TetR", "p_CI"])}
    for j in problem.present:
        r = R.REPORTERS[j]
        starts.update({f"r_{r}(0)": R.R_IDX[j], f"D_{r}(0)": R.D_IDX[j], f"F_{r}(0)": R.F_IDX[j]})
    for i, name in enumerate(problem.names):
        if name in starts:
            theta[i] = torch.log(x0[species.index(starts[name])].clamp_min(1e-12))
    return torch.maximum(torch.minimum(theta, problem.hi), problem.lo)


# ---------------------------------------------------------------------------
# MAGI
# ---------------------------------------------------------------------------


def matern52(t, var, ls, jitter=1e-6):
    """Covariance C, cov(x'(s), x(t)) = dC/ds, and d2C/ds dt on the grid ``t``."""
    r = t.unsqueeze(1) - t.unsqueeze(0)
    a = math.sqrt(5.0) / ls
    ar = a * r.abs()
    e = torch.exp(-ar)
    c = var * (1 + ar + ar**2 / 3) * e
    dc = -var * (a**2 * r / 3) * (1 + ar) * e
    ddc = var * (a**2 / 3) * (1 + ar - ar**2) * e
    eye = torch.eye(t.numel(), dtype=t.dtype)
    return c + jitter * var * eye, dc, ddc + jitter * var * a**2 * eye


def gp_fit(t, y):
    """Variance, length scale and noise sd of one series by marginal likelihood (grid)."""
    yc = y - y.mean()
    sd, span = float(yc.std()), float(t[-1] - t[0])
    best = (-math.inf, sd**2, span / 10, 0.05 * sd)
    for ls in torch.logspace(math.log10(span / 100), math.log10(span / 2), 30).tolist():
        for vf in (0.5, 1.0, 2.0):
            for nf in (0.01, 0.03, 0.1):
                var, noise = vf * sd**2, nf * sd
                c, _, _ = matern52(t, var, ls)
                L = torch.linalg.cholesky(c + noise**2 * torch.eye(t.numel(), dtype=t.dtype))
                alpha = torch.cholesky_solve(yc.unsqueeze(-1), L)
                ll = float(-0.5 * (yc @ alpha.squeeze(-1)) - torch.log(L.diagonal()).sum())
                if ll > best[0]:
                    best = (ll, var, ls, noise)
    return best[1:]


def fit_magi(problem: Problem, refine: int = 2, restarts: int = 3, map_iters: int = 30000,
             lr: float = 0.01, nuts_seconds: float = 600.0, seed: int = 0) -> tuple[torch.Tensor, dict]:
    """MAGI: MAP over (trajectories, parameters), then NUTS. Returns ``theta`` (posterior median)."""
    g = torch.Generator().manual_seed(seed)
    species = modelled_species(problem)
    obs_rows = [species.index(s) for s in problem.observed_state]
    t_obs = problem.t
    t = torch.linspace(float(t_obs[0]), float(t_obs[-1]), (t_obs.numel() - 1) * refine + 1, dtype=torch.float64)
    mean, sd = species_scales(problem, species)

    noise = torch.zeros(len(obs_rows), dtype=torch.float64)
    hyper = [(float(sd[k]) ** 2, float(t[-1] - t[0]) / 10) for k in range(len(species))]
    fitted_ls = []
    for c, row in enumerate(obs_rows):
        var, ls, nz = gp_fit(t_obs, problem.y[:, c])
        hyper[row], noise[c] = (var, ls), nz
        fitted_ls.append(ls)
    median_ls = sorted(fitted_ls)[len(fitted_ls) // 2]
    hyper = [(v, ls if k in obs_rows else median_ls) for k, (v, ls) in enumerate(hyper)]

    Cinv, M, Kinv = [], [], []
    for var, ls in hyper:
        c, dc, ddc = matern52(t, var, ls)
        ci = torch.linalg.inv(c)
        m = dc @ ci
        kd = ddc - m @ dc.T
        Cinv.append(ci); M.append(m); Kinv.append(torch.linalg.inv(0.5 * (kd + kd.T)))
    Cinv, M, Kinv = torch.stack(Cinv), torch.stack(M), torch.stack(Kinv)
    temperature = t.numel() * len(species) / problem.y.numel()
    n_kin = len(kinetic_index(problem))

    def log_post(w):  # w = [x / sd (flattened, species-major), z_kin]
        x = w[: -n_kin].reshape(len(species), t.numel()) * sd.unsqueeze(-1)
        z = w[-n_kin:]
        xc = x - mean.unsqueeze(-1)
        f = rhs_on(problem, species, x.T.clamp_min(0.0), params_from_z(problem, z)).T
        dev = f - torch.einsum("snm,sm->sn", M, xc)
        gp = torch.einsum("sn,snm,sm->", xc, Cinv, xc) + torch.einsum("sn,snm,sm->", dev, Kinv, dev)
        data = (((x[obs_rows][:, ::refine].T - problem.y) / noise) ** 2).sum()
        prior = (nn.functional.logsigmoid(z) + nn.functional.logsigmoid(-z)).sum()
        return -0.5 * gp / temperature - 0.5 * data + prior

    def initial():
        x = mean.unsqueeze(-1).expand(len(species), t.numel()).clone()
        for c, row in enumerate(obs_rows):
            x[row] = torch.from_numpy(__import__("numpy").interp(t.numpy(), t_obs.numpy(), problem.y[:, c].numpy()))
        kin = kinetic_index(problem)
        z = problem.to_z(problem.sample_prior(1, g)[0])[kin]
        return torch.cat([(x / sd.unsqueeze(-1)).reshape(-1), z])

    best_w, best_lp = None, -math.inf
    for _ in range(restarts):
        w = initial().requires_grad_(True)
        opt = torch.optim.Adam([w], lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=map_iters, eta_min=lr / 50)
        for _ in range(map_iters):
            opt.zero_grad()
            loss = -log_post(w)
            if not torch.isfinite(loss):
                break
            loss.backward()
            opt.step()
            sched.step()
        with torch.no_grad():
            lp = float(log_post(w))
        if math.isfinite(lp) and lp > best_lp:
            best_lp, best_w = lp, w.detach().clone()

    info = {"map_logpost": best_lp, "temperature": temperature}
    draws = best_w.unsqueeze(0)
    if nuts_seconds > 0:
        hess = torch.autograd.functional.hessian(lambda w: -log_post(w), best_w)
        evals, evecs = torch.linalg.eigh(0.5 * (hess + hess.T))
        chol = evecs / evals.clamp_min(1e-6 * float(evals.abs().max())).sqrt()

        def logp_grad(v):
            v = v.detach().requires_grad_(True)
            lp = log_post(best_w + chol @ v)
            if not torch.isfinite(lp):
                return -math.inf, torch.zeros_like(v)
            (gr,) = torch.autograd.grad(lp, v)
            return float(lp), gr

        vs, diag = nuts(logp_grad, torch.zeros_like(best_w), n_warmup=150, n_samples=250, max_depth=7,
                        seed=seed, max_seconds=nuts_seconds)
        draws = best_w + vs @ chol.T
        info.update({k: v for k, v in diag.items()})
    thetas = torch.stack([full_theta(problem, w[-n_kin:],
                                     w[: -n_kin].reshape(len(species), t.numel())[:, 0] * sd, species)
                          for w in draws])
    info["draws"] = thetas
    return thetas.median(0).values, info


# ---------------------------------------------------------------------------
# PINN
# ---------------------------------------------------------------------------


class _Sine(nn.Module):
    def forward(self, x):
        return torch.sin(x)


def _network(out: int, width: int = 100, depth: int = 5) -> nn.Sequential:
    layers, fan_in = [], 1
    for _ in range(depth):
        layers += [nn.Linear(fan_in, width), _Sine()]
        fan_in = width
    layers.append(nn.Linear(fan_in, out))
    net = nn.Sequential(*layers).double()
    for m in net:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    return net


def fit_pinn(problem: Problem, restarts: int = 3, iterations: int = 20000, lr: float = 1e-3,
             n_colloc: int = 512, t_scale: float = 10.0, seed: int = 0) -> tuple[torch.Tensor, dict]:
    """Inverse PINN; the restart with the lowest final loss wins. Returns ``theta``."""
    species = modelled_species(problem)
    obs_rows = [species.index(s) for s in problem.observed_state]
    _, sd = species_scales(problem, species)
    scale = sd * 3  # softplus output times a per-species scale
    obs_scale = problem.range
    t_obs = problem.t
    t0, t1 = float(t_obs[0]), float(t_obs[-1])
    best = None
    for r in range(restarts):
        g = torch.Generator().manual_seed(seed * 100 + r)
        torch.manual_seed(seed * 100 + r)
        net = _network(len(species))
        z = problem.to_z(problem.sample_prior(1, g)[0])[kinetic_index(problem)].clone().requires_grad_(True)
        opt = torch.optim.Adam([{"params": net.parameters()}, {"params": [z]}], lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iterations, eta_min=lr / 10)
        states = lambda tt: nn.functional.softplus(net((tt / t_scale).unsqueeze(-1))) * scale
        for _ in range(iterations):
            tc = t0 + (t1 - t0) * torch.rand(n_colloc, generator=g, dtype=torch.float64)
            u, du = torch.func.jvp(states, (tc,), (torch.ones_like(tc),))
            residual = ((du - rhs_on(problem, species, u, params_from_z(problem, z))) / scale).pow(2).mean()
            misfit = ((states(t_obs)[:, obs_rows] - problem.y) / obs_scale).pow(2).mean()
            loss = residual + misfit
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        final = float(residual.detach() + misfit.detach())
        if best is None or final < best[0]:
            with torch.no_grad():
                x0 = states(torch.tensor([t0], dtype=torch.float64))[0]
            best = (final, full_theta(problem, z.detach(), x0, species))
    return best[1], {"final_loss": best[0]}
