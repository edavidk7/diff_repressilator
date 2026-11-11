"""Baselines that never integrate the ODE: MAGI (Yang, Wong & Kou 2021) and an inverse PINN (Casajuana et al. 2026).

Both represent every species as a smooth curve and enforce the ODE as a residual on it. They return ``theta`` in the
layout of ``Problem.names``, with the start read off the fitted curves at t = 0. Equation numbers in the MAGI and
PINN sections refer to those papers. Where we depart from them:

MAGI  Matern nu = 5/2 (paper: 2.01); hyperparameters by a grid over the marginal likelihood (paper: optimiser with a
      Fourier prior on the length scale); hidden species get the median fitted length scale and a data-derived
      variance (paper: optimised jointly with theta); noise sd fixed at its GP fit (paper: sampled); NUTS from the
      MAP with the Hessian as metric (paper: HMC); posterior median (paper: mean); curves floored at 1e-9 in f.
PINN  no separate initial-condition term (the start is unknown here; the data term covers frame 0); residuals
      divided by a per-species scale; 20000 Adam steps (paper: 3000-5000), cosine lr, best of 3 restarts.
"""

import math

import torch
from torch import nn

from differentiable_cell import reporters as R
from differentiable_cell.fit import Problem
from differentiable_cell.nuts import nuts


def modelled_species(problem) -> list[int]:
    if not isinstance(problem, Problem):  # a circuit: every species
        return list(range(len(problem.c.species)))
    out = list(range(6))
    for j in problem.present:
        out += [R.R_IDX[j], R.D_IDX[j], R.F_IDX[j]]
    return out


def species_scales(problem: Problem, species: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and sd per modelled species: observed ones from their channel, hidden ones from related channels
    (reporter r, D from the reporter's F, D at ~5%; circuit species from the fusions, else all channels)."""
    y = problem.y
    if not isinstance(problem, Problem):  # a circuit: hidden species get the scale of all observed channels
        obs = {s: c for c, s in enumerate(problem.observed_state)}
        cols = lambda s: [obs[s]] if s in obs else list(range(y.shape[1]))
        means = [float(y[:, cols(s)].mean()) for s in species]
        sds = [float(y[:, cols(s)].std()) for s in species]
        return torch.tensor(means, dtype=torch.float64), torch.tensor(sds, dtype=torch.float64).clamp_min(1e-6)
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
    if not isinstance(problem, Problem):
        return {k: theta[i].exp() for i, k in enumerate(problem.c.params)}
    p, _ = problem.params_and_start(theta)
    return p


def rhs_on(problem, species: list[int], t: torch.Tensor, x: torch.Tensor, p: dict) -> torch.Tensor:
    """Vector field for the modelled species at times ``t (N,)``, ``x`` of shape ``(N, S)``."""
    if not isinstance(problem, Problem):
        return problem.c.rhs(t, x, p)
    full = torch.zeros(x.shape[0], R.N_STATE, dtype=x.dtype)
    full[:, species] = x
    return R.rhs(full, p, problem.mask)[:, species]


def full_theta(problem: Problem, z_kin: torch.Tensor, x0: torch.Tensor, species: list[int]) -> torch.Tensor:
    """``theta`` in the layout of ``problem.names``: kinetic parameters + starting values."""
    kin = kinetic_index(problem)
    theta = torch.zeros(len(problem.names), dtype=torch.float64)
    theta[kin] = problem.lo[kin] + torch.sigmoid(z_kin) * (problem.hi[kin] - problem.lo[kin])
    if not isinstance(problem, Problem):
        theta[len(kin):] = torch.log(x0.clamp_min(1e-12))
        return torch.maximum(torch.minimum(theta, problem.hi), problem.lo)
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
    """K(I, I), 'K(I, I) = dK/ds and K''(I, I) = d2K/ds dt of MAGI Eq. (6), on the grid ``t``."""
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
    """phi_1 (variance), phi_2 (length scale) and noise sd of one observed series: MAGI Eq. (7), on a grid."""
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
    """MAP of MAGI's tempered posterior over (curves, parameters), then NUTS; returns the posterior median."""
    g = torch.Generator().manual_seed(seed)
    species = modelled_species(problem)
    obs_rows = [species.index(s) for s in problem.observed_state]
    t_obs = problem.t
    # discretisation set I: the frames refined ``refine`` times
    t = torch.linspace(float(t_obs[0]), float(t_obs[-1]), (t_obs.numel() - 1) * refine + 1, dtype=torch.float64)
    mean, sd = species_scales(problem, species)  # mu_d (constant) and the scale of each curve

    noise = torch.zeros(len(obs_rows), dtype=torch.float64)
    hyper = [(float(sd[k]) ** 2, float(t[-1] - t[0]) / 10) for k in range(len(species))]
    fitted_ls = []
    for c, row in enumerate(obs_rows):
        var, ls, nz = gp_fit(t_obs, problem.y[:, c])
        hyper[row], noise[c] = (var, ls), nz
        fitted_ls.append(ls)
    median_ls = sorted(fitted_ls)[len(fitted_ls) // 2]
    hyper = [(v, ls if k in obs_rows else median_ls) for k, (v, ls) in enumerate(hyper)]

    Cinv, M, Kinv = [], [], []  # Eq. (6): C_d^-1, m_d = 'K C^-1, K_d^-1 with K_d = K'' - 'K C^-1 K'
    for var, ls in hyper:
        c, dc, ddc = matern52(t, var, ls)
        ci = torch.linalg.inv(c)
        m = dc @ ci
        kd = ddc - m @ dc.T
        Cinv.append(ci); M.append(m); Kinv.append(torch.linalg.inv(0.5 * (kd + kd.T)))
    Cinv, M, Kinv = torch.stack(Cinv), torch.stack(M), torch.stack(Kinv)
    temperature = t.numel() * len(species) / problem.y.numel()  # beta = D |I| / N ("Prior tempering")
    n_kin = len(kinetic_index(problem))

    def log_post(w):  # tempered Eq. (5); w = [x / sd (flattened, species-major), z_kin]
        x = w[: -n_kin].reshape(len(species), t.numel()) * sd.unsqueeze(-1)
        z = w[-n_kin:]
        xc = x - mean.unsqueeze(-1)
        f = rhs_on(problem, species, t, x.T.clamp_min(1e-9), params_from_z(problem, z)).T  # > 0: d(0^n)/dn is NaN
        dev = f - torch.einsum("snm,sm->sn", M, xc)  # f - m_d (x - mu); mu' = 0
        gp = torch.einsum("sn,snm,sm->", xc, Cinv, xc) + torch.einsum("sn,snm,sm->", dev, Kinv, dev)  # terms (1), (3)
        data = (((x[obs_rows][:, ::refine].T - problem.y) / noise) ** 2).sum()  # term (2)
        prior = (nn.functional.logsigmoid(z) + nn.functional.logsigmoid(-z)).sum()  # uniform on the log box
        return -0.5 * gp / temperature - 0.5 * data + prior

    def initial():  # observed curves: linear interpolation of the frames (as in MAGI); hidden: their mean
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
             n_colloc: int = 512, t_scale: float = 10.0, seed: int = 0, causal: bool = False,
             bins: int = 32) -> tuple[torch.Tensor, dict]:
    """Inverse PINN (5 x 100 sine units, softplus output, Adam); the restart with the lowest final loss wins."""
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
        for it in range(iterations):
            tc = t0 + (t1 - t0) * torch.rand(n_colloc, generator=g, dtype=torch.float64)
            u, du = torch.func.jvp(states, (tc,), (torch.ones_like(tc),))
            # PINN Eq. (4) with lambda_f = lambda_y = 1: ODE residual r of Eq. (3) at collocation points + data misfit
            residual = ((du - rhs_on(problem, species, tc, u, params_from_z(problem, z))) / scale).pow(2)
            if causal:  # Wang et al. 2022, Eq. (12)-(14): a time bin counts only once the earlier ones are solved
                which = ((tc - t0) / (t1 - t0) * bins).long().clamp_max(bins - 1)
                count = torch.zeros(bins, dtype=torch.float64).index_add_(0, which, torch.ones_like(tc))
                per_bin = torch.zeros(bins, dtype=torch.float64).index_add_(0, which, residual.mean(-1)) / count.clamp_min(1)
                eps = 10.0 ** (-2 + min(4 * it // iterations, 4))  # causality parameter annealed 1e-2 ... 1e2
                weight = torch.exp(-eps * (per_bin.cumsum(0) - per_bin)).detach()
                residual = (weight * per_bin).sum() / (count > 0).sum()
            else:
                residual = residual.mean()
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
