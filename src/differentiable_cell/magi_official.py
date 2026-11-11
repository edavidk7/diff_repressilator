"""MAGI baseline run with the authors' own software (Yang, Wong & Kou 2021; github.com/wongswk/magi, pymagi).

The package (modules ``pymagi`` (C++), ``arma``, ``magi``) is built from source; it is not on PyPI. We supply the
vector field on the modelled species with analytic derivatives and leave the rest to the package: Matern 2.01
kernel, its own hyperparameter fit (hidden species included), sampled noise, HMC, posterior mean.
"""

import time

import numpy as np
import torch

from differentiable_cell import reporters as R
from differentiable_cell.fit import Problem
from differentiable_cell.solver_free import full_theta, kinetic_index, modelled_species


def ode_functions(problem: Problem, scale: float = 1.0):
    """``fOde, fOdeDx, fOdeDtheta`` in pymagi's layout for the modelled species (m x3, p x3, then r, D, F per built
    reporter), in units of ``scale``. The package's theta is u in [0, 2], log parameter = lo + u (hi - lo) / 2 in
    ``problem.names`` order: the package starts its optimiser at theta = 1, which must lie inside the bounds."""
    kin = kinetic_index(problem)
    lo, half = problem.lo[kin].numpy(), (problem.hi[kin] - problem.lo[kin]).numpy() / 2
    present = list(problem.present)
    K = 1.0 + problem.tau_t.numpy() * np.asarray(problem.mask, dtype=float)  # titration, as in reporters.repression
    k, delta = R.K_MAT.numpy()[present], R.DELTA.numpy()[present]
    rep = np.array(R.GENE_REPRESSOR)
    n_x, n_theta = 6 + 3 * len(present), 4 + 2 * len(present)
    gene = np.arange(3)

    def hill(theta, x):
        q = np.exp(lo + theta * half)
        q[[1, *range(5, n_theta, 2)]] /= scale  # the additive leaks alpha_0, a_0 in the scaled units
        prot = np.maximum(x[:, 3:6] * scale, 1e-12)
        h = 1.0 / (1.0 + (prot / K) ** q[2])
        return q, prot, h

    def f(theta, x, tvec):
        q, _, h = hill(theta, x)
        out = np.empty((x.shape[0], n_x))
        out[:, 0:3] = -x[:, 0:3] + q[0] / scale * h[:, rep] + q[1]
        out[:, 3:6] = -q[3] * (x[:, 3:6] - x[:, 0:3])
        for i, j in enumerate(present):
            r, D, F = 6 + 3 * i, 7 + 3 * i, 8 + 3 * i
            out[:, r] = -x[:, r] + q[4 + 2 * i] / scale * h[:, j] + q[5 + 2 * i]
            out[:, D] = delta[i] * x[:, r] - (k[i] + delta[i]) * x[:, D]
            out[:, F] = k[i] * x[:, D] - delta[i] * x[:, F]
        return out

    def dx(theta, x, tvec):  # [time, with respect to, component]
        q, prot, h = hill(theta, x)
        dh = -q[2] * h * (1.0 - h) / prot  # dh_j / dp_j, unscaled: the Jacobian does not depend on the units
        out = np.zeros((x.shape[0], n_x, n_x))
        out[:, gene, gene] = -1.0
        out[:, 3 + rep, gene] = q[0] * dh[:, rep]
        out[:, 3 + gene, 3 + gene] = -q[3]
        out[:, gene, 3 + gene] = q[3]
        for i, j in enumerate(present):
            r, D, F = 6 + 3 * i, 7 + 3 * i, 8 + 3 * i
            out[:, r, r] = -1.0
            out[:, 3 + j, r] = q[4 + 2 * i] * dh[:, j]
            out[:, r, D] = delta[i]
            out[:, D, D] = -(k[i] + delta[i])
            out[:, D, F] = k[i]
            out[:, F, F] = -delta[i]
        return out

    def dtheta(theta, x, tvec):  # [time, parameter, component]
        q, prot, h = hill(theta, x)
        dn = -h * (1.0 - h) * np.log(prot / K)  # dh_j / dn
        out = np.zeros((x.shape[0], n_theta, n_x))
        out[:, 0, 0:3] = h[:, rep] / scale
        out[:, 1, 0:3] = 1.0
        out[:, 2, 0:3] = q[0] / scale * dn[:, rep]
        out[:, 3, 3:6] = -(x[:, 3:6] - x[:, 0:3])
        for i, j in enumerate(present):
            r = 6 + 3 * i
            out[:, 2, r] = q[4 + 2 * i] / scale * dn[:, j]
            out[:, 4 + 2 * i, r] = h[:, j] / scale
            out[:, 5 + 2 * i, r] = 1.0
        return out * (q * half)[None, :, None]

    return f, dx, dtheta


def fit_magi_official(problem: Problem, seed: int = 0, level: int = 0, niter_hmc: int = 20000,
                      nsteps_hmc: int = 200, burnin_ratio: float = 0.5, positive_system: bool = False,
                      verbose: bool = False, control: dict | None = None) -> tuple[torch.Tensor, dict]:
    """The official ``MagiSolver`` on a ``Problem``; returns the posterior mean as ``theta`` and plain-number info.

    ``level``: each level inserts a point between neighbouring frames (the package's ``setDiscretization``).
    ``niter_hmc``, ``nsteps_hmc``, ``burnin_ratio``: the package's HMC settings, at its defaults.
    ``positive_system``: the package's switch bounding the curves at 0 in HMC (its default: off). Its initialisation
    ignores the bound, and a start with a negative hidden curve is then never left (acceptance 0).
    The prior on the parameters is the package's flat one, here on the problem's log box.
    ``control``: further entries for the package's control list (curves in units of ``info["scale"]``), e.g. to
    resume from a saved initialisation: phi, sigma, xInit, thetaInit, skipMissingComponentOptimization.
    ``seed`` is unused: the Python binding exposes no seed and the chain is deterministic.
    """
    from arma import ode_system, setDiscretization  # the official package's Python front end
    from magi import MagiSolver

    species = modelled_species(problem)
    n_kin = len(kinetic_index(problem))
    # the package assumes curves of order 1 (bounds on hidden-species hyperparameters, jitter): one common unit
    scale = float(problem.y.abs().max())
    system = ode_system("repressilator-reporters", *ode_functions(problem, scale),
                        thetaLowerBound=np.zeros(n_kin), thetaUpperBound=np.full(n_kin, 2.0))

    data = np.full((problem.t.numel(), 1 + len(species)), np.nan)  # hidden species: NaN columns
    data[:, 0] = problem.t.numpy()
    for c, s in enumerate(problem.observed_state):
        data[:, 1 + species.index(s)] = problem.y[:, c].numpy() / scale
    data = setDiscretization(data, level=level)

    control = dict(niterHmc=niter_hmc, nstepsHmc=nsteps_hmc, burninRatio=burnin_ratio,
                   positiveSystem=positive_system, verbose=verbose, **(control or {}))  # the rest at the package's defaults
    started = time.time()
    result = MagiSolver(y=data[:, 1:].copy(), odeModel=system, tvec=data[:, 0].copy(), control=control)
    seconds = time.time() - started

    u = torch.from_numpy(result["theta"].mean(-1))  # the package's point estimate: the posterior mean
    curves = result["xsampled"].mean(-1) * scale  # (species, time)
    theta = full_theta(problem, torch.logit((u / 2).clamp(1e-9, 1 - 1e-9)),
                       torch.from_numpy(curves[:, 0].copy()), species)

    info = {"seconds": seconds, "niter_hmc": niter_hmc, "nsteps_hmc": nsteps_hmc, "burnin_ratio": burnin_ratio,
            "level": level, "grid_points": int(data.shape[0]),
            "kernel": "generalMatern", "band_size": 20, "positive_system": positive_system,
            "prior_temperature": float(1.0 / np.mean(np.isfinite(data[:, 1:]))),
            "kept_draws": int(result["theta"].shape[1]), "mean_log_posterior": float(result["lp"].mean()),
            "u_mean": u.tolist(), "u_sd": result["theta"].std(-1).tolist(),  # u: position in the log box, [0, 2]
            "sigma_mean": (result["sigma"].mean(-1) * scale).tolist(), "scale": scale,
            "phi": result["phi"].tolist(),  # in the scaled units
            "x0_mean": curves[:, 0].tolist(), "species": species}
    return theta, info
