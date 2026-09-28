"""Stochastic forward models and summary statistics for inference.

For stochastic data an estimator can either keep the ODE (misspecified: it
has no intrinsic noise) or simulate the stochastic model itself.  This module
is the second option, shared by :mod:`.abc_smc` and :mod:`.grad_stochastic`:

* :func:`simulate` - batched CLE or DGA trajectories for a batch of
  ``theta``, at the problem's system size ``omega``, with the measurement
  noise added, so simulated and observed data are directly comparable;
* :func:`summaries` - per-species mean, standard deviation and
  autocorrelation up to ``max_lag`` frames: the statistics of a single-cell
  trace that do not depend on its (random) phase;
* :func:`distance` - how far simulated data are from the observed data,
  for each regime.
"""

import torch

from differentiable_cell.data import RepressilatorParams
from differentiable_cell.gillespie import run_dga
from differentiable_cell.inference.problem import Problem
from differentiable_cell.langevin import run_langevin


def simulate(
    problem: Problem,
    theta: torch.Tensor,
    n_cells: int,
    kind: str = "cle",
    generator: torch.Generator | None = None,
    substeps: int = 10,
    measurement_noise: bool | None = None,
    n_obs_times: int | None = None,
    dga_kwargs: dict | None = None,
) -> torch.Tensor:
    """Stochastic trajectories of the observed species.

    Args:
        problem: supplies time grid, observed species, ``omega`` and noise.
        theta: ``(B, dim)``.
        n_cells: independent cells per parameter set.
        kind: ``"cle"`` (Langevin) or ``"dga"`` (differentiable Gillespie).
        generator: random source (fixing it gives common random numbers).
        substeps: CLE steps per observation interval.
        measurement_noise: add the problem's Gaussian measurement noise to
            each cell.  Default: yes, except in the ensemble regime, where the
            data's noise sits on the across-cell mean (and :func:`distance`
            accounts for it there) rather than on each cell.
        n_obs_times: simulate only this many leading observation times.
        dga_kwargs: extra arguments for :func:`run_dga`.

    Returns:
        ``(B, n_cells, G, n_obs)``.
    """
    params, state = problem.params_state(theta)
    b = theta.shape[0]
    params = RepressilatorParams(
        **{k: getattr(params, k).unsqueeze(-1) for k in params.keys()}, batch_size=(b, 1)
    )
    state = state.unsqueeze(-1).expand(b, n_cells).clone()
    n = problem.t.numel() if n_obs_times is None else n_obs_times
    t = problem.t[:n]
    omega = problem.omega or 40.0
    if kind == "cle":
        traj = run_langevin(state, params, t, omega=omega, substeps=substeps, generator=generator)
    elif kind == "dga":
        if b != 1:
            raise ValueError("the DGA runs one parameter set at a time")
        flat_params = RepressilatorParams(
            **{k: getattr(params, k).reshape(()) for k in params.keys()}, batch_size=()
        )
        traj = run_dga(state[0], flat_params, t, omega=omega, generator=generator,
                       **(dga_kwargs or {}))
        traj = traj.unsqueeze(1)  # (G, 1, C)
    else:
        raise ValueError(f"unknown kind {kind!r}")
    y = torch.stack([getattr(traj, f) for f in problem.observed], dim=-1)  # (G, B, C, k)
    y = y.permute(1, 2, 0, 3)
    problem.n_sims += b * n_cells
    if measurement_noise is None:
        measurement_noise = problem.regime != "ensemble"
    if measurement_noise:
        y = y + problem.sigma * torch.randn(y.shape, generator=generator, dtype=y.dtype)
    return y


def summaries(y: torch.Tensor, max_lag: int = 30) -> torch.Tensor:
    """Phase-free statistics of traces ``(..., G, k)`` -> ``(..., k, 2 + max_lag)``.

    Mean, standard deviation, and the autocorrelation at lags
    ``1..max_lag`` frames of each species.  30 frames at the default cadence
    is ~52 model units, more than one period in the truth box.
    """
    mean = y.mean(-2)
    centred = y - mean.unsqueeze(-2)
    var = centred.pow(2).mean(-2).clamp_min(1e-12)
    g = y.shape[-2]
    acf = torch.stack(
        [(centred[..., lag:, :] * centred[..., : g - lag, :]).mean(-2) / var
         for lag in range(1, max_lag + 1)],
        dim=-1,
    )
    return torch.cat([mean.unsqueeze(-1), var.sqrt().unsqueeze(-1), acf], dim=-1)


def distance(problem: Problem, sim: torch.Tensor, max_lag: int = 30) -> torch.Tensor:
    """Distance of simulated data ``(B, C, G, k)`` to the problem's data -> ``(B,)``.

    * ``"single_cell"``: RMS difference of :func:`summaries`, the cell-average
      of the simulated summaries against the observed trace's.  Mean and
      standard deviation are measured in units of the observed range.
    * ``"ensemble"``: RMS of the mean-trajectory error in units of its
      standard error, plus the same for the standard deviation trajectory.
    * ``"ode"``: RMS standardised residual of each cell's trajectory, averaged
      over cells (1 at the truth, for an exact model).
    """
    y = problem.y
    if problem.regime == "single_cell":
        scale = problem.scales()
        s_sim = summaries(sim, max_lag).mean(1)  # (B, k, L+2)
        s_obs = summaries(y, max_lag)
        diff = s_sim - s_obs
        diff = torch.cat([diff[..., :2] / scale.unsqueeze(-1), diff[..., 2:]], dim=-1)
        return diff.pow(2).mean((-1, -2)).sqrt()
    if problem.regime == "ensemble":
        n = problem.n_cells
        m_sim, v_sim = sim.mean(1), sim.var(1)
        c = sim.shape[1]
        # Data mean: cell-to-cell spread over n cells plus the measurement
        # noise on the mean itself; simulated mean: its own Monte Carlo error.
        se_mean = (problem.y_var / n + problem.sigma**2 + v_sim.detach() / c).sqrt()
        # Standard error of a standard deviation is ~ sd / sqrt(2 (n - 1)).
        sd_obs = problem.y_var.clamp_min(1e-12).sqrt()
        se_sd = (sd_obs / (2 * (n - 1)) ** 0.5).clamp_min(1e-3 * problem.scales())
        d_mean = ((m_sim - y) / se_mean).pow(2).mean((-1, -2))
        d_sd = ((v_sim.clamp_min(1e-12).sqrt() - sd_obs) / se_sd).pow(2).mean((-1, -2))
        return (0.5 * (d_mean + d_sd)).sqrt()
    return ((sim - y) / problem.sigma).pow(2).mean((-1, -2)).sqrt().mean(1)
