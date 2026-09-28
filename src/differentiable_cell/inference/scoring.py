"""Scoring a :class:`Result` against the truth of its :class:`Case`.

Per free parameter:

* ``rel_err`` - ``|estimate - truth| / truth`` of the point estimate;
* ``log_err`` - ``|log estimate - log truth|`` (symmetric in over/under);
* ``covered`` / ``ci_width_log`` - whether the central 90% interval of the
  samples contains the truth, and its width in log units (methods with
  samples only).

Per trajectory (the ODE run at the point estimate against the noise-free
truth, each species normalised by its true range):

* ``nrmse_observed`` / ``nrmse_hidden`` over the fitted record;
* ``nrmse_forecast`` for the observed species over the next half-record,
  which no method saw - the test of whether the period came out right.
"""

import math
from dataclasses import replace
from typing import Any

import torch

from differentiable_cell.data import RepressilatorParams, RepressilatorState
from differentiable_cell.inference.base import Result
from differentiable_cell.inference.problem import Case
from differentiable_cell.model import Repressilator
from differentiable_cell.run import FIELDS, PARAM_FIELDS, run_torchdiffeq


def _truth_trajectory(case: Case, t: torch.Tensor) -> torch.Tensor:
    y0 = torch.tensor([case.truth[f"{f}(0)"] for f in FIELDS], dtype=torch.float64)
    params = RepressilatorParams(
        **{k: torch.tensor(case.truth[k], dtype=torch.float64) for k in PARAM_FIELDS},
        batch_size=())
    state = RepressilatorState(**{f: y0[i] for i, f in enumerate(FIELDS)}, batch_size=())
    with torch.no_grad():
        traj = run_torchdiffeq(Repressilator(), state, params, t)
    return torch.stack([getattr(traj, f) for f in FIELDS], -1)


def score(case: Case, result: Result, forecast_fraction: float = 0.5) -> dict[str, Any]:
    p = case.problem
    k = len(p.free_params)
    out: dict[str, Any] = {"method": result.method, "wall": result.wall, "n_sims": result.n_sims,
                           **case.meta, "params": {}}
    est = result.theta.detach()
    for i, name in enumerate(p.free_params):
        truth = case.truth[name]
        value = math.exp(float(est[i]))
        row = {"truth": truth, "estimate": value,
               "rel_err": abs(value - truth) / truth,
               "log_err": abs(math.log(value) - math.log(truth))}
        if result.samples is not None and result.samples.shape[0] > 1:
            s = result.samples[:, i]
            lo, hi = float(s.quantile(0.05)), float(s.quantile(0.95))
            row.update(covered=bool(lo <= math.log(truth) <= hi), ci_width_log=hi - lo,
                       post_sd_log=float(s.std()))
        out["params"][name] = row

    dt = float(p.t[1] - p.t[0])
    n_extra = int(round(forecast_fraction * (p.t.numel() - 1)))
    t_ext = torch.arange(p.t.numel() + n_extra, dtype=torch.float64) * dt + float(p.t[0])
    truth_traj = _truth_trajectory(case, t_ext)
    ext = replace(p, t=t_ext, y=torch.cat([p.y, p.y[-1:].expand(n_extra, -1)]))
    with torch.no_grad():
        pred = ext.simulate(est, all_species=True)
    rng = (truth_traj.max(0).values - truth_traj.min(0).values).clamp_min(1e-8)
    err = ((pred - truth_traj) / rng) ** 2
    g = p.t.numel()
    obs, hid = p.obs_idx, [FIELDS.index(h) for h in p.hidden]
    nrmse = lambda e: float(e.mean().sqrt()) if e.numel() else float("nan")
    out["nrmse_observed"] = nrmse(err[:g, obs])
    out["nrmse_hidden"] = nrmse(err[:g, hid])
    out["nrmse_forecast"] = nrmse(err[g:, obs])
    return out
