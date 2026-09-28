"""Inverse physics-informed neural network (Casajuana et al., 2026).

A network ``u_w(t)`` represents all eight species; the kinetic parameters
are trained jointly with its weights ``w`` on

    L = lambda_f * mean_j |du/dt(t_j) - f(u(t_j); theta)|^2     (collocation)
      + lambda_y * mean_(obs) |u(t_i) - y(t_i)|^2                (data)

No ODE solver is involved: the dynamics enter only as a residual.  Setup as
in the paper - fully connected, five hidden layers of 100 units, sine
activations, Glorot initialisation, softplus output for positivity, Adam.
The paper's initial-condition term ``lambda_0`` presumes a measured
``y(0)`` for every species; here only observed ones have one, and their
``t = 0`` frame is already in the data term, so it is dropped.

Adaptations to this benchmark, all recorded here:

* 8 species instead of the paper's 3-protein model, and a horizon of
  ~3.4 periods; time enters the network as ``t / t_scale`` (default 10
  model units) so that sine features span the record;
* every term is expressed in units of a per-species scale (observed:
  range of the data; hidden: mean observed range) so the loss is balanced;
* parameters live in the shared prior box via ``theta = box(sigmoid(z))``;
* the hidden initial values are read off the trained network at ``t = 0``.
"""

import math

import torch
from torch import nn

from differentiable_cell.inference.base import Result, Timer, clip_to_box
from differentiable_cell.inference.problem import Problem
from differentiable_cell.run import FIELDS


class _Sine(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


def _network(width: int, depth: int, out: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    fan_in = 1
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


_M = [0, 1, 2, 6]
_P = [3, 4, 5, 7]
_REP = [5, 3, 4, 4]


def _rhs(u: torch.Tensor, v: dict[str, torch.Tensor]) -> torch.Tensor:
    """Repressilator right-hand side on ``(N, 8)`` states (FIELDS order)."""
    m, p, rep = u[:, _M], u[:, _P], u[:, _REP]
    alpha = torch.stack([v["alpha"]] * 3 + [v["alpha_GFP"]])
    beta = torch.stack([v["beta"]] * 3 + [v["beta_GFP"]])
    dm = -m + alpha / (1 + rep ** v["n"]) + v["alpha_0"]
    dp = beta * (m - p)
    out = torch.empty_like(u)
    out[:, _M] = dm
    out[:, _P] = dp
    return out


def _train_once(problem: Problem, seed: int, iterations: int, lr: float, n_colloc: int,
                width: int, depth: int, t_scale: float, lambda_f: float, lambda_y: float,
                max_seconds: float | None, timer: Timer) -> dict:
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    k = len(problem.free_params)
    net = _network(width, depth, len(FIELDS))
    z_par = problem.to_z(problem.sample_prior(1, g)[0])[:k].clone().requires_grad_(True)
    opt = torch.optim.Adam([{"params": net.parameters()}, {"params": [z_par]}], lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda e: 0.1 ** (min(e, iterations) / iterations))

    obs_scale = problem.scales()
    scale = torch.full((len(FIELDS),), float(obs_scale.mean()), dtype=torch.float64)
    scale[problem.obs_idx] = obs_scale
    t_obs = problem.t
    t0, t1 = float(t_obs[0]), float(t_obs[-1])
    lo, hi = problem.bounds[:k].unbind(-1)

    def states(t: torch.Tensor) -> torch.Tensor:
        return nn.functional.softplus(net((t / t_scale).unsqueeze(-1))) * scale

    def params() -> dict[str, torch.Tensor]:
        vals = (lo + torch.sigmoid(z_par) * (hi - lo)).exp()
        v = {name: vals[i] for i, name in enumerate(problem.free_params)}
        for name, val in problem.fixed_params.items():
            v[name] = torch.tensor(val, dtype=torch.float64)
        return v

    losses = []
    for it in range(iterations):
        tc = t0 + (t1 - t0) * torch.rand(n_colloc, generator=g, dtype=torch.float64)
        # Each output row depends only on its own time, so one forward-mode
        # pass with a tangent of ones gives du/dt for every species at once.
        u, du = torch.func.jvp(states, (tc,), (torch.ones_like(tc),))
        res = ((du - _rhs(u, params())) / scale).pow(2).mean()
        fit = ((states(t_obs)[:, problem.obs_idx] - problem.y) / obs_scale).pow(2).mean()
        loss = lambda_f * res + lambda_y * fit
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if it % 100 == 0:
            losses.append((float(res.detach()), float(fit.detach())))
        if max_seconds is not None and timer.elapsed() > max_seconds:
            break
    with torch.no_grad():
        u0 = states(torch.tensor([t0], dtype=torch.float64))[0]
        ic_idx = [FIELDS.index(f) for f in problem.ic_fields]
        theta = torch.cat([lo + torch.sigmoid(z_par) * (hi - lo),
                           u0[ic_idx].clamp_min(1e-12).log()])
        final = (float(res), float(fit))
    return {"theta": clip_to_box(problem, theta.detach()), "loss": lambda_f * final[0] + lambda_y * final[1],
            "final_residual": final[0], "final_data": final[1], "history": losses[::10]}


def fit(
    problem: Problem,
    restarts: int = 3,
    iterations: int = 20000,
    lr: float = 1e-3,
    n_colloc: int = 512,
    width: int = 100,
    depth: int = 5,
    t_scale: float = 10.0,
    lambda_f: float = 1.0,
    lambda_y: float = 1.0,
    seed: int = 0,
    max_seconds: float | None = None,
) -> Result:
    """Train ``restarts`` inverse PINNs and keep the lowest final loss.

    Args:
        problem: the inverse problem (the PINN uses only ``t``, ``y`` and the
            model structure; the sampling cadence enters via the data term).
        restarts: independent networks/parameter initialisations.
        iterations: Adam steps per restart (the paper uses 3000-5000 on a
            shorter, 3-species problem; learning rate decays 10x).
        lr: Adam learning rate.
        n_colloc: collocation points, resampled uniformly every step.
        width, depth: network size (paper: 100 x 5).
        t_scale: time units per unit network input.
        lambda_f, lambda_y: loss weights.
        seed: base seed.
        max_seconds: optional wall-clock cap per restart.
    """
    runs = []
    with Timer(problem) as timer:
        for r in range(restarts):
            start = timer.elapsed()
            sub = Timer(problem)
            sub.t0 = timer.t0 + start
            runs.append(_train_once(problem, seed * 1000 + r, iterations, lr, n_colloc, width,
                                    depth, t_scale, lambda_f, lambda_y, max_seconds, sub))
    best = min(runs, key=lambda d: d["loss"] if math.isfinite(d["loss"]) else math.inf)
    return Result(
        method="pinn",
        theta=best["theta"],
        samples=None,
        n_sims=0,
        wall=timer.wall,
        info={"restart_losses": [r["loss"] for r in runs],
              "final_residual": best["final_residual"], "final_data": best["final_data"],
              "restart_theta": [r["theta"].tolist() for r in runs]},
    )
