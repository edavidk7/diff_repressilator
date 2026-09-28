"""Synthetic benchmark cases: truths, data regimes and reporter panels.

Truths are sampled, not fixed, so that a method is not scored on one lucky
point.  Each truth is drawn inside the oscillatory region (Box 1 stability
condition) and its initial state is taken *on the limit cycle*, after a
burn-in from the standard asymmetric start: imaging starts on cells that
are already oscillating, and a start state on the prior boundary (exact
zeros) would unfairly favour methods that clamp.

Three data regimes:

* ``"ode"`` - one deterministic trajectory plus Gaussian measurement noise;
* ``"single_cell"`` - one exact-SSA cell at system size ``omega`` plus the
  same measurement noise;
* ``"ensemble"`` - ``n_cells`` exact-SSA cells from a common (synchronised)
  start, summarised by the across-cell mean and variance of each observed
  species.

Measurement noise follows ``experiments/run_partial.py``: standard deviation
``noise`` times each species' range over the record.
"""

import math

import torch

from differentiable_cell.data import RepressilatorParams, RepressilatorState
from differentiable_cell.gillespie import run_gillespie
from differentiable_cell.helpers import estimate_period
from differentiable_cell.inference.problem import Case, Problem
from differentiable_cell.model import Repressilator
from differentiable_cell.run import FIELDS, PARAM_FIELDS, run_torchdiffeq

PANELS: dict[str, list[str]] = {
    "all": list(FIELDS),
    "gfp": ["p_GFP"],
    "tetR+gfp": ["p_TetR", "p_GFP"],
    "lacI": ["p_LacI"],
    "tetR": ["p_TetR"],
    "cI": ["p_CI"],
    "lacI+gfp": ["p_LacI", "p_GFP"],
    "cI+gfp": ["p_CI", "p_GFP"],
    "3proteins": ["p_LacI", "p_TetR", "p_CI"],
}
"""Reporter panels: which species carry a fluorophore."""

TRUTH_BOX: dict[str, tuple[float, float]] = {
    # Narrower than the prior: where the real circuit plausibly sits.
    "alpha": (50.0, 1000.0),
    "n": (1.6, 3.0),
    "beta": (0.1, 1.0),
    "alpha_GFP": (50.0, 1000.0),
    "beta_GFP": (0.02, 0.1),
}
LEAK_RATIO = (1e-4, 1e-2)  # alpha_0 / alpha


def _loguniform(lo: float, hi: float, g: torch.Generator) -> float:
    return math.exp(math.log(lo) + float(torch.rand((), generator=g)) * math.log(hi / lo))


def steady_state_unstable(alpha: float, alpha_0: float, n: float, beta: float) -> bool:
    """Whether the symmetric steady state is linearly unstable (so the loop oscillates).

    The fixed point solves ``p = alpha / (1 + p**n) + alpha_0``.  Rather
    than transcribe Box 1's closed-form boundary, this checks the eigenvalues
    of the 6x6 Jacobian of the repressor loop directly.
    """
    lo, hi = 0.0, alpha + alpha_0 + 1.0
    for _ in range(200):  # bisection on a monotone function
        p = 0.5 * (lo + hi)
        if p - alpha / (1 + p**n) - alpha_0 > 0:
            hi = p
        else:
            lo = p
    slope = -alpha * n * p ** (n - 1) / (1 + p**n) ** 2
    jac = torch.zeros(6, 6, dtype=torch.float64)
    for i, j in ((0, 5), (1, 3), (2, 4)):  # m_i repressed by protein j
        jac[i, i] = -1.0
        jac[i, j] = slope
        jac[3 + i, 3 + i] = -beta
        jac[3 + i, i] = beta
    return bool(torch.linalg.eigvals(jac).real.max() > 0)


def sample_truth(generator: torch.Generator, max_tries: int = 1000) -> dict[str, float]:
    """Draw kinetic parameters inside :data:`TRUTH_BOX` that oscillate."""
    for _ in range(max_tries):
        v = {k: _loguniform(*TRUTH_BOX[k], generator) for k in TRUTH_BOX}
        v["alpha_0"] = v["alpha"] * _loguniform(*LEAK_RATIO, generator)
        # A margin inside the unstable region, so the oscillation is robust.
        if steady_state_unstable(v["alpha"], v["alpha_0"], v["n"], v["beta"] / 1.5) and \
                steady_state_unstable(v["alpha"], v["alpha_0"], v["n"], v["beta"] * 1.5):
            return v
    raise RuntimeError("no oscillating truth found")


def paper_truth() -> dict[str, float]:
    """The parameters of Box 1 (the README's running example)."""
    return dict(alpha=216.0, alpha_0=0.216, n=2.0, beta=0.2, alpha_GFP=216.0,
                beta_GFP=2.0 / 90.0)


def _params(v: dict[str, float]) -> RepressilatorParams:
    return RepressilatorParams(
        **{k: torch.tensor(v[k], dtype=torch.float64) for k in PARAM_FIELDS}, batch_size=()
    )


def _state(y: torch.Tensor) -> RepressilatorState:
    return RepressilatorState(**{f: y[..., i] for i, f in enumerate(FIELDS)},
                              batch_size=y.shape[:-1])


def limit_cycle_start(truth: dict[str, float], phase: float) -> torch.Tensor:
    """A state on the limit cycle, at fraction ``phase`` of a period after burn-in."""
    start = torch.tensor([0, 0, 0, 5.0, 0, 15.0, 0, 0], dtype=torch.float64)
    t = torch.linspace(0, 400, 4001, dtype=torch.float64)
    with torch.no_grad():
        traj = run_torchdiffeq(Repressilator(), _state(start), _params(truth), t)
    y = torch.stack([getattr(traj, f) for f in FIELDS], -1)
    period = estimate_period(t[2000:], y[2000:, 3]) or 40.0
    i = 3000 + int(phase * period / 0.1)
    return y[min(i, t.numel() - 1)]


def make_case(
    truth: dict[str, float],
    panel: str | list[str],
    regime: str = "ode",
    noise: float = 0.01,
    n_frames: int = 87,
    horizon: float = 150.0,
    omega: float = 40.0,
    n_cells: int = 200,
    seed: int = 0,
    phase: float | None = None,
    fit_device_params: bool | None = None,
) -> Case:
    """Build one benchmark case.

    Args:
        truth: kinetic parameters.
        panel: a key of :data:`PANELS` or an explicit species list.
        regime: ``"ode"``, ``"single_cell"`` or ``"ensemble"``.
        noise: measurement noise, as a fraction of each species' range.
        n_frames: observation times (87 frames over 150 units ~ one every
            5 min, the README's cadence).
        horizon: record length in model units (~3.4 periods at Box 1 values).
        omega: system size for the stochastic regimes.
        n_cells: cells in the ensemble regime.
        seed: seeds the start phase, the stochastic run and the noise.
        phase: start phase on the limit cycle; drawn from ``seed`` if None.
        fit_device_params: whether ``alpha_GFP``/``beta_GFP`` are free;
            default: free exactly when ``p_GFP`` is observed.
    """
    g = torch.Generator().manual_seed(seed)
    observed = list(PANELS[panel]) if isinstance(panel, str) else list(panel)
    if phase is None:
        phase = float(torch.rand((), generator=g))
    y0 = limit_cycle_start(truth, phase)
    t = torch.linspace(0, horizon, n_frames, dtype=torch.float64)
    with torch.no_grad():
        clean = run_torchdiffeq(Repressilator(), _state(y0), _params(truth), t)
    clean = torch.stack([getattr(clean, f) for f in FIELDS], -1)
    obs_idx = [FIELDS.index(s) for s in observed]

    y_var = None
    if regime == "ode":
        signal = clean[:, obs_idx]
    elif regime in ("single_cell", "ensemble"):
        cells = 1 if regime == "single_cell" else n_cells
        with torch.no_grad():
            traj = run_gillespie(_state(y0.expand(cells, 8).clone()), _params(truth), t,
                                 omega=omega, generator=g)
        stoch = torch.stack([getattr(traj, f) for f in FIELDS], -1)[..., obs_idx]  # (G, C, k)
        if regime == "single_cell":
            signal = stoch[:, 0]
        else:
            signal = stoch.mean(1)
            y_var = stoch.var(1)
    else:
        raise ValueError(f"unknown regime {regime!r}")

    ranges = (signal.max(0).values - signal.min(0).values).clamp_min(1e-8)
    sigma = noise * ranges
    y = signal + sigma * torch.randn(signal.shape, generator=g, dtype=signal.dtype)

    if fit_device_params is None:
        fit_device_params = "p_GFP" in observed
    free = ["alpha", "alpha_0", "n", "beta"]
    if fit_device_params:
        free += ["alpha_GFP", "beta_GFP"]
    fixed = {k: v for k, v in truth.items() if k not in free}
    # Device parameters that are not free and not observed do not affect the
    # data at all; they are passed at their true value only so the ODE runs.

    problem = Problem(
        t=t, y=y, observed=observed, free_params=free, fixed_params=fixed,
        sigma=sigma.clamp_min(1e-6), regime=regime,
        omega=omega if regime != "ode" else None,
        n_cells=n_cells if regime == "ensemble" else 1, y_var=y_var,
    )
    # The truth in theta space, clipped into the prior box (a true value that
    # sits below the box floor, e.g. an almost-zero mRNA, is scored at the floor).
    lo, hi = problem.bounds.unbind(-1)
    theta = torch.tensor(
        [math.log(truth[p]) for p in free]
        + [math.log(max(float(y0[FIELDS.index(f)]), 1e-12)) for f in problem.ic_fields],
        dtype=torch.float64,
    ).clamp(lo, hi)
    full_truth = dict(truth)
    full_truth.update({f"{f}(0)": float(y0[i]) for i, f in enumerate(FIELDS)})
    meta = dict(panel=panel if isinstance(panel, str) else "+".join(panel), regime=regime,
                noise=noise, seed=seed, phase=phase, omega=omega, n_cells=n_cells)
    return Case(problem=problem, truth=full_truth, truth_theta=theta, meta=meta, clean=clean)
