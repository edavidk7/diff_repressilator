"""The inverse problem every estimator in the benchmark is handed.

A :class:`Problem` holds the data and everything an estimator is allowed to
know: time grid, which species are observed, which parameters are free, the
prior box, the noise level.  It never holds the truth; that lives next to it
in a :class:`Case`, and only the scoring code reads it.

All estimators work on one shared unknown vector ``theta``:

* ``log`` of each free kinetic parameter, then
* ``log`` of every species' value at ``t = 0`` (all eight).

The prior is uniform on a box in ``theta`` (i.e. log-uniform in value), so

* nested sampling and ABC draw ``u ~ U(0,1)^d`` and map it with
  :meth:`Problem.from_unit`;
* gradient methods and HMC work on ``z = logit(u)``, unconstrained, via
  :meth:`Problem.from_z`, which cannot leave the box.

Observed species' initial values are unknowns too, with a prior box of the
first measurement +/- 5 sigma.  Plugging the noisy first frame in as the
start state (as ``experiments/run_partial.py`` does) looks harmless but is
not: at 1% noise the resulting phase error alone costs ~2000 nats of
log-likelihood at the true parameters on the Box 1 case.
"""

from dataclasses import dataclass, field
from typing import Any

import torch

from differentiable_cell.data import RepressilatorParams, RepressilatorState
from differentiable_cell.inference.fast_ode import rk4
from differentiable_cell.run import FIELDS, PARAM_FIELDS

PRIORS: dict[str, tuple[float, float]] = {
    # Biologically bounded boxes, in value space; see experiments/run_partial.py
    # for the reasoning behind each bound.  Not derived from any truth.
    "alpha": (1.0, 2000.0),
    "alpha_0": (1e-4, 10.0),
    "n": (1.0, 4.0),
    "beta": (0.03, 2.0),
    "alpha_GFP": (1.0, 2000.0),
    "beta_GFP": (0.02, 0.2),
}

IC_PRIOR: tuple[float, float] = (1e-3, 3000.0)
"""Log-uniform box for a hidden species' initial value, in model units.

The upper end exceeds any level reachable under the ``alpha`` prior; the
lower end sits below the repressed mRNA floor ``alpha_0``.
"""

CANONICAL: tuple[str, ...] = ("alpha", "n", "beta")
"""Network parameters scored as the headline result.  ``alpha_0`` is scored
separately: at a percent of noise it is known to be unidentifiable."""


@dataclass
class Problem:
    """Data plus what an estimator may know about how it was generated.

    Attributes:
        t: observation times, model units, shape ``(G,)``.
        y: observed data, shape ``(G, n_obs)``.  For ``regime="ensemble"`` it
            is the across-cell mean and ``y_var`` holds the variance.
        observed: names of the observed species, in column order of ``y``.
        free_params: kinetic parameters to estimate.
        fixed_params: values of the others (they do not affect the observed
            species or are treated as known device constants).
        sigma: measurement noise standard deviation per observed column.
        regime: ``"ode"``, ``"single_cell"`` or ``"ensemble"``.
        omega: system size of the generator (stochastic regimes).
        n_cells: cells behind an ensemble summary.
        y_var: across-cell variance for ``"ensemble"``.
        rk4_dt: target RK4 step of :meth:`simulate` (error ~1e-3 of the
            signal at 0.35; measurement noise is ~1e-2).
        n_sims: running count of trajectories simulated, for budgets.
        known_ic: initial values treated as known (excluded from ``theta``),
            e.g. to reproduce a paper that assumed them known.
        priors: per-problem override of :data:`PRIORS` (value-space boxes).
        ic_prior: override of :data:`IC_PRIOR` for unobserved initial values.
    """

    t: torch.Tensor
    y: torch.Tensor
    observed: list[str]
    free_params: list[str]
    fixed_params: dict[str, float]
    sigma: torch.Tensor
    regime: str = "ode"
    omega: float | None = None
    n_cells: int = 1
    y_var: torch.Tensor | None = None
    rk4_dt: float = 0.35
    n_sims: int = 0
    known_ic: dict[str, float] = field(default_factory=dict)
    priors: dict[str, tuple[float, float]] | None = None
    ic_prior: tuple[float, float] | None = None

    # ------------------------------------------------------------------ layout
    @property
    def obs_idx(self) -> list[int]:
        return [FIELDS.index(s) for s in self.observed]

    @property
    def hidden(self) -> list[str]:
        return [f for f in FIELDS if f not in self.observed]

    @property
    def ic_fields(self) -> list[str]:
        """Species whose initial value is unknown (part of ``theta``)."""
        return [f for f in FIELDS if f not in self.known_ic]

    @property
    def names(self) -> list[str]:
        """Names of the entries of ``theta``; initial values are ``"<field>(0)"``."""
        return [*self.free_params, *(f"{f}(0)" for f in self.ic_fields)]

    @property
    def dim(self) -> int:
        return len(self.free_params) + len(self.ic_fields)

    def ic_box(self, name: str) -> tuple[float, float]:
        """Prior box (value space) for one species' initial value."""
        if name not in self.observed:
            return self.ic_prior or IC_PRIOR
        j = self.observed.index(name)
        y0, s = float(self.y[0, j]), float(self.sigma[j])
        lo = max(y0 - 5 * s, IC_PRIOR[0])
        return lo, max(y0 + 5 * s, 2 * lo)

    @property
    def bounds(self) -> torch.Tensor:
        """``(dim, 2)`` box in ``theta`` (log) space."""
        priors = {**PRIORS, **(self.priors or {})}
        rows = [priors[p] for p in self.free_params] + [self.ic_box(f) for f in self.ic_fields]
        return torch.tensor(rows, dtype=self.t.dtype).log()

    # ---------------------------------------------------------- transforms
    def from_unit(self, u: torch.Tensor) -> torch.Tensor:
        lo, hi = self.bounds.unbind(-1)
        return lo + u * (hi - lo)

    def to_unit(self, theta: torch.Tensor) -> torch.Tensor:
        lo, hi = self.bounds.unbind(-1)
        return (theta - lo) / (hi - lo)

    def from_z(self, z: torch.Tensor) -> torch.Tensor:
        return self.from_unit(torch.sigmoid(z))

    def to_z(self, theta: torch.Tensor) -> torch.Tensor:
        u = self.to_unit(theta).clamp(1e-9, 1 - 1e-9)
        return torch.log(u) - torch.log1p(-u)

    def log_prior_z(self, z: torch.Tensor) -> torch.Tensor:
        """Log density of the uniform-box prior pushed to ``z`` (up to a constant)."""
        return (torch.nn.functional.logsigmoid(z) + torch.nn.functional.logsigmoid(-z)).sum(-1)

    def sample_prior(self, n: int, generator: torch.Generator | None = None) -> torch.Tensor:
        u = torch.rand(n, self.dim, generator=generator, dtype=self.t.dtype)
        return self.from_unit(u)

    def values(self, theta: torch.Tensor) -> dict[str, torch.Tensor]:
        """``theta`` -> named values (parameters and hidden ICs), batched."""
        v = theta.exp()
        return {name: v[..., i] for i, name in enumerate(self.names)}

    # -------------------------------------------------------------- forward
    def params_state(
        self, theta: torch.Tensor
    ) -> tuple[RepressilatorParams, RepressilatorState]:
        """Build parameters and initial state for a batch of ``theta``."""
        batch = theta.shape[:-1]
        v = theta.exp()
        k = len(self.free_params)
        pvals = {}
        for name in PARAM_FIELDS:
            if name in self.free_params:
                pvals[name] = v[..., self.free_params.index(name)]
            else:
                pvals[name] = torch.full(batch, self.fixed_params[name], dtype=theta.dtype)
        ics = self.ic_fields
        svals = {
            name: (v[..., k + ics.index(name)] if name in ics
                   else torch.full(batch, float(self.known_ic[name]), dtype=theta.dtype))
            for name in FIELDS
        }
        return (
            RepressilatorParams(**pvals, batch_size=batch),
            RepressilatorState(**svals, batch_size=batch),
        )

    def simulate(self, theta: torch.Tensor, n_obs_times: int | None = None,
                 all_species: bool = False, compiled: bool = True) -> torch.Tensor:
        """Deterministic ODE prediction for a batch of ``theta``.

        RK4 (:mod:`.fast_ode`) with a step of about ``rk4_dt`` that divides
        the (uniform) observation spacing.

        Args:
            theta: shape ``(..., dim)``.
            n_obs_times: simulate only the first this-many observation times
                (a curriculum window); default all.
            all_species: return all eight species instead of the observed ones.
            compiled: use the compiled step (``False`` for forward-mode AD).

        Returns:
            ``(..., G, n_obs)`` (or ``(..., G, 8)``).  Parameter sets the
            solver cannot integrate come back as non-finite values.
        """
        params, state = self.params_state(theta)
        n = self.t.numel() if n_obs_times is None else n_obs_times
        dt_obs = float(self.t[1] - self.t[0])
        stride = max(1, round(dt_obs / self.rk4_dt))
        y0 = torch.stack([getattr(state, f) for f in FIELDS], dim=-1)
        y = rk4(y0, params.alpha, params.alpha_0, params.n, params.beta,
                params.alpha_GFP, params.beta_GFP, h=dt_obs / stride,
                n_steps=(n - 1) * stride, stride=stride, compiled=compiled)
        self.n_sims += int(theta[..., 0].numel())
        return y if all_species else y[..., self.obs_idx]

    def obs_variance(self) -> torch.Tensor:
        """Per-point variance of the data about the ODE prediction, ``(G, k)``.

        Measurement noise, plus - for an ensemble mean - the standard error
        from cell-to-cell spread.  Shared by the likelihood and ABC's distance.
        """
        var = (self.sigma**2).expand_as(self.y)
        if self.regime == "ensemble" and self.y_var is not None:
            var = var + self.y_var / self.n_cells
        return var

    def log_likelihood(self, theta: torch.Tensor, compiled: bool = True) -> torch.Tensor:
        """Gaussian log-likelihood of the data under the ODE, batched.

        For the ensemble regime the mean trajectory is compared against the
        across-cell mean with its standard error added to the noise.
        """
        pred = self.simulate(theta, compiled=compiled)
        var = self.obs_variance()
        r2 = (pred - self.y) ** 2 / var
        ll = -0.5 * (r2 + torch.log(2 * torch.pi * var)).sum((-1, -2))
        return torch.nan_to_num(ll, nan=-torch.inf)

    def scales(self) -> torch.Tensor:
        """Per-observed-species ranges, for normalised losses."""
        return (self.y.max(0).values - self.y.min(0).values).clamp_min(1e-8)


@dataclass
class Case:
    """A problem together with the truth that generated it (for scoring only)."""

    problem: Problem
    truth: dict[str, float]
    truth_theta: torch.Tensor
    meta: dict[str, Any] = field(default_factory=dict)
    clean: torch.Tensor | None = None  # noise-free all-species trajectory (G, 8)
