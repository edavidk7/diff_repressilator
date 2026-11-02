"""The inverse problem and its fit by backpropagation through the ODE solver.

Unknowns ``theta`` are log values on a box (names in ``Problem.names``): the circuit (alpha, alpha_0, n,
beta), each built reporter's strength and leak (a, a_0), and the full starting state. Observed species
start within +-10% of the channel's range around their first frame; hidden ones are free. Titration
(``tau``) is known from the plasmid design.
"""

import math
from dataclasses import dataclass

import torch
from torchdiffeq import odeint

from differentiable_cell import reporters as R
from differentiable_cell.reporter_sim import CHANNEL_STATE, mask

CIRCUIT_BOX = {
    "alpha": (20.0, 1000.0),  # must oscillate; 1000 = 40k monomers per cell
    "alpha_0": (1e-3, 1.0),  # a leak of order K_M stops the oscillation
    "n": (1.0, 4.0),  # two to four operator sites
    "beta": (0.1, 1.0),  # ssrA-tagged repressors: protein half-life 2-20 min
}
CIRCUIT_START_BOX = (1e-3, 1000.0)
CIRCUIT_SPECIES = ["m_lacI", "m_tetR", "m_cI", "p_LacI", "p_TetR", "p_CI"]
# Fixed-step RK4 for fitting (~5 steps per frame); data come from tight dopri5, so the solver's own error
# is not fitted.
RK4_STEP = 0.35


def solve(f, y0, t):
    return odeint(f, y0, t, method="rk4", options={"step_size": RK4_STEP})


@dataclass
class Problem:
    t: torch.Tensor  # frame times (G,)
    y: torch.Tensor  # calibrated frames of the observed channels, K_M units (G, k)
    channels: list[int]  # observed channels, indices into reporter_sim.CHANNELS
    present: list[int]  # reporters built (they titrate and have parameters)
    tau: list[float]  # known titration, per reporter

    def __post_init__(self):
        self.mask = mask(self.present)
        self.observed_state = [CHANNEL_STATE[c] for c in self.channels]
        self.tau_t = torch.tensor(self.tau, dtype=torch.float64)
        self.range = (self.y.max(0).values - self.y.min(0).values).clamp_min(1e-9)

        def first_frame_box(state, default):
            if state not in self.observed_state:
                return default
            c = self.observed_state.index(state)
            first, rng = float(self.y[0, c]), float(self.range[c])
            return max(first - 0.1 * rng, 1e-3), first + 0.1 * rng

        def reading_range(j):
            c = self.observed_state.index(R.F_IDX[j])
            return max(float(self.y[:, c].min()), 1e-3), float(self.y[:, c].max())

        boxes = list(CIRCUIT_BOX.values())
        for j in self.present:
            low, high = reading_range(j)
            boxes += [(0.8 * high, 10 * high),  # a: F <= a when fully derepressed
                      (1e-4 * high, low)]  # a_0: F >= a_0 when fully repressed
        boxes += [first_frame_box(i, CIRCUIT_START_BOX) for i in range(6)]
        for j in self.present:
            low, high = reading_range(j)
            boxes += [(1e-4 * high, 10 * high),  # r(0): between ~a_0 and ~a
                      (5e-6 * high, 0.5 * high),  # D(0): dark FP is ~5% of r
                      first_frame_box(R.F_IDX[j], (0.1 * low, 3 * high))]  # F(0)
        self.lo = torch.tensor([math.log(lo) for lo, _ in boxes], dtype=torch.float64)
        self.hi = torch.tensor([math.log(hi) for _, hi in boxes], dtype=torch.float64)

    @property
    def names(self) -> list[str]:
        reporters = [R.REPORTERS[j] for j in self.present]
        return (list(CIRCUIT_BOX) + [f"{k}_{r}" for r in reporters for k in ("a", "a0")]
                + [f"{s}(0)" for s in CIRCUIT_SPECIES]
                + [f"{k}_{r}(0)" for r in reporters for k in ("r", "D", "F")])

    # theta (log values) = lo + sigmoid(z) (hi - lo): z is unconstrained, the box exact.
    def from_z(self, z):
        return self.lo + torch.sigmoid(z) * (self.hi - self.lo)

    def to_z(self, theta):
        u = ((theta - self.lo) / (self.hi - self.lo)).clamp(1e-9, 1 - 1e-9)
        return torch.log(u) - torch.log1p(-u)

    def sample_prior(self, n, generator=None):
        return self.lo + torch.rand(n, len(self.lo), generator=generator, dtype=torch.float64) * (self.hi - self.lo)

    def values(self, theta) -> dict[str, torch.Tensor]:
        return {name: theta[..., i].exp() for i, name in enumerate(self.names)}

    def params_and_start(self, theta):
        """Model parameters and the 15-species start ``(..., 15)``; absent reporters stay at zero."""
        v = self.values(theta)
        one, zero = torch.ones_like(v["alpha"]), torch.zeros_like(v["alpha"])

        def per_reporter(key, absent):
            return torch.stack([v[f"{key}_{r}"] if j in self.present else absent
                                for j, r in enumerate(R.REPORTERS)], dim=-1)

        p = {"alpha": v["alpha"], "alpha_0": v["alpha_0"], "n": v["n"], "beta": v["beta"],
             "a": per_reporter("a", one), "a_0": per_reporter("a0", zero),
             "tau": self.tau_t.expand(*theta.shape[:-1], 3)}
        start = [v[f"{s}(0)"] for s in CIRCUIT_SPECIES]
        for j, r in enumerate(R.REPORTERS):
            start += [v[f"{k}_{r}(0)"] for k in ("r", "D", "F")] if j in self.present else [zero] * 3
        return p, torch.stack(start, dim=-1)

    def predict(self, theta, n_frames=None, t=None):
        """Observed channels ``(..., G, k)`` at the first ``n_frames`` frames, or at times ``t``."""
        p, y0 = self.params_and_start(theta)
        if t is None:
            t = self.t if n_frames is None else self.t[:n_frames]
        traj = solve(lambda _t, y: R.rhs(y, p, self.mask), y0, t)  # (G, ..., 15)
        return traj[..., self.observed_state].movedim(0, -2)

    def loss(self, theta, n_frames=None, kind: str = "mse"):
        """Mean over frames and channels of a loss on residuals divided by the channel's range.
        ``kind``: "mse", "l1" or "huber" (quadratic below 0.05, a few noise levels)."""
        y = self.y if n_frames is None else self.y[:n_frames]
        r = (self.predict(theta, n_frames) - y) / self.range
        if kind == "mse":
            per_point = r**2
        elif kind == "l1":
            per_point = r.abs()
        elif kind == "huber":
            per_point = torch.nn.functional.huber_loss(r, torch.zeros_like(r), reduction="none", delta=0.05)
        else:
            raise ValueError(kind)
        return per_point.mean((-1, -2))


def fit_gradient(problem: Problem, restarts: int = 64, epochs: int = 1500, lr: float = 0.05,
                 lr_final: float = 2.5e-4, stages=(0.25, 0.5, 1.0), seed: int = 0, loss: str = "mse",
                 optimizer: str = "adam", schedule: str = "exp", return_all: bool = False,
                 history: list | None = None, history_every: int = 25):
    """Backpropagation through the solver from ``restarts`` prior draws, optimised as one batch.

    The loss first sees 25% of the frames, then 50%, then all: a period error becomes a phase error that
    grows over the record and flattens the loss. ``optimizer``: adam, rmsprop or sgd (Nesterov);
    ``schedule``: lr from ``lr`` to ``lr_final``, exponential or cosine. The best restart is chosen by
    full-record MSE whatever the training loss. Returns its theta (all restarts' with ``return_all``) and
    every restart's final MSE; ``history`` collects ``(epoch, MSE per restart, theta per restart)``.
    """
    g = torch.Generator().manual_seed(seed)
    z = problem.to_z(problem.sample_prior(restarts, g)).requires_grad_(True)
    opt = {"adam": lambda: torch.optim.Adam([z], lr=lr),
           "rmsprop": lambda: torch.optim.RMSprop([z], lr=lr),
           "sgd": lambda: torch.optim.SGD([z], lr=lr, momentum=0.9, nesterov=True)}[optimizer]()
    if schedule == "exp":
        sched = torch.optim.lr_scheduler.ExponentialLR(opt, (lr_final / lr) ** (1.0 / max(epochs - 1, 1)))
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr_final)
    windows = [max(4, math.ceil(f * problem.t.numel())) for f in stages]

    def full_mse():
        with torch.no_grad():
            theta = problem.from_z(z)
            return torch.nan_to_num(problem.loss(theta), nan=math.inf), theta.clone()

    for epoch in range(epochs):
        if history is not None and epoch % history_every == 0:
            history.append((epoch, *full_mse()))
        opt.zero_grad()
        per_restart = problem.loss(problem.from_z(z), windows[epoch * len(windows) // epochs], kind=loss)
        # Restarts are independent; one that goes non-finite stops moving.
        torch.where(torch.isfinite(per_restart), per_restart, torch.zeros_like(per_restart)).sum().backward()
        z.grad = torch.nan_to_num(z.grad, nan=0.0, posinf=0.0, neginf=0.0)
        opt.step()
        sched.step()
    final, theta = full_mse()
    if history is not None:
        history.append((epochs, final, theta))
    if return_all:
        return theta, final
    return theta[int(final.argmin())], final


def fit_laplace(problem: Problem, theta_map: torch.Tensor, n_draws: int = 4000, seed: int = 0):
    """Laplace approximation at the fit: Gaussian draws ``(n_draws, dim)`` in z with the inverse Hessian of
    the log posterior (per-channel noise from the fit's residuals, uniform prior on the box). Eigenvalues
    are floored at the prior's curvature (0.25), so undetermined directions get prior-width spread."""
    with torch.no_grad():
        noise_var = ((problem.predict(theta_map) - problem.y) / problem.range).pow(2).mean(0)
    z_map = problem.to_z(theta_map).detach()

    def neg_log_post(z):
        r = (problem.predict(problem.from_z(z)) - problem.y) / problem.range
        log_prior = (torch.nn.functional.logsigmoid(z) + torch.nn.functional.logsigmoid(-z)).sum()
        return 0.5 * (r**2 / noise_var).sum() - log_prior

    hess = torch.autograd.functional.hessian(neg_log_post, z_map)
    evals, evecs = torch.linalg.eigh(0.5 * (hess + hess.T))
    chol = evecs / evals.clamp_min(0.25).sqrt()
    eps = torch.randn(n_draws, len(z_map), generator=torch.Generator().manual_seed(seed), dtype=z_map.dtype)
    return problem.from_z(z_map + eps @ chol.T), {"negative_hessian_eigs": int((evals < 0).sum())}
