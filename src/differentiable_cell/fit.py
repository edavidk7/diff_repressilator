"""The inverse problem: from one cell's reporter frames, find its starting state
and the parameters; the forward simulation determines everything after that.

Everything is in model units: the readout is the calibrated fluorescence,
i.e. mature FP in K_M units. Titration is known from the plasmid design
(``tau``, not estimated).

Unknowns ``theta`` (names in :attr:`Problem.names`), each log-uniform on a box:

    alpha, alpha_0, n, beta        the circuit (symmetric, Box 1)
    per reporter present:
        a, a_0                     production and leak
    the starting state:
        m_*(0), p_*(0)             the six circuit species
        r(0), D(0), F(0)           each present reporter's mRNA, dark and mature protein

Observed channels are any of the six in ``reporter_sim.CHANNELS``: reporter
fluorescence, or a repressor fused to an FP. Every observed species' starting
value is boxed tightly around its first measured frame, so the data pin where
in its cycle the cell starts; everything hidden is free.

:func:`fit_gradient` is backpropagation through the ODE solver
(``torchdiffeq``): range-normalised MSE on the frames, Adam from a batch of
prior draws, over a growing horizon.
"""

import math
from dataclasses import dataclass

import torch
from torchdiffeq import odeint

from differentiable_cell import reporters as R
from differentiable_cell.reporter_sim import CHANNEL_STATE, mask

# Prior boxes. None uses the true answer; each comes from physics or the data.
CIRCUIT_BOX = {
    "alpha": (20.0, 1000.0),  # the data oscillate (very low alpha cannot); 1000 = 40k monomers/cell
    "alpha_0": (1e-3, 1.0),  # a leak of order K_M (= 1) kills the oscillation (Box 1)
    "n": (1.0, 4.0),  # two to four operator sites
    "beta": (0.1, 1.0),  # ssrA-tagged repressors: protein half-life ~2-20 min
}
CIRCUIT_START_BOX = (1e-3, 1000.0)  # hidden mRNA / protein cannot exceed alpha_max + alpha_0_max
CIRCUIT_SPECIES = ["m_lacI", "m_tetR", "m_cI", "p_LacI", "p_TetR", "p_CI"]

# torchdiffeq settings for fitting. Data are generated with tight dopri5
# (reporter_sim.RTOL); fitting with a different, cheaper solver avoids fitting
# the solver's own error. "rk4" steps are in model time units.
SOLVER = {"method": "rk4", "step": 0.35}


def solve(f, y0, t):
    """``odeint`` with :data:`SOLVER`."""
    if SOLVER["method"] == "rk4":
        return odeint(f, y0, t, method="rk4", options={"step_size": SOLVER["step"]})
    return odeint(f, y0, t, method=SOLVER["method"], rtol=SOLVER["rtol"], atol=SOLVER["atol"])


@dataclass
class Problem:
    t: torch.Tensor  # frame times (G,)
    y: torch.Tensor  # calibrated readout of the observed channels, K_M units (G, k)
    channels: list[int]  # observed channels, indices into reporter_sim.CHANNELS
    present: list[int]  # reporters built (they titrate and have parameters)
    tau: list[float]  # known titration per reporter (all three; only present ones act)

    def __post_init__(self):
        self.mask = mask(self.present)
        self.observed_state = [CHANNEL_STATE[c] for c in self.channels]
        self.tau_t = torch.tensor(self.tau, dtype=torch.float64)
        self.range = (self.y.max(0).values - self.y.min(0).values).clamp_min(1e-9)
        def first_frame_box(state_index, default):
            """Observed species: its first frame +- 10% of range; hidden: ``default``."""
            if state_index not in self.observed_state:
                return default
            c = self.observed_state.index(state_index)
            first, rng = float(self.y[0, c]), float(self.range[c])
            return (max(first - 0.1 * rng, 1e-3), first + 0.1 * rng)

        def reporter_range(j):
            """Lowest and highest reading of reporter j's own channel."""
            c = self.observed_state.index(R.F_IDX[j])
            return max(float(self.y[:, c].min()), 1e-3), float(self.y[:, c].max())

        boxes = list(CIRCUIT_BOX.values())
        for j in self.present:
            low, high = reporter_range(j)
            boxes += [(0.8 * high, 10 * high),  # a: F <= a at full derepression, so a >~ the top reading
                      (1e-4 * high, low)]  # a_0: F >~ a_0 at full repression, so a_0 <= the lowest reading
        boxes += [first_frame_box(i, CIRCUIT_START_BOX) for i in range(6)]
        for j in self.present:
            low, high = reporter_range(j)
            # r swings between ~a_0 (repressed) and ~a (derepressed): the same range as
            # production. The FP is a smoothed view of r, so the readings don't bound it.
            boxes += [(1e-4 * high, 10 * high),  # r(0)
                      (5e-6 * high, 0.5 * high),  # D(0): dark FP ~ delta / (k + delta) ~ 5% of r
                      first_frame_box(R.F_IDX[j], (0.1 * low, 3 * high))]  # F(0)
        self.lo = torch.tensor([math.log(lo) for lo, _ in boxes], dtype=torch.float64)
        self.hi = torch.tensor([math.log(hi) for _, hi in boxes], dtype=torch.float64)

    @property
    def names(self) -> list[str]:
        out = list(CIRCUIT_BOX)
        for j in self.present:
            r = R.REPORTERS[j]
            out += [f"a_{r}", f"a0_{r}"]
        out += [f"{s}(0)" for s in CIRCUIT_SPECIES]
        for j in self.present:
            r = R.REPORTERS[j]
            out += [f"r_{r}(0)", f"D_{r}(0)", f"F_{r}(0)"]
        return out

    # theta = log values, inside the box; z = logit(position in the box) is unconstrained.
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
        """Model parameters and the starting state ``(..., 15)``."""
        v = self.values(theta)
        one, zero = torch.ones_like(v["alpha"]), torch.zeros_like(v["alpha"])

        def per_reporter(key, absent):  # 3-vector; absent reporters get a dummy value
            return torch.stack([v[f"{key}_{r}"] if j in self.present else absent
                                for j, r in enumerate(R.REPORTERS)], dim=-1)

        p = {"alpha": v["alpha"], "alpha_0": v["alpha_0"], "n": v["n"], "beta": v["beta"],
             "a": per_reporter("a", one), "a_0": per_reporter("a0", zero),
             "tau": self.tau_t.expand(*theta.shape[:-1], 3)}

        start = [v[f"{s}(0)"] for s in CIRCUIT_SPECIES]
        for j, r in enumerate(R.REPORTERS):  # absent reporters start (and stay) at zero
            if j in self.present:
                start += [v[f"r_{r}(0)"], v[f"D_{r}(0)"], v[f"F_{r}(0)"]]
            else:
                start += [zero, zero, zero]
        return p, torch.stack(start, dim=-1)

    def predict(self, theta, n_frames=None, t=None):
        """Model readout ``(..., G, k)`` at the first ``n_frames`` frames (or at times ``t``)."""
        p, y0 = self.params_and_start(theta)
        if t is None:
            t = self.t if n_frames is None else self.t[:n_frames]
        traj = solve(lambda _t, y: R.rhs(y, p, self.mask), y0, t)  # (G, ..., 15)
        return traj[..., self.observed_state].movedim(0, -2)

    def loss(self, theta, n_frames=None, kind: str = "mse"):
        """Per batch element, on range-normalised residuals r = (prediction - data) / range.

        ``"mse"``: mean r^2.  ``"l1"``: mean |r|.  ``"huber"``: quadratic for
        |r| < 0.05 (a few noise levels), linear beyond - robust to outlying
        frames and to the long tails a phase error produces.
        """
        pred = self.predict(theta, n_frames)
        y = self.y if n_frames is None else self.y[:n_frames]
        r = (pred - y) / self.range
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
                 lr_final: float = 2.5e-4, stages=(0.25, 0.5, 1.0), seed: int = 0,
                 loss: str = "mse", optimizer: str = "adam", schedule: str = "exp",
                 log_every: int = 0, return_all: bool = False,
                 history: list | None = None, history_every: int = 25):
    """Backprop through the ODE: a first-order optimiser on a batch of prior draws.

    Args:
        loss: ``"mse"``, ``"l1"`` or ``"huber"`` (see :meth:`Problem.loss`).
        optimizer: ``"adam"``, ``"adamw"`` (no weight decay: parameters are
            physical), ``"rmsprop"`` or ``"sgd"`` (Nesterov momentum 0.9).
        schedule: learning rate from ``lr`` to ``lr_final``, ``"exp"``
            (exponential) or ``"cosine"``.

    The loss sees the first quarter of the frames, then half, then all of
    them: a small period error grows into a phase error over long windows,
    which flattens the loss. The best restart is chosen by MSE on the full
    record whatever the training loss, so variants are compared on one scale.
    Returns the best restart's ``theta`` (every restart's, ``(restarts, D)``,
    with ``return_all``) and every restart's final MSE. If ``history`` is a
    list, every ``history_every`` epochs it gets ``(epoch, full-record MSE per
    restart, theta per restart)``.
    """
    g = torch.Generator().manual_seed(seed)
    z = problem.to_z(problem.sample_prior(restarts, g)).requires_grad_(True)
    opt = {"adam": lambda: torch.optim.Adam([z], lr=lr),
           "adamw": lambda: torch.optim.AdamW([z], lr=lr, weight_decay=0.0),
           "rmsprop": lambda: torch.optim.RMSprop([z], lr=lr),
           "sgd": lambda: torch.optim.SGD([z], lr=lr, momentum=0.9, nesterov=True)}[optimizer]()
    if schedule == "exp":
        sched = torch.optim.lr_scheduler.ExponentialLR(opt, (lr_final / lr) ** (1.0 / max(epochs - 1, 1)))
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr_final)
    windows = [max(4, math.ceil(f * problem.t.numel())) for f in stages]
    for epoch in range(epochs):
        window = windows[min(epoch * len(windows) // epochs, len(windows) - 1)]
        if history is not None and epoch % history_every == 0:
            with torch.no_grad():
                theta = problem.from_z(z)
                history.append((epoch, torch.nan_to_num(problem.loss(theta), nan=math.inf), theta.clone()))
        opt.zero_grad()
        per_restart = problem.loss(problem.from_z(z), window, kind=loss)
        ok = torch.isfinite(per_restart)
        # Restarts are independent: the summed loss gives each its own gradient.
        # A restart that goes non-finite simply stops moving.
        torch.where(ok, per_restart, torch.zeros_like(per_restart)).sum().backward()
        z.grad = torch.nan_to_num(z.grad, nan=0.0, posinf=0.0, neginf=0.0)
        opt.step()
        sched.step()
        if log_every and epoch % log_every == 0:
            print(f"epoch {epoch:5d}  window {window:4d}  best loss {float(per_restart.detach().min()):.5f}", flush=True)
    with torch.no_grad():
        final = torch.nan_to_num(problem.loss(problem.from_z(z)), nan=math.inf)
        if history is not None:
            history.append((epochs, final, problem.from_z(z).clone()))
    if return_all:
        return problem.from_z(z).detach(), final
    best = int(final.argmin())
    return problem.from_z(z[best]).detach(), final
