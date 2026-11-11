"""The four further circuits and the dilution-only repressilator (paper Sec. "Circuits, simulation and imaging").

A ``Circuit`` is a vector field with true parameters, prior boxes and observed species; ``CircuitProblem`` gives it
the interface of ``fit.Problem``, so every estimator runs on it unchanged.
"""

import math
from dataclasses import dataclass, field
from typing import Callable

import torch
from torchdiffeq import odeint

from differentiable_cell.fit import solve


def pulse(t, on: float, off: float, width: float = 0.2):
    """Smooth 0 -> 1 -> 0 inducer between ``on`` and ``off``."""
    return torch.sigmoid((t - on) / width) - torch.sigmoid((t - off) / width)


def hill(x, n):
    """x^n / (1 + x^n) for x >= 0."""
    xn = x.clamp_min(0.0) ** n
    return xn / (1 + xn)


@dataclass
class Circuit:
    name: str
    species: list[str]
    params: dict[str, tuple[float, float, float]]  # name: (truth, low, high)
    rhs: Callable  # rhs(t, y (..., S), p: dict of (...)) -> (..., S)
    observed: list[int]
    start: list[float]  # typical starting state
    start_box: tuple[float, float] = (1e-3, 1e3)  # for hidden species
    minutes_per_unit: float = 2.0 / math.log(2.0)
    frame_min: float = 5.0
    hours: float = 10.0
    burn_in: float = 0.0  # oscillators: run this long plus a random phase before imaging
    phase_max: float = 0.0
    notes: str = ""

    @property
    def dt_frame(self) -> float:
        return self.frame_min / self.minutes_per_unit

    def truth(self) -> dict:
        return {k: torch.tensor(v[0], dtype=torch.float64) for k, v in self.params.items()}


def simulate(c: Circuit, p: dict, y0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return odeint(lambda s, y: c.rhs(s, y, p), y0, t, method="dopri5", rtol=1e-8, atol=1e-10)


@dataclass
class CellData:
    t: torch.Tensor  # frame times
    y: torch.Tensor  # noisy frames of the observed species (G, k)
    fine_t: torch.Tensor
    fine_state: torch.Tensor  # (Gf, S) hidden noise-free state
    meta: dict = field(default_factory=dict)


def generate(c: Circuit, seed: int, noise: float = 0.03, fine_per_frame: int = 10) -> CellData:
    """One synthetic cell: start jittered by +-20% (oscillators: burn-in and a random phase), frames, noise."""
    g = torch.Generator().manual_seed(seed)
    p = c.truth()
    y0 = torch.tensor(c.start, dtype=torch.float64) * torch.exp(0.2 * (2 * torch.rand(len(c.start), generator=g, dtype=torch.float64) - 1))
    if c.burn_in:
        phase = float(torch.rand((), generator=g, dtype=torch.float64)) * c.phase_max
        y0 = simulate(c, p, y0, torch.tensor([0.0, c.burn_in + phase], dtype=torch.float64))[-1]
    n_frames = int(round(c.hours * 60 / c.frame_min))
    fine_t = torch.arange(n_frames * fine_per_frame + 1, dtype=torch.float64) * (c.dt_frame / fine_per_frame)
    state = simulate(c, p, y0, fine_t)
    frames = state[::fine_per_frame, c.observed]
    rng = frames.max(0).values - frames.min(0).values
    y = frames + noise * rng * torch.randn(frames.shape, generator=g, dtype=torch.float64)
    return CellData(t=fine_t[::fine_per_frame], y=y, fine_t=fine_t, fine_state=state, meta={"seed": seed})


class CircuitProblem:
    """Same interface as ``fit.Problem`` for a ``Circuit`` and one cell's frames."""

    def __init__(self, circuit: Circuit, t: torch.Tensor, y: torch.Tensor):
        self.c, self.t, self.y = circuit, t, y
        self.observed_state = circuit.observed
        self.range = (y.max(0).values - y.min(0).values).clamp_min(1e-9)
        boxes = [(lo, hi) for _, lo, hi in circuit.params.values()]
        for i in range(len(circuit.species)):
            if i in circuit.observed:
                k, first = circuit.observed.index(i), float(y[0, circuit.observed.index(i)])
                rng = float(self.range[k])
                boxes.append((max(first - 0.1 * rng, 1e-4), max(first + 0.1 * rng, 2e-4)))
            else:
                boxes.append(circuit.start_box)
        self.lo = torch.tensor([math.log(lo) for lo, _ in boxes], dtype=torch.float64)
        self.hi = torch.tensor([math.log(hi) for _, hi in boxes], dtype=torch.float64)

    @property
    def names(self) -> list[str]:
        return list(self.c.params) + [f"{s}(0)" for s in self.c.species]

    def from_z(self, z):
        return self.lo + torch.sigmoid(z) * (self.hi - self.lo)

    def to_z(self, theta):
        u = ((theta - self.lo) / (self.hi - self.lo)).clamp(1e-9, 1 - 1e-9)
        return torch.log(u) - torch.log1p(-u)

    def sample_prior(self, n, generator=None):
        return self.lo + torch.rand(n, len(self.lo), generator=generator, dtype=torch.float64) * (self.hi - self.lo)

    def values(self, theta) -> dict[str, torch.Tensor]:
        return {name: theta[..., i].exp() for i, name in enumerate(self.names)}

    def predict(self, theta, n_frames=None, t=None):
        n_p = len(self.c.params)
        p = {k: theta[..., i].exp() for i, k in enumerate(self.c.params)}
        y0 = theta[..., n_p:].exp()
        if t is None:
            t = self.t if n_frames is None else self.t[:n_frames]
        traj = solve(lambda s, y: self.c.rhs(s, y, p), y0, t)  # (G, ..., S)
        return traj[..., self.observed_state].movedim(0, -2)

    def loss(self, theta, n_frames=None, kind: str = "mse"):  # paper Eq. (5)
        y = self.y if n_frames is None else self.y[:n_frames]
        r = (self.predict(theta, n_frames) - y) / self.range
        if kind == "huber":
            return torch.nn.functional.huber_loss(r, torch.zeros_like(r), reduction="none", delta=0.05).mean((-1, -2))
        return (r**2).mean((-1, -2)) if kind == "mse" else r.abs().mean((-1, -2))

    def true_theta(self, data: CellData) -> torch.Tensor:
        vals = [v[0] for v in self.c.params.values()] + data.fine_state[0].tolist()
        return torch.tensor([math.log(max(v, 1e-12)) for v in vals])


# --------------------------------------------------------------------------- the circuits
# Time in mRNA lifetimes (2.885 min) unless noted; protein in units of its repression threshold.


def _nar(t, y, p):
    """Negative autoregulation: TetR-GFP represses its own promoter; aTc (on 3-6 h) relieves repression."""
    m, prot = y[..., 0], y[..., 1]
    free = prot / (1 + 20 * pulse(t, 62.4, 124.8))  # aTc frees 95% of the repressor (known from design)
    dm = -m + p["alpha"] / (1 + free ** p["n"]) + p["alpha_0"]
    dp = -p["beta"] * (prot - m)
    return torch.stack([dm, dp], -1)


NAR = Circuit(
    name="negative autoregulation", species=["m", "p"],
    params={"alpha": (50.0, 5.0, 500.0), "alpha_0": (0.05, 1e-3, 1.0), "n": (2.0, 1.0, 4.0), "beta": (0.05, 0.01, 1.0)},
    rhs=_nar, observed=[1], start=[0.5, 0.5],
    notes="TetR-GFP fusion observed (protein lifetime ~1 h, dilution); starts near zero, aTc pulse 3-6 h.")


def _toggle(t, y, p):
    """Toggle switch (Gardner et al. 2000): u = LacI, v = TetR. IPTG pulse (2-4 h) inactivates LacI,
    aTc pulse (10-12 h) inactivates TetR. Time in protein lifetimes (dilution, 30 min)."""
    u, v = y[..., 0], y[..., 1]
    u_free = u / (1 + 30 * pulse(t, 4.0, 8.0))
    v_free = v / (1 + 30 * pulse(t, 20.0, 24.0))
    du = p["alpha_1"] / (1 + v_free ** p["beta"]) - u
    dv = p["alpha_2"] / (1 + u_free ** p["gamma"]) - v
    return torch.stack([du, dv], -1)


TOGGLE = Circuit(
    name="toggle switch", species=["u", "v"],
    params={"alpha_1": (15.0, 2.0, 200.0), "alpha_2": (12.0, 2.0, 200.0), "beta": (2.5, 1.0, 4.0), "gamma": (2.0, 1.0, 4.0)},
    rhs=_toggle, observed=[0], start=[15.0, 0.05], start_box=(1e-3, 300.0), minutes_per_unit=30.0, hours=15.0,
    notes="GFP co-expressed with LacI observed; starts in the LacI-high state; IPTG 2-4 h, aTc 10-12 h.")


def _goodwin(t, y, p):
    """Goodwin oscillator: x (mRNA) -> y (protein) -> z (repressor) represses x. Time unit 8 min (period ~2 h)."""
    x, yy, z = y[..., 0], y[..., 1], y[..., 2]
    dx = p["a"] / (1 + z ** p["n"]) - p["b1"] * x
    dy = x - p["b2"] * yy
    dz = yy - p["b3"] * z
    return torch.stack([dx, dy, dz], -1)


GOODWIN = Circuit(
    name="Goodwin oscillator", species=["x", "y", "z"],
    params={"a": (20.0, 1.0, 500.0), "n": (10.0, 4.0, 16.0), "b1": (0.2, 0.05, 2.0), "b2": (0.3, 0.05, 2.0),
            "b3": (0.45, 0.05, 2.0)},  # distinct rates: at equal ones the data's symmetry in b hides in 2nd order
    rhs=_goodwin, observed=[2], start=[1.0, 1.0, 1.0], minutes_per_unit=8.0, burn_in=300.0, phase_max=25.0,
    notes="Repressor z observed (a fusion); burn-in to the limit cycle plus a random phase.")


def _iffl(t, y, p):
    """Incoherent type-1 feed-forward loop: input S activates X; X activates Y and Z; Y represses Z.
    S on 1-6 h. Time in protein lifetimes (30 min)."""
    Y, Z = y[..., 0], y[..., 1]
    X = 3.0 * pulse(t, 2.0, 12.0)  # active X, known from the input (S) schedule
    dY = p["beta_y"] * hill(X / p["K_xy"], 2.0) - Y
    dZ = p["beta_z"] * hill(X / p["K_xz"], 2.0) / (1 + (Y / p["K_yz"]) ** 2) - p["alpha_z"] * Z
    return torch.stack([dY, dZ], -1)


IFFL = Circuit(
    name="incoherent feed-forward loop", species=["Y", "Z"],
    params={"beta_y": (2.0, 0.2, 20.0), "K_xy": (1.0, 0.1, 10.0), "beta_z": (5.0, 0.5, 50.0), "K_xz": (0.5, 0.05, 5.0),
            "K_yz": (0.5, 0.05, 5.0), "alpha_z": (1.0, 0.2, 5.0)},
    rhs=_iffl, observed=[0, 1], start=[0.01, 0.01], minutes_per_unit=30.0, hours=8.0,
    notes="Y and Z observed (two reporters); input pulse 1-6 h gives a Z pulse (adaptation).")

CIRCUITS = {"nar": NAR, "toggle": TOGGLE, "goodwin": GOODWIN, "iffl": IFFL}


def _repressilator_dilution(t, y, p):
    """The improved repressilator (Potvin-Trottier et al. 2016, LPT117): no protein degradation, so every
    protein (repressors and reporters) is lost by dilution only, at rate beta; a TetR titration sponge raises
    TetR's threshold to 1 + tau_tet. State: m (lacI, tetR, cI), P (LacI, TetR, CI), then r, D, F for the
    reporters of LacI (CFP), TetR (Venus) and CI (mKate2)."""
    m, P, r, D, F = (y[..., 3 * i:3 * i + 3] for i in range(5))
    K = torch.stack(torch.broadcast_tensors(torch.ones_like(p["tau_tet"]), 1 + p["tau_tet"],
                                            torch.ones_like(p["tau_tet"])), -1)
    h = 1 / (1 + (P.clamp_min(0.0) / K) ** p["n"].unsqueeze(-1))  # promoter activity per repressor
    beta = p["beta"].unsqueeze(-1)
    a = torch.stack([p["a_lac"], p["a_tet"], p["a_cI"]], -1)
    a0 = torch.stack([p["a0_lac"], p["a0_tet"], p["a0_cI"]], -1)
    k = torch.stack([p["k_lac"], p["k_tet"], p["k_cI"]], -1)
    dm = -m + p["alpha"].unsqueeze(-1) * h[..., [2, 0, 1]] + p["alpha_0"].unsqueeze(-1)  # lacI <- CI, tetR <- LacI, cI <- TetR
    dP = -beta * (P - m)
    dr = -r + a * h + a0  # each reporter copies its repressor's target promoter
    dD = beta * r - (k + beta) * D
    dF = k * D - beta * F
    return torch.cat([dm, dP, dr, dD, dF], -1)


_mid = lambda lo, hi: (math.sqrt(lo * hi), lo, hi)  # real data: no truth, only a box
REPRESSILATOR_DILUTION = Circuit(
    name="improved repressilator (dilution only, TetR sponge)",
    species=[f"{k}_{g}" for k in ("m", "P", "r", "D", "F") for g in ("lac", "tet", "cI")],
    params={"alpha": _mid(10.0, 1e4), "alpha_0": _mid(1e-3, 10.0), "n": _mid(1.0, 4.0), "beta": _mid(0.01, 0.5),
            "tau_tet": _mid(0.1, 100.0),
            **{f"a_{g}": _mid(1.0, 1e4) for g in ("lac", "tet", "cI")},
            **{f"a0_{g}": _mid(1e-3, 50.0) for g in ("lac", "tet", "cI")},
            **{f"k_{g}": _mid(0.01, 2.0) for g in ("lac", "tet", "cI")}},  # maturation half-time ~1-200 min
    rhs=_repressilator_dilution, observed=[12, 13, 14], start=[1.0] * 15, start_box=(1e-3, 1e4), frame_min=8.0,
    notes="Reporters: CFP (P_LlacO1, LacI), Venus (P_LtetO1, TetR), mKate2 (P_R, CI); 8-min frames.")
