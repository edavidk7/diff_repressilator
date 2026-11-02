"""Synthetic cells: the reporter model (reporters.py) simulated as ODE or exact Gillespie, imaged by a camera.

Units: time in mRNA lifetimes (2.885 min), protein in K_M. A cell starts after a burn-in plus a random
phase, is simulated at 30-s resolution and imaged every 5 min; frames are calibrated back to K_M units.
"""

import math
from dataclasses import dataclass, field

import torch
from torchdiffeq import odeint

from differentiable_cell import reporters as R
from differentiable_cell.gillespie import ssa_exact

FRAME_MIN: float = 5.0
DT_FRAME: float = FRAME_MIN / R.TAU_M_MIN  # 1.733 model units per frame
BURN_IN: float = 150.0  # >= 5 reporter decay times
PHASE_MAX: float = 60.0  # > one period (~42 units)
REFERENCE = torch.tensor([0, 0, 0, 5.0, 0, 15.0] + [0.0] * 9, dtype=torch.float64)
RTOL, ATOL = 1e-8, 1e-10

# Observable channels: (name, state index, FP colour). Reporters show their mature FP; a fusion shows the
# repressor itself (idealised: the fusion does not perturb the circuit).
CHANNELS = [("lac", R.F_IDX[0], "lac"), ("tet", R.F_IDX[1], "tet"), ("cI", R.F_IDX[2], "cI"),
            ("LacI-fusion", R.P_IDX[0], "lac"), ("TetR-fusion", R.P_IDX[1], "tet"),
            ("CI-fusion", R.P_IDX[2], "cI")]
CHANNEL_STATE = [c[1] for c in CHANNELS]


def box1_truth() -> dict:
    """Box 1 circuit (Elowitz & Leibler 2000). Reporters on p15A (~3x pSC101 copies, so a = 3 alpha),
    titration tau = 0.75 (~30 operator sites per repressor against K_M ~ 40 monomers)."""
    return dict(alpha=216.0, alpha_0=0.216, n=2.0, beta=0.2,
                a=[648.0] * 3, a_0=[0.648] * 3, tau=[0.75] * 3)


def params(v: dict) -> dict:
    return {k: torch.as_tensor(val, dtype=torch.float64) for k, val in v.items()}


def mask(present: list[int]) -> torch.Tensor:
    m = torch.zeros(3, dtype=torch.float64)
    m[list(present)] = 1.0
    return m


def simulate_ode(p: dict, present: torch.Tensor, y0: torch.Tensor, t: torch.Tensor,
                 rtol: float = RTOL, atol: float = ATOL) -> torch.Tensor:
    """Deterministic cell from ``y0`` at times ``t`` -> ``(len(t), ..., 15)``; differentiable."""
    return odeint(lambda _t, y: R.rhs(y, p, present), y0, t, method="dopri5",
                  rtol=rtol, atol=atol)


def start_state(p: dict, present: torch.Tensor, phase: float | torch.Tensor,
                burn_in: float = BURN_IN) -> torch.Tensor:
    """State after ``burn_in + phase`` from the fixed reference state."""
    t = torch.stack([torch.zeros((), dtype=torch.float64),
                     torch.as_tensor(burn_in, dtype=torch.float64) + phase])
    return simulate_ode(p, present, REFERENCE, t)[-1]


def simulate_ssa(p: dict, present: torch.Tensor, y0: torch.Tensor, t: torch.Tensor,
                 omega: float, generator: torch.Generator | None = None,
                 settle: float = 30.0) -> torch.Tensor:
    """One exact Gillespie cell in K_M units at times ``t`` -> ``(len(t), 15)``. Runs ``settle`` units
    first so the cell has its own stochastic state, not the deterministic start."""
    scales = R.count_scales(p, omega)
    S = R.stoichiometry()
    prop = lambda x: R.propensities(x, p, present, omega)
    x = (y0 * scales).round().unsqueeze(0)
    if settle > 0:
        x = ssa_exact(prop, S, x, torch.tensor([0.0, settle], dtype=torch.float64),
                      generator=generator)[-1]
    return ssa_exact(prop, S, x, t - t[0], generator=generator)[:, 0] / scales


@dataclass
class Camera:
    """Readout in AU: Poisson photons (brightness relative to GFPmut3) x gain + offset + read noise."""

    gain: float = 2.0
    photons_per_molecule: float = 0.5
    offset: float = 100.0
    read_noise: float = 5.0

    def eps(self) -> torch.Tensor:
        """Photons per molecule per frame, per channel."""
        return torch.tensor([self.photons_per_molecule * R.FP_BRIGHTNESS[colour] / R.FP_BRIGHTNESS["tet"]
                             for _, _, colour in CHANNELS], dtype=torch.float64)

    def au_per_km(self, omega: float) -> torch.Tensor:
        """AU per K_M of fluorescent molecule, per channel."""
        return self.gain * self.eps() * omega

    def read(self, F: torch.Tensor, omega: float, noise: bool,
             generator: torch.Generator | None = None) -> torch.Tensor:
        """AU from fluorescent molecules ``F (..., 6)`` in K_M; ``noise=False``: the expected value."""
        if not noise:
            return F * self.au_per_km(omega) + self.offset
        photons = torch.poisson((F * omega * self.eps()).clamp_min(0.0), generator=generator)
        read = self.read_noise * torch.randn(photons.shape, generator=generator, dtype=torch.float64)
        return self.gain * photons + self.offset + read


@dataclass
class CellData:
    """One cell: the hidden full-resolution state and the recorded frames."""

    fine_t: torch.Tensor  # (Gf,) model time, 0 = first frame
    fine_state: torch.Tensor  # (Gf, 15) K_M units, before the camera
    t: torch.Tensor  # (G,) frame times
    y: torch.Tensor  # (G, 6) calibrated frames of all CHANNELS, K_M units
    y_au: torch.Tensor  # (G, 6) raw camera readout, AU
    truth: dict
    meta: dict = field(default_factory=dict)


def generate(truth: dict, present: list[int], kind: str = "ode", noise: bool = True,
             hours: float = 10.0, omega: float = 40.0, seed: int = 0,
             camera: Camera | None = None, fine_per_frame: int = 10) -> CellData:
    """One cell with reporters ``present`` (indices into reporters.REPORTERS) imaged for ``hours``.

    ``kind``: "ode" or "ssa" (exact Gillespie at system size ``omega``); ``seed`` sets the start phase,
    the Gillespie path and the camera noise. All six channels are returned; use only those the
    construct has.
    """
    camera = camera or Camera()
    g = torch.Generator().manual_seed(seed)
    p = params(truth)
    m = mask(present)
    phase = float(torch.rand((), generator=g, dtype=torch.float64)) * PHASE_MAX
    y0 = start_state(p, m, phase)
    n_frames = int(round(hours * 60 / FRAME_MIN))
    fine_t = torch.arange(n_frames * fine_per_frame + 1, dtype=torch.float64) * (DT_FRAME / fine_per_frame)
    if kind == "ode":
        state = simulate_ode(p, m, y0, fine_t)
    elif kind == "ssa":
        state = simulate_ssa(p, m, y0, fine_t, omega, generator=g)
    else:
        raise ValueError(f"kind must be 'ode' or 'ssa', got {kind!r}")
    frames = state[::fine_per_frame]
    y_au = camera.read(frames[:, CHANNEL_STATE], omega, noise, generator=g)
    # gain and offset are known: calibrate back to K_M units (noise included)
    y = (y_au - camera.offset) / camera.au_per_km(omega)
    return CellData(fine_t=fine_t, fine_state=state, t=fine_t[::fine_per_frame], y=y, y_au=y_au,
                    truth=truth, meta=dict(kind=kind, noise=noise, seed=seed, phase=phase,
                                           omega=omega, present=list(present), hours=hours,
                                           au_per_km=camera.au_per_km(omega).tolist()))
