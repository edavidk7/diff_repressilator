"""Simulation and data generation for the three-reporter repressilator.

The model itself (vector field, reaction network) is in
:mod:`differentiable_cell.reporters`; this module turns it into data:

* :func:`simulate_ode` - the deterministic cell, integrated with
  ``torchdiffeq`` (dopri5, tight tolerances) at arbitrary times;
* :func:`simulate_ssa` - one exact Gillespie cell at system size ``omega``;
* :func:`start_state` - where imaging starts: the cell has been oscillating
  for a while, so it is run from a fixed reference state through a burn-in
  and a random extra time (its phase);
* :class:`Camera` - fluorescence in arbitrary units, with a noise switch;
* :func:`generate` - one cell, at full resolution, plus the 5-min frames an
  experiment would record, calibrated back to model units (the camera's
  gain and offset are known).

Units: time in mRNA lifetimes (2.885 min), circuit protein in units of K_M,
reporter protein in K_M units until the camera converts it to arbitrary
units.
"""

import math
from dataclasses import dataclass, field

import torch
from torchdiffeq import odeint

from differentiable_cell import reporters as R
from differentiable_cell.gillespie import ssa_exact

FRAME_MIN: float = 5.0
DT_FRAME: float = FRAME_MIN / R.TAU_M_MIN  # 1.733 model units per frame
BURN_IN: float = 150.0  # >= 5 reporter decay times (tagged FPs, 40-60 min)
PHASE_MAX: float = 60.0  # > one period at the Box 1 values (~42 units)
REFERENCE = torch.tensor([0, 0, 0, 5.0, 0, 15.0] + [0.0] * 9, dtype=torch.float64)
RTOL, ATOL = 1e-8, 1e-10

# What a microscope can see: the three reporters' mature FP, and the three
# repressors themselves if they are fused to an FP (idealised: the fusion does
# not perturb the circuit). Each channel: (name, state index, FP colour).
CHANNELS = [("lac", R.F_IDX[0], "lac"), ("tet", R.F_IDX[1], "tet"), ("cI", R.F_IDX[2], "cI"),
            ("LacI-fusion", R.P_IDX[0], "lac"), ("TetR-fusion", R.P_IDX[1], "tet"),
            ("CI-fusion", R.P_IDX[2], "cI")]
CHANNEL_STATE = [c[1] for c in CHANNELS]


def box1_truth() -> dict:
    """Box 1 circuit (Elowitz & Leibler 2000) with three reporters on p15A.

    Reporter promoters at ~3x the circuit's output (p15A ~15 copies vs the
    repressilator's pSC101 ~5), leak 1e-3 of that, and titration
    ``tau = 0.75`` per reporter (~30 operator sites per repressor against
    K_M ~ 40 monomers).
    """
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
    """The deterministic cell from ``y0`` at times ``t`` (``t[0]`` is ``y0``'s time).

    Returns ``(len(t), ..., 15)``.  Differentiable in ``p`` and ``y0``.
    """
    return odeint(lambda _t, y: R.rhs(y, p, present), y0, t, method="dopri5",
                  rtol=rtol, atol=atol)


def start_state(p: dict, present: torch.Tensor, phase: float | torch.Tensor,
                burn_in: float = BURN_IN) -> torch.Tensor:
    """The state after a burn-in of ``burn_in`` plus ``phase`` from the reference state."""
    t = torch.stack([torch.zeros((), dtype=torch.float64),
                     torch.as_tensor(burn_in, dtype=torch.float64) + phase])
    return simulate_ode(p, present, REFERENCE, t)[-1]


def simulate_ssa(p: dict, present: torch.Tensor, y0: torch.Tensor, t: torch.Tensor,
                 omega: float, generator: torch.Generator | None = None,
                 settle: float = 30.0) -> torch.Tensor:
    """One exact Gillespie cell in K_M units at times ``t``, ``(len(t), 15)``.

    ``y0`` is rounded to whole molecules and run for ``settle`` time units
    first, so the recorded cell has its own stochastic state rather than the
    deterministic one it was started from.
    """
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
    """Fluorescence readout in arbitrary units (AU).

    Each mature FP molecule yields ``eps_j`` detected photons per frame
    (``photons_per_molecule`` scaled by the FP's FPbase brightness relative
    to GFPmut3); photons are Poisson; the camera multiplies by ``gain`` and
    adds ``offset`` plus Gaussian read noise.  With ``noise=False`` the
    readout is the expected value.
    """

    gain: float = 2.0
    photons_per_molecule: float = 0.5
    offset: float = 100.0
    read_noise: float = 5.0

    def eps(self) -> torch.Tensor:
        """Photons per molecule per frame for each of the six :data:`CHANNELS`."""
        return torch.tensor([self.photons_per_molecule * R.FP_BRIGHTNESS[colour] / R.FP_BRIGHTNESS["tet"]
                             for _, _, colour in CHANNELS], dtype=torch.float64)

    def au_per_km(self, omega: float) -> torch.Tensor:
        """AU per K_M unit of fluorescent molecule, for each channel."""
        return self.gain * self.eps() * omega

    def read(self, F: torch.Tensor, omega: float, noise: bool,
             generator: torch.Generator | None = None) -> torch.Tensor:
        """Readout ``(..., 6)`` in AU from fluorescent molecules ``F`` ``(..., 6)`` in K_M units."""
        if not noise:
            return F * self.au_per_km(omega) + self.offset
        photons = torch.poisson((F * omega * self.eps()).clamp_min(0.0), generator=generator)
        read = self.read_noise * torch.randn(photons.shape, generator=generator, dtype=torch.float64)
        return self.gain * photons + self.offset + read


@dataclass
class CellData:
    """One cell: the full-resolution truth and the frames a microscope records."""

    fine_t: torch.Tensor  # (Gf,) model units, 0 = first frame
    fine_state: torch.Tensor  # (Gf, 15) K_M units, noise-free w.r.t. the camera
    t: torch.Tensor  # (G,) frame times
    y: torch.Tensor  # (G, 6) calibrated readout of the six CHANNELS, K_M units, camera noise
    y_au: torch.Tensor  # (G, 6) the raw camera readout in AU
    truth: dict
    meta: dict = field(default_factory=dict)


def generate(truth: dict, present: list[int], kind: str = "ode", noise: bool = True,
             hours: float = 10.0, omega: float = 40.0, seed: int = 0,
             camera: Camera | None = None, fine_per_frame: int = 10) -> CellData:
    """Simulate one cell with reporters ``present`` for ``hours`` of imaging.

    Args:
        truth: parameters (see :func:`box1_truth`).
        present: indices into :data:`reporters.REPORTERS` of the reporters
            built (titration acts only through those).
        kind: ``"ode"`` (deterministic cell) or ``"ssa"`` (exact Gillespie).
        noise: camera noise on or off.
        hours: recording length.
        omega: system size, K_M in molecules.
        seed: sets the start phase, the Gillespie path and the camera noise.
        camera: readout model.
        fine_per_frame: full-resolution points per 5-min frame.

    Returns:
        :class:`CellData` with the full-resolution state and the frames.
        All six :data:`CHANNELS` are returned; a panel selects the ones its
        construct has. Reporter channels not in ``present`` must not be used
        (those reporters would not exist).
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
    # Calibrated: the camera's gain and offset are known, so the readout converts
    # back to model units (K_M units of mature FP) - noise included.
    y = (y_au - camera.offset) / camera.au_per_km(omega)
    return CellData(fine_t=fine_t, fine_state=state, t=fine_t[::fine_per_frame], y=y, y_au=y_au,
                    truth=truth, meta=dict(kind=kind, noise=noise, seed=seed, phase=phase,
                                           omega=omega, present=list(present), hours=hours,
                                           au_per_km=camera.au_per_km(omega).tolist()))
