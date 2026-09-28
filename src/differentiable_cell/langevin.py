"""Chemical Langevin simulation of the repressilator.

The chemical Langevin equation (CLE) is the diffusion approximation to the
reaction network of :mod:`differentiable_cell.units`: in molecule counts,
over a step ``dt``::

    X <- | X + S^T a(X) dt + S^T (sqrt(a(X) dt) * xi) |,    xi ~ N(0, I)

with ``S`` the ``(reactions, species)`` stoichiometry and ``a`` the
propensities.  Each reaction gets its own Gaussian increment, so the noise
has the correct covariance ``S^T diag(a) S``.  The CLE can step a count
below zero, which a concentration cannot do; the absolute value reflects
such a step back into the positive orthant.

Everything is an ordinary tensor op, so gradients flow through the whole
path.  Pass the same ``noise`` tensor (or a generator in the same state) to
two calls and they share their random numbers: differentiating then gives a
pathwise (reparameterised) gradient of that noise realisation.
"""

from typing import Callable

import torch

from differentiable_cell.data import RepressilatorParams, RepressilatorState
from differentiable_cell.units import (
    K_M,
    STOICHIOMETRY,
    TRANSLATION_EFFICIENCY,
    counts_to_state,
    propensities,
    state_to_counts,
)


def _uniform_step(t: torch.Tensor) -> torch.Tensor:
    if t.ndim != 1 or t.numel() < 2:
        raise ValueError("`t` must be a 1-D tensor with at least two times.")
    dt = (t[-1] - t[0]) / (t.numel() - 1)
    if ((torch.diff(t) - dt).abs() > 1e-3 * dt.abs()).any():
        raise ValueError("The Langevin runner needs a uniform time grid.")
    return dt


def simulate_cle(
    propensity_fn: Callable[[torch.Tensor], torch.Tensor],
    stoichiometry: torch.Tensor,
    x0: torch.Tensor,
    t: torch.Tensor,
    substeps: int = 10,
    noise: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Euler-Maruyama integration of a generic chemical Langevin equation.

    Args:
        propensity_fn: maps counts ``(..., D)`` to rates ``(..., K)``.
        stoichiometry: ``(K, D)`` change in each species per reaction.
        x0: initial counts, shape ``(..., D)``.
        t: uniform observation grid, 1-D.
        substeps: Euler-Maruyama steps per observation interval.
        noise: optional standard normals of shape
            ``((len(t) - 1) * substeps, ..., K)``; drawn if omitted.
        generator: random source when ``noise`` is drawn here.

    Returns:
        Counts on the grid, shape ``(len(t), ..., D)``; index 0 is ``x0``.
    """
    dt = _uniform_step(t) / substeps
    n_steps = (t.numel() - 1) * substeps
    s = stoichiometry.to(x0)
    if noise is None:
        noise = torch.randn(
            (n_steps, *x0.shape[:-1], s.shape[0]),
            generator=generator,
            dtype=x0.dtype,
            device=x0.device,
        )
    elif noise.shape[0] != n_steps:
        raise ValueError(f"`noise` needs {n_steps} steps, got {noise.shape[0]}.")

    sqrt_dt = dt.sqrt()
    x = x0
    out = [x0]
    for k in range(n_steps):
        a = propensity_fn(x)
        # sqrt has an infinite derivative at 0; a switched-off reaction
        # contributes no noise, so a floor changes nothing in the forward pass.
        kick = a * dt + a.clamp_min(1e-12).sqrt() * sqrt_dt * noise[k]
        x = (x + kick @ s).abs()
        if (k + 1) % substeps == 0:
            out.append(x)
    return torch.stack(out, dim=0)


def run_langevin(
    state: RepressilatorState,
    params: RepressilatorParams,
    t: torch.Tensor,
    omega: float = K_M,
    eff: float = TRANSLATION_EFFICIENCY,
    substeps: int = 10,
    noise: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> RepressilatorState:
    """Simulate the repressilator with the chemical Langevin equation.

    Same calling convention as :func:`differentiable_cell.run.run_fixed_step`:
    dimensionless state and parameters in, dimensionless trajectory out.

    Args:
        state: initial condition in model units, any batch shape (e.g.
            ``(n_cells,)`` for independent cells).
        params: kinetic parameters, broadcast against ``state``.
        t: uniform grid of observation times in model units.
        omega: system size (``K_M`` in molecules).  Intrinsic noise scales as
            ``1 / sqrt(omega)``; ``omega -> inf`` recovers the ODE.
        eff: translation efficiency (proteins per transcript).
        substeps: Euler-Maruyama steps per observation interval.
        noise: standard normals, shape ``((len(t)-1)*substeps, *batch, 16)``,
            for common random numbers.  Drawn from ``generator`` if omitted.
        generator: random source for the noise.

    Returns:
        Trajectory with batch shape ``(len(t), *state.batch_size)``.
    """
    x0 = state_to_counts(state, params, omega, eff)
    x = simulate_cle(
        lambda x: propensities(x, params, omega, eff),
        STOICHIOMETRY,
        x0,
        t,
        substeps=substeps,
        noise=noise,
        generator=generator,
    )
    return counts_to_state(
        x, params, torch.Size((t.numel(), *state.batch_size)), omega, eff
    )
