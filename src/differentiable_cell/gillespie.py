"""Exact and differentiable Gillespie simulation of the repressilator.

Two simulators over the reaction network of :mod:`differentiable_cell.units`:

* :func:`ssa_exact` - Gillespie's direct method.  Integer counts, exact
  sample paths, no gradients.  It generates stochastic "ground truth" data.
* :func:`ssa_dga` - the differentiable Gillespie algorithm (DGA) of Rijal &
  Mehta (2025).  It keeps the direct method's two random numbers per event
  but replaces its two discontinuous operations with smooth ones:

  - reaction selection, ``i' = sum_i Theta(u' - c_i)`` with ``c_i`` the
    cumulative normalised propensities, becomes
    ``sum_i sigmoid((u' - c_i) / a)`` - a real-valued index;
  - the update ``x += S[i']`` (a Kronecker delta over reactions) becomes
    ``x += sum_i exp(-(i - i')**2 / (2 b**2)) S[i]``.

  Both are exact again as ``a, b -> 0``; the paper uses ``1/a = 200`` and
  ``1/b = 20``.  Waiting times ``-log(u) / R`` are already smooth in the
  rates.

Reading a trajectory out on an observation grid needs one more smoothing
that the paper does not spell out (it fits end-time moments).  A hard
readout (the state after the last event before ``t``) passes gradients only
through the counts, not through event *times*, so it is blind to anything
that only rescales time - e.g. doubling every rate.  :func:`ssa_dga` therefore
reads out ``x(t) = x0 + sum_k dx_k * sigmoid((t - t_k) / c)``, which carries
the waiting-time gradient; ``c -> 0`` recovers the hard readout.

Cost: one loop iteration per event.  At the paper's ``K_M = 40`` the
repressilator fires ~10^3-10^4 events per mRNA lifetime, so the DGA (which
keeps every event on the autograd tape) is meant for reduced system size
``omega`` and/or short windows; ``max_events`` guards against runaway cost.
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

PropensityFn = Callable[[torch.Tensor], torch.Tensor]


def ssa_exact(
    propensity_fn: PropensityFn,
    stoichiometry: torch.Tensor,
    x0: torch.Tensor,
    t: torch.Tensor,
    generator: torch.Generator | None = None,
    max_events: int = 50_000_000,
) -> torch.Tensor:
    """Gillespie's direct method for a batch of independent cells.

    Args:
        propensity_fn: maps counts ``(B, D)`` to rates ``(B, K)``.
        stoichiometry: ``(K, D)``.
        x0: initial counts, shape ``(B, D)``; rounded to integers.
        t: increasing observation times, 1-D; ``t[0]`` is the time of ``x0``.
        generator: random source.
        max_events: abort if any cell needs more events than this.

    Returns:
        Counts on the grid, shape ``(len(t), B, D)``.
    """
    with torch.no_grad():
        s = stoichiometry.to(x0)
        x = x0.round().clone()
        batch = x.shape[0]
        clock = torch.full((batch,), float(t[0]), dtype=x.dtype, device=x.device)
        out = torch.empty((t.numel(), *x.shape), dtype=x.dtype, device=x.device)
        out[0] = x
        next_obs = torch.ones(batch, dtype=torch.long, device=x.device)
        rows = torch.arange(batch, device=x.device)
        n_obs = t.numel()
        t = t.to(x)

        for _ in range(max_events):
            if bool((next_obs >= n_obs).all()):
                return out
            a = propensity_fn(x).clamp_min(0.0)
            total = a.sum(-1)
            u = torch.rand((2, batch), generator=generator, dtype=x.dtype, device=x.device)
            tau = -torch.log1p(-u[0]) / total  # inf when nothing can fire
            new_clock = clock + tau

            # Every observation time passed by this jump sees the old state.
            while True:
                pending = next_obs < n_obs
                passed = pending & (t[next_obs.clamp(max=n_obs - 1)] < new_clock)
                if not bool(passed.any()):
                    break
                out[next_obs[passed], rows[passed]] = x[passed]
                next_obs = next_obs + passed.long()

            cum = a.cumsum(-1)
            idx = (cum < (u[1] * total).unsqueeze(-1)).sum(-1).clamp(max=a.shape[-1] - 1)
            fires = torch.isfinite(new_clock).unsqueeze(-1)
            x = torch.where(fires, x + s[idx], x)
            clock = new_clock
        raise RuntimeError(f"ssa_exact exceeded max_events={max_events}.")


def ssa_dga(
    propensity_fn: PropensityFn,
    stoichiometry: torch.Tensor,
    x0: torch.Tensor,
    t: torch.Tensor,
    a: float = 1.0 / 200.0,
    b: float = 1.0 / 20.0,
    c: float | None = None,
    generator: torch.Generator | None = None,
    max_events: int = 200_000,
) -> torch.Tensor:
    """Differentiable Gillespie algorithm (Rijal & Mehta, 2025).

    Args:
        propensity_fn: maps counts ``(B, D)`` to rates ``(B, K)``; must be
            differentiable in whatever parameters it closes over.
        stoichiometry: ``(K, D)``.
        x0: initial counts, shape ``(B, D)``.
        t: increasing observation times, 1-D.
        a: sigmoid width of the reaction-index selection.
        b: Gaussian width of the abundance update.
        c: sigmoid width of the time readout, in model time.  Defaults to a
            tenth of the mean observation spacing; ``0`` gives a hard readout.
        generator: random source.  Two calls with generators in the same
            state and the same number of events share their random numbers.
        max_events: abort beyond this many events (every event is kept on
            the autograd tape).

    Returns:
        Real-valued counts on the grid, shape ``(len(t), B, D)``.
    """
    s = stoichiometry.to(x0)
    n_reactions = s.shape[0]
    t = t.to(x0)
    if c is None:
        c = float((t[-1] - t[0]) / max(t.numel() - 1, 1)) / 10.0
    batch = x0.shape[0]
    index = torch.arange(n_reactions, dtype=x0.dtype, device=x0.device)

    x = x0
    clock = torch.full((batch,), float(t[0]), dtype=x0.dtype, device=x0.device)
    jumps, times = [], []
    for _ in range(max_events):
        if bool((clock.detach() >= t[-1]).all()):
            break
        rates = propensity_fn(x).clamp_min(0.0)
        total = rates.sum(-1)
        u = torch.rand((2, batch), generator=generator, dtype=x0.dtype, device=x0.device)
        clock = clock + (-torch.log1p(-u[0]) / total.clamp_min(1e-30))
        # sum_{i=1}^{K-1} sigmoid((u' - c_i) / a): the index of the reaction
        # whose cumulative bin contains u', as a real number in [0, K-1].
        boundaries = (rates.cumsum(-1) / total.clamp_min(1e-30).unsqueeze(-1))[:, :-1]
        chosen = torch.sigmoid((u[1].unsqueeze(-1) - boundaries) / a).sum(-1)
        weights = torch.exp(-((index - chosen.unsqueeze(-1)) ** 2) / (2.0 * b**2))
        dx = weights @ s
        x = x + dx
        jumps.append(dx)
        times.append(clock)
    else:
        raise RuntimeError(f"ssa_dga exceeded max_events={max_events}.")

    if not jumps:
        return x0.expand(t.numel(), *x0.shape)
    dx = torch.stack(jumps, dim=0)  # (E, B, D)
    tk = torch.stack(times, dim=0)  # (E, B)
    lag = t.view(-1, 1, 1) - tk.unsqueeze(0)  # (G, E, B)
    gate = (lag >= 0).to(dx) if c == 0 else torch.sigmoid(lag / c)
    return x0.unsqueeze(0) + torch.einsum("geb,ebd->gbd", gate, dx)


def _flatten_cells(state: RepressilatorState, params, omega, eff):
    x0 = state_to_counts(state, params, omega, eff)
    return x0.reshape(-1, x0.shape[-1])


def run_gillespie(
    state: RepressilatorState,
    params: RepressilatorParams,
    t: torch.Tensor,
    omega: float = K_M,
    eff: float = TRANSLATION_EFFICIENCY,
    generator: torch.Generator | None = None,
    max_events: int = 50_000_000,
) -> RepressilatorState:
    """Exact stochastic repressilator trajectories, in model units.

    ``state`` may carry a batch of cells; ``params`` must be a single set
    (no batch), since cells are flattened for the event loop.  Initial
    values are rounded to whole molecules.
    """
    x0 = _flatten_cells(state, params, omega, eff)
    x = ssa_exact(
        lambda x: propensities(x, params, omega, eff),
        STOICHIOMETRY, x0, t, generator=generator, max_events=max_events,
    )
    x = x.reshape(t.numel(), *state.batch_size, x.shape[-1])
    return counts_to_state(
        x, params, torch.Size((t.numel(), *state.batch_size)), omega, eff
    )


def run_dga(
    state: RepressilatorState,
    params: RepressilatorParams,
    t: torch.Tensor,
    omega: float = K_M,
    eff: float = TRANSLATION_EFFICIENCY,
    a: float = 1.0 / 200.0,
    b: float = 1.0 / 20.0,
    c: float | None = None,
    generator: torch.Generator | None = None,
    max_events: int = 200_000,
) -> RepressilatorState:
    """Differentiable Gillespie trajectories of the repressilator, in model units.

    See :func:`ssa_dga` for ``a``, ``b``, ``c``.  Unlike :func:`run_gillespie`
    the initial counts are not rounded, so they stay differentiable.
    """
    x0 = _flatten_cells(state, params, omega, eff)
    x = ssa_dga(
        lambda x: propensities(x, params, omega, eff),
        STOICHIOMETRY, x0, t, a=a, b=b, c=c,
        generator=generator, max_events=max_events,
    )
    x = x.reshape(t.numel(), *state.batch_size, x.shape[-1])
    return counts_to_state(
        x, params, torch.Size((t.numel(), *state.batch_size)), omega, eff
    )
