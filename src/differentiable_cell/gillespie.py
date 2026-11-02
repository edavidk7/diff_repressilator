"""Gillespie simulation of a reaction network: exact, and differentiable (DGA).

Both take ``propensity_fn`` (counts ``(B, D)`` -> rates ``(B, K)``), the
stoichiometry ``(K, D)``, start counts ``x0 (B, D)`` and observation times
``t`` (``t[0]`` is the time of ``x0``), and return counts ``(len(t), B, D)``.

The DGA (Rijal & Mehta 2025) smooths the two discrete steps of the direct method:
reaction choice ``sum_i sigmoid((u - c_i) / a)`` instead of a step function, and
update ``x += sum_i exp(-(i - i*)^2 / 2b^2) S[i]`` instead of ``S[i*]``. It also
reads the state out as ``x0 + sum_k dx_k sigmoid((t - t_k) / c)`` so gradients
flow through event times too (``c = 0``: hard readout).
"""

from typing import Callable

import torch

PropensityFn = Callable[[torch.Tensor], torch.Tensor]


def ssa_exact(propensity_fn: PropensityFn, stoichiometry: torch.Tensor, x0: torch.Tensor, t: torch.Tensor,
              generator: torch.Generator | None = None, max_events: int = 50_000_000) -> torch.Tensor:
    """Gillespie's direct method, batched over independent cells. Integer counts, no gradients."""
    with torch.no_grad():
        s = stoichiometry.to(x0)
        x = x0.round().clone()
        batch, n_obs = x.shape[0], t.numel()
        t = t.to(x)
        clock = torch.full((batch,), float(t[0]), dtype=x.dtype)
        out = torch.empty((n_obs, *x.shape), dtype=x.dtype)
        out[0] = x
        next_obs = torch.ones(batch, dtype=torch.long)
        rows = torch.arange(batch)
        for _ in range(max_events):
            if bool((next_obs >= n_obs).all()):
                return out
            a = propensity_fn(x).clamp_min(0.0)
            total = a.sum(-1)
            u = torch.rand((2, batch), generator=generator, dtype=x.dtype)
            new_clock = clock - torch.log1p(-u[0]) / total  # inf when nothing can fire
            while True:  # observation times passed by this jump see the old state
                passed = (next_obs < n_obs) & (t[next_obs.clamp(max=n_obs - 1)] < new_clock)
                if not bool(passed.any()):
                    break
                out[next_obs[passed], rows[passed]] = x[passed]
                next_obs = next_obs + passed.long()
            idx = (a.cumsum(-1) < (u[1] * total).unsqueeze(-1)).sum(-1).clamp(max=a.shape[-1] - 1)
            x = torch.where(torch.isfinite(new_clock).unsqueeze(-1), x + s[idx], x)
            clock = new_clock
        raise RuntimeError(f"ssa_exact exceeded max_events={max_events}")


def ssa_dga(propensity_fn: PropensityFn, stoichiometry: torch.Tensor, x0: torch.Tensor, t: torch.Tensor,
            a: float = 1 / 200, b: float = 1 / 20, c: float | None = None,
            generator: torch.Generator | None = None, max_events: int = 200_000) -> torch.Tensor:
    """Differentiable Gillespie. ``a``, ``b``: the paper's smoothing widths; ``c``: readout width
    (default: a tenth of the observation spacing). Every event stays on the autograd tape."""
    s = stoichiometry.to(x0)
    t = t.to(x0)
    if c is None:
        c = float((t[-1] - t[0]) / max(t.numel() - 1, 1)) / 10
    batch = x0.shape[0]
    index = torch.arange(s.shape[0], dtype=x0.dtype)
    x, clock = x0, torch.full((batch,), float(t[0]), dtype=x0.dtype)
    jumps, times = [], []
    for _ in range(max_events):
        if bool((clock.detach() >= t[-1]).all()):
            break
        rates = propensity_fn(x).clamp_min(0.0)
        total = rates.sum(-1).clamp_min(1e-30)
        u = torch.rand((2, batch), generator=generator, dtype=x0.dtype)
        clock = clock - torch.log1p(-u[0]) / total
        bins = (rates.cumsum(-1) / total.unsqueeze(-1))[:, :-1]
        chosen = torch.sigmoid((u[1].unsqueeze(-1) - bins) / a).sum(-1)  # real-valued reaction index
        dx = torch.exp(-((index - chosen.unsqueeze(-1)) ** 2) / (2 * b**2)) @ s
        x = x + dx
        jumps.append(dx)
        times.append(clock)
    else:
        raise RuntimeError(f"ssa_dga exceeded max_events={max_events}")
    if not jumps:
        return x0.expand(t.numel(), *x0.shape)
    lag = t.view(-1, 1, 1) - torch.stack(times).unsqueeze(0)  # (G, E, B)
    gate = (lag >= 0).to(x0) if c == 0 else torch.sigmoid(lag / c)
    return x0.unsqueeze(0) + torch.einsum("geb,ebd->gbd", gate, torch.stack(jumps))
