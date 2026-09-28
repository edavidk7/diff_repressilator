"""Molecule counts and the repressilator as a chemical reaction network.

The ODE of :mod:`differentiable_cell.model` is dimensionless (Box 1 of
Elowitz & Leibler, 2000).  A stochastic simulation needs the same system in
*molecule counts*, and the conversion is less obvious than it looks, so it
lives here once and both stochastic simulators use it.

Let ``Omega`` be ``K_M``, the number of repressor monomers per cell that
half-represses a promoter (40 in the paper), and ``eff`` the translation
efficiency, the mean number of proteins made per transcript (20).  Box 1
writes protein in units of ``K_M`` and "mRNA rescaled by translation
efficiency", which pins the scales to::

    P = Omega * p                     (protein copies)
    M = m * Omega * beta / eff        (mRNA copies)

The second one contains ``beta``: with ``dp/dt = -beta (p - m)`` the
steady-state protein level ``p = m`` must equal ``eff * M / (beta * Omega)``
(translation over protein decay, in units of ``K_M``).  At the paper's values
``M ~ 0.4 m``, so ``alpha = 216`` means ~87 transcripts per cell at full
induction - which is indeed 0.5 transcripts/s times a 173 s mRNA lifetime.

In model time (units of the mRNA lifetime) each gene then has four
reactions, with propensities in events per model time unit::

    transcription   0 -> M     s_m * (alpha / (1 + (P_j / Omega)**n) + alpha_0)
    mRNA decay      M -> 0     M
    translation     M -> M + P eff * M
    protein decay   P -> 0     beta * P

where ``s_m = Omega * beta / eff``.  Dividing the drift of this network by the
scales returns the ODE exactly (checked in ``tests/test_units.py``), so the
ODE is the ``Omega -> infinity`` limit and ``Omega`` is the system size that
sets the intrinsic noise (relative fluctuations ~ ``1 / sqrt(Omega)``).

The GFP reporter gets the same four reactions with ``alpha_GFP`` and
``beta_GFP``, and like *cI* it is repressed by TetR.

A caveat on the paper's own numbers: it quotes both "20 proteins per
transcript" and a translation rate of 0.167 per mRNA per second.  Over a
2 min half-life that is 20, over the 173 s *lifetime* it is 29.  We keep
``eff = 20`` (the number the paper uses for ``alpha``) and the lifetime
convention of :mod:`differentiable_cell.helpers`.
"""

import torch

from differentiable_cell.data import RepressilatorParams, RepressilatorState
from differentiable_cell.run import FIELDS

K_M: float = 40.0
"""Repressor monomers per cell for half-maximal repression (paper, Box 1)."""

TRANSLATION_EFFICIENCY: float = 20.0
"""Mean proteins made per transcript (paper, Box 1)."""

GENES: tuple[tuple[str, str, str], ...] = (
    # (mRNA field, protein field, repressor field)
    ("m_lacI", "p_LacI", "p_CI"),
    ("m_tetR", "p_TetR", "p_LacI"),
    ("m_cI", "p_CI", "p_TetR"),
    ("m_gfp", "p_GFP", "p_TetR"),
)
"""The four transcription units, each with the repressor that acts on it."""

REACTIONS: tuple[str, ...] = tuple(
    f"{kind}:{gene[0][2:]}"
    for gene in GENES
    for kind in ("transcription", "mRNA_decay", "translation", "protein_decay")
)
"""Reaction names, in the column order of :func:`propensities`."""


def _stoichiometry() -> torch.Tensor:
    s = torch.zeros(len(REACTIONS), len(FIELDS))
    for g, (m_name, p_name, _) in enumerate(GENES):
        m, p = FIELDS.index(m_name), FIELDS.index(p_name)
        s[4 * g + 0, m] = 1.0
        s[4 * g + 1, m] = -1.0
        s[4 * g + 2, p] = 1.0
        s[4 * g + 3, p] = -1.0
    return s


STOICHIOMETRY: torch.Tensor = _stoichiometry()
"""Change in each species' count per reaction, shape ``(16, 8)``."""


def count_scales(
    params: RepressilatorParams,
    omega: float = K_M,
    eff: float = TRANSLATION_EFFICIENCY,
) -> torch.Tensor:
    """Molecules per model unit for each species, shape ``(..., 8)``.

    Multiply a packed dimensionless state by this to get molecule counts.
    The mRNA scales depend on ``beta`` (see the module docstring), so this
    is a function of the parameters, not a constant.
    """
    beta = torch.as_tensor(params.beta)
    beta_gfp = torch.as_tensor(params.beta_GFP)
    beta, beta_gfp = torch.broadcast_tensors(beta, beta_gfp)
    protein = torch.full_like(beta, omega)
    by_field = {
        "m_lacI": omega * beta / eff,
        "m_tetR": omega * beta / eff,
        "m_cI": omega * beta / eff,
        "p_LacI": protein,
        "p_TetR": protein,
        "p_CI": protein,
        "m_gfp": omega * beta_gfp / eff,
        "p_GFP": protein,
    }
    return torch.stack([by_field[name] for name in FIELDS], dim=-1)


def state_to_counts(
    state: RepressilatorState,
    params: RepressilatorParams,
    omega: float = K_M,
    eff: float = TRANSLATION_EFFICIENCY,
) -> torch.Tensor:
    """Pack a dimensionless state into molecule counts, shape ``(..., 8)``."""
    y = torch.stack([getattr(state, name) for name in FIELDS], dim=-1)
    return y * count_scales(params, omega, eff)


def counts_to_state(
    x: torch.Tensor,
    params: RepressilatorParams,
    batch_size: torch.Size,
    omega: float = K_M,
    eff: float = TRANSLATION_EFFICIENCY,
) -> RepressilatorState:
    """Inverse of :func:`state_to_counts`."""
    y = x / count_scales(params, omega, eff)
    return RepressilatorState(
        **{name: y[..., i] for i, name in enumerate(FIELDS)},
        batch_size=batch_size,
    )


def propensities(
    x: torch.Tensor,
    params: RepressilatorParams,
    omega: float = K_M,
    eff: float = TRANSLATION_EFFICIENCY,
) -> torch.Tensor:
    """Reaction rates in events per model time unit, shape ``(..., 16)``.

    Args:
        x: molecule counts, shape ``(..., 8)`` in :data:`FIELDS` order.  May
            be real-valued (Langevin, differentiable Gillespie); negative
            values are clipped to zero so every propensity is non-negative.
        params: kinetic parameters, broadcastable against ``x[..., 0]``.
        omega: system size, i.e. ``K_M`` in molecules.
        eff: translation efficiency.
    """
    x = x.clamp_min(0.0)
    rates = []
    for m_name, p_name, rep_name in GENES:
        gfp = m_name == "m_gfp"
        alpha = params.alpha_GFP if gfp else params.alpha
        beta = params.beta_GFP if gfp else params.beta
        m = x[..., FIELDS.index(m_name)]
        p = x[..., FIELDS.index(p_name)]
        rep = x[..., FIELDS.index(rep_name)]
        s_m = omega * beta / eff
        rates += [
            s_m * (alpha / (1.0 + (rep / omega) ** params.n) + params.alpha_0),
            m,
            eff * m,
            beta * p,
        ]
    return torch.stack(torch.broadcast_tensors(*rates), dim=-1)
