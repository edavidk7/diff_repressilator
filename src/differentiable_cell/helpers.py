"""Small conversions between model units and experimental units.

The model of :mod:`differentiable_cell.model` is dimensionless: following
Box 1 of Elowitz & Leibler (2000), time is measured in units of the mRNA
lifetime, so the mRNA decay term is a bare ``-m_i`` with no rate constant.
Nothing in the simulation therefore comes out in minutes, and the helpers
here do that last step - which is what you need to put a simulated
trajectory next to a microscopy timecourse.
"""

import math

import torch

MRNA_HALF_LIFE_MIN: float = 2.0
"""Average mRNA half-life in *E. coli*, in minutes (paper, Box 1)."""

TAU_M_MIN: float = MRNA_HALF_LIFE_MIN / math.log(2.0)
"""mRNA *lifetime* (1 / decay rate) in minutes - the model's time unit.

Note the factor of ``ln 2``: the lifetime is the reciprocal decay rate, not
the half-life.  With a 2 min half-life it is about 2.885 min.
"""


def to_minutes(
    t: torch.Tensor, tau_m_min: float = TAU_M_MIN
) -> torch.Tensor:
    """Convert dimensionless model time to minutes.

    Args:
        t: times in model units, i.e. in multiples of the mRNA lifetime.
        tau_m_min: the mRNA lifetime in minutes.  Defaults to
            :data:`TAU_M_MIN` (~2.885 min); override it when modelling a
            strain or medium with different mRNA turnover.

    Returns:
        The same tensor expressed in minutes.

    Example:
        A period of 43.5 model units is ``43.5 * 2.885`` ~ 125 min, in the
        range of the 160 +/- 40 min the paper measures in single cells.
    """
    return t * tau_m_min


def from_minutes(
    t_min: torch.Tensor, tau_m_min: float = TAU_M_MIN
) -> torch.Tensor:
    """Convert minutes to dimensionless model time.

    The inverse of :func:`to_minutes`; useful for building an integration
    grid that covers a given number of minutes of real growth.

    Args:
        t_min: times in minutes.
        tau_m_min: the mRNA lifetime in minutes.

    Returns:
        The same tensor in model units.
    """
    return t_min / tau_m_min
