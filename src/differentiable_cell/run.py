"""Integration runners for the repressilator.

Two ways to turn the vector field of :class:`~differentiable_cell.model.
Repressilator` into a trajectory:

* :func:`run_fixed_step` - an explicit fixed-step scheme written out here.
  Every operation is an ordinary tensor op recorded on the autograd tape, so
  gradients flow back through the whole unrolled trajectory.  Cheap, exact
  about *what* it computed, and the memory cost grows with the number of
  steps.
* :func:`run_torchdiffeq` - hands the same vector field to ``torchdiffeq``'s
  adaptive solvers, which control local error and can back-propagate with
  the adjoint method at constant memory.  Use it as the reference when
  checking whether a fixed step size is small enough.

Both take a single :class:`~differentiable_cell.data.RepressilatorParams`
and a single initial :class:`~differentiable_cell.data.RepressilatorState`,
and return the whole trajectory as one ``RepressilatorState`` whose leading
batch dimension is time.

Times are in model units (multiples of the mRNA lifetime); use
:func:`differentiable_cell.helpers.to_minutes` to plot against real time.
"""

from typing import Callable

import torch
from torchdiffeq import odeint, odeint_adjoint

from differentiable_cell.data import RepressilatorParams, RepressilatorState
from differentiable_cell.model import Repressilator

FIELDS: tuple[str, ...] = (
    "m_lacI",
    "m_tetR",
    "m_cI",
    "p_LacI",
    "p_TetR",
    "p_CI",
    "m_gfp",
    "p_GFP",
)
"""Field order used when a state is packed into a single tensor."""

PARAM_FIELDS: tuple[str, ...] = (
    "alpha",
    "alpha_0",
    "n",
    "beta",
    "alpha_GFP",
    "beta_GFP",
)
"""Kinetic parameters, in the order the adjoint solver is told to track."""


def _pack(state: RepressilatorState) -> torch.Tensor:
    """Stack a state's eight fields into one tensor of shape ``(..., 8)``."""
    return torch.stack([getattr(state, name) for name in FIELDS], dim=-1)


def _unpack(y: torch.Tensor, batch_size: torch.Size) -> RepressilatorState:
    """Inverse of :func:`_pack`, with an explicit batch shape."""
    return RepressilatorState(
        **{name: y[..., i] for i, name in enumerate(FIELDS)},
        batch_size=batch_size,
    )


def _make_rhs(
    model: Repressilator, params: RepressilatorParams
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Close ``model`` and ``params`` over a packed right-hand side.

    Returns a function mapping a packed state ``(..., 8)`` to its packed
    derivative, which is the form both runners iterate on.
    """

    def rhs(y: torch.Tensor) -> torch.Tensor:
        derivatives = model.forward_tensor(
            **{name: y[..., i] for i, name in enumerate(FIELDS)},
            **{name: getattr(params, name) for name in PARAM_FIELDS},
        )
        return torch.stack(derivatives, dim=-1)

    return rhs


def run_fixed_step(
    model: Repressilator,
    state: RepressilatorState,
    params: RepressilatorParams,
    t: torch.Tensor,
    method: str = "rk4",
) -> RepressilatorState:
    """Integrate on a uniform grid with an explicit fixed-step scheme.

    The loop is unrolled in Python, so the returned trajectory carries the
    full autograd graph of every step: differentiating a loss on the
    trajectory with respect to ``params`` works directly, at a memory cost
    linear in ``len(t)``.

    Args:
        model: the vector field to integrate.
        state: initial condition, with any batch shape (a single cell, or a
            population).  Not modified.
        params: one set of kinetic parameters, broadcast against ``state``.
        t: 1-D tensor of times in model units, uniformly spaced (to within
            floating-point rounding) and increasing.  ``t[0]`` is the time of
            ``state``; the step is taken as ``(t[-1] - t[0]) / (len(t) - 1)``.
        method: ``"euler"`` (first order, cheapest) or ``"rk4"`` (classical
            fourth-order Runge-Kutta, the default).

    Returns:
        The trajectory as a :class:`RepressilatorState` with batch shape
        ``(len(t), *state.batch_size)``; index 0 is ``state`` itself.

    Raises:
        ValueError: if ``t`` is not 1-D and uniformly spaced, or if
            ``method`` is unknown.
    """
    if t.ndim != 1 or t.numel() < 2:
        raise ValueError("`t` must be a 1-D tensor with at least two times.")

    # Average rather than take the first gap: a float32 `linspace` grid is
    # uniform only to within rounding, and the mean is the better estimate.
    dt = (t[-1] - t[0]) / (t.numel() - 1)
    if ((torch.diff(t) - dt).abs() > 1e-3 * dt.abs()).any():
        raise ValueError(
            "`run_fixed_step` needs a uniform grid; use `run_torchdiffeq` "
            "for arbitrary evaluation times."
        )

    rhs = _make_rhs(model, params)

    if method == "euler":

        def step(y: torch.Tensor) -> torch.Tensor:
            return y + dt * rhs(y)

    elif method == "rk4":

        def step(y: torch.Tensor) -> torch.Tensor:
            k1 = rhs(y)
            k2 = rhs(y + 0.5 * dt * k1)
            k3 = rhs(y + 0.5 * dt * k2)
            k4 = rhs(y + dt * k3)
            return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    else:
        raise ValueError(f"Unknown method {method!r}; use 'euler' or 'rk4'.")

    y = _pack(state)
    trajectory = [y]
    for _ in range(t.numel() - 1):
        y = step(y)
        trajectory.append(y)

    return _unpack(
        torch.stack(trajectory, dim=0),
        batch_size=torch.Size((t.numel(), *state.batch_size)),
    )


def run_torchdiffeq(
    model: Repressilator,
    state: RepressilatorState,
    params: RepressilatorParams,
    t: torch.Tensor,
    method: str = "dopri5",
    rtol: float = 1e-7,
    atol: float = 1e-9,
    adjoint: bool = False,
) -> RepressilatorState:
    """Integrate with ``torchdiffeq``, evaluating at arbitrary times.

    The solver chooses its own internal steps to meet the requested
    tolerances and interpolates onto ``t``, so this is the runner to trust
    when checking a fixed step size, and the one to use over long horizons.

    Args:
        model: the vector field to integrate.
        state: initial condition, with any batch shape.  Not modified.
        params: one set of kinetic parameters, broadcast against ``state``.
        t: 1-D tensor of increasing times in model units at which to report
            the solution; ``t[0]`` is the time of ``state``.  It need not be
            uniformly spaced.
        method: any ``torchdiffeq`` solver name, e.g. ``"dopri5"`` (default,
            adaptive) or ``"rk4"`` (fixed step).
        rtol: relative error tolerance for adaptive solvers.
        atol: absolute error tolerance for adaptive solvers.
        adjoint: if True, use ``odeint_adjoint``, which recomputes the
            backward pass by solving the adjoint ODE.  Memory then does not
            grow with the number of steps, at the cost of a second solve and
            of gradients that are themselves approximate.  Note that the
            adjoint tracks tensors reached through ``model``'s parameters
            and the explicit inputs, so ``params`` must require grad for
            parameter gradients to come back.

    Returns:
        The trajectory as a :class:`RepressilatorState` with batch shape
        ``(len(t), *state.batch_size)``; index 0 is ``state`` itself.
    """
    rhs = _make_rhs(model, params)

    class _Field(torch.nn.Module):
        """Adapter giving ``rhs`` the ``(t, y)`` signature odeint expects."""

        def forward(self, _t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return rhs(y)

    kwargs = {"method": method, "rtol": rtol, "atol": atol}
    if adjoint:
        # `odeint_adjoint` differentiates only w.r.t. tensors it is told
        # about; ours live in `params`, not in a module's parameter list.
        kwargs["adjoint_params"] = tuple(
            getattr(params, name) for name in PARAM_FIELDS
        )
        y = odeint_adjoint(_Field(), _pack(state), t, **kwargs)
    else:
        y = odeint(_Field(), _pack(state), t, **kwargs)

    return _unpack(
        y, batch_size=torch.Size((t.numel(), *state.batch_size))
    )
