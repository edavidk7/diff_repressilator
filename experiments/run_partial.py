"""Recover hidden parameters (and initial conditions) of a clean run.

Loads a directory written by ``run_clean.py``, hides some of the parameters
and/or some of the species, and fits them back by differentiating through the
model.  Everything else - the time grid, the integration call, the untouched
parameters, the precision - is taken from the clean run's log, so the inverse
problem is posed against exactly the forward problem that generated the data.

The fit is by shooting: integrate from an initial state and compare against
the observed species.  A dropped species contributes exactly one unknown, its
value at ``t = 0``; the ODE supplies the rest of its trajectory, so a hidden
species is reconstructed rather than fitted point by point.

The horizon grows over training, and that is load-bearing rather than
decorative.  Because a small period error accumulates into a phase flip, the
loss landscape of a period-setting parameter such as ``beta`` saturates over
long horizons: measured at 11 periods, the loss at ``beta = 0.19`` is 815
against 855 at ``0.21``, with the truth at 0.20 - almost no gradient signal.
At 2 periods the same landscape is cleanly curved (104 / 0 / 101).  So the
curriculum fits one period first, then widens.

Examples:
    python experiments/run_partial.py run_results/<clean-run> --drop_params alpha
    python experiments/run_partial.py run_results/<clean-run> \\
        --drop_params alpha beta --drop_state m_cI p_CI --restarts 5
"""

import argparse
import csv
import inspect
import logging
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from differentiable_cell.data import RepressilatorState
from differentiable_cell.helpers import TAU_M_MIN
from differentiable_cell.model import Repressilator
from differentiable_cell.run import FIELDS, PARAM_FIELDS

from _common import (
    REPO_ROOT,
    RESULTS_DIR,
    add_time_axes,
    close_figure,
    integrate,
    load_run,
    params_from_dict,
    plot_species_panels,
    provenance,
    seed_everything,
    setup_logging,
    write_json,
)

LOG = logging.getLogger("run_partial")

PRIORS: dict[str, tuple[float, float]] = {
    # Bounds for `--init_method uniform` and for the box the optimiser is kept
    # inside, in value space, set at the biologically sensible limits.  They
    # are NOT derived from the stored truth - initialising near the answer
    # would make any "recovery" meaningless.
    #
    # alpha is the steady-state protein level at full induction, in units of
    # K_M (~40 monomers/cell).  Sanity check on the definition: 0.5
    # transcripts/s x 20 proteins/transcript / (ln2 / 10 min) = 8,656
    # copies/cell, / 40 = 216, the paper's value.  The span is 40 copies (a
    # barely-expressed gene) to 80,000 (~3% of the proteome, already strong
    # overexpression).
    "alpha": (1.0, 2000.0),
    # Leakiness, same units: from effectively nothing to a few percent of full
    # induction for a poorly repressed promoter.
    "alpha_0": (1e-4, 10.0),
    # Hill coefficient.  The upper end is set by the operator count and the
    # repressor's oligomeric state.  The lower end is 1 for a NUMERICAL
    # reason as much as a biological one: the Hill term is p**n, whose
    # derivative n*p**(n-1) is infinite at p = 0, and this system's troughs -
    # and its initial condition - contain exact zeros.  Below 1 the gradient
    # is inf, which becomes NaN through the chain and is written straight into
    # the parameters, after which the solver fails on NaN inputs.
    "n": (1.0, 4.0),
    # beta = protein decay / mRNA decay = tau_mRNA / tau_protein, so with a
    # 2 min mRNA half-life, beta = 2 / (protein half-life in min).  The span
    # runs from growth dilution alone (~67 min) to the fastest real
    # ssrA-mediated degradation (~1 min).
    "beta": (0.03, 2.0),
    "alpha_GFP": (1.0, 2000.0),
    # GFP-AAV has a ~90 min half-life (beta ~ 0.022); the least stable GFP
    # variants reach ~10 min (beta ~ 0.2).
    "beta_GFP": (0.02, 0.2),
}

LOSSES: dict[str, Callable[..., torch.Tensor]] = {
    "mse": lambda a, b, **kw: F.mse_loss(a, b),
    "mae": lambda a, b, **kw: F.l1_loss(a, b),
    "huber": lambda a, b, delta=1.0: F.huber_loss(a, b, delta=delta),
}


# --------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------


def parse_overrides(pairs: list[str] | None) -> dict[str, float]:
    """Parse repeated ``--set NAME=VALUE`` arguments."""
    overrides: dict[str, float] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"--set expects NAME=VALUE, got {pair!r}")
        name, _, raw = pair.partition("=")
        name = name.strip()
        if name not in PARAM_FIELDS:
            raise SystemExit(
                f"--set: unknown parameter {name!r}; choose from {list(PARAM_FIELDS)}"
            )
        try:
            overrides[name] = float(raw)
        except ValueError:
            raise SystemExit(f"--set {name}: {raw!r} is not a number") from None
    return overrides


def apply_overrides(
    stored: dict[str, torch.Tensor], overrides: dict[str, float]
) -> dict[str, torch.Tensor]:
    """Override stored parameters, warning loudly about each one.

    An override makes the model inconsistent with the data it is being fitted
    to - the trajectory was generated with the stored value - so this is worth
    a warning rather than a silent substitution.
    """
    used = dict(stored)
    for name, value in overrides.items():
        old = float(stored[name])
        delta = "" if old == 0 else f" ({(value - old) / abs(old):+.1%})"
        LOG.warning(
            "overriding stored parameter %s: %g -> %g%s.\n"
            "         The loaded trajectory was generated with the stored "
            "value; the model is now out of distribution with respect to its "
            "own data.",
            name,
            old,
            value,
            delta,
        )
        used[name] = torch.tensor(value, dtype=stored[name].dtype)
    return used


def inverse_softplus(y: torch.Tensor) -> torch.Tensor:
    """Inverse of ``softplus``, numerically safe for large and tiny values.

    Initial concentrations are optimised through a softplus so they can never
    go negative: a negative concentration is unphysical, and it also makes
    ``p ** n`` NaN for the non-integer Hill coefficients the optimiser
    explores, which silently kills a whole restart.
    """
    y = y.clamp(min=1e-12)
    return torch.where(y > 20.0, y, torch.log(torch.expm1(y)))


def init_log_param(name: str, method: str, generator: torch.Generator) -> torch.Tensor:
    """Initial value of a free parameter, in the log space it is optimised in.

    ``--init_method`` is interpreted in the space the variable is optimised in,
    which is why ``zero`` means ``log p = 0``, i.e. ``p = 1``, rather than the
    invalid ``p = 0``.
    """
    if method == "zero":
        return torch.zeros((), dtype=torch.get_default_dtype())
    if method == "normal":
        return torch.randn((), generator=generator, dtype=torch.get_default_dtype())
    lo, hi = PRIORS[name]
    u = torch.rand((), generator=generator, dtype=torch.get_default_dtype())
    return math.log(lo) + u * (math.log(hi) - math.log(lo))


def init_hidden_ic(
    method: str, scale: float, generator: torch.Generator
) -> torch.Tensor:
    """Initial guess for a dropped species' value at ``t = 0``, in value space.

    Concentrations are non-negative, so ``normal`` is folded and ``uniform``
    spans zero to the scale of the observed data.
    """
    dtype = torch.get_default_dtype()
    if method == "zero":
        return torch.zeros((), dtype=dtype)
    if method == "normal":
        return (torch.randn((), generator=generator, dtype=dtype) * scale).abs()
    return torch.rand((), generator=generator, dtype=dtype) * scale


def make_optimizer(
    name: str, variables: list[torch.Tensor], lr: float
) -> torch.optim.Optimizer:
    """Instantiate any ``torch.optim`` class by name, with decay disabled.

    Weight decay pulls parameters toward zero, which is meaningless for
    physical rate constants and - since they are optimised in log space -
    would actively drag every one of them toward 1.  It is therefore forced
    off wherever the optimiser accepts it.
    """
    cls = getattr(torch.optim, name, None)
    if not (isinstance(cls, type) and issubclass(cls, torch.optim.Optimizer)):
        available = sorted(
            n
            for n in dir(torch.optim)
            if isinstance(getattr(torch.optim, n), type)
            and issubclass(getattr(torch.optim, n), torch.optim.Optimizer)
            and n != "Optimizer"
        )
        raise SystemExit(
            f"--optim {name!r} is not a torch.optim class. Available: {available}"
        )

    kwargs: dict[str, Any] = {"lr": lr}
    accepted = inspect.signature(cls).parameters
    disabled = []
    if "weight_decay" in accepted:
        kwargs["weight_decay"] = 0.0
        disabled.append("weight_decay=0.0")
    if "decoupled_weight_decay" in accepted:
        kwargs["decoupled_weight_decay"] = False
        disabled.append("decoupled_weight_decay=False")
    if "line_search_fn" in accepted:
        # LBFGS without a line search takes raw `lr`-scaled quasi-Newton steps,
        # which overshoot immediately on a landscape this curved.  Strong-Wolfe
        # makes it usable; it is the only reason `--optim LBFGS` is worth trying.
        kwargs["line_search_fn"] = "strong_wolfe"
        LOG.info("%s: using strong_wolfe line search", name)
    if disabled:
        LOG.info("%s: parameter decay disabled (%s)", name, ", ".join(disabled))
    return cls(variables, **kwargs)


def lr_lambda(args: argparse.Namespace) -> Callable[[int], float]:
    """Linear decay from ``lr_max`` to ``lr_min``; constant if no ``lr_min``."""
    if args.lr_min is None or args.epochs < 2:
        return lambda _epoch: 1.0
    floor = args.lr_min / args.lr_max
    return lambda epoch: 1.0 - (1.0 - floor) * min(epoch, args.epochs - 1) / (
        args.epochs - 1
    )


def curriculum_windows(
    t: torch.Tensor, period: float | None, stages: list[float], epochs: int
) -> Callable[[int], int]:
    """Map an epoch to the number of timesteps included in the loss.

    Stages are given in periods and split evenly over the epochs.  Without a
    measured period (a run that never oscillated) the whole horizon is used
    throughout, since there is nothing to grow relative to.
    """
    n_steps = t.numel()
    if period is None:
        LOG.info("no period in the clean run's summary; curriculum disabled")
        return lambda _epoch: n_steps

    bounds: list[int] = []
    for stage in stages:
        idx = int(torch.searchsorted(t, torch.tensor(stage * period)).item()) + 1
        bounds.append(max(4, min(idx, n_steps)))
    per_stage = max(1, math.ceil(epochs / len(bounds)))

    LOG.info(
        "curriculum: %s periods -> %s steps, %d epochs each", stages, bounds, per_stage
    )
    return lambda epoch: bounds[min(epoch // per_stage, len(bounds) - 1)]


def species_scales(data: torch.Tensor, observed: list[int], mode: str) -> torch.Tensor:
    """Per-species divisors for the loss.

    mRNA spans 0-125 while GFP spans 0-25, so an unnormalised MSE is dominated
    by the mRNAs and barely sees the reporter - the one species that is
    actually observable in the experiment.

    Hidden species get the mean of the observed scales rather than their own
    range, which would leak the very trajectory being reconstructed.
    """
    scales = torch.ones(len(FIELDS), dtype=data.dtype)
    if mode == "none":
        return scales
    per_species = (
        data.max(dim=0).values - data.min(dim=0).values
        if mode == "range"
        else data.std(dim=0)
    )
    fallback = per_species[observed].mean().clamp(min=1e-8)
    for i in range(len(FIELDS)):
        scales[i] = per_species[i] if i in observed else fallback
    return scales.clamp(min=1e-8)


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------


def fit_once(
    args: argparse.Namespace,
    model: Repressilator,
    t: torch.Tensor,
    data: torch.Tensor,
    fixed_params: dict[str, torch.Tensor],
    spec: dict[str, Any],
    observed: list[int],
    hidden: list[int],
    scales: torch.Tensor,
    window_of: Callable[[int], int],
    seed: int,
) -> dict[str, Any]:
    """One restart: initialise the free variables and optimise them.

    Free variables are the log of each dropped parameter and the softplus-raw
    initial value of each dropped species.  Observed species take their
    initial value from the data, which is the honest choice under ``--noise``:
    a measured initial condition is as noisy as the rest of the record.
    """
    seed_everything(seed)
    generator = torch.Generator().manual_seed(seed)

    log_params = {
        name: init_log_param(name, args.init_method, generator).requires_grad_(True)
        for name in args.drop_params
    }
    hidden_scale = float(scales[observed].mean()) if observed else 1.0
    raw_ic = {
        i: inverse_softplus(
            init_hidden_ic(args.init_method, hidden_scale, generator)
        ).requires_grad_(True)
        for i in hidden
    }
    variables: list[torch.Tensor] = [*log_params.values(), *raw_ic.values()]

    initial_values = {name: float(v.detach().exp()) for name, v in log_params.items()}
    initial_ics = {
        FIELDS[i]: float(F.softplus(v.detach())) for i, v in raw_ic.items()
    }

    if not variables:
        LOG.warning("nothing to fit: no --drop_params and no --drop_state")
        return {
            "seed": seed,
            "initial": initial_values,
            "initial_ics": initial_ics,
            "recovered": {},
            "recovered_ics": {},
            "history": [],
            "final_loss": 0.0,
            "status": "no-free-variables",
        }

    optimizer = make_optimizer(args.optim, variables, args.lr_max)
    # A step can land somewhere the adaptive solver cannot integrate at all.
    # That is a step to reject, not a run to lose, so the schedule carries a
    # backoff factor that shrinks whenever that happens.
    backoff = {"factor": 1.0, "failures": 0}
    base_schedule = lr_lambda(args)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: base_schedule(epoch) * backoff["factor"]
    )

    def clamp_to_priors() -> None:
        """Keep parameters inside the plausible box PRIORS already defines.

        Unbounded exploration reaches absurdly stiff corners - a huge `alpha`
        with a large Hill coefficient - where the adaptive solver fails with a
        step-size underflow rather than merely a bad loss.  The box is the
        same one `--init_method uniform` samples from, so it constrains
        nothing that was ever considered plausible.
        """
        with torch.no_grad():
            for name, raw in log_params.items():
                lo, hi = PRIORS[name]
                raw.clamp_(math.log(lo), math.log(hi))
    loss_fn = LOSSES[args.loss]
    loss_kwargs = {"delta": args.huber_delta} if args.loss == "huber" else {}

    def current_params() -> dict[str, torch.Tensor]:
        values = dict(fixed_params)
        for name, raw in log_params.items():
            values[name] = raw.exp()
        return values

    def current_state() -> RepressilatorState:
        # Observed species start from the measurement, clamped to be physical.
        # Noise on a species whose true initial value is 0 - six of the eight
        # here - readily makes the measurement negative, and while a negative
        # *measurement* is legitimate data, a negative *concentration* is not a
        # state the model can be started from: the Hill term p**n is NaN for a
        # negative base whenever n is not an integer, which it is not as soon
        # as n is being fitted.
        values = {name: data[0, i].clamp(min=0.0) for i, name in enumerate(FIELDS)}
        for i, raw in raw_ic.items():
            values[FIELDS[i]] = F.softplus(raw)
        return RepressilatorState(**values, batch_size=())

    def compute_loss(n_window: int) -> torch.Tensor:
        simulated = integrate(
            model,
            current_state(),
            params_from_dict(current_params()),
            t[:n_window],
            spec,
        )
        sim = torch.stack([getattr(simulated, name) for name in FIELDS], dim=-1)
        return loss_fn(
            sim[:, observed] / scales[observed],
            data[:n_window, observed] / scales[observed],
            **loss_kwargs,
        )

    history: list[dict[str, float]] = []
    status = "completed"
    last_loss = float("nan")
    # The last point at which the loss was successfully evaluated.  `step`
    # evaluates the closure at the *current* parameters before moving, so a
    # solver failure condemns the point the previous step left behind - and
    # rolling back to the start of the failing step would restore exactly
    # that bad point.  This is the point that is actually known integrable.
    last_verified = [v.detach().clone() for v in variables]

    for epoch in range(args.epochs):
        n_window = window_of(epoch)

        def closure() -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            loss = compute_loss(n_window)
            loss.backward()
            # A non-finite gradient must never reach the parameters: Adam would
            # write NaN into them and every later solve would fail on NaN
            # inputs, which surfaces confusingly as a solver error.  Note that
            # clip_grad_norm_ does not help - a NaN gradient gives a NaN norm
            # and stays NaN.
            for variable in variables:
                if variable.grad is not None and not torch.isfinite(variable.grad).all():
                    LOG.warning(
                        "non-finite gradient at epoch %d; zeroing it for this step",
                        epoch,
                    )
                    variable.grad = torch.zeros_like(variable.grad)
            torch.nn.utils.clip_grad_norm_(variables, args.clip)
            return loss

        # The closure evaluates at these values, so on success they are the
        # ones proven integrable.
        pre_step = [v.detach().clone() for v in variables]
        try:
            loss = optimizer.step(closure)
            if loss is None:  # optimisers that do not return the closure value
                with torch.no_grad():
                    loss = compute_loss(n_window)
        except (AssertionError, RuntimeError) as exc:
            # The step landed where the solver cannot integrate.  Undo it, drop
            # the momentum that pointed there, and continue with a smaller
            # step rather than throwing the restart away.
            failed_at = ", ".join(
                f"{k}={float(v.detach().exp()):.4g}" for k, v in log_params.items()
            )
            with torch.no_grad():
                for variable, saved in zip(variables, last_verified):
                    variable.copy_(saved)
            optimizer.state.clear()
            backoff["factor"] *= 0.5
            backoff["failures"] += 1
            LOG.warning(
                "restart seed=%d epoch %d: solver failed at {%s} (%s); step "
                "rejected, learning rate backed off to %.3g of schedule",
                seed,
                epoch,
                failed_at,
                str(exc).splitlines()[0],
                backoff["factor"],
            )
            scheduler.step()
            if backoff["factor"] < 1e-3:
                LOG.error(
                    "restart seed=%d: still unintegrable after %d backoffs; "
                    "aborting it", seed, backoff["failures"],
                )
                status = f"solver-failure-at-epoch-{epoch}"
                break
            continue
        last_verified = pre_step
        clamp_to_priors()
        scheduler.step()

        last_loss = float(loss.detach())
        history.append(
            {
                "epoch": epoch,
                "loss": last_loss,
                "window": n_window,
                "lr": optimizer.param_groups[0]["lr"],
                **{name: float(raw.detach().exp()) for name, raw in log_params.items()},
                **{
                    FIELDS[i]: float(F.softplus(raw.detach()))
                    for i, raw in raw_ic.items()
                },
            }
        )

        if not math.isfinite(last_loss):
            LOG.error("restart seed=%d diverged at epoch %d; aborting it", seed, epoch)
            status = f"diverged-at-epoch-{epoch}"
            break

        if epoch % max(1, args.epochs // 10) == 0 or epoch == args.epochs - 1:
            shown = " ".join(
                f"{k}={float(v.detach().exp()):.4g}" for k, v in log_params.items()
            )
            LOG.info(
                "  epoch %5d/%d  window %5d  loss %.6g  %s",
                epoch,
                args.epochs,
                n_window,
                last_loss,
                shown,
            )

    return {
        "seed": seed,
        "initial": initial_values,
        "initial_ics": initial_ics,
        "recovered": {k: float(v.detach().exp()) for k, v in log_params.items()},
        "recovered_ics": {
            FIELDS[i]: float(F.softplus(v.detach())) for i, v in raw_ic.items()
        },
        "history": history,
        "final_loss": last_loss,
        "status": status,
        "solver_failures": backoff["failures"],
    }


def full_horizon_error(
    model: Repressilator,
    t: torch.Tensor,
    clean: torch.Tensor,
    params: dict[str, torch.Tensor],
    result: dict[str, Any],
    spec: dict[str, Any],
    scales: torch.Tensor,
) -> tuple[float, dict[str, float], RepressilatorState]:
    """Simulate the recovered parameters over the whole stored horizon.

    Early stopping was not wanted, so this is a diagnostic rather than a
    stopping rule - but it is the number that separates a fit which matched
    its training window from one that actually got the period right.
    """
    values = {name: clean[0, i] for i, name in enumerate(FIELDS)}
    for name, value in result["recovered_ics"].items():
        values[name] = torch.tensor(value, dtype=t.dtype)

    try:
        with torch.no_grad():
            simulated = integrate(
                model,
                RepressilatorState(**values, batch_size=()),
                params_from_dict(params),
                t,
                spec,
            )
    except (AssertionError, RuntimeError) as exc:
        # A diverged fit can leave parameters the solver cannot integrate at
        # all.  That is a result to report, not a reason to lose the run.
        LOG.error(
            "full-horizon diagnostic could not be simulated with the recovered "
            "parameters (%s); reporting it as non-finite",
            str(exc).splitlines()[0],
        )
        nan = float("nan")
        return nan, {name: nan for name in FIELDS}, None
    sim = torch.stack([getattr(simulated, name) for name in FIELDS], dim=-1)
    err = (sim - clean).abs()
    per_species = {name: float(err[:, i].max()) for i, name in enumerate(FIELDS)}
    normalised = float(((sim - clean) / scales).pow(2).mean())
    return normalised, per_species, simulated


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def spread(records: list[dict[str, Any]], key: str, name: str) -> float | None:
    """Standard deviation of a recovered quantity across restarts."""
    values = [r[key][name] for r in records if r[key] and math.isfinite(r["final_loss"])]
    return float(torch.tensor(values).std()) if len(values) > 1 else None


def build_metrics(
    args: argparse.Namespace,
    stored: dict[str, torch.Tensor],
    used: dict[str, torch.Tensor],
    stored_ic: dict[str, torch.Tensor],
    best: dict[str, Any],
    restarts: list[dict[str, Any]],
    full_err: float,
    per_species: dict[str, float],
) -> dict[str, Any]:
    """Per-parameter truth / initial / recovered, plus the fit diagnostics."""

    def entry(truth: float, initial: float, recovered: float) -> dict[str, Any]:
        return {
            "truth": truth,
            "initial": initial,
            "recovered": recovered,
            "abs_error": abs(recovered - truth),
            "rel_error": abs(recovered - truth) / abs(truth) if truth else None,
            "initial_rel_error": abs(initial - truth) / abs(truth) if truth else None,
        }

    parameters = {}
    for name in PARAM_FIELDS:
        record: dict[str, Any] = {
            "truth": float(stored[name]),
            "used_in_model": float(used[name]),
            "free": name in args.drop_params,
        }
        if name in args.drop_params:
            record |= entry(
                float(stored[name]), best["initial"][name], best["recovered"][name]
            )
            record["spread_over_restarts"] = spread(restarts, "recovered", name)
            # A value sitting on its prior box is not a fit, it is a fit that
            # ran out of room: either the prior is wrong or the optimiser was
            # driven into the wall and held there by its own momentum.
            lo, hi = PRIORS[name]
            recovered = best["recovered"][name]
            record["at_prior_bound"] = (
                "lower" if recovered <= lo * 1.001
                else "upper" if recovered >= hi * 0.999
                else None
            )
        parameters[name] = record

    initial_conditions = {
        name: entry(
            float(stored_ic[name]),
            best["initial_ics"][name],
            best["recovered_ics"][name],
        )
        | {"spread_over_restarts": spread(restarts, "recovered_ics", name)}
        for name in args.drop_state
    }

    return {
        "observed_species": [f for f in FIELDS if f not in args.drop_state],
        "hidden_species": list(args.drop_state),
        "final_window_loss": best["final_loss"],
        "full_horizon_normalised_mse": full_err,
        "full_horizon_max_abs_error": per_species,
        "restarts": [
            {"seed": r["seed"], "final_loss": r["final_loss"], "status": r["status"]}
            for r in restarts
        ],
        "best_seed": best["seed"],
        "parameters": parameters,
        "initial_conditions": initial_conditions,
    }


def write_restart_csv(
    path: Path, args: argparse.Namespace, restarts: list[dict[str, Any]]
) -> Path:
    """One row per restart: the spread is the identifiability answer."""
    free = list(args.drop_params) + list(args.drop_state)
    columns = ["seed", "status", "final_loss"] + [
        f"{kind}_{name}" for name in free for kind in ("init", "recovered")
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in restarts:
            row: dict[str, Any] = {
                "seed": record["seed"],
                "status": record["status"],
                "final_loss": record["final_loss"],
            }
            for name in args.drop_params:
                row[f"init_{name}"] = record["initial"].get(name)
                row[f"recovered_{name}"] = record["recovered"].get(name)
            for name in args.drop_state:
                row[f"init_{name}"] = record["initial_ics"].get(name)
                row[f"recovered_{name}"] = record["recovered_ics"].get(name)
            writer.writerow(row)
    return path


def plot_fit(
    run_dir: Path,
    args: argparse.Namespace,
    t: torch.Tensor,
    clean: torch.Tensor,
    recovered_traj: RepressilatorState,
    restarts: list[dict[str, Any]],
    best: dict[str, Any],
    truth: dict[str, float],
    metrics: dict[str, Any],
) -> Path:
    """Loss curves, traces of every free variable against truth, trajectories."""
    fig = plt.figure(figsize=(11, 16))
    # Two blocks: diagnostics on top, species panels below.  They are laid out
    # separately because the species block carries a secondary top axis whose
    # label needs room that a single even grid does not leave.
    outer = fig.add_gridspec(2, 1, height_ratios=[2.0, 3.6], hspace=0.28)
    grid = [
        *outer[0].subgridspec(2, 1, hspace=0.55),
        *outer[1].subgridspec(3, 1, hspace=0.3),
    ]

    # --- loss curves ----------------------------------------------------
    ax_loss = fig.add_subplot(grid[0])
    for record in restarts:
        if not record["history"]:
            continue
        ax_loss.semilogy(
            [h["epoch"] for h in record["history"]],
            [h["loss"] for h in record["history"]],
            linewidth=1.8 if record is best else 1.0,
            alpha=1.0 if record is best else 0.5,
            label=f"seed {record['seed']}" + (" (best)" if record is best else ""),
        )
    if best["history"]:
        windows = [h["window"] for h in best["history"]]
        for epoch in range(1, len(windows)):
            if windows[epoch] != windows[epoch - 1]:
                ax_loss.axvline(epoch, color="grey", linestyle=":", linewidth=1)
    ax_loss.set_title(
        "Training loss (dotted lines: curriculum widens the window)",
        fontsize=10,
        loc="left",
    )
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel(f"{args.loss} loss\n(normalised: {args.normalize})", fontsize=9)
    ax_loss.grid(True, which="both", alpha=0.4, linewidth=0.5)
    if ax_loss.get_legend_handles_labels()[0]:
        ax_loss.legend(fontsize=8, ncols=4)

    # --- free-variable traces -------------------------------------------
    ax_par = fig.add_subplot(grid[1])
    free = list(args.drop_params) + list(args.drop_state)
    if free:
        colours = plt.cm.tab10.colors
        for k, name in enumerate(free):
            colour = colours[k % len(colours)]
            reference = truth[name]
            for record in restarts:
                if not record["history"] or reference == 0:
                    continue
                ax_par.plot(
                    [h["epoch"] for h in record["history"]],
                    [h[name] / reference for h in record["history"]],
                    color=colour,
                    linewidth=1.6 if record is best else 0.8,
                    alpha=1.0 if record is best else 0.4,
                    label=name if record is best else None,
                )
        ax_par.axhline(1.0, color="black", linestyle="--", linewidth=1, label="truth")
        ax_par.set_yscale("log")
        ax_par.legend(fontsize=8, ncols=4)
    else:
        ax_par.text(0.5, 0.5, "no free variables", ha="center", va="center")
    ax_par.set_title(
        "Free variables (parameters and hidden initial conditions), "
        "relative to truth",
        fontsize=10,
        loc="left",
    )
    ax_par.set_xlabel("epoch")
    ax_par.set_ylabel("value / truth", fontsize=9)
    ax_par.grid(True, which="both", alpha=0.4, linewidth=0.5)

    # --- trajectories -----------------------------------------------------
    axes = [fig.add_subplot(grid[2 + i]) for i in range(3)]
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
    if recovered_traj is None:
        axes[0].text(
            0.5, 0.5, "recovered parameters could not be simulated",
            ha="center", va="center", transform=axes[0].transAxes,
        )
    plot_species_panels(
        axes,
        t,
        {name: clean[:, i] for i, name in enumerate(FIELDS)},
        linewidth=3.5,
        alpha=0.25,
        label_suffix=" (true)",
        solid_only=True,
    )
    if recovered_traj is not None:
        plot_species_panels(
            axes,
            t,
            {name: getattr(recovered_traj, name) for name in FIELDS},
            linewidth=1.3,
            label_suffix=" (fit)",
        )
    for ax in axes:
        ax.legend(loc="upper right", fontsize=7, ncols=3, framealpha=0.9)
    add_time_axes(axes, axes[-1], axes[0])

    params_text = ", ".join(args.drop_params) or "none"
    hidden_text = ", ".join(args.drop_state) or "none"
    fig.suptitle(
        f"Parameter recovery - {args.optim}, {args.loss}, {args.epochs} epochs, "
        f"{args.restarts} restart(s)\n"
        f"free params: {params_text}   |   hidden species: {hidden_text}   |   "
        f"full-horizon normalised MSE: "
        f"{metrics['full_horizon_normalised_mse']:.4g}",
        fontsize=11,
    )
    # tight_layout cannot handle the secondary axis, so place the margins
    # explicitly rather than let it warn and guess.
    fig.subplots_adjust(top=0.93, bottom=0.05, left=0.10, right=0.97)
    return close_figure(fig, run_dir / "fit.png")


def run_dir_name(args: argparse.Namespace) -> str:
    """Timestamp plus the arguments that define the experiment."""
    parts = [
        datetime.now().strftime("%Y%m%d-%H%M%S"),
        "partial",
        "p-" + ("+".join(args.drop_params) if args.drop_params else "none"),
        "s-" + ("+".join(args.drop_state) if args.drop_state else "none"),
        args.optim,
        args.loss,
        f"ep{args.epochs}",
        f"lr{args.lr_max:g}" + (f"-{args.lr_min:g}" if args.lr_min else ""),
        f"init{args.init_method}",
        f"r{args.restarts}",
        f"seed{args.seed}",
    ]
    if args.sample_hz:
        parts.append(f"hz{args.sample_hz:g}")
    if args.noise:
        parts.append(f"noise{args.noise:g}")
    if args.tag:
        parts.append(args.tag)
    return "_".join(parts)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "run_dir", type=Path, help="clean-run directory from run_clean.py"
    )
    parser.add_argument(
        "--set",
        action="append",
        metavar="NAME=VALUE",
        dest="overrides",
        help="override a stored parameter (repeatable); warns, as the data no "
        "longer matches the model",
    )
    parser.add_argument(
        "--drop_params",
        nargs="*",
        default=[],
        choices=PARAM_FIELDS,
        metavar="NAME",
        help=f"parameters to hide and estimate {list(PARAM_FIELDS)}",
    )
    parser.add_argument(
        "--drop_state",
        nargs="*",
        default=[],
        choices=FIELDS,
        metavar="NAME",
        help="species to mask from the data; their initial value becomes a free "
        f"variable and the ODE supplies the rest {list(FIELDS)}",
    )
    parser.add_argument(
        "--init_method",
        choices=["uniform", "normal", "zero"],
        default="uniform",
        help="initialisation for every free variable, in the space it is optimised in",
    )
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr_max", type=float, default=0.05)
    parser.add_argument(
        "--lr_min",
        type=float,
        default=None,
        help="linear decay target; constant learning rate when omitted",
    )
    parser.add_argument("--loss", choices=list(LOSSES), default="mse")
    parser.add_argument("--huber_delta", type=float, default=1.0)
    parser.add_argument("--optim", default="Adam", help="any torch.optim class name")
    parser.add_argument(
        "--normalize",
        choices=["range", "std", "none"],
        default="range",
        help="per-species loss scaling, so mRNA does not drown out the reporter",
    )
    parser.add_argument(
        "--sample_hz",
        type=float,
        default=None,
        help="subsample the record to a real imaging cadence, given as a "
        "frequency in Hz of wall-clock time. Time-lapse microscopy every 5 min "
        "is 1/300 s = 0.00333 Hz. Converted to a stride on the stored grid via "
        "the mRNA lifetime (1 model unit = %.3f min); the full grid is used "
        "when omitted" % TAU_M_MIN,
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0.0,
        help="Gaussian sigma, relative to each species' range, added to the data",
    )
    parser.add_argument("--restarts", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--curriculum",
        default="1,2,4",
        help="window sizes in periods, split evenly over the epochs; growing the "
        "horizon avoids the flat loss landscape of a many-period fit",
    )
    parser.add_argument("--clip", type=float, default=10.0, help="gradient-norm clip")
    parser.add_argument(
        "--adjoint",
        action="store_true",
        help="back-propagate with torchdiffeq's adjoint solver: memory stops "
        "growing with the window, at the cost of a second solve per step. Use "
        "it for long windows or many concurrent fits - plain odeint keeps the "
        "whole solver tape and can exhaust RAM",
    )
    parser.add_argument(
        "--tag", default=None, help="appended to the output directory name"
    )
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    args.overrides = parse_overrides(args.overrides)
    args.stages = [float(s) for s in args.curriculum.split(",") if s.strip()]

    clash = set(args.overrides) & set(args.drop_params)
    if clash:
        raise SystemExit(
            f"--set and --drop_params both name {sorted(clash)}: an override of a "
            "parameter that is about to be re-estimated would be discarded "
            "immediately. Drop it or override it, not both."
        )
    return args


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    payload = load_run(args.run_dir)
    config = payload["config"]
    torch.set_default_dtype(getattr(torch, config["dtype"]))
    seed_everything(args.seed)

    LOG.info(
        "fitting from %s (%s, %d steps, dt=%g, %s)",
        args.run_dir,
        config["method"],
        config["steps"],
        config["dt"],
        config["dtype"],
    )

    model = Repressilator()
    spec = config["integration"]

    if args.adjoint:
        if spec["runner"] != "run_torchdiffeq":
            raise SystemExit(
                "--adjoint needs a clean run integrated with --ode; this one used "
                f"{spec['runner']}."
            )
        spec = {**spec, "kwargs": {**spec["kwargs"], "adjoint": True}}
        LOG.info("using the adjoint solver: constant memory, one extra solve per step")

    t = payload["t"]
    clean = torch.stack([payload["trajectory"][name] for name in FIELDS], dim=-1)

    if args.sample_hz:
        # A real experiment samples in wall-clock time, so the cadence is given
        # in Hz and converted here: 1 model unit is one mRNA lifetime.
        interval_s = 1.0 / args.sample_hz
        interval_model = interval_s / 60.0 / TAU_M_MIN
        grid_dt = float(t[1] - t[0])
        stride = round(interval_model / grid_dt)
        if stride < 1:
            LOG.warning(
                "--sample_hz %g asks for a frame every %.3g min, finer than the "
                "stored grid's %.3g min; using every stored point instead",
                args.sample_hz, interval_s / 60.0, grid_dt * TAU_M_MIN,
            )
            stride = 1
        if stride > 1 and spec["runner"] == "run_fixed_step":
            raise SystemExit(
                "--sample_hz needs a clean run integrated with --ode. Subsampling "
                "a fixed-step run would coarsen the integration step itself, not "
                "just the observation times, and change the dynamics being fitted."
            )
        t = t[::stride]
        clean = clean[::stride]
        LOG.info(
            "sampling at %g Hz = one frame every %.3g min = %.3g model units "
            "(stride %d): %d frames over %.0f min",
            args.sample_hz, interval_s / 60.0, interval_model, stride,
            t.numel(), float(t[-1]) * TAU_M_MIN,
        )
        args.sample_stride, args.sample_frames = stride, int(t.numel())

    stored_params = payload["params"]
    used_params = apply_overrides(stored_params, args.overrides)
    fixed_params = {k: v for k, v in used_params.items() if k not in args.drop_params}

    observed = [i for i, name in enumerate(FIELDS) if name not in args.drop_state]
    hidden = [i for i, name in enumerate(FIELDS) if name in args.drop_state]
    scales = species_scales(clean, observed, args.normalize)

    data = clean.clone()
    if args.noise:
        generator = torch.Generator().manual_seed(args.seed)
        ranges = clean.max(dim=0).values - clean.min(dim=0).values
        data = data + args.noise * ranges * torch.randn(
            data.shape, generator=generator, dtype=data.dtype
        )
        LOG.info("added Gaussian noise, sigma = %g x each species' range", args.noise)

    window_of = curriculum_windows(
        t, payload["summary"].get("period"), args.stages, args.epochs
    )

    started = time.perf_counter()
    restarts = []
    for k in range(args.restarts):
        seed = args.seed + k
        LOG.info("restart %d/%d (seed %d)", k + 1, args.restarts, seed)
        restarts.append(
            fit_once(
                args, model, t, data, fixed_params, spec,
                observed, hidden, scales, window_of, seed,
            )
        )
    wall_s = time.perf_counter() - started

    # A restart that was aborted keeps the last loss it managed to compute, so
    # it must not win on that stale value while a completed restart exists.
    completed = [
        r
        for r in restarts
        if r["status"] in ("completed", "no-free-variables")
        and math.isfinite(r["final_loss"])
    ]
    if not completed:
        LOG.error(
            "no restart completed (%s); reported values come from an aborted run",
            ", ".join(sorted({r["status"] for r in restarts})),
        )
    best = min(completed or restarts, key=lambda r: r["final_loss"])

    recovered_params = dict(fixed_params)
    for name, value in best["recovered"].items():
        recovered_params[name] = torch.tensor(value, dtype=t.dtype)

    full_err, per_species, recovered_traj = full_horizon_error(
        model, t, clean, recovered_params, best, spec, scales
    )
    metrics = build_metrics(
        args, stored_params, used_params, payload["initial_state"], best,
        restarts, full_err, per_species,
    )
    metrics["wall_s"] = wall_s

    # --- report -----------------------------------------------------------
    print(f"\noptimiser       : {args.optim}   loss: {args.loss}   epochs: {args.epochs}")
    print(f"observed species: {[FIELDS[i] for i in observed]}")
    print(f"hidden species  : {[FIELDS[i] for i in hidden] or 'none'}")
    print(f"wall time       : {wall_s:.1f} s over {args.restarts} restart(s)")
    print(f"final window loss: {best['final_loss']:.6g}   (seed {best['seed']})")
    print(f"full-horizon normalised MSE: {full_err:.6g}")

    rows = [(n, metrics["parameters"][n]) for n in args.drop_params]
    rows += [(n, metrics["initial_conditions"][n]) for n in args.drop_state]
    if rows:
        print("\nfree variable      truth      initial    recovered    rel. error")
        for name, entry in rows:
            rel = entry["rel_error"]
            rel_text = "         n/a" if rel is None else f"{rel:13.2%}"
            bound = entry.get("at_prior_bound")
            print(
                f"  {name:<14}{entry['truth']:10.4g}{entry['initial']:12.4g}"
                f"{entry['recovered']:13.4g}{rel_text}"
                + (f"   <-- pinned to {bound} prior bound" if bound else "")
            )
        pinned = [n for n, e in rows if e.get("at_prior_bound")]
        if pinned:
            LOG.warning(
                "%s finished on the prior box. That is not a converged value: "
                "widen PRIORS if the bound is wrong, or lower --lr_max, since a "
                "large early step drives a parameter into the wall and the "
                "optimiser's momentum keeps it there.",
                ", ".join(pinned),
            )

    # --- save -------------------------------------------------------------
    run_dir = RESULTS_DIR / run_dir_name(args)
    run_dir.mkdir(parents=True, exist_ok=True)

    resolved = {
        "format_version": config["format_version"],
        "source_run": str(args.run_dir),
        "source_provenance": config["provenance"],
        "args": {k: v for k, v in vars(args).items() if k != "overrides"},
        "overrides": args.overrides,
        "priors": {k: PRIORS[k] for k in args.drop_params},
        "provenance": provenance(args.seed),
    }
    written = [run_dir / "fit.pt"]
    torch.save(
        {
            "config": resolved,
            "metrics": metrics,
            "restarts": restarts,
            "recovered_params": recovered_params,
            "scales": scales,
            "t": t,
        },
        written[0],
    )
    written.append(write_json(run_dir / "config.json", resolved))
    written.append(write_json(run_dir / "metrics.json", metrics))
    written.append(write_restart_csv(run_dir / "restarts.csv", args, restarts))
    if not args.no_plot:
        truth = {n: float(stored_params[n]) for n in args.drop_params}
        truth |= {n: float(payload["initial_state"][n]) for n in args.drop_state}
        written.append(
            plot_fit(run_dir, args, t, clean, recovered_traj, restarts, best,
                     truth, metrics)
        )

    print(f"\nsaved -> {run_dir.relative_to(REPO_ROOT)}/")
    for path in written:
        print(f"           {path.name}  ({path.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
