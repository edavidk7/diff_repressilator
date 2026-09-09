"""Run a clean (deterministic, noise-free) repressilator simulation.

Integrates the model for a fixed number of steps with either the hand-rolled
fixed-step solver or ``torchdiffeq``, prints a summary, and writes the
parameters, the full trajectory - initial state included - and a plot of all
eight species into a per-run directory under ``run_results/``, named
``<timestamp>_<args>``.

The log is written to be replayable: it carries the exact time grid, the
integration call, the parameters, the initial state and the provenance of the
run, and a round-trip check is performed at save time so that "1:1" is a
verified number rather than an intention.  ``run_partial.py`` consumes it.

Examples:
    python experiments/run_clean.py --fixed --steps 20000
    python experiments/run_clean.py --ode --steps 2000 --method dopri5
"""

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files without needing a display

import matplotlib.pyplot as plt
import torch

from differentiable_cell.data import RepressilatorParams, RepressilatorState
from differentiable_cell.helpers import TAU_M_MIN, to_minutes
from differentiable_cell.model import Repressilator
from differentiable_cell.run import FIELDS, PARAM_FIELDS

from _common import (
    FORMAT_VERSION,
    RESULTS_DIR,
    REPO_ROOT,
    add_time_axes,
    close_figure,
    default_params,
    default_state,
    estimate_period,
    integrate,
    integration_spec,
    params_caption,
    plot_species_panels,
    provenance,
    seed_everything,
    setup_logging,
    stack_states,
    verify_roundtrip,
    write_json,
)

LOG = logging.getLogger("run_clean")

ROUNDTRIP_TOL = {"float64": 1e-8, "float32": 1e-2}
"""Acceptable replay deviation per dtype; float32 accumulates over 20k steps."""


def summarise(
    t: torch.Tensor,
    trajectory: RepressilatorState,
    args: argparse.Namespace,
    wall_s: float,
) -> dict[str, float | None]:
    """Print a human-readable summary and return the derived quantities."""
    period = estimate_period(t, trajectory.p_LacI)
    period_min = None if period is None else float(to_minutes(torch.tensor(period)))

    solver = f"torchdiffeq/{args.method}" if args.ode else f"fixed/{args.method}"
    print(f"solver          : {solver}")
    print(f"steps           : {args.steps}   dt = {args.dt}")
    print(
        f"horizon         : {float(t[-1]):.2f} model units "
        f"= {float(to_minutes(t[-1])):.1f} min  (tau_m = {TAU_M_MIN:.3f} min)"
    )
    print(f"wall time       : {wall_s:.2f} s")
    if period is None:
        print("period          : no oscillation detected")
    else:
        print(f"period          : {period:.2f} model units = {period_min:.1f} min")
        print("                  (paper reports 160 +/- 40 min in single cells)")

    print("\nspecies             min        max       final")
    for name in FIELDS:
        x = getattr(trajectory, name)
        print(
            f"  {name:<10} {float(x.min()):10.3f} {float(x.max()):10.3f} "
            f"{float(x[-1]):11.3f}"
        )

    if not torch.isfinite(stack_states(trajectory)).all():
        print("\nWARNING: trajectory contains non-finite values - reduce dt.")

    return {"period": period, "period_min": period_min, "wall_s": wall_s}


def run_dir_name(args: argparse.Namespace) -> str:
    """Directory name for a run: timestamp followed by the arguments.

    The arguments are folded into the name so that two runs differing only in
    step size or solver cannot overwrite each other, and so that the contents
    of ``run_results/`` are readable without opening any file.
    """
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    parts = [
        stamp,
        "ode" if args.ode else "fixed",
        args.method,
        f"steps{args.steps}",
        f"dt{args.dt:g}",
        args.dtype,
        f"seed{args.seed}",
    ]
    if args.ode:
        parts += [f"rtol{args.rtol:g}", f"atol{args.atol:g}"]
    return "_".join(parts)


def build_payload(
    t: torch.Tensor,
    trajectory: RepressilatorState,
    params: RepressilatorParams,
    args: argparse.Namespace,
    summary: dict[str, float | None],
) -> dict:
    """Assemble the saved dictionary.

    Everything is stored as plain tensors rather than as TensorClasses, so the
    file loads with ``torch.load(..., weights_only=True)`` and does not depend
    on this package's classes still existing in their current form.
    """
    return {
        # Trajectory, index 0 being the initial state.
        "t": t,
        "trajectory": {name: getattr(trajectory, name) for name in FIELDS},
        "initial_state": {name: getattr(trajectory, name)[0] for name in FIELDS},
        "params": {name: getattr(params, name) for name in PARAM_FIELDS},
        "config": {
            "format_version": FORMAT_VERSION,
            "solver": "ode" if args.ode else "fixed",
            "method": args.method,
            "steps": args.steps,
            "dt": args.dt,
            "rtol": args.rtol,
            "atol": args.atol,
            "dtype": args.dtype,
            "tau_m_min": TAU_M_MIN,
            "field_order": list(FIELDS),
            "param_order": list(PARAM_FIELDS),
            # The exact call needed to replay this run, so a reader does not
            # have to re-derive it from the CLI flags.
            "integration": integration_spec(
                args.ode, args.method, args.rtol, args.atol
            ),
            "model": {
                "class": Repressilator.__name__,
                "module": Repressilator.__module__,
                "fields": list(FIELDS),
                "params": list(PARAM_FIELDS),
            },
            "provenance": provenance(args.seed),
        },
        "summary": summary,
    }


def plot(
    run_dir: Path,
    t: torch.Tensor,
    trajectory: RepressilatorState,
    params: RepressilatorParams,
    args: argparse.Namespace,
    summary: dict[str, float | None],
) -> Path:
    """Plot all eight species against real time.

    Three stacked panels: the three repressor mRNAs, the three repressor
    proteins, and the reporter.  The x axis is in minutes, with model time
    (multiples of the mRNA lifetime) on a secondary axis on top, so the plot
    can be read against both the equations and a microscopy timecourse.
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    plot_species_panels(
        axes, t, {name: getattr(trajectory, name) for name in FIELDS}
    )
    for ax in axes:
        ax.legend(loc="upper right", fontsize=9, ncols=3, framealpha=0.9)
    add_time_axes(axes, axes[-1], axes[0])

    period = summary["period_min"]
    period_text = (
        "no oscillation detected" if period is None else f"period ~ {period:.0f} min"
    )
    solver = f"torchdiffeq/{args.method}" if args.ode else f"fixed-step/{args.method}"
    fig.suptitle(
        f"Repressilator, clean run - {solver}, {args.steps} steps, dt = {args.dt:g}\n"
        f"{params_caption(params)} - {period_text}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return close_figure(fig, run_dir / "trajectory.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    solver = parser.add_mutually_exclusive_group()
    solver.add_argument(
        "--fixed",
        action="store_true",
        help="integrate with the hand-rolled fixed-step solver (default)",
    )
    solver.add_argument(
        "--ode",
        action="store_true",
        help="integrate with torchdiffeq's adaptive solvers",
    )
    parser.add_argument(
        "--steps", type=int, default=20000, help="number of steps to take"
    )
    parser.add_argument(
        "--dt", type=float, default=0.005, help="step size in model units"
    )
    parser.add_argument(
        "--method",
        default=None,
        help="solver: 'rk4'/'euler' for --fixed, any torchdiffeq name for --ode "
        "(defaults to 'rk4' and 'dopri5' respectively)",
    )
    parser.add_argument(
        "--rtol", type=float, default=1e-7, help="--ode relative tolerance"
    )
    parser.add_argument(
        "--atol", type=float, default=1e-9, help="--ode absolute tolerance"
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed, recorded in the log")
    parser.add_argument(
        "--no-plot", action="store_true", help="skip writing the trajectory plot"
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="skip the replay round-trip check (which doubles the runtime)",
    )
    parser.add_argument(
        "--dtype",
        default="float64",
        choices=["float32", "float64"],
        help="working precision (float64 by default: these are stiff-ish equations)",
    )

    args = parser.parse_args()
    if args.method is None:
        args.method = "dopri5" if args.ode else "rk4"
    return args


def main() -> None:
    args = parse_args()
    setup_logging()
    torch.set_default_dtype(getattr(torch, args.dtype))
    seed_everything(args.seed)

    model = Repressilator()
    params = default_params()
    state = default_state()
    spec = integration_spec(args.ode, args.method, args.rtol, args.atol)

    # `steps` steps means `steps + 1` reported states, the initial one included.
    t = torch.arange(args.steps + 1, dtype=torch.get_default_dtype()) * args.dt

    started = time.perf_counter()
    with torch.no_grad():
        trajectory = integrate(model, state, params, t, spec)
    wall_s = time.perf_counter() - started

    summary = summarise(t, trajectory, args, wall_s)
    payload = build_payload(t, trajectory, params, args, summary)

    if not args.no_verify:
        err = verify_roundtrip(model, payload)
        summary["roundtrip_max_abs_err"] = err
        payload["summary"] = summary
        tol = ROUNDTRIP_TOL[args.dtype]
        print(f"\nround-trip      : max abs replay error {err:.3g} (tol {tol:g})")
        if err > tol:
            LOG.warning(
                "replaying this log does not reproduce the stored trajectory "
                "(%.3g > %.3g); the run is not 1:1 reconstructable.",
                err,
                tol,
            )

    run_dir = RESULTS_DIR / run_dir_name(args)
    run_dir.mkdir(parents=True, exist_ok=True)

    written = [run_dir / "trajectory.pt"]
    torch.save(payload, written[0])
    written.append(write_json(run_dir / "config.json", payload["config"] | {"summary": summary}))
    if not args.no_plot:
        written.append(plot(run_dir, t, trajectory, params, args, summary))

    print(f"\nsaved -> {run_dir.relative_to(REPO_ROOT)}/")
    for path in written:
        print(f"           {path.name}  ({path.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
