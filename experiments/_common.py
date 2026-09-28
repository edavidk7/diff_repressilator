"""Shared pieces of the experiment scripts.

Holds everything that both the forward experiment (``run_clean.py``) and the
inverse one (``run_partial.py``) need: the default parameters and initial
state, the trajectory-log format, provenance capture, and the plotting panel
spec, so the two scripts cannot drift apart in how they read and write runs.
"""

import json
import logging
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch

from differentiable_cell.data import RepressilatorParams, RepressilatorState
from differentiable_cell.helpers import TAU_M_MIN, estimate_period, to_minutes  # noqa: F401  (re-exported)
from differentiable_cell.model import Repressilator
from differentiable_cell.run import (
    FIELDS,
    PARAM_FIELDS,
    run_fixed_step,
    run_torchdiffeq,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "run_results"

FORMAT_VERSION = 1
"""Schema version of the saved trajectory payload.

Bumped whenever the layout of the ``.pt`` dictionary changes.  Readers refuse
a version they do not know rather than silently mis-reading fields.
"""

# One colour per gene, shared between its mRNA and its protein so the phase
# relationship between the two is easy to read across panels.
COLOURS = {
    "lacI": "tab:blue",
    "tetR": "tab:red",
    "cI": "tab:purple",
    "gfp": "tab:green",  # GFP keeps green, so cI takes purple
}

PANELS: tuple[tuple[str, str, list[tuple[str, str, str]]], ...] = (
    (
        "Repressor mRNA",
        "mRNA\n(x translation efficiency)",
        [
            ("m_lacI", "lacI", "$m_{lacI}$"),
            ("m_tetR", "tetR", "$m_{tetR}$"),
            ("m_cI", "cI", "$m_{cI}$"),
        ],
    ),
    (
        "Repressor protein",
        "protein\n(x $K_M$ ~ 40 monomers/cell)",
        [
            ("p_LacI", "lacI", "LacI"),
            ("p_TetR", "tetR", "TetR"),
            ("p_CI", "cI", "CI"),
        ],
    ),
    (
        "GFP reporter (P$_{Ltet01}$, repressed by TetR)",
        "reporter\n(rescaled units)",
        [("m_gfp", "gfp", "$m_{gfp}$"), ("p_GFP", "gfp", "GFP")],
    ),
)
"""Plot layout: (panel title, y label, [(field, gene colour key, label)])."""


# --------------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------------


def default_params() -> RepressilatorParams:
    """Parameters in the oscillatory regime, following Box 1 of the paper."""
    return RepressilatorParams(
        alpha=torch.tensor(216.0),
        alpha_0=torch.tensor(0.216),  # 1e-3 * alpha: tight repression
        n=torch.tensor(2.0),  # two operator sites per promoter
        beta=torch.tensor(0.2),  # 2 min mRNA vs 10 min ssrA-tagged protein
        alpha_GFP=torch.tensor(216.0),
        beta_GFP=torch.tensor(2.0 / 90.0),  # GFP-AAV half-life ~90 min
        batch_size=(),
    )


def default_state() -> RepressilatorState:
    """An asymmetric initial condition, so the loop leaves its steady state."""
    zero = torch.tensor(0.0)
    return RepressilatorState(
        m_lacI=zero.clone(),
        m_tetR=zero.clone(),
        m_cI=zero.clone(),
        p_LacI=torch.tensor(5.0),
        p_TetR=zero.clone(),
        p_CI=torch.tensor(15.0),
        m_gfp=zero.clone(),
        p_GFP=zero.clone(),
        batch_size=(),
    )


# --------------------------------------------------------------------------
# Packing helpers
# --------------------------------------------------------------------------


def stack_states(trajectory: RepressilatorState) -> torch.Tensor:
    """Stack a state (or trajectory) into a tensor of shape ``(..., 8)``."""
    return torch.stack([getattr(trajectory, name) for name in FIELDS], dim=-1)


def unstack_states(y: torch.Tensor, batch_size: torch.Size) -> RepressilatorState:
    """Inverse of :func:`stack_states`."""
    return RepressilatorState(
        **{name: y[..., i] for i, name in enumerate(FIELDS)},
        batch_size=batch_size,
    )


def params_from_dict(values: dict[str, torch.Tensor]) -> RepressilatorParams:
    """Build a :class:`RepressilatorParams` from a plain field -> tensor map."""
    return RepressilatorParams(
        **{name: values[name] for name in PARAM_FIELDS}, batch_size=()
    )


# --------------------------------------------------------------------------
# Integration, described by a spec so a saved run can be replayed exactly
# --------------------------------------------------------------------------


def integration_spec(
    ode: bool, method: str, rtol: float, atol: float
) -> dict[str, Any]:
    """Describe an integration call in a form that can be saved and replayed."""
    if ode:
        return {
            "runner": "run_torchdiffeq",
            "kwargs": {"method": method, "rtol": rtol, "atol": atol},
        }
    return {"runner": "run_fixed_step", "kwargs": {"method": method}}


def integrate(
    model: Repressilator,
    state: RepressilatorState,
    params: RepressilatorParams,
    t: torch.Tensor,
    spec: dict[str, Any],
) -> RepressilatorState:
    """Integrate according to a spec produced by :func:`integration_spec`.

    Keeping the dispatch in one place is what lets ``run_partial`` reproduce a
    clean run's forward problem without re-deriving it from that run's CLI.
    """
    runner = {"run_fixed_step": run_fixed_step, "run_torchdiffeq": run_torchdiffeq}
    if spec["runner"] not in runner:
        raise ValueError(f"Unknown runner {spec['runner']!r} in integration spec.")
    return runner[spec["runner"]](model, state, params, t, **spec["kwargs"])


def verify_roundtrip(
    model: Repressilator,
    payload: dict[str, Any],
) -> float:
    """Re-integrate a saved run from its own log and report the max deviation.

    This is what turns "the log is 1:1 with the run" from an intention into a
    checked number: it rebuilds the initial state, parameters and integration
    call from the payload alone and compares against the stored trajectory.
    """
    state = RepressilatorState(**payload["initial_state"], batch_size=())
    params = params_from_dict(payload["params"])
    with torch.no_grad():
        replay = integrate(
            model, state, params, payload["t"], payload["config"]["integration"]
        )
    stored = torch.stack([payload["trajectory"][name] for name in FIELDS], dim=-1)
    return float((stack_states(replay) - stored).abs().max())


# --------------------------------------------------------------------------
# Provenance and reproducibility
# --------------------------------------------------------------------------


def git_state() -> dict[str, Any]:
    """Current commit and whether the tree is dirty; tolerant of no git."""
    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(
                args, cwd=REPO_ROOT, capture_output=True, text=True, timeout=5
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    commit = run("git", "rev-parse", "HEAD")
    status = run("git", "status", "--porcelain")
    return {
        "git_commit": commit,
        "git_dirty": None if status is None else bool(status),
    }


def provenance(seed: int) -> dict[str, Any]:
    """Everything needed to explain where a run came from."""
    return {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "argv": sys.argv,
        "script": Path(sys.argv[0]).name,
        "seed": seed,
        # str(): TorchVersion is not a weights_only-safe global.
        "torch_version": str(torch.__version__),
        "python_version": platform.python_version(),
        "hostname": platform.node(),
        **git_state(),
    }


def seed_everything(seed: int) -> None:
    """Seed the RNGs used here.  Cheap, and keeps the field from ever being absent."""
    torch.manual_seed(seed)


def setup_logging(verbose: bool = False) -> None:
    """Consistent log formatting across the experiment scripts."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)-8s %(message)s",
    )


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    """Write a JSON file, converting tensors and paths to plain values."""

    def default(obj: Any) -> Any:
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        return str(obj)

    path.write_text(json.dumps(payload, indent=2, default=default) + "\n")
    return path


# --------------------------------------------------------------------------
# Analysis and plotting
# --------------------------------------------------------------------------


def plot_species_panels(
    axes,
    t: torch.Tensor,
    series: dict[str, torch.Tensor],
    linewidth: float = 1.4,
    alpha: float = 1.0,
    label_suffix: str = "",
    solid_only: bool = False,
) -> None:
    """Draw the three species panels of :data:`PANELS` onto ``axes``.

    mRNA is dashed and protein solid throughout, so the translation lag is
    readable; ``solid_only`` overrides that when a second, contrasting
    trajectory is drawn on the same axes.

    Args:
        axes: three axes, one per panel.
        t: times in model units; converted to minutes for the x axis.
        series: field name -> values, for every field in :data:`FIELDS`.
        linewidth, alpha: line style, for over-plotting two trajectories.
        label_suffix: appended to each legend entry, e.g. " (true)".
        solid_only: draw every species solid rather than dashing the mRNAs.
    """
    minutes = to_minutes(t).detach().cpu().numpy()
    for ax, (title, ylabel, entries) in zip(axes, PANELS):
        for field, gene, label in entries:
            style = "-" if solid_only or not field.startswith("m_") else "--"
            ax.plot(
                minutes,
                series[field].detach().cpu().numpy(),
                style,
                color=COLOURS[gene],
                linewidth=linewidth,
                alpha=alpha,
                label=f"{label}{label_suffix}",
            )
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, which="major", linewidth=0.6, alpha=0.5)
        ax.grid(True, which="minor", linewidth=0.3, alpha=0.3)
        ax.minorticks_on()
        ax.margins(x=0)


def add_time_axes(axes, bottom_ax, top_ax) -> None:
    """Label the shared x axis in minutes, with model units on a top axis."""
    bottom_ax.set_xlabel(f"time (min, at $\\tau_m$ = {TAU_M_MIN:.2f} min)")
    top = top_ax.secondary_xaxis(
        "top", functions=(lambda m: m / TAU_M_MIN, lambda u: u * TAU_M_MIN)
    )
    top.set_xlabel("time (mRNA lifetimes, model units)", fontsize=9)


def params_caption(params: RepressilatorParams) -> str:
    """One-line LaTeX summary of a parameter set, for figure titles."""
    return (
        rf"$\alpha$={float(params.alpha):g}, $\alpha_0$={float(params.alpha_0):g}, "
        rf"$n$={float(params.n):g}, $\beta$={float(params.beta):g}, "
        rf"$\alpha_{{GFP}}$={float(params.alpha_GFP):g}, "
        rf"$\beta_{{GFP}}$={float(params.beta_GFP):.4g}"
    )


def load_run(run_dir: Path) -> dict[str, Any]:
    """Load a clean-run log, checking the schema version.

    Args:
        run_dir: a directory written by ``run_clean.py``, or the ``.pt`` itself.

    Returns:
        The saved payload.

    Raises:
        FileNotFoundError: if no ``trajectory.pt`` is there.
        ValueError: if the payload was written by an incompatible version.
    """
    path = run_dir if run_dir.suffix == ".pt" else run_dir / "trajectory.pt"
    if not path.exists():
        raise FileNotFoundError(f"No trajectory log at {path}")

    payload = torch.load(path, weights_only=True)
    version = payload.get("config", {}).get("format_version")
    if version != FORMAT_VERSION:
        raise ValueError(
            f"{path} has format_version {version!r}, but this code reads "
            f"{FORMAT_VERSION}. Re-generate the clean run with the current "
            "run_clean.py."
        )
    return payload


def close_figure(fig, path: Path) -> Path:
    """Save and close a figure."""
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
