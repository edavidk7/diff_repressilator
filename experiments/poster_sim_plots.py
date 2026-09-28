"""Poster-styled simulation figures: one ODE cell and one Gillespie cell, camera noise on.

    python experiments/poster_sim_plots.py      -> run_results/figures/poster_sim_{ode,gillespie}[_dotted].{png,pdf,svg}

10 h of imaging at 5-min frames, the three reporters (Box 1 parameters), styled
for the poster: black ink, gray (#b3b3b3) gridlines, channel
colours as in the circuit schematic, transparent background.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from differentiable_cell.reporter_sim import FRAME_MIN, box1_truth, generate

torch.set_default_dtype(torch.float64)
OUT = Path(__file__).resolve().parent.parent / "run_results" / "figures"
INK = "black"
GRID = (179 / 255, 179 / 255, 179 / 255)  # #b3b3b3, as in the schematics
CHANNELS = [("mJuniper (LacI reporter)", "#1f8a7a"), ("GFP (TetR reporter)", "#3f7f2a"),
            ("mScarlet-I3 (CI reporter)", "#b8323f")]
HOURS = 10.0


def figure(kind: str, title: str, name: str) -> None:
    """The plain figure, and a ``_dotted`` one: a dot per 5-min frame, the line at 50% opacity."""
    d = generate(box1_truth(), [0, 1, 2], kind=kind, noise=True, seed=1, hours=HOURS)
    for dotted in (False, True):
        draw(d, title, name + ("_dotted" if dotted else ""), dotted)


def draw(d, title: str, name: str, dotted: bool) -> None:
    hours = torch.arange(d.t.numel()) * FRAME_MIN / 60
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for j, (label, colour) in enumerate(CHANNELS):
        if dotted:
            ax.plot(hours, d.y[:, j], color=colour, lw=1.4, alpha=0.5)
            ax.scatter(hours, d.y[:, j], color=colour, s=9, linewidths=0, label=label, zorder=3)
        else:
            ax.plot(hours, d.y[:, j], color=colour, lw=1.8, label=label)
    ax.set_title(title, loc="left", color=INK, fontsize=14, fontweight="bold", pad=10)
    ax.set_xlabel("time [h]", color=INK, fontsize=11)
    ax.set_ylabel("fluorescence [K$_M$ units]", color=INK, fontsize=11)
    ax.set_xlim(0, HOURS)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK, labelsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK)
    legend = ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    for text in legend.get_texts():
        text.set_color(INK)
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200, transparent=True)
    plt.close(fig)
    print(OUT / f"{name}.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    figure("ode", "ODE cell (5-min frames)", "poster_sim_ode")
    figure("ssa", "Gillespie cell (5-min frames)", "poster_sim_gillespie")


if __name__ == "__main__":
    main()
