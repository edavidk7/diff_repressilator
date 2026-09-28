"""Poster figure: one Gillespie cell fitted by the differentiable ODE and by ABC-SMC.

    python experiments/plot_fit_comparison.py   -> run_results/figures/fit_comparison.{png,pdf,svg}

Two reporters (GFP and mScarlet-I3), seed 1: of the three seeds, the one where
ABC-SMC's fit is its median (fit MSE 2.6x that at the true parameters), so the
cell is representative rather than picked for contrast. Dots: the 5-min
frames; line: the ODE integrated at each method's estimate; gray: the cell's
hidden noise-free fluorescence.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent / "run_results"
IN, OUT = ROOT / "panel_benchmark", ROOT / "figures"
CELL = "2_reporters__ssa__seed1.json"
HOURS_PER_UNIT = 2.0 / 0.6931471805599453 / 60  # model time unit = mRNA lifetime
GRID = (179 / 255, 179 / 255, 179 / 255)
CHANNELS = {"tet": ("GFP (TetR reporter)", "#3f7f2a"), "cI": ("mScarlet-I3 (CI reporter)", "#b8323f")}
METHODS = [("gradient", "diff. ODE (ours)"), ("abc", "ABC-SMC")]


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 3.4), sharey=True)
    for ax, (method, label) in zip(axes, METHODS):
        r = json.loads((IN / method / CELL).read_text())
        frames = [t * HOURS_PER_UNIT for t in r["t"]]
        fine = [t * HOURS_PER_UNIT for t in r["fine_t"]]
        for c, channel in enumerate(r["channels"]):
            name, colour = CHANNELS[channel]
            ax.plot(fine, [row[c] for row in r["truth_fine"]], color="black", lw=1.0, alpha=0.3,
                    label="hidden truth" if c == 0 else None)
            ax.scatter(frames, [row[c] for row in r["y"]], s=10, color=colour, linewidths=0, zorder=3)
            ax.plot(fine, [row[c] for row in r["fit_fine"]], color=colour, lw=2.2, label=name)
        ratio = r["loss_fit"] / r["loss_truth"]
        ax.set_title(label, loc="left", fontsize=16, fontweight="bold", pad=24)
        ax.text(0, 1.02, f"fit MSE {ratio:.2g}× that at the true parameters", transform=ax.transAxes,
                fontsize=13, va="bottom")
        ax.set_xlabel("time [h]", fontsize=14)
        ax.set_xlim(0, frames[-1])
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=12)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("fluorescence [K$_M$ units]", fontsize=14)
    handles, labels = axes[1].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], ls="", marker="o", ms=4, color="black"))
    labels.append("5-min frames")
    axes[1].legend(handles, labels, frameon=False, fontsize=12, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"fit_comparison.{ext}", dpi=200, transparent=True)
    print(OUT / "fit_comparison.png")


if __name__ == "__main__":
    main()
