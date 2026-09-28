"""Plot one generated cell: ODE vs Gillespie, camera noise on vs off.

    python experiments/show_reporter_data.py        -> run_results/reporter_data.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from differentiable_cell.reporter_sim import FRAME_MIN, box1_truth, generate

torch.set_default_dtype(torch.float64)
OUT = Path(__file__).resolve().parent.parent / "run_results"
CHANNELS = [("mJuniper (LacI reporter)", "#1baf7a"), ("GFP (TetR reporter)", "#008300"),
            ("mScarlet-I3 (CI reporter)", "#e34948")]


def main() -> None:
    OUT.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True)
    for row, kind in enumerate(("ode", "ssa")):
        for col, noise in enumerate((False, True)):
            d = generate(box1_truth(), [0, 1, 2], kind=kind, noise=noise, seed=1)
            minutes = torch.arange(d.t.numel()) * FRAME_MIN
            ax = axes[row, col]
            for j, (label, colour) in enumerate(CHANNELS):
                ax.plot(minutes, d.y[:, j], color=colour, lw=1.6, label=label)
            ax.set_title(f"{'ODE' if kind == 'ode' else 'Gillespie'} cell, camera noise "
                         f"{'on' if noise else 'off'}", loc="left", fontsize=10)
            ax.grid(color="#e6e6e1", lw=0.6)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if col == 0:
                ax.set_ylabel("fluorescence [AU]")
            if row == 1:
                ax.set_xlabel("time [min]")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("One cell, three reporters, 5-min frames (Box 1 parameters)", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "reporter_data.png", dpi=120)
    print(OUT / "reporter_data.png")


if __name__ == "__main__":
    main()
