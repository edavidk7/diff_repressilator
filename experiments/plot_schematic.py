"""Schematic of the modelled construct.

    python experiments/plot_schematic.py      -> run_results/figures/schematic.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent.parent / "run_results" / "figures"
INK, MUTED = "#2b2b2b", "#6b6b6b"
GENE = {"lacI": "#1baf7a", "tetR": "#008300", "cI": "#e34948"}  # coloured as their reporters


def box(ax, xy, w, h, text, colour, fill="white", size=10, weight="normal"):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor=fill, edgecolor=colour, lw=1.8))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=size, color=INK,
            weight=weight)


def repress(ax, start, end, colour):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-[", mutation_scale=12, color=colour, lw=1.8,
                                 connectionstyle="arc3,rad=0.25"))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")

    # Circuit on pSC101 (~5 copies)
    ax.text(0.2, 4.7, "Repressilator (pSC101, ~5 copies)", fontsize=12, weight="bold", color=INK)
    pos = {"lacI": (1.0, 3.2), "tetR": (4.0, 3.2), "cI": (2.5, 1.0)}
    label = {"lacI": "lacI → LacI", "tetR": "tetR → TetR", "cI": "cI → CI"}
    for g, (x, y) in pos.items():
        box(ax, (x, y), 1.9, 0.8, label[g], GENE[g], size=11)
    repress(ax, (2.9, 3.6), (4.0, 3.6), GENE["lacI"])  # LacI --| tetR
    repress(ax, (5.0, 3.2), (4.2, 1.6), GENE["tetR"])  # TetR --| cI
    repress(ax, (2.5, 1.4), (1.6, 3.2), GENE["cI"])  # CI --| lacI
    ax.text(0.2, 0.25, "dm/dt = -m + α / (1 + (p/K)ⁿ) + α₀     dp/dt = -β (p - m)",
            fontsize=10, color=MUTED, family="monospace")

    # Reporters on p15A (~15 copies)
    ax.text(6.6, 4.7, "Reporter plasmid (p15A, ~15 copies)", fontsize=12, weight="bold", color=INK)
    rows = [("P_Llac → mJuniper-LVA", "lacI", "LacI"), ("P_Ltet → GFP-AAV", "tetR", "TetR"),
            ("P_R → mScarlet-I3-LVA", "cI", "CI")]
    for i, (text, g, rep) in enumerate(rows):
        y = 3.6 - 0.95 * i
        box(ax, (6.8, y), 3.1, 0.7, text, GENE[g], size=10)
        ax.text(10.05, y + 0.35, f"⊣ by {rep}   (titrates {rep}, τ)", fontsize=9, color=MUTED, va="center")
    ax.text(6.8, 0.95, "each: mRNA r → dark D → fluorescent F  (maturation k, tag decay δ)",
            fontsize=9, color=MUTED)

    # Fusions
    ax.text(6.8, 0.45, "Fusion channels: the repressor itself carries an FP (e.g. TetR-FP)",
            fontsize=9, color=MUTED)
    ax.text(6.8, 0.1, "Observed: calibrated fluorescence, 5-min frames, 10 h, camera noise",
            fontsize=9, color=MUTED)
    fig.tight_layout()
    fig.savefig(OUT / "schematic.png", dpi=150)
    print(OUT / "schematic.png")


if __name__ == "__main__":
    main()
