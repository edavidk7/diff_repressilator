"""Paper schematics: the method and the circuits (style of figures4papers: Arial, semantic palette, no top/right
spines, frameless legends, 300 dpi).

    python experiments/plot_schematics.py      -> run_results/figures/{method_schematic,circuits_schematic}.{pdf,png}

The method figure draws the real curriculum and learning rate of fit.fit_gradient and the recorded training curves of
all restarts on one cell (restart_spread.py).
"""

import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import style  # noqa: F401  (applies the figure style)
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from common import FIGURES, RESULTS

PALETTE = style.PALETTE


def save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=1)
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"{name}.{ext}")
    plt.close(fig)
    print(FIGURES / f"{name}.pdf")


def box(ax, xy, w, h, text, face, edge=None, size=10, weight="normal", color=None):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.06", facecolor=face,
                                edgecolor=edge or face, lw=1.4))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=size, weight=weight,
            color=color or PALETTE["ink"])


def arrow(ax, a, b, color, style="-|>", lw=1.6, rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=16, color=color, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))


NODE_R = 0.11


def edge(ax, a, b, kind="activate", bend=0.0, color=None, r_a=NODE_R, r_b=NODE_R, lw=1.8):
    """Curved edge between node centres a and b, clipped to the node circles; an arrowhead (activation) or a bar
    (repression) at b. ``bend``: curvature as a fraction of the distance (positive bends to the left)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    ctrl = (a + b) / 2 + bend * np.array([-d[1], d[0]])
    t = np.linspace(0, 1, 200)[:, None]
    curve = (1 - t) ** 2 * a + 2 * (1 - t) * t * ctrl + t**2 * b
    keep = (np.linalg.norm(curve - a, axis=1) > r_a) & (np.linalg.norm(curve - b, axis=1) > r_b + 0.015)
    curve = curve[keep]
    color = color or (PALETTE["ink"] if kind == "activate" else PALETTE["red_strong"])
    ax.plot(curve[:, 0], curve[:, 1], color=color, lw=lw, solid_capstyle="round", zorder=2)
    end, tangent = curve[-1], curve[-1] - curve[-6]
    tangent /= np.linalg.norm(tangent)
    normal = np.array([-tangent[1], tangent[0]])
    if kind == "activate":
        tip = end + 0.015 * tangent
        ax.add_patch(plt.Polygon([tip, end - 0.06 * tangent + 0.035 * normal, end - 0.06 * tangent - 0.035 * normal],
                                 color=color, zorder=2))
    else:
        bar = np.array([end + 0.055 * normal, end - 0.055 * normal])
        ax.plot(bar[:, 0], bar[:, 1], color=color, lw=lw + 1.0, solid_capstyle="round", zorder=2)


# ------------------------------------------------------------------------------------------------ the method

def windows(epochs=500, hold=0.1, full=0.9, start=0.1, lr=0.05, lr_final=2.5e-4):
    e = np.arange(epochs)
    progress = np.clip((e / (epochs - 1) - hold) / (full - hold), 0, 1)
    window = start + (1 - start) * progress
    first = int(np.argmax(window >= 1))
    r = lr_final / lr
    rate = np.where(e < first, lr, lr * (r + (1 - r) * 0.5 * (1 + np.cos(math.pi * (e - first) / (epochs - first)))))
    return e, window, rate


def method_schematic() -> None:
    fig = plt.figure(figsize=(15.5, 3.9))
    grid = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.3, 1.1, 1.1], wspace=0.62)

    # (a) the problem
    ax = fig.add_subplot(grid[0])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("a  Unknowns", loc="left", weight="bold")
    box(ax, (0.0, 0.56), 1.0, 0.3, "kinetic parameters $\\theta$\nrates, Hill coefficients,\nreporter strength, leak",
        PALETTE["green_1"], PALETTE["green_3"], size=8.5)
    box(ax, (0.0, 0.18), 1.0, 0.3, "starting state $x_0$\nobserved species: near frame 1\nhidden species: free",
        PALETTE["green_1"], PALETTE["green_3"], size=8.5)
    ax.text(0.5, 0.03, "64 restarts drawn log-uniformly\nfrom the prior box, optimised as one batch", ha="center",
            va="center", fontsize=9, color=PALETTE["grey"])

    # (b) the solver, unrolled
    ax = fig.add_subplot(grid[1])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("b  Differentiable simulation", loc="left", weight="bold")
    ax.text(0.5, 0.9, "$\\dot x = f(x, \\theta, u(t))$,  RK4 in PyTorch", ha="center", fontsize=10.5)
    xs = [0.08, 0.3, 0.52, 0.92]
    for i, (x, lab) in enumerate(zip(xs, ["$x_0$", "$x_1$", "$x_2$", "$x_G$"])):
        box(ax, (x - 0.06, 0.56), 0.12, 0.14, lab, "white", PALETTE["blue_main"], size=11)
        if i < 3:
            arrow(ax, (x + 0.06, 0.63), (xs[i + 1] - 0.06 if i < 2 else 0.66, 0.63), PALETTE["blue_main"])
    ax.text(0.73, 0.63, "$\\cdots$", ha="center", va="center", fontsize=14)
    arrow(ax, (0.78, 0.63), (0.86, 0.63), PALETTE["blue_main"])
    t = np.linspace(0.05, 0.95, 200)
    ax.plot(t, 0.33 + 0.08 * np.sin(2 * np.pi * 2.2 * t), color=PALETTE["blue_main"], lw=1.6)
    rng = np.random.default_rng(1)
    tf = np.linspace(0.07, 0.93, 22)
    ax.scatter(tf, 0.33 + 0.08 * np.sin(2 * np.pi * 2.2 * tf) + 0.02 * rng.standard_normal(22), s=9,
               color=PALETTE["ink"], zorder=3)
    ax.text(0.5, 0.17, "Huber loss on the frames of the current window", ha="center", fontsize=9.5)
    arrow(ax, (0.9, 0.48), (0.1, 0.48), PALETTE["red_strong"], rad=0.0, ls="--")
    ax.text(0.5, 0.04, "$\\nabla_{\\theta, x_0}\\mathcal{L}$ by backpropagation through every solver step", ha="center",
            fontsize=9.5, color=PALETTE["red_strong"])

    # (c) curriculum and coupled learning rate
    ax = fig.add_subplot(grid[2])
    e, window, rate = windows()
    ax.plot(e, 100 * window, color=PALETTE["blue_main"], lw=2.2)
    ax.set_xlabel("epoch")
    ax.set_ylabel("frames in the loss [%]", color=PALETTE["blue_main"])
    ax.set_ylim(0, 105)
    ax.set_title("c  Curriculum and learning rate", loc="left", weight="bold")
    ax2 = ax.twinx()
    ax2.plot(e, rate, color=PALETTE["red_strong"], lw=2.2)
    ax2.set_ylabel("learning rate", color=PALETTE["red_strong"])
    ax2.set_ylim(0, 0.055)
    ax2.spines["right"].set_visible(True)
    ax.axvspan(450, 499, color=PALETTE["neutral"], alpha=0.35, lw=0)
    ax.text(474, 50, "anneal on\nwhole record", ha="center", va="center", fontsize=8.5, rotation=90)

    # (d) restarts
    ax = fig.add_subplot(grid[3])
    r = json.loads((RESULTS / "restart_spread" / "3_reporters__ode__seed0.json").read_text())
    ep = [h["epoch"] for h in r["history"]]
    curves = np.array([h["mse_ratio"] for h in r["history"]]).T
    best = int(np.argmin(curves[:, -1]))
    for i, c in enumerate(curves):
        if i != best:
            ax.plot(ep, c, color=PALETTE["blue_secondary"], alpha=0.25, lw=0.8)
    ax.plot(ep, curves[best], color=PALETTE["blue_main"], lw=2.4, label="best restart (the estimate)")
    ax.axhline(1, color=PALETTE["ink"], lw=1, ls="--", label="MSE at the true parameters")
    ax.set_yscale("log")
    ax.set_ylim(0.8, 3e3)
    ax.set_xlabel("epoch")
    ax.set_ylabel("full-record MSE / truth")
    ax.set_title("d  64 restarts, best kept", loc="left", weight="bold")
    ax.legend(fontsize=8.5, loc="upper right")
    save(fig, "method_schematic")


# ------------------------------------------------------------------------------------------------ the circuits

def node(ax, xy, label, colour, observed=False):
    ax.add_patch(plt.Circle(xy, NODE_R, facecolor=colour if observed else "white", edgecolor=colour, lw=2, zorder=3))
    ax.text(*xy, label, ha="center", va="center", fontsize=9, zorder=4,
            color="white" if observed else PALETTE["ink"], weight="bold" if observed else "normal")


def inducer(ax, xy, target, label, kind="repress"):
    """A known inducer above a node: it inactivates (repress) or drives (activate) the target."""
    ax.text(xy[0], xy[1] + 0.05, label, ha="center", va="bottom", fontsize=8.5, color=PALETTE["violet"], weight="bold")
    edge(ax, xy, target, kind, color=PALETTE["violet"], r_a=0.0)


def circuits_schematic() -> None:
    fig, axes = plt.subplots(1, 6, figsize=(15.5, 3.1))
    blue, green = PALETTE["blue_main"], PALETTE["teal"]
    for ax in axes:
        ax.set_axis_off()
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-0.6, 0.6)
        ax.set_aspect("equal")

    ax = axes[0]
    ax.set_title("repressilator", fontsize=11, weight="bold")
    pts = {g: (0.36 * math.cos(a), 0.36 * math.sin(a) - 0.05)
           for g, a in zip(["LacI", "TetR", "CI"], [math.pi / 2, -math.pi / 6, 7 * math.pi / 6])}
    for g, xy in pts.items():
        node(ax, xy, g, blue, observed=True)
    for a, b in [("LacI", "TetR"), ("TetR", "CI"), ("CI", "LacI")]:
        edge(ax, pts[a], pts[b], "repress", bend=-0.25)
    ax.text(0, -0.55, "via up to 3 reporters or fusions", ha="center", fontsize=8.5, color=PALETTE["grey"])

    ax = axes[1]
    ax.set_title("negative autoregulation", fontsize=11, weight="bold")
    c = np.array([0.0, -0.1])
    node(ax, c, "TetR", blue, observed=True)
    loop = np.array([[math.cos(t), math.sin(t)] for t in np.linspace(math.radians(-20), math.radians(200), 120)])
    loop = c + np.array([0.0, 0.2]) + 0.15 * loop
    loop = loop[np.linalg.norm(loop - c, axis=1) > NODE_R + 0.015]
    ax.plot(loop[:, 0], loop[:, 1], color=PALETTE["red_strong"], lw=1.8)
    end, tan = loop[-1], loop[-1] - loop[-4]
    tan /= np.linalg.norm(tan)
    nrm = np.array([-tan[1], tan[0]])
    bar = np.array([end + 0.055 * nrm, end - 0.055 * nrm])
    ax.plot(bar[:, 0], bar[:, 1], color=PALETTE["red_strong"], lw=2.8, solid_capstyle="round")
    inducer(ax, (0.36, 0.3), c, "aTc")

    ax = axes[2]
    ax.set_title("toggle switch", fontsize=11, weight="bold")
    u, v = (-0.27, -0.1), (0.27, -0.1)
    node(ax, u, "LacI", blue, observed=True)
    node(ax, v, "TetR", green)
    edge(ax, u, v, "repress", bend=-0.3)
    edge(ax, v, u, "repress", bend=-0.3)
    inducer(ax, (-0.27, 0.36), u, "IPTG")
    inducer(ax, (0.27, 0.36), v, "aTc")

    ax = axes[3]
    ax.set_title("Goodwin oscillator", fontsize=11, weight="bold")
    x, y, z = (-0.36, -0.1), (0.0, -0.1), (0.36, -0.1)
    node(ax, x, "x", green)
    node(ax, y, "y", green)
    node(ax, z, "z", blue, observed=True)
    edge(ax, x, y)
    edge(ax, y, z)
    edge(ax, z, x, "repress", bend=0.55)

    ax = axes[4]
    ax.set_title("incoherent feed-forward loop", fontsize=11, weight="bold")
    X, Y, Z = (0.0, 0.3), (-0.28, -0.2), (0.28, -0.2)
    node(ax, X, "X", green)
    node(ax, Y, "Y", blue, observed=True)
    node(ax, Z, "Z", blue, observed=True)
    edge(ax, X, Y)
    edge(ax, X, Z)
    edge(ax, Y, Z, "repress")
    ax.text(-0.42, 0.52, "input pulse", ha="center", fontsize=8.5, color=PALETTE["violet"], weight="bold")
    edge(ax, (-0.42, 0.5), X, color=PALETTE["violet"], r_a=0.0)

    ax = axes[5]
    ax.set_title("reporter readout", fontsize=11, weight="bold")
    gold = PALETTE["gold"]
    r_, d_, f_ = (-0.38, 0.15), (0.0, 0.15), (0.38, 0.15)
    node(ax, r_, "r", gold)
    node(ax, d_, "D", gold)
    node(ax, f_, "F", gold, observed=True)
    edge(ax, r_, d_)
    edge(ax, d_, f_)
    for xy, lab in [(r_, "mRNA"), (d_, "dark FP"), (f_, "mature FP")]:
        ax.text(xy[0], xy[1] - 0.2, lab, ha="center", fontsize=8.5, color=PALETTE["grey"])
    ax.text(0.19, 0.3, "$k$", ha="center", fontsize=9)
    ax.text(0, -0.2, "transcribed from a circuit promoter;\nimaged: photons, gain, offset,\nread noise, 5-min frames",
            ha="center", va="top", fontsize=8.5, color=PALETTE["grey"])
    fig.text(0.5, 0.03, "filled: observed   open: hidden   arrow: activation   bar: repression   purple: known inducer",
             ha="center", fontsize=9, color=PALETTE["grey"])
    save(fig, "circuits_schematic")


if __name__ == "__main__":
    method_schematic()
    circuits_schematic()
