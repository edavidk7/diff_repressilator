"""Snapshots of the two Potvin-Trottier et al. (2016) movies, where the traces are read, and the traces themselves.

    python experiments/plot_movies.py      -> run_results/figures/movies.{png,pdf}

Needs the movies and traces cached by real_repressilator.py and real_improved.py.
"""

import subprocess

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

import real_improved as imp
import real_repressilator as orig
from common import FIGURES
from style import CHANNEL_COLOUR, PALETTE, RGB_REPORTERS


def frames(movie, w, h, fmt):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(movie), "-f", "rawvideo", "-pix_fmt", fmt, "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3 if fmt == "rgb24" else 1)


def boxes(ax, xs, rows, half, offset, width):
    """Mother-cell box (solid) and background box beside the trench (dashed), as in the trace extraction."""
    for x in xs:
        bg = x + offset if x + offset < width - half - 1 else x - offset
        for cx, ls in ((x, "-"), (bg, "--")):
            ax.add_patch(Rectangle((cx - half - 0.5, rows.start - 0.5), 2 * half + 1, rows.stop - rows.start,
                                   fill=False, ec="white", lw=0.9, ls=ls))


def main() -> None:
    o = frames(orig.OUT / "movie.mov", 468, 146, "gray")[..., 0]
    m = frames(imp.OUT / "movie.mov", 932, 364, "rgb24")
    fig = plt.figure(figsize=(12, 7.4))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 0.62, 1.0], width_ratios=[1.0, 1.0], hspace=0.38, wspace=0.18)

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(o[40], cmap="gray", vmin=0, vmax=255)
    boxes(ax, orig.TRENCH_X, orig.MOTHER_ROWS, 3, 27, o.shape[2])
    ax.set_title("a  Original repressilator (GFP), frame at 3.3 h", fontsize=11)
    ax.axis("off")

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(m[40, :250])
    boxes(ax, imp.TRENCH_X, imp.MOTHER_ROWS, 5, 53, m.shape[2])
    ax.set_title("b  Improved repressilator (mKate2, Venus, CFP), 5.3 h", fontsize=11)
    ax.axis("off")

    # c: the mother cell of one trench of the improved strain, every 5 frames (40 min)
    x, step, n = imp.TRENCH_X[0], 5, 30
    crops = [m[i, 30:115, x - 14:x + 15] for i in range(0, step * n, step)]
    ax = fig.add_subplot(gs[1, :])
    ax.imshow(np.concatenate([np.pad(c, ((0, 0), (0, 2), (0, 0))) for c in crops], 1), aspect="equal")
    wcrop = crops[0].shape[1] + 2
    ax.set_xticks([wcrop * k + wcrop / 2 for k in range(0, n, 3)],
                  [f"{k * step * 8 / 60:.0f}" for k in range(0, n, 3)])
    ax.set_yticks([])
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("time (h)")
    ax.set_title("c  Improved repressilator, mother cell of trench 1, every 40 min", fontsize=11)

    ax = fig.add_subplot(gs[2, 0])
    tr = orig.traces()
    t = np.arange(tr.shape[1]) * 5 / 60
    ax.plot(t[:240], tr[0, :240], color=CHANNEL_COLOUR["tet"], lw=1.4)
    ax.set_xlabel("time (h)")
    ax.set_ylabel("GFP (a.u., 8-bit)")
    ax.set_title("d  Original strain, trench 1", fontsize=11)

    ax = fig.add_subplot(gs[2, 1])
    tr = imp.traces()
    t = np.arange(tr.shape[1]) * 8 / 60
    for j, name in enumerate(["mKate2 (CI)", "Venus (TetR)", "CFP (LacI)"]):
        ax.plot(t, tr[0, :, j], color=RGB_REPORTERS[j], lw=1.4, label=name)
    ax.axvspan(0, step * n * 8 / 60, color=PALETTE["neutral"], alpha=0.35, lw=0)
    ax.set_xlabel("time (h)")
    ax.set_ylabel("intensity (a.u., 8-bit)")
    ax.set_title("e  Improved strain, trench 1 (shaded: panel c)", fontsize=11)
    ax.set_ylim(-10, 310)
    ax.legend(fontsize=8, ncol=3, loc="upper right", handlelength=1.2)

    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"movies.{ext}", bbox_inches="tight")
    print(FIGURES / "movies.png")


if __name__ == "__main__":
    main()
