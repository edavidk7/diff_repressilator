"""How precisely can each panel determine each parameter, in principle?

    python experiments/plot_identifiability.py            -> run_results/figures/identifiability.{png,pdf,svg}
    python experiments/plot_identifiability.py --replot   restyle from identifiability.json

For every panel of the benchmark: the Fisher information of one cell's frames
at the true parameters (Box 1, ODE cell, camera noise, 10 h), the starting
state profiled out and the prior box added, gives the 1-sigma uncertainty of
alpha, n, beta, alpha_0 in log units (~ relative error). Lower is better; a
bar at the prior line means the data add nothing.
"""

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from matplotlib.ticker import FuncFormatter

from identifiability_reporters import panel, profiled_fisher, true_theta
from panel_benchmark import PANELS

from differentiable_cell.fit import Problem
from differentiable_cell.reporter_sim import box1_truth, generate

torch.set_default_dtype(torch.float64)
OUT = Path(__file__).resolve().parent.parent / "run_results" / "figures"
# Neutral tones: the method colours (blue, orange, pink, purple) mean something else on the poster.
PARAMS = [("alpha", "α", "#1a1a1a"), ("n", "n", "#5c5c5c"), ("beta", "β", "#9a9a9a"), ("alpha_0", "α₀", "#d0d0d0")]
GRID = "#b3b3b3"


def uncertainties() -> dict:
    out = {}
    for name, spec in PANELS.items():
        channels, present = panel(spec)
        data = generate(box1_truth(), present, kind="ode", noise=True, seed=1)
        problem = Problem(t=data.t, y=data.y[:, channels], channels=channels, present=present,
                          tau=data.truth["tau"])
        theta = true_theta(problem, data)
        fisher, names = profiled_fisher(problem, theta, float(problem.loss(theta)))
        kin = [problem.names.index(n) for n in names]
        width = (problem.hi - problem.lo)[kin]
        cov = torch.linalg.inv(fisher + torch.diag(12.0 / width**2))
        out[name] = {k: 100 * math.sqrt(float(cov[names.index(k), names.index(k)])) for k, _, _ in PARAMS}
        out[name]["prior"] = {k: 100 * float(width[names.index(k)]) / math.sqrt(12) for k, _, _ in PARAMS}
        print(name, {k: round(v, 1) for k, v in out[name].items() if k != "prior"}, flush=True)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cached = OUT / "identifiability.json"
    if "--replot" in sys.argv and cached.exists():  # restyle without recomputing
        u = json.loads(cached.read_text())
    else:
        u = uncertainties()
        cached.write_text(json.dumps(u, indent=2))
    names = list(u)
    shown = PARAMS[:3]  # alpha_0 is unidentifiable everywhere; the caption says so
    fig, ax = plt.subplots(figsize=(12, 4.0))
    width = 0.26
    for i, (key, sym, colour) in enumerate(shown):
        xs = [p + (i - (len(shown) - 1) / 2) * width for p in range(len(names))]
        ax.bar(xs, [u[n][key] for n in names], width=width * 0.9, color=colour, label=sym)
    ax.set_yscale("log")
    ax.set_ylim(top=100)  # show the 100% tick
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}%"))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax.axhline(10, color="black", lw=0.9, ls="--")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right", rotation_mode="anchor", fontsize=13)
    ax.tick_params(axis="y", labelsize=13)
    ax.set_ylabel("achievable 1σ uncertainty", fontsize=15)
    ax.set_title("Identifiability per reporter configuration", loc="left", fontsize=18, pad=34)
    ax.text(0, 1.02, "Fisher information at the true parameters (dashed: 10%)",
            transform=ax.transAxes, fontsize=13, va="bottom")
    ax.legend(frameon=False, ncol=3, loc="lower right", bbox_to_anchor=(1.0, 0.99), fontsize=15,
              handlelength=1.2)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"identifiability.{ext}", dpi=200, transparent=True)
    print(OUT / "identifiability.png")


if __name__ == "__main__":
    main()
