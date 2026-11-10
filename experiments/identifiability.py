"""How precisely could one cell's frames determine each parameter? (Fisher information at the truth)

    python experiments/identifiability.py                 -> run_results/figures/identifiability.{png,pdf,svg}
    python experiments/identifiability.py --replot        restyle from identifiability.json
    python experiments/identifiability.py --combinations "lac,tet,cI"   eigen-combinations for one set of channels

Fisher information J^T J / sigma^2 of the range-normalised frames (sigma^2: residual variance at the truth),
the starting state profiled out (Schur complement), the log-uniform prior box added (variance width^2 / 12).
1-sigma in log units ~ relative error. ODE cell, seed 1, camera noise on.
"""

import argparse
import json
import math

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import style  # noqa: F401  (applies the figure style)
from matplotlib.ticker import FuncFormatter

from differentiable_cell.fit import Problem
from differentiable_cell.reporter_sim import CHANNELS, box1_truth, generate

from common import FIGURES, PANELS, panel, true_theta

torch.set_default_dtype(torch.float64)
# Neutral tones: the method colours mean something else in the other figures.
PARAMS = [("alpha", "α", "#1a1a1a"), ("n", "n", "#5c5c5c"), ("beta", "β", "#9a9a9a"), ("alpha_0", "α₀", "#d0d0d0")]


def profiled_fisher(problem: Problem, theta: torch.Tensor, noise_var: float):
    """Fisher information of the kinetic parameters with the starting state profiled out, and their names."""
    f = lambda th: (problem.predict(th) / problem.range).reshape(-1)
    J = torch.autograd.functional.jacobian(f, theta, vectorize=True, strategy="forward-mode")
    fisher = J.T @ J / noise_var
    kin = [i for i, n in enumerate(problem.names) if not n.endswith("(0)")]
    nui = [i for i, n in enumerate(problem.names) if n.endswith("(0)")]
    F_kk, F_kn, F_nn = fisher[kin][:, kin], fisher[kin][:, nui], fisher[nui][:, nui]
    schur = F_kk - F_kn @ torch.linalg.pinv(F_nn) @ F_kn.T
    return 0.5 * (schur + schur.T), [problem.names[i] for i in kin]


def fisher_for(spec: str):
    channels, present = panel(spec)
    data = generate(box1_truth(), present, kind="ode", noise=True, seed=1)
    problem = Problem(t=data.t, y=data.y[:, channels], channels=channels, present=present, tau=data.truth["tau"])
    theta = true_theta(problem, data)
    return problem, *profiled_fisher(problem, theta, float(problem.loss(theta)))


def uncertainties() -> dict:
    """Per configuration: 1-sigma (%) of each parameter with the prior box, and of the box alone."""
    out = {}
    for name, spec in PANELS.items():
        problem, fisher, names = fisher_for(spec)
        width = (problem.hi - problem.lo)[[problem.names.index(n) for n in names]]
        cov = torch.linalg.inv(fisher + torch.diag(12.0 / width**2))
        out[name] = {k: 100 * math.sqrt(float(cov[names.index(k), names.index(k)])) for k, _, _ in PARAMS}
        out[name]["prior"] = {k: 100 * float(width[names.index(k)]) / math.sqrt(12) for k, _, _ in PARAMS}
        print(name, {k: round(v, 1) for k, v in out[name].items() if k != "prior"}, flush=True)
    return out


def combinations(spec: str) -> None:
    """Eigenvectors of the profiled Fisher: which parameter combinations the frames determine."""
    problem, fisher, names = fisher_for(spec)
    evals, evecs = torch.linalg.eigh(fisher)
    print(f"channels {spec}\n1-sigma (log units)   combination")
    for k in reversed(range(len(evals))):
        top = evecs[:, k].abs().argsort(descending=True)[:4]
        combo = "  ".join(f"{float(evecs[i, k]):+.2f} log({names[i]})" for i in top if abs(evecs[i, k]) > 0.05)
        print(f"{1 / math.sqrt(max(float(evals[k]), 1e-300)):10.3g}            {combo}")


def plot(u: dict) -> None:
    names, shown, width = list(u), PARAMS[:3], 0.26  # alpha_0: unidentifiable everywhere (caption says so)
    fig, ax = plt.subplots(figsize=(12, 4.0))
    for i, (key, sym, colour) in enumerate(shown):
        xs = [p + (i - (len(shown) - 1) / 2) * width for p in range(len(names))]
        ax.bar(xs, [u[n][key] for n in names], width=width * 0.9, color=colour, label=sym)
    ax.set_yscale("log")
    ax.set_ylim(top=100)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}%"))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax.axhline(10, color="black", lw=0.9, ls="--")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right", rotation_mode="anchor", fontsize=13)
    ax.tick_params(axis="y", labelsize=13)
    ax.set_ylabel("achievable 1σ uncertainty", fontsize=15)
    ax.set_title("Identifiability per reporter configuration", loc="left", fontsize=18, pad=34)
    ax.text(0, 1.02, "Fisher information at the true parameters (dashed: 10%)", transform=ax.transAxes,
            fontsize=13, va="bottom")
    ax.legend(frameon=False, ncol=3, loc="lower right", bbox_to_anchor=(1.0, 0.99), fontsize=15, handlelength=1.2)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIGURES / f"identifiability.{ext}")
    print(FIGURES / "identifiability.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replot", action="store_true")
    ap.add_argument("--combinations", metavar="CHANNELS", help=f"comma-separated, from {[c[0] for c in CHANNELS]}")
    args = ap.parse_args()
    if args.combinations:
        return combinations(args.combinations)
    FIGURES.mkdir(parents=True, exist_ok=True)
    cached = FIGURES / "identifiability.json"
    if args.replot and cached.exists():
        u = json.loads(cached.read_text())
    else:
        u = uncertainties()
        cached.write_text(json.dumps(u, indent=2))
    plot(u)


if __name__ == "__main__":
    main()
