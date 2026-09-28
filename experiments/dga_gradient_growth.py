"""How large do differentiable-Gillespie gradients get as the time window grows?

    python experiments/dga_gradient_growth.py     -> run_results/figures/dga_gradients.png

Differentiable Gillespie (Rijal & Mehta 2025, 1/a = 200, 1/b = 20) on the
three-reporter repressilator (Box 1 parameters, all reporters present):
|d mean(state at T) / d log alpha| over 8 cells, for growing windows T and two
system sizes. The same derivative of the ODE, for comparison, stays O(1-100).
"""

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from differentiable_cell import reporters as R
from differentiable_cell.gillespie import ssa_dga
from differentiable_cell.reporter_sim import box1_truth, mask, params, simulate_ode, start_state

torch.set_default_dtype(torch.float64)
OUT = Path(__file__).resolve().parent.parent / "run_results" / "figures"
WINDOWS = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0)
OMEGAS = (2.0, 5.0)


def dga_gradient(omega: float, window: float, cells: int = 8) -> float:
    p0 = params(box1_truth())
    m = mask([0, 1, 2])
    log_alpha = torch.tensor(math.log(p0["alpha"]), requires_grad=True)
    p = dict(p0, alpha=log_alpha.exp())
    y0 = start_state(p0, m, 0.0)
    x0 = (y0 * R.count_scales(p0, omega)).expand(cells, R.N_STATE).clone()
    t = torch.tensor([0.0, window])
    x = ssa_dga(lambda x: R.propensities(x, p, m, omega), R.stoichiometry(), x0, t, c=0.05,
                generator=torch.Generator().manual_seed(0), max_events=2_000_000)
    (grad,) = torch.autograd.grad(x[-1].mean(), log_alpha)
    return abs(float(grad))


def ode_gradient(window: float) -> float:
    p0 = params(box1_truth())
    m = mask([0, 1, 2])
    log_alpha = torch.tensor(math.log(p0["alpha"]), requires_grad=True)
    p = dict(p0, alpha=log_alpha.exp())
    y0 = start_state(p0, m, 0.0)
    scales = R.count_scales(p0, 1.0)  # same "counts" weighting as the DGA, per unit omega
    y = simulate_ode(p, m, y0, torch.tensor([0.0, window]))[-1] * scales
    (grad,) = torch.autograd.grad(y.mean(), log_alpha)
    return abs(float(grad))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = {f"omega={o:g}": [dga_gradient(o, w) for w in WINDOWS] for o in OMEGAS}
    rows["ODE (per unit omega)"] = [ode_gradient(w) for w in WINDOWS]
    for k, v in rows.items():
        print(k, [f"{g:.2e}" for g in v], flush=True)
    (OUT / "dga_gradients.json").write_text(json.dumps({"windows": WINDOWS, **rows}, indent=2))

    fig, ax = plt.subplots(figsize=(7, 4.2))
    minutes = [w * R.TAU_M_MIN for w in WINDOWS]
    for (k, v), colour in zip(rows.items(), ("#2a78d6", "#eb6834", "#9a9a93")):
        ax.plot(minutes, v, marker="o", lw=2, color=colour, label=k if "ODE" in k else f"DGA, {k}")
    ax.set_yscale("log")
    ax.set_xlabel("simulated window [min]  (one period ~ 120 min)")
    ax.set_ylabel("|d mean state / d log α|")
    ax.set_title("Differentiable-Gillespie gradients explode with the window", loc="left")
    ax.grid(color="#e6e6e1", lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "dga_gradients.png", dpi=130)
    print(OUT / "dga_gradients.png")


if __name__ == "__main__":
    main()
