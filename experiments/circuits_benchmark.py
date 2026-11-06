"""All methods on other circuits (synthetic ODE cells): does each fit, and what can it recover?

    python experiments/circuits_benchmark.py --method gradient,abc,magi,pinn --workers 6   5 seeds per circuit (resumes)
    python experiments/circuits_benchmark.py --report        table, Fisher bounds, run_results/circuits/fits_<circuit>.png

Each method with the benchmark's settings (panel_benchmark.estimate). Per fit: the MSE
relative to the MSE at the truth (<= 1: at the optimum), the NRMSE to the hidden trajectory, and parameter
errors next to the 1-sigma Fisher bound (start profiled out, prior box added; seed 0 of each circuit).
"""

import argparse
import json
import math

import numpy as np
import torch

from common import RESULTS, run_pool, setup_torch

OUT = RESULTS / "circuits"
SEEDS = range(5)


def cell(key: str, seed: int):
    from differentiable_cell.circuits import CIRCUITS, CircuitProblem, generate

    c = CIRCUITS[key]
    data = generate(c, seed)
    return c, data, CircuitProblem(c, data.t, data.y)


def job_file(method: str, key: str, seed: int):
    return OUT / method / f"{key}__seed{seed}.json"


def run_one(method: str, key: str, seed: int) -> None:
    import time

    from panel_benchmark import estimate

    c, data, problem = cell(key, seed)
    truth = problem.true_theta(data).clamp(problem.lo, problem.hi)
    t0 = time.perf_counter()
    theta = estimate(method, problem, seed)[0]
    wall = time.perf_counter() - t0
    fit, true = problem.values(theta), problem.values(truth)
    with torch.no_grad():
        fine = problem.predict(theta, t=data.fine_t)
        hidden = data.fine_state[:, c.observed]
        row = {"method": method, "circuit": key, "seed": seed, "wall": wall, "loss_fit": float(problem.loss(theta)), "loss_truth": float(problem.loss(truth)),
               "nrmse_hidden": float((((fine - hidden) / problem.range) ** 2).mean().sqrt()),
               "rel_err": {k: abs(float(fit[k]) - float(true[k])) / float(true[k]) for k in c.params},
               "theta": theta.tolist(), "t": data.t.tolist(), "y": data.y.tolist(), "fine_t": data.fine_t.tolist(),
               "fit_fine": fine.tolist(), "truth_fine": hidden.tolist()}
    job_file(method, key, seed).parent.mkdir(parents=True, exist_ok=True)
    job_file(method, key, seed).write_text(json.dumps(row))
    print(method, key, seed, round(row["loss_fit"] / row["loss_truth"], 3), {k: round(v, 3) for k, v in row["rel_err"].items()},
          flush=True)


def bounds(key: str) -> dict:
    """1-sigma relative uncertainty of each parameter from one cell (seed 0)."""
    from identifiability import profiled_fisher

    c, data, problem = cell(key, 0)
    theta = problem.true_theta(data).clamp(problem.lo, problem.hi)
    fisher, names = profiled_fisher(problem, theta, float(problem.loss(theta)))
    width = (problem.hi - problem.lo)[[problem.names.index(n) for n in names]]
    cov = torch.linalg.inv(fisher + torch.diag(12.0 / width**2))
    return {n: math.sqrt(float(cov[i, i])) for i, n in enumerate(names)}


METHODS = ["gradient", "abc", "magi", "pinn"]
LABEL = {"gradient": "diff. ODE", "abc": "ABC-SMC", "magi": "MAGI", "pinn": "inverse PINN"}
COLOUR = {"gradient": "#2a78d6", "abc": "#eb6834", "magi": "#e87ba4", "pinn": "#4a3aa7"}


def report() -> None:
    from differentiable_cell.circuits import CIRCUITS

    rows = [json.loads(p.read_text()) for m in METHODS for p in sorted((OUT / m).glob("*__seed*.json"))]
    summary = {}
    for key, c in CIRCUITS.items():
        b = bounds(key)
        print(f"\n{c.name}: median over seeds (fit = MSE / MSE at truth; hidden = NRMSE to the hidden trajectory)")
        print(f"  {'method':12s} {'n':>2s} {'fit':>8s} {'hidden':>8s}  " + "  ".join(f"{k:>8s}" for k in c.params))
        print(f"  {'Fisher 1σ':12s} {'':2s} {'':8s} {'':8s}  " + "  ".join(f"{b[k]:8.1%}" for k in c.params))
        summary[key] = {"bound": b}
        for m in METHODS:
            rs = [r for r in rows if r["circuit"] == key and r["method"] == m]
            if not rs:
                continue
            fit = [r["loss_fit"] / r["loss_truth"] for r in rs]
            err = {k: [r["rel_err"][k] for r in rs] for k in c.params}
            print(f"  {LABEL[m]:12s} {len(rs):2d} {np.nanmedian(fit):8.3g} {np.nanmedian([r['nrmse_hidden'] for r in rs]):8.3g}  "
                  + "  ".join(f"{np.nanmedian(err[k]):8.1%}" for k in c.params))
            summary[key][m] = {"fit": fit, "nrmse_hidden": [r["nrmse_hidden"] for r in rs], "rel_err": err,
                               "wall": [r.get("wall") for r in rs]}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    for key in CIRCUITS:
        plot(rows, key)


def plot(rows, key: str, seed: int = 0) -> None:
    """One cell (seed fixed in advance), one panel per method: frames, hidden truth, the ODE at the estimate."""
    import matplotlib

    from differentiable_cell.circuits import CIRCUITS

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = CIRCUITS[key]
    found = {r["method"]: r for r in rows if r["circuit"] == key and r["seed"] == seed}
    methods = [m for m in METHODS if m in found]
    if not methods:
        return
    fig, axes = plt.subplots(1, len(methods), figsize=(4 * len(methods), 2.8), sharey=True, squeeze=False)
    for ax, m in zip(axes[0], methods):
        r = found[m]
        h, hf = np.array(r["fine_t"]) * c.minutes_per_unit / 60, np.array(r["t"]) * c.minutes_per_unit / 60
        for j, i in enumerate(c.observed):
            ax.plot(h, np.array(r["truth_fine"])[:, j], color="black", lw=0.9, alpha=0.35)
            ax.scatter(hf, np.array(r["y"])[:, j], s=3, alpha=0.6, linewidths=0, color=f"C{j}")
            fine = np.array(r["fit_fine"], dtype=float)[:, j]
            if np.isfinite(fine).all():
                ax.plot(h, fine, lw=1.4, color=f"C{j}", label=c.species[i])
        ax.set_title(f"{LABEL[m]}: fit {r['loss_fit'] / r['loss_truth']:.3g}x", loc="left", fontsize=10, color=COLOUR[m])
        ax.set_xlabel("time [h]")
        ax.grid(color="#e0e0e0", lw=0.5)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"{c.name}, seed {seed} (dots: frames; grey: hidden truth; line: ODE at the estimate)",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / f"fits_{key}.png", dpi=110)
    print(OUT / f"fits_{key}.png")


def main() -> None:
    from differentiable_cell.circuits import CIRCUITS

    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="gradient", help="comma-separated: gradient, abc, magi, pinn")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--one", nargs=3, metavar=("METHOD", "CIRCUIT", "SEED"))
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        return report()
    if args.one:
        setup_torch(args.threads)
        return run_one(args.one[0], args.one[1], int(args.one[2]))
    jobs = [["--threads", str(args.threads), "--one", m, k, str(s)] for s in SEEDS for k in CIRCUITS
            for m in args.method.split(",") if not job_file(m, k, s).exists()]
    run_pool(__file__, jobs, args.workers, OUT / "log.txt")


if __name__ == "__main__":
    main()
