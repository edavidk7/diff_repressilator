"""Estimation methods across reporter/fusion panels.

    python experiments/panel_benchmark.py --method gradient --workers 3    run (resumes)
    python experiments/panel_benchmark.py --report                         table of results

For each panel x {ODE cell, Gillespie cell} x seed: generate one cell (Box 1
parameters, camera noise on, 10 h at 5-min frames) and fit it with

* ``gradient`` - backprop through the ODE (:func:`differentiable_cell.fit.fit_gradient`);
* ``nuts``     - NUTS through the ODE from the gradient fit (:mod:`differentiable_cell.nuts`),
                 point estimate = posterior median (hours per case: not in the benchmark);
* ``laplace``  - the gradient fit plus a Laplace (Hessian) approximation for intervals;
* ``abc``      - ABC-SMC, no gradients (:func:`differentiable_cell.abc.fit_abc`),
                 point estimate = weighted posterior median;
* ``magi``     - MAGI, solver-free (:func:`differentiable_cell.solver_free.fit_magi`);
* ``pinn``     - inverse PINN, solver-free (:func:`differentiable_cell.solver_free.fit_pinn`).

Each fit is saved with the parameter errors, its MSE next to the MSE at the
true parameters (<= 1: it reached the likelihood optimum), and what the
figures need: the frames, the fitted trajectory and the hidden truth for the
observed channels at full resolution.
"""

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from identifiability_reporters import panel, true_theta  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "run_results" / "panel_benchmark"
PANELS = {
    "GFP only": "tet",
    "2 reporters": "tet,cI",
    "3 reporters": "lac,tet,cI",
    "TetR fusion": "TetR-fusion",
    "TetR fusion + GFP": "TetR-fusion,tet",
    "TetR fusion + cI reporter": "TetR-fusion,cI",
    "TetR fusion + 3 reporters": "TetR-fusion,lac,tet,cI",
    "3 fusions": "LacI-fusion,TetR-fusion,CI-fusion",
}
KINDS = ("ode", "ssa")
SEEDS = (0, 1, 2)
PARAMS = ("alpha", "n", "beta", "alpha_0")
# From experiments/tuning.py: the only variant that reached the likelihood optimum on
# all six tuning datasets (chosen on fit quality, not on errors against the truth).
GRADIENT_SETTINGS: dict = {"loss": "huber", "schedule": "cosine"}


REDUCED = ("GFP only", "3 reporters", "TetR fusion + GFP")  # the three that tell the story


def jobs(reduced: bool = False, seeds: int = 0) -> list[tuple[str, str, int]]:
    """The grid: all panels x both kinds x SEEDS; ``reduced``: 3 panels x ODE x SEEDS;
    ``seeds`` > 0: 3 panels x both kinds x that many seeds (seed-major, so a partial
    run has complete seeds)."""
    if seeds:
        return [(name, kind, seed) for seed in range(seeds) for kind in KINDS for name in REDUCED]
    if reduced:
        return [(name, "ode", seed) for seed in SEEDS for name in REDUCED]
    return [(name, kind, seed) for seed in SEEDS for kind in KINDS for name in PANELS]


def job_file(method: str, name: str, kind: str, seed: int) -> Path:
    slug = "".join(c if c.isalnum() else "_" for c in name)
    return OUT / method / f"{slug}__{kind}__seed{seed}.json"


def run_one(method: str, name: str, kind: str, seed: int, threads: int) -> None:
    torch.set_num_threads(threads)
    torch.set_default_dtype(torch.float64)
    from differentiable_cell.fit import Problem, fit_gradient
    from differentiable_cell.reporter_sim import CHANNELS, box1_truth, generate

    channels, present = panel(PANELS[name])
    data = generate(box1_truth(), present, kind=kind, noise=True, seed=seed)
    problem = Problem(t=data.t, y=data.y[:, channels], channels=channels, present=present,
                      tau=data.truth["tau"])
    truth = true_theta(problem, data).clamp(problem.lo, problem.hi)

    t0 = time.perf_counter()
    draws, info = None, {}
    if method == "gradient":
        theta, _ = fit_gradient(problem, seed=seed, **GRADIENT_SETTINGS)
    elif method == "nuts":
        from differentiable_cell.nuts import fit_nuts
        theta_map, _ = fit_gradient(problem, seed=seed, **GRADIENT_SETTINGS)  # same MAP as "gradient"
        draws, info = fit_nuts(problem, theta_map, max_seconds=1200, seed=seed)
        theta = draws.median(0).values
    elif method == "laplace":
        from differentiable_cell.nuts import fit_laplace
        theta, _ = fit_gradient(problem, seed=seed, **GRADIENT_SETTINGS)  # same MAP as "gradient"
        draws, info = fit_laplace(problem, theta, seed=seed)
    elif method == "abc":
        from differentiable_cell.abc import fit_abc
        particles, weights, history = fit_abc(problem, max_sims=300_000, seed=seed)
        theta = weighted_median(particles, weights)
        draws = particles[torch.multinomial(weights, 2000, replacement=True)]
    elif method == "magi":
        from differentiable_cell.solver_free import fit_magi
        theta, info = fit_magi(problem, seed=seed)
        draws = info.pop("draws")
    elif method == "pinn":
        from differentiable_cell.solver_free import fit_pinn
        theta, info = fit_pinn(problem, seed=seed)
    else:
        raise ValueError(method)
    wall = time.perf_counter() - t0
    with torch.no_grad():
        loss_fit = float(problem.loss(theta))  # MSE of the ODE at the estimate, for every method

    fit, true = problem.values(theta), problem.values(truth)
    with torch.no_grad():
        fit_fine = problem.predict(theta, t=data.fine_t)
    row = {"method": method, "panel": name, "kind": kind, "seed": seed, "wall": wall,
           "unknowns": len(problem.names), "loss_fit": loss_fit, "loss_truth": float(problem.loss(truth)),
           "estimate": {k: float(fit[k]) for k in PARAMS},
           "truth": {k: float(true[k]) for k in PARAMS},
           "rel_err": {k: abs(float(fit[k]) - float(true[k])) / float(true[k]) for k in PARAMS},
           "theta": theta.tolist(), "names": problem.names,
           "info": {k: v for k, v in info.items() if isinstance(v, (int, float))},
           "channels": [CHANNELS[c][0] for c in channels],
           "t": data.t.tolist(), "y": data.y[:, channels].tolist(),
           "fine_t": data.fine_t.tolist(), "fit_fine": fit_fine.tolist(),
           "truth_fine": data.fine_state[:, problem.observed_state].tolist()}
    if draws is not None:  # 90% intervals and whether they cover the truth
        vals = problem.values(draws)
        row["ci90"] = {k: [float(vals[k].quantile(0.05)), float(vals[k].quantile(0.95))] for k in PARAMS}
        row["covered"] = {k: row["ci90"][k][0] <= row["truth"][k] <= row["ci90"][k][1] for k in PARAMS}
    path = job_file(method, name, kind, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row))
    print(method, name, kind, seed, {k: round(v, 3) for k, v in row["rel_err"].items()}, flush=True)


def weighted_median(particles: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Per-coordinate weighted median of ABC particles."""
    out = []
    for i in range(particles.shape[1]):
        order = particles[:, i].argsort()
        cdf = weights[order].cumsum(0)
        out.append(particles[order[int((cdf >= 0.5).nonzero()[0])], i])
    return torch.stack(out)


def report() -> None:
    rows = [json.loads(p.read_text()) for p in OUT.glob("*/*.json")]
    for method in sorted({r["method"] for r in rows}):
        for kind in KINDS:
            rs_k = [r for r in rows if r["method"] == method and r["kind"] == kind]
            if not rs_k:
                continue
            print(f"\n{method} - {kind.upper()} cell - median relative error over seeds\n")
            print("| panel | seeds | α | n | β | α₀ | fit/truth MSE | unknowns | wall [min] |")
            print("|---|---|---|---|---|---|---|---|---|")
            for name in PANELS:
                rs = [r for r in rs_k if r["panel"] == name]
                if not rs:
                    continue
                med = lambda f: statistics.median(f(r) for r in rs)
                print(f"| {name} | {len(rs)} | "
                      + " | ".join(f"{med(lambda r: r['rel_err'][k]):.1%}" for k in PARAMS)
                      + f" | {med(lambda r: r['loss_fit'] / r['loss_truth']):.2f} | {rs[0]['unknowns']} "
                      f"| {med(lambda r: r['wall']) / 60:.0f} |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="gradient",
                    help="one of gradient, nuts, laplace, abc, magi, pinn, or several comma-separated "
                         "(one shared queue, interleaved by job)")
    ap.add_argument("--reduced", action="store_true", help="3 panels x ODE cells x 3 seeds")
    ap.add_argument("--seeds", type=int, default=0, help="3 panels x both kinds x this many seeds")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--job", type=int)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        return report()
    todo = jobs(args.reduced, args.seeds)
    methods = args.method.split(",")
    if args.job is not None:
        return run_one(methods[0], *todo[args.job], threads=args.threads)
    grid = (["--reduced"] if args.reduced else []) + (["--seeds", str(args.seeds)] if args.seeds else [])
    for m in methods:
        (OUT / m).mkdir(parents=True, exist_ok=True)
    pending = [(m, i) for i, j in enumerate(todo) for m in methods if not job_file(m, *j).exists()]
    print(f"{len(pending)} fits pending", flush=True)
    logs = {m: open(OUT / f"log_{m}.txt", "a") for m in methods}
    running: list[subprocess.Popen] = []
    while pending or running:
        running = [p for p in running if p.poll() is None]
        while pending and len(running) < args.workers:
            m, i = pending.pop(0)
            running.append(subprocess.Popen([sys.executable, __file__, "--method", m, "--job", str(i),
                                             "--threads", str(args.threads)] + grid,
                                            stdout=logs[m], stderr=subprocess.STDOUT))
        time.sleep(10)
    print("done", flush=True)


if __name__ == "__main__":
    main()
