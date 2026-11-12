"""Controls for the fit comparison: does the gap survive a fairer read-out and other budgets?

    python experiments/fit_controls.py refine --workers 4                        start-only re-fit, local refinement
    python experiments/fit_controls.py abc --workers 2                           ABC-SMC again, keeping its population
    python experiments/fit_controls.py abc --particles 64 --max-sims 96000       ABC-SMC at another budget

The benchmark scores each estimate by integrating the ODE from it, start included.
* refine: for each saved estimate (abc, magi, pinn; gradient on seeds 3-5 as a check), 600 epochs with the
  benchmark's loss and curriculum: (a) kinetic parameters frozen, only the start moves, from the estimate's
  own start plus 15 prior draws; (b) everything moves, from the estimate (one start).
* abc: ABC-SMC with the benchmark's seed (it reproduces the saved estimate), seeds 0-2, comparing the
  weighted median with the population's best particle. Other --particles/--max-sims go to abc_budget/.

Results: run_results/fit_controls/{refine/<method>, abc, abc_budget/p<P>_sims<S>}/<cell>.json
"""

import argparse
import json
import math
import time

import torch

from common import KINDS, MAIN_PANELS, RESULTS, cell, run_pool, setup_torch
from panel_benchmark import job_file

OUT = RESULTS / "fit_controls"
METHODS = ("abc", "magi", "pinn", "gradient")


def refine(problem, theta: torch.Tensor, seed: int, epochs: int = 600, random_starts: int = 15,
           lr: float = 0.05, lr_final: float = 2.5e-4, stages=(0.25, 0.5, 1.0)):
    """One batch: rows 0..random_starts move only the start (row 0 from the estimate's), the last row moves
    everything. Returns (best start-only theta, its MSE, local theta, its MSE)."""
    start = torch.tensor([n.endswith("(0)") for n in problem.names])
    rows = theta.expand(random_starts + 2, -1).clone()
    rows[1:-1, start] = problem.sample_prior(random_starts, torch.Generator().manual_seed(seed))[:, start]
    z = problem.to_z(rows).requires_grad_(True)
    frozen = torch.zeros_like(rows, dtype=torch.bool)
    frozen[:-1] = ~start
    opt = torch.optim.Adam([z], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr_final)
    windows = [max(4, math.ceil(f * problem.t.numel())) for f in stages]
    for epoch in range(epochs):
        opt.zero_grad()
        per = problem.loss(problem.from_z(z), windows[epoch * len(windows) // epochs], kind="huber")
        torch.where(torch.isfinite(per), per, torch.zeros_like(per)).sum().backward()
        z.grad = torch.nan_to_num(z.grad, nan=0.0, posinf=0.0, neginf=0.0).masked_fill(frozen, 0.0)
        opt.step()
        sched.step()
    with torch.no_grad():
        final = problem.from_z(z)
        mse = torch.nan_to_num(problem.loss(final), nan=math.inf)
    best = int(mse[:-1].argmin())
    return final[best], float(mse[best]), final[-1], float(mse[-1])


def run_refine(method: str, name: str, kind: str, seed: int) -> None:
    saved = json.loads(job_file(method, name, kind, seed).read_text())
    if "theta" not in saved:  # older ABC fits kept no theta; the ABC rerun reproduces it
        saved["theta"] = json.loads((OUT / "abc" / job_file(method, name, kind, seed).name).read_text())["theta_median"]
    problem, data = cell(name, kind, seed)
    theta = torch.tensor(saved["theta"])
    row = {"method": method, "panel": name, "kind": kind, "seed": seed, "loss_truth": saved["loss_truth"]}
    with torch.no_grad():
        row["loss_saved"] = float(problem.loss(theta))
    if not torch.isfinite(theta).all():
        row["failed"] = True
    else:
        th_s, row["loss_start_only"], th_l, row["loss_local"] = refine(problem, theta, seed)
        vals = problem.values(th_l)
        with torch.no_grad():
            row["fit_fine_start_only"] = problem.predict(th_s, t=data.fine_t).tolist()
            row["fit_fine_local"] = problem.predict(th_l, t=data.fine_t).tolist()
        row.update(theta_start_only=th_s.tolist(), theta_local=th_l.tolist(),
                   rel_err_local={k: abs(float(vals[k]) - v) / v for k, v in saved["truth"].items()})
    write(OUT / "refine" / method / job_file(method, name, kind, seed).name, row)


def abc_dir(particles: int, max_sims: int):
    return OUT / "abc" if (particles, max_sims) == (500, 300_000) else OUT / "abc_budget" / f"p{particles}_sims{max_sims}"


def run_abc(name: str, kind: str, seed: int, particles: int, max_sims: int) -> None:
    from differentiable_cell.abc import fit_abc, weighted_median

    problem, _ = cell(name, kind, seed)
    saved = json.loads(job_file("abc", name, kind, seed).read_text())
    t0 = time.perf_counter()
    pop, weights, history = fit_abc(problem, particles=particles, batch=particles, max_sims=max_sims, seed=seed)
    with torch.no_grad():
        median = weighted_median(pop, weights)
        per_particle = torch.nan_to_num(problem.loss(pop), nan=math.inf)
    q = per_particle.sort().values
    write(abc_dir(particles, max_sims) / job_file("abc", name, kind, seed).name,
          {"particles": particles, "max_sims": max_sims, "panel": name, "kind": kind, "seed": seed,
           "loss_truth": saved["loss_truth"], "loss_median": float(problem.loss(median)),
           "loss_median_saved": saved["loss_fit"], "loss_best_particle": float(q[0]),
           "loss_particle_quantiles": {str(p): float(q[int(p * (len(q) - 1))]) for p in (0.05, 0.25, 0.5, 0.75, 0.95)},
           "final_eps": history[-1][0], "populations": len(history),
           "theta_best": pop[per_particle.argmin()].tolist(), "theta_median": median.tolist(),
           "wall": time.perf_counter() - t0})


def write(path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row))
    print(path, flush=True)


def saved_seeds(name: str, kind: str) -> list[int]:
    stem = job_file("gradient", name, kind, 0).name.rsplit("seed", 1)[0]
    return sorted(int(p.stem.rsplit("seed", 1)[1]) for p in job_file("gradient", name, kind, 0).parent.glob(stem + "*"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["refine", "abc"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--particles", type=int, default=500)
    ap.add_argument("--max-sims", type=int, default=300_000)
    ap.add_argument("--one", nargs="+", metavar="ARG", help="run a single job")
    args = ap.parse_args()
    if args.one:
        setup_torch(args.threads)
        if args.what == "refine":
            method, name, kind, seed = args.one
            return run_refine(method, name, kind, int(seed))
        name, kind, seed = args.one
        return run_abc(name, kind, int(seed), args.particles, args.max_sims)
    cells = [(n, k, s) for n in MAIN_PANELS for k in KINDS for s in saved_seeds(n, k)]
    base = [args.what, "--threads", str(args.threads), "--particles", str(args.particles),
            "--max-sims", str(args.max_sims), "--one"]
    if args.what == "refine":  # ABC seeds 0-2 need the ABC rerun first (their saved fits kept no theta)
        jobs = [(m, n, k, s) for n, k, s in cells for m in METHODS
                if job_file(m, n, k, s).exists() and (m != "gradient" or 3 <= s < 6)
                and not (OUT / "refine" / m / job_file(m, n, k, s).name).exists()]
    else:
        jobs = [(n, k, s) for n, k, s in cells if s < 3
                and not (abc_dir(args.particles, args.max_sims) / job_file("abc", n, k, s).name).exists()]
    run_pool(__file__, [base + [str(x) for x in j] for j in sorted(jobs, key=lambda j: j[-1])],
             args.workers, OUT / f"log_{args.what}.txt")


if __name__ == "__main__":
    main()
