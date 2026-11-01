"""Controls for the fit-quality comparison (does the gap survive a fairer read-out?).

    python experiments/fit_controls.py refine --workers 4   re-fit the start of every saved estimate
    python experiments/fit_controls.py abc --workers 4      ABC-SMC again, keeping the particles
    python experiments/fit_controls.py budget --particles 64 --max-sims 96000   ABC-SMC at other budgets

The benchmark scores every method by integrating the ODE at its point estimate.
Two things in that read-out are not the kinetic parameters themselves:

* ``refine``: the starting state. MAGI and the PINN read it off their fitted
  curves at t = 0, ABC and MAGI take a coordinate-wise median over particles or
  draws. A wrong starting phase alone ruins the fit of a limit cycle. For each
  saved estimate of ``abc``, ``magi``, ``pinn`` (and ``gradient`` on seeds 3-5, as a
  check that the extra budget changes nothing for it) two refinements are run in one
  batch with the benchmark's loss, curriculum and lr schedule:

    - start-only: the kinetic parameters are frozen at the estimate, only the
      starting state moves; from the estimate's own start plus 15 prior draws.
    - local: every unknown moves, from the estimate itself (one start).

* ``abc``: ABC-SMC rerun with the benchmark's settings and seed (it reproduces
  the saved estimate), keeping the final population, so the fit of the
  weighted median can be compared with the population's best particle.

* ``budget``: ABC-SMC with another population size and simulation budget (same
  cells as ``abc``), e.g. 64 particles - the same number as the differentiable
  ODE's restarts - and 96k simulations, its number of (forward + backward) solves.

Results: ``run_results/fit_controls/{refine,abc,abc_budget/p<P>_sims<S>}/...json``.
"""

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from identifiability_reporters import panel  # noqa: E402
from panel_benchmark import PANELS, REDUCED, job_file, weighted_median  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent / "run_results"
OUT = ROOT / "fit_controls"
METHODS = ("abc", "magi", "pinn", "gradient")
RANDOM_STARTS = 15
EPOCHS = 600


def problem_for(name: str, kind: str, seed: int):
    from differentiable_cell.fit import Problem
    from differentiable_cell.reporter_sim import box1_truth, generate

    channels, present = panel(PANELS[name])
    data = generate(box1_truth(), present, kind=kind, noise=True, seed=seed)
    return Problem(t=data.t, y=data.y[:, channels], channels=channels, present=present,
                   tau=data.truth["tau"]), data


def refine(problem, theta: torch.Tensor, seed: int, epochs: int = EPOCHS, lr: float = 0.05,
           lr_final: float = 2.5e-4, stages=(0.25, 0.5, 1.0)):
    """Rows 0..RANDOM_STARTS: start-only (row 0 from the estimate's own start); last row: local."""
    start = torch.tensor([n.endswith("(0)") for n in problem.names])
    g = torch.Generator().manual_seed(seed)
    draws = problem.sample_prior(RANDOM_STARTS, g)
    rows = theta.expand(RANDOM_STARTS + 2, -1).clone()
    rows[1:-1, start] = draws[:, start]
    z = problem.to_z(rows).requires_grad_(True)
    frozen = torch.zeros_like(rows, dtype=torch.bool)
    frozen[:-1] = ~start  # start-only rows keep their kinetic parameters
    opt = torch.optim.Adam([z], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr_final)
    windows = [max(4, math.ceil(f * problem.t.numel())) for f in stages]
    for epoch in range(epochs):
        window = windows[min(epoch * len(windows) // epochs, len(windows) - 1)]
        opt.zero_grad()
        per = problem.loss(problem.from_z(z), window, kind="huber")
        torch.where(torch.isfinite(per), per, torch.zeros_like(per)).sum().backward()
        z.grad = torch.nan_to_num(z.grad, nan=0.0, posinf=0.0, neginf=0.0).masked_fill(frozen, 0.0)
        opt.step()
        sched.step()
    with torch.no_grad():
        final = problem.from_z(z)
        mse = torch.nan_to_num(problem.loss(final), nan=math.inf)
    best_start = int(mse[:-1].argmin())
    return final[best_start], float(mse[best_start]), final[-1], float(mse[-1]), float(mse[0])


def run_refine(method: str, name: str, kind: str, seed: int) -> None:
    saved = json.loads(job_file(method, name, kind, seed).read_text())
    problem, data = problem_for(name, kind, seed)
    assert torch.allclose(problem.y, torch.tensor(saved["y"]), rtol=1e-9), "data not reproduced"
    if "theta" not in saved:  # older ABC fits kept no theta: the ABC rerun reproduces it
        rerun = OUT / "abc" / f"{job_file(method, name, kind, seed).stem}.json"
        if method != "abc" or not rerun.exists():
            print(f"no theta for {method} {name} {kind} {seed}: run `abc` first", flush=True)
            return
        saved["theta"] = json.loads(rerun.read_text())["theta_median"]
    theta = torch.tensor(saved["theta"], dtype=torch.float64)
    t0 = time.perf_counter()
    with torch.no_grad():
        loss_saved = float(problem.loss(theta))
    if not math.isfinite(loss_saved) and not torch.isfinite(theta).all():
        row = {"failed": True}
    else:
        th_s, mse_s, th_l, mse_l, mse_own = refine(problem, theta, seed)
        with torch.no_grad():
            fine_s = problem.predict(th_s, t=data.fine_t).tolist()
            fine_l = problem.predict(th_l, t=data.fine_t).tolist()
        vals_l = problem.values(th_l)
        row = {"loss_start_only": mse_s, "loss_start_only_own": mse_own, "loss_local": mse_l,
               "fit_fine_start_only": fine_s, "fit_fine_local": fine_l,
               "theta_start_only": th_s.tolist(), "theta_local": th_l.tolist(),
               "rel_err_local": {k: abs(float(vals_l[k]) - saved["truth"][k]) / saved["truth"][k]
                                 for k in saved["truth"]}}
    row.update({"method": method, "panel": name, "kind": kind, "seed": seed, "loss_saved": loss_saved,
                "loss_saved_json": saved["loss_fit"], "loss_truth": saved["loss_truth"],
                "wall": time.perf_counter() - t0})
    write(OUT / "refine" / method / f"{job_file(method, name, kind, seed).stem}.json", row)


def budget_dir(particles: int, max_sims: int) -> Path:
    return OUT / "abc_budget" / f"p{particles}_sims{max_sims}"


def run_abc(name: str, kind: str, seed: int, particles: int = 500, max_sims: int = 300_000,
            out_dir: Path | None = None) -> None:
    from differentiable_cell.abc import fit_abc

    problem, _ = problem_for(name, kind, seed)
    saved = json.loads(job_file("abc", name, kind, seed).read_text())
    t0 = time.perf_counter()
    n_particles = particles
    particles, weights, history = fit_abc(problem, particles=n_particles, batch=n_particles,
                                          max_sims=max_sims, seed=seed)
    with torch.no_grad():
        median = weighted_median(particles, weights)
        per_particle = problem.loss(particles)
        loss_median = float(problem.loss(median))
    order = per_particle.argsort()
    best = particles[order[0]]
    q = torch.nan_to_num(per_particle, nan=math.inf).sort().values
    write((out_dir or OUT / "abc") / f"{job_file('abc', name, kind, seed).stem}.json",
          {"particles": n_particles, "max_sims": max_sims, "panel": name, "kind": kind, "seed": seed, "loss_truth": saved["loss_truth"],
           "loss_median": loss_median, "loss_median_saved": saved["loss_fit"],
           "loss_best_particle": float(q[0]), "loss_particle_quantiles":
               {str(p): float(q[int(p * (len(q) - 1))]) for p in (0.05, 0.25, 0.5, 0.75, 0.95)},
           "final_eps": history[-1][0], "populations": len(history), "theta_best": best.tolist(),
           "theta_median": median.tolist(),
           "wall": time.perf_counter() - t0})


def write(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row))
    print(path.name, flush=True)


def todo(what: str) -> list[tuple]:
    out = []
    for name in REDUCED:
        for kind in ("ode", "ssa"):
            stem = job_file("gradient", name, kind, 0).stem.rsplit("seed", 1)[0]
            seeds = sorted(int(p.stem.rsplit("seed", 1)[1]) for p in (ROOT / "panel_benchmark" / "gradient").glob(stem + "seed*.json"))
            for seed in seeds:
                if what in ("abc", "budget"):
                    if seed < 3:
                        out.append(("abc", name, kind, seed))
                else:
                    # gradient is only a check (the extra budget should change nothing): 3 seeds
                    out += [(m, name, kind, seed) for m in METHODS if job_file(m, name, kind, seed).exists()
                            and (m != "gradient" or 3 <= seed < 6)]
    return out


def out_file(what: str, job: tuple, particles: int = 0, max_sims: int = 0) -> Path:
    method, name, kind, seed = job
    stem = job_file(method, name, kind, seed).stem
    if what == "budget":
        return budget_dir(particles, max_sims) / f"{stem}.json"
    return OUT / "abc" / f"{stem}.json" if what == "abc" else OUT / "refine" / method / f"{stem}.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["refine", "abc", "budget"])
    ap.add_argument("--particles", type=int, default=64, help="budget: ABC population size")
    ap.add_argument("--max-sims", type=int, default=96_000, help="budget: ABC simulation budget")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--job", type=int)
    ap.add_argument("--methods", default=",".join(METHODS))
    args = ap.parse_args()
    jobs = [j for j in todo(args.what) if j[0] in args.methods.split(",") or args.what in ("abc", "budget")]
    if args.job is not None:
        torch.set_num_threads(args.threads)
        torch.set_default_dtype(torch.float64)
        method, name, kind, seed = jobs[args.job]
        if args.what == "budget":
            return run_abc(name, kind, seed, args.particles, args.max_sims, budget_dir(args.particles, args.max_sims))
        return run_abc(name, kind, seed) if args.what == "abc" else run_refine(method, name, kind, seed)
    # interleave methods so a partial run covers all of them
    pending = [i for i, j in enumerate(jobs) if not out_file(args.what, j, args.particles, args.max_sims).exists()]
    pending.sort(key=lambda i: (jobs[i][3], jobs[i][2], jobs[i][1], jobs[i][0]))
    print(f"{len(pending)} of {len(jobs)} pending", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    log = open(OUT / f"log_{args.what}.txt", "a")
    extra = ["--particles", str(args.particles), "--max-sims", str(args.max_sims)]
    running: list[subprocess.Popen] = []
    while pending or running:
        running = [p for p in running if p.poll() is None]
        while pending and len(running) < args.workers:
            i = pending.pop(0)
            running.append(subprocess.Popen([sys.executable, __file__, args.what, "--job", str(i),
                                             "--threads", str(args.threads), "--methods", args.methods] + extra,
                                            stdout=log, stderr=subprocess.STDOUT))
        time.sleep(5)
    print("done", flush=True)


if __name__ == "__main__":
    main()
