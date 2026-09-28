"""Light tuning of the gradient fit: loss, optimiser, learning-rate schedule.

    python experiments/tuning.py --workers 3      run (resumes; one JSON per fit)
    python experiments/tuning.py --report         table: variant x (fit MSE / truth MSE, errors)

Every variant is judged on the same scale: the plain MSE of its final fit
relative to the MSE at the true parameters (<= 1: it reached the likelihood
optimum), and the relative errors of alpha, n, beta. Reduced budget (32
restarts x 1000 epochs); the comparison is relative.
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
from panel_benchmark import PANELS  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "run_results" / "tuning"
DATASETS = [(p, s) for s in (0, 1) for p in ("GFP only", "3 reporters", "TetR fusion + GFP")]
VARIANTS = {
    "baseline (adam, exp, mse, lr .05)": {},
    "l1 loss": {"loss": "l1"},
    "huber loss": {"loss": "huber"},
    "cosine schedule": {"schedule": "cosine"},
    "rmsprop": {"optimizer": "rmsprop", "lr": 0.01, "lr_final": 5e-5},
    "sgd nesterov": {"optimizer": "sgd", "lr": 0.02, "lr_final": 1e-4},
    "adam lr .02": {"lr": 0.02, "lr_final": 1e-4},
    "adam lr .1": {"lr": 0.1, "lr_final": 5e-4},
    "huber + cosine": {"loss": "huber", "schedule": "cosine"},
}
PARAMS = ("alpha", "n", "beta")


def jobs():
    return [(v, p, s) for v in VARIANTS for p, s in DATASETS]


def job_file(v, p, s) -> Path:
    slug = lambda x: "".join(c if c.isalnum() else "_" for c in x)
    return OUT / f"{slug(v)}__{slug(p)}__seed{s}.json"


def run_one(variant: str, name: str, seed: int, threads: int) -> None:
    torch.set_num_threads(threads)
    torch.set_default_dtype(torch.float64)
    from differentiable_cell.fit import Problem, fit_gradient
    from differentiable_cell.reporter_sim import box1_truth, generate

    channels, present = panel(PANELS[name])
    data = generate(box1_truth(), present, kind="ode", noise=True, seed=seed)
    problem = Problem(t=data.t, y=data.y[:, channels], channels=channels, present=present,
                      tau=data.truth["tau"])
    truth = true_theta(problem, data).clamp(problem.lo, problem.hi)
    t0 = time.perf_counter()
    theta, final = fit_gradient(problem, restarts=32, epochs=1000, seed=seed, **VARIANTS[variant])
    fit, true = problem.values(theta), problem.values(truth)
    row = {"variant": variant, "panel": name, "seed": seed, "wall": time.perf_counter() - t0,
           "mse_ratio": float(final.min()) / float(problem.loss(truth)),
           "rel_err": {k: abs(float(fit[k]) - float(true[k])) / float(true[k]) for k in PARAMS}}
    job_file(variant, name, seed).write_text(json.dumps(row, indent=2))
    print(variant, name, seed, round(row["mse_ratio"], 3), flush=True)


def report() -> None:
    rows = [json.loads(p.read_text()) for p in OUT.glob("*.json")]
    print("| variant | fits | median fit/truth MSE | reached optimum (<= 1.02) | median α err | median n err | median β err | median wall [min] |")
    print("|---|---|---|---|---|---|---|---|")
    for v in VARIANTS:
        rs = [r for r in rows if r["variant"] == v]
        if not rs:
            continue
        med = lambda f: statistics.median(f(r) for r in rs)
        reached = sum(r["mse_ratio"] <= 1.02 for r in rs)
        print(f"| {v} | {len(rs)} | {med(lambda r: r['mse_ratio']):.3f} | {reached}/{len(rs)} | "
              + " | ".join(f"{med(lambda r: r['rel_err'][k]):.1%}" for k in PARAMS)
              + f" | {med(lambda r: r['wall']) / 60:.1f} |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--job", type=int)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        return report()
    todo = jobs()
    if args.job is not None:
        return run_one(*todo[args.job], threads=args.threads)
    OUT.mkdir(parents=True, exist_ok=True)
    pending = [i for i, j in enumerate(todo) if not job_file(*j).exists()]
    running: list[subprocess.Popen] = []
    log = open(OUT / "log.txt", "a")
    while pending or running:
        running = [p for p in running if p.poll() is None]
        while pending and len(running) < args.workers:
            i = pending.pop(0)
            running.append(subprocess.Popen([sys.executable, __file__, "--job", str(i),
                                             "--threads", str(args.threads)],
                                            stdout=log, stderr=subprocess.STDOUT))
        time.sleep(10)
    print("done", flush=True)


if __name__ == "__main__":
    main()
