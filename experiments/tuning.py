"""Light tuning of the gradient fit (loss, optimiser, schedule), judged on fit quality only.

    python experiments/tuning.py --workers 3      run (resumes)
    python experiments/tuning.py --report         variant x (fit MSE / MSE at truth, errors)

6 ODE cells, reduced budget (32 restarts x 1000 epochs); the comparison is relative.
"""

import argparse
import json
import statistics
import time

from common import RESULTS, cell, run_pool, setup_torch, slug, true_theta

OUT = RESULTS / "tuning"
CELLS = [(p, s) for s in (0, 1) for p in ("GFP only", "3 reporters", "TetR fusion + GFP")]
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


def job_file(variant: str, name: str, seed: int):
    return OUT / f"{slug(variant)}__{slug(name)}__seed{seed}.json"


def run_one(variant: str, name: str, seed: int) -> None:
    from differentiable_cell.fit import fit_gradient

    problem, data = cell(name, "ode", seed)
    truth = true_theta(problem, data).clamp(problem.lo, problem.hi)
    t0 = time.perf_counter()
    old = {"loss": "mse", "schedule": "exp", "curriculum": "stages"}  # the baseline the variants modify
    theta, final = fit_gradient(problem, restarts=32, epochs=1000, seed=seed, **(old | VARIANTS[variant]))
    fit, true = problem.values(theta), problem.values(truth)
    row = {"variant": variant, "panel": name, "seed": seed, "wall": time.perf_counter() - t0,
           "mse_ratio": float(final.min()) / float(problem.loss(truth)),
           "rel_err": {k: abs(float(fit[k]) - float(true[k])) / float(true[k]) for k in PARAMS}}
    job_file(variant, name, seed).write_text(json.dumps(row, indent=2))
    print(variant, name, seed, round(row["mse_ratio"], 3), flush=True)


def report() -> None:
    rows = [json.loads(p.read_text()) for p in OUT.glob("*.json")]
    print("| variant | fits | median fit/truth MSE | at optimum (<= 1.02) | α | n | β | wall [min] |")
    print("|---|---|---|---|---|---|---|---|")
    for v in VARIANTS:
        rs = [r for r in rows if r["variant"] == v]
        if rs:
            med = lambda f: statistics.median(f(r) for r in rs)
            errs = " | ".join(f"{med(lambda r: r['rel_err'][k]):.1%}" for k in PARAMS)
            print(f"| {v} | {len(rs)} | {med(lambda r: r['mse_ratio']):.3f} | "
                  f"{sum(r['mse_ratio'] <= 1.02 for r in rs)}/{len(rs)} | {errs} | {med(lambda r: r['wall']) / 60:.1f} |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--one", nargs=3, metavar=("VARIANT", "CONFIG", "SEED"))
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        return report()
    if args.one:
        setup_torch(args.threads)
        return run_one(args.one[0], args.one[1], int(args.one[2]))
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [["--threads", str(args.threads), "--one", v, n, str(s)]
            for v in VARIANTS for n, s in CELLS if not job_file(v, n, s).exists()]
    run_pool(__file__, jobs, args.workers, OUT / "log.txt")


if __name__ == "__main__":
    main()
