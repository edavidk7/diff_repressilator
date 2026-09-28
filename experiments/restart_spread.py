"""How do the 64 restarts of the gradient fit spread out?

    python experiments/restart_spread.py --workers 3     run (resumes; one JSON per cell)
    python experiments/restart_spread.py --report        table
    add --epochs 600 and/or --lr-final 1e-5 to either for another schedule (own output folder)

Same cells and settings as ``panel_benchmark.py --method gradient``, but every
restart's final MSE (relative to the MSE at the true parameters) and
parameter errors are kept, not only the best one's, plus the same every 25
epochs of training (full-record MSE, whatever the curriculum window).
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
from panel_benchmark import GRADIENT_SETTINGS, PANELS, REDUCED  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "run_results" / "restart_spread"
JOBS = [(name, kind, 0) for kind in ("ode", "ssa") for name in REDUCED]
PARAMS = ("alpha", "n", "beta")
OPTIMUM = 1.02  # fit/truth MSE at or below this: the restart reached the optimum


EPOCHS = 1500  # set by --epochs; the default 1500 is fit_gradient's
LR_FINAL = 2.5e-4  # set by --lr-final; the default is fit_gradient's


def out_dir() -> Path:
    name = "".join([f"epochs{EPOCHS}" if EPOCHS != 1500 else "",
                    f"_lrfinal{LR_FINAL:g}" if LR_FINAL != 2.5e-4 else ""]).strip("_")
    return OUT / name if name else OUT


def job_file(name: str, kind: str, seed: int) -> Path:
    slug = "".join(c if c.isalnum() else "_" for c in name)
    return out_dir() / f"{slug}__{kind}__seed{seed}.json"


def run_one(name: str, kind: str, seed: int, threads: int) -> None:
    torch.set_num_threads(threads)
    torch.set_default_dtype(torch.float64)
    from differentiable_cell.fit import Problem, fit_gradient
    from differentiable_cell.reporter_sim import box1_truth, generate

    channels, present = panel(PANELS[name])
    data = generate(box1_truth(), present, kind=kind, noise=True, seed=seed)
    problem = Problem(t=data.t, y=data.y[:, channels], channels=channels, present=present,
                      tau=data.truth["tau"])
    truth = true_theta(problem, data).clamp(problem.lo, problem.hi)
    t0 = time.perf_counter()
    history: list = []
    thetas, final = fit_gradient(problem, epochs=EPOCHS, lr_final=LR_FINAL, seed=seed, return_all=True, history=history,
                                 **GRADIENT_SETTINGS)
    mse_truth = float(problem.loss(truth))
    true = problem.values(truth)

    def errors(theta):
        fit = problem.values(theta)
        return {k: [abs(float(v) - float(true[k])) / float(true[k]) for v in fit[k]] for k in PARAMS}

    row = {"panel": name, "kind": kind, "seed": seed, "wall": time.perf_counter() - t0,
           "mse_ratio": [float(f) / mse_truth for f in final], "rel_err": errors(thetas),
           "history": [{"epoch": e, "mse_ratio": [float(f) / mse_truth for f in mse], "rel_err": errors(th)}
                       for e, mse, th in history]}
    job_file(name, kind, seed).write_text(json.dumps(row, indent=2))
    print(name, kind, seed, "done", flush=True)


def report() -> None:
    print(f"| cell | at optimum (<= {OPTIMUM}) | MSE ratio: best / median / worst "
          "| α err: best restart / median of optimum restarts / median of others |")
    print("|---|---|---|---|")
    for name, kind, seed in JOBS:
        f = job_file(name, kind, seed)
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        ratio, a = r["mse_ratio"], r["rel_err"]["alpha"]
        best = min(range(len(ratio)), key=ratio.__getitem__)
        good = [i for i, x in enumerate(ratio) if x <= OPTIMUM]
        bad = [i for i, x in enumerate(ratio) if x > OPTIMUM]
        med = lambda idx, v: f"{statistics.median(v[i] for i in idx):.0%}" if idx else "-"
        print(f"| {name}, {kind} | {len(good)}/{len(ratio)} | {min(ratio):.2f} / "
              f"{statistics.median(ratio):.2f} / {max(ratio):.3g} | "
              f"{a[best]:.0%} / {med(good, a)} / {med(bad, a)} |")


def convergence() -> None:
    """When does training stop paying off? Per cell, at each checkpoint: the best
    restart so far (by full-record MSE) and how many restarts sit at the optimum."""
    print("\n| cell | epoch | best MSE ratio | restarts at optimum | α / n / β err of best |")
    print("|---|---|---|---|---|")
    for name, kind, seed in JOBS:
        f = job_file(name, kind, seed)
        if not f.exists():
            continue
        for h in json.loads(f.read_text())["history"]:
            if h["epoch"] % (EPOCHS // 6) and h["epoch"] not in (100, EPOCHS):
                continue
            ratio = h["mse_ratio"]
            best = min(range(len(ratio)), key=ratio.__getitem__)
            errs = " / ".join(f"{h['rel_err'][k][best]:.0%}" for k in PARAMS)
            print(f"| {name}, {kind} | {h['epoch']} | {ratio[best]:.3f} | "
                  f"{sum(x <= OPTIMUM for x in ratio)}/{len(ratio)} | {errs} |")


def plot() -> None:
    """Full-record MSE ratio of every restart over training, one panel per cell."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [json.loads(job_file(*j).read_text()) for j in JOBS if job_file(*j).exists()]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharex=True, squeeze=False)
    for ax, r in zip(axes.flat, rows):
        epochs = [h["epoch"] for h in r["history"]]
        curves = list(zip(*[h["mse_ratio"] for h in r["history"]]))
        final_best = min(range(len(curves)), key=lambda i: curves[i][-1])
        for i, c in enumerate(curves):
            ax.plot(epochs, c, color="#325ba9" if i == final_best else "#abbddc",
                    lw=2 if i == final_best else 0.7, zorder=3 if i == final_best else 1)
        ax.axhline(1.0, color="#325ba9", lw=0.8, ls="--")
        for e in (EPOCHS // 3, 2 * EPOCHS // 3):  # curriculum window changes (25% -> 50% -> 100%)
            ax.axvline(e, color="#abbddc", lw=0.8, ls=":")
        ax.set_yscale("log")
        ax.set_ylim(0.8 * min(min(c) for c in curves), 1e3)
        ax.set_title(f"{r['panel']}, {'ODE' if r['kind'] == 'ode' else 'Gillespie'} cell", loc="left")
    for ax in axes[-1]:
        ax.set_xlabel("epoch")
    for ax in axes[:, 0]:
        ax.set_ylabel("MSE / MSE at truth")
    fig.tight_layout()
    path = out_dir() / "restart_curves.png"
    fig.savefig(path, dpi=130)
    print(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--job", type=int)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--lr-final", type=float, default=2.5e-4)
    args = ap.parse_args()
    global EPOCHS, LR_FINAL
    EPOCHS, LR_FINAL = args.epochs, args.lr_final
    if args.report:
        report()
        convergence()
        return plot()
    if args.job is not None:
        return run_one(*JOBS[args.job], threads=args.threads)
    out_dir().mkdir(parents=True, exist_ok=True)
    pending = [i for i, j in enumerate(JOBS) if not job_file(*j).exists()]
    running: list[subprocess.Popen] = []
    log = open(out_dir() / "log.txt", "a")
    while pending or running:
        running = [p for p in running if p.poll() is None]
        while pending and len(running) < args.workers:
            i = pending.pop(0)
            running.append(subprocess.Popen([sys.executable, __file__, "--job", str(i),
                                             "--threads", str(args.threads), "--epochs", str(EPOCHS),
                                             "--lr-final", str(LR_FINAL)],
                                            stdout=log, stderr=subprocess.STDOUT))
        time.sleep(10)
    print("done", flush=True)


if __name__ == "__main__":
    main()
