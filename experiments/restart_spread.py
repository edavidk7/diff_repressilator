"""How do the 64 restarts of the gradient fit spread out?

    python experiments/restart_spread.py --workers 3     run (resumes)
    python experiments/restart_spread.py --report        tables and run_results/restart_spread/restart_curves.png
    add --epochs 600 and/or --lr-final 1e-5 for another schedule (own output folder)

Same cells and settings as the benchmark's gradient fit (seed 0), but every restart's MSE (relative to the
MSE at the truth) and errors are kept, at the end and every 25 epochs.
"""

import argparse
import json
import statistics
import time

from common import KINDS, MAIN_PANELS, RESULTS, cell, run_pool, setup_torch, slug, true_theta
from panel_benchmark import GRADIENT_SETTINGS

CELLS = [(name, kind, 0) for kind in KINDS for name in MAIN_PANELS]
PARAMS = ("alpha", "n", "beta")
OPTIMUM = 1.02  # MSE ratio at or below this: the restart reached the optimum


def out_dir(epochs: int, lr_final: float):
    name = "_".join(([f"epochs{epochs}"] if epochs != 1500 else []) + ([f"lrfinal{lr_final:g}"] if lr_final != 2.5e-4 else []))
    return RESULTS / "restart_spread" / name


def job_file(folder, name: str, kind: str, seed: int):
    return folder / f"{slug(name)}__{kind}__seed{seed}.json"


def run_one(name: str, kind: str, seed: int, epochs: int, lr_final: float) -> None:
    from differentiable_cell.fit import fit_gradient

    problem, data = cell(name, kind, seed)
    truth = true_theta(problem, data).clamp(problem.lo, problem.hi)
    t0 = time.perf_counter()
    history: list = []
    thetas, final = fit_gradient(problem, epochs=epochs, lr_final=lr_final, seed=seed, return_all=True,
                                 history=history, **GRADIENT_SETTINGS)
    mse_truth, true = float(problem.loss(truth)), problem.values(truth)

    def errors(theta):
        fit = problem.values(theta)
        return {k: [abs(float(v) - float(true[k])) / float(true[k]) for v in fit[k]] for k in PARAMS}

    row = {"panel": name, "kind": kind, "seed": seed, "wall": time.perf_counter() - t0,
           "mse_ratio": [float(f) / mse_truth for f in final], "rel_err": errors(thetas),
           "history": [{"epoch": e, "mse_ratio": [float(f) / mse_truth for f in mse], "rel_err": errors(th)}
                       for e, mse, th in history]}
    job_file(out_dir(epochs, lr_final), name, kind, seed).write_text(json.dumps(row, indent=2))
    print(name, kind, seed, "done", flush=True)


def report(folder, epochs: int) -> None:
    rows = [json.loads(job_file(folder, *c).read_text()) for c in CELLS if job_file(folder, *c).exists()]
    print(f"| cell | at optimum (<= {OPTIMUM}) | MSE ratio best / median / worst | α err: best / median at optimum / others |")
    print("|---|---|---|---|")
    for r in rows:
        ratio, a = r["mse_ratio"], r["rel_err"]["alpha"]
        good = [i for i, x in enumerate(ratio) if x <= OPTIMUM]
        bad = [i for i, x in enumerate(ratio) if x > OPTIMUM]
        med = lambda idx: f"{statistics.median(a[i] for i in idx):.0%}" if idx else "-"
        print(f"| {r['panel']}, {r['kind']} | {len(good)}/{len(ratio)} | {min(ratio):.2f} / "
              f"{statistics.median(ratio):.2f} / {max(ratio):.3g} | {a[ratio.index(min(ratio))]:.0%} / {med(good)} / {med(bad)} |")

    print("\n| cell | epoch | best MSE ratio | restarts at optimum | α / n / β err of best |\n|---|---|---|---|---|")
    for r in rows:
        for h in r["history"]:
            if h["epoch"] % (epochs // 6) and h["epoch"] not in (100, epochs):
                continue
            ratio = h["mse_ratio"]
            best = ratio.index(min(ratio))
            errs = " / ".join(f"{h['rel_err'][k][best]:.0%}" for k in PARAMS)
            print(f"| {r['panel']}, {r['kind']} | {h['epoch']} | {ratio[best]:.3f} | "
                  f"{sum(x <= OPTIMUM for x in ratio)}/{len(ratio)} | {errs} |")
    plot(rows, folder, epochs)


def plot(rows, folder, epochs: int) -> None:
    """Full-record MSE ratio of every restart over training, one panel per cell; the final best in bold."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharex=True, squeeze=False)
    for ax, r in zip(axes.flat, rows):
        x = [h["epoch"] for h in r["history"]]
        curves = list(zip(*[h["mse_ratio"] for h in r["history"]]))
        best = min(range(len(curves)), key=lambda i: curves[i][-1])
        for i, c in enumerate(curves):
            ax.plot(x, c, color="#325ba9" if i == best else "#abbddc", lw=2 if i == best else 0.7,
                    zorder=3 if i == best else 1)
        ax.axhline(1.0, color="#325ba9", lw=0.8, ls="--")
        for e in (epochs // 3, 2 * epochs // 3):  # curriculum window changes
            ax.axvline(e, color="#abbddc", lw=0.8, ls=":")
        ax.set_yscale("log")
        ax.set_ylim(0.8 * min(min(c) for c in curves), 1e3)
        ax.set_title(f"{r['panel']}, {'ODE' if r['kind'] == 'ode' else 'Gillespie'} cell", loc="left")
    for ax in axes[-1]:
        ax.set_xlabel("epoch")
    for ax in axes[:, 0]:
        ax.set_ylabel("MSE / MSE at truth")
    fig.tight_layout()
    fig.savefig(folder / "restart_curves.png", dpi=130)
    print(folder / "restart_curves.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--lr-final", type=float, default=2.5e-4)
    ap.add_argument("--one", nargs=3, metavar=("CONFIG", "KIND", "SEED"))
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    folder = out_dir(args.epochs, args.lr_final)
    if args.report:
        return report(folder, args.epochs)
    if args.one:
        setup_torch(args.threads)
        return run_one(args.one[0], args.one[1], int(args.one[2]), args.epochs, args.lr_final)
    folder.mkdir(parents=True, exist_ok=True)
    jobs = [["--threads", str(args.threads), "--epochs", str(args.epochs), "--lr-final", str(args.lr_final),
             "--one", n, k, str(s)] for n, k, s in CELLS if not job_file(folder, n, k, s).exists()]
    run_pool(__file__, jobs, args.workers, folder / "log.txt")


if __name__ == "__main__":
    main()
