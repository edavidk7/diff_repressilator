"""The benchmark: every method on the same simulated cells.

    python experiments/panel_benchmark.py --method gradient,abc,magi,pinn --seeds 13 --workers 5
    python experiments/panel_benchmark.py --method gradient,abc --workers 3    all 8 configurations, 3 seeds
    python experiments/panel_benchmark.py --report

Methods: gradient (backprop through the ODE), laplace (the same fit + Laplace intervals), abc (ABC-SMC,
weighted median), magi and pinn (solver-free). One JSON per fit in run_results/panel_benchmark/<method>/
with the parameter errors, the fit MSE next to the MSE at the true parameters (<= 1: at the optimum), and
the frames, fitted trajectory and hidden truth for figures. Resumable: finished fits are skipped.
"""

import argparse
import os
import json
import statistics
import time

import torch

from differentiable_cell.reporter_sim import CHANNELS

from common import KINDS, MAIN_PANELS, PANELS, RESULTS, cell, current, run_pool, setup_torch, slug, true_theta

OUT = RESULTS / "panel_benchmark"
PARAMS = ("alpha", "n", "beta", "alpha_0")
# The method's settings are fit_gradient's defaults (ramp curriculum, coupled schedule, 64 x 500; chosen on fit
# quality, never on truth). The first benchmark used these (results in */gradient_v1, */laplace_v1):
# the fit's loss: the squared error by default, FIT_LOSS=huber for the Huber penalty
GRADIENT_SETTINGS: dict = {"loss": os.environ.get("FIT_LOSS", "mse")}
FIDES_STARTS = 16
V1_SETTINGS: dict = {"loss": "huber", "schedule": "cosine", "curriculum": "stages", "restarts": 64, "epochs": 1500,
                     "box": "sigmoid", "dtype": torch.float64}


def job_file(method: str, name: str, kind: str, seed: int):
    return OUT / method / f"{slug(name)}__{kind}__seed{seed}.json"


def grid(seeds: int) -> list[tuple[str, str, int]]:
    """``seeds`` > 0: the main configurations with that many seeds (seed-major, so a partial run has whole
    seeds); otherwise all configurations with 3 seeds."""
    if seeds:
        return [(n, k, s) for s in range(seeds) for k in KINDS for n in MAIN_PANELS]
    return [(n, k, s) for s in range(3) for k in KINDS for n in PANELS]


def estimate(method: str, problem, seed: int):
    """Point estimate, posterior draws (or None) and numeric diagnostics."""
    from differentiable_cell.abc import fit_abc, weighted_median
    from differentiable_cell.fit import fit_gradient, fit_laplace
    from differentiable_cell.solver_free import fit_magi, fit_pinn

    if method in ("gradient", "laplace"):
        theta, _ = fit_gradient(problem, seed=seed, **GRADIENT_SETTINGS)
        return (theta, *fit_laplace(problem, theta, seed=seed)) if method == "laplace" else (theta, None, {})
    if method == "abc":
        particles, weights, _ = fit_abc(problem, seed=seed)
        g = torch.Generator().manual_seed(seed)
        return weighted_median(particles, weights), particles[torch.multinomial(weights, 2000, True, generator=g)], {}
    if method == "magi":
        theta, info = fit_magi(problem, seed=seed)
        return theta, info.pop("draws"), info
    if method == "pinn":
        theta, info = fit_pinn(problem, seed=seed)
        return theta, None, info
    if method == "abc_summary":  # ABC-SMC with a summary-statistic distance; estimate as for "abc"
        from differentiable_cell.abc import summary_distance

        particles, weights, _ = fit_abc(problem, seed=seed, distance=summary_distance(problem))
        g = torch.Generator().manual_seed(seed)
        best = float(problem.loss(particles).min())  # the best particle's trajectory MSE, for reference
        return (weighted_median(particles, weights), particles[torch.multinomial(weights, 2000, True, generator=g)],
                {"best_particle_mse": best})
    if method == "pinn_causal":
        theta, info = fit_pinn(problem, seed=seed, causal=True)
        return theta, None, info
    if method == "snpe":  # truncated sequential neural posterior estimation, 1e5 simulations; estimate = best draw
        from differentiable_cell.neural_sbi import fit_snpe

        theta, info = fit_snpe(problem, seed=seed, rounds=10, sims_per_round=10_000, max_epochs=7, threads=2)
        info.pop("history", None)
        return theta, info.pop("draws"), info
    if method == "lna":  # linear-noise / Kalman likelihood, for Gillespie cells
        from differentiable_cell.lna import fit_lna

        theta, info = fit_lna(problem, seed=seed)
        return theta, None, info
    if method == "fides":  # multistart trust-region least squares from the first 16 of the gradient fit's starts
        from differentiable_cell.second_order import fit_fides

        theta, info = fit_fides(problem, restarts=FIDES_STARTS, maxiter=300, seed=seed)
        return theta, None, info
    raise ValueError(method)


def run_one(method: str, name: str, kind: str, seed: int) -> None:
    problem, data = cell(name, kind, seed)
    truth = true_theta(problem, data).clamp(problem.lo, problem.hi)
    t0 = time.perf_counter()
    theta, draws, info = estimate(method, problem, seed)
    wall = time.perf_counter() - t0
    fit, true = problem.values(theta), problem.values(truth)
    with torch.no_grad():
        row = {"method": method, "panel": name, "kind": kind, "seed": seed, "wall": wall,
               "unknowns": len(problem.names),
               "loss_fit": float(problem.loss(theta)),  # MSE of the ODE at the estimate, for every method
               "loss_truth": float(problem.loss(truth)),
               "estimate": {k: float(fit[k]) for k in PARAMS}, "truth": {k: float(true[k]) for k in PARAMS},
               "rel_err": {k: abs(float(fit[k]) - float(true[k])) / float(true[k]) for k in PARAMS},
               "theta": theta.tolist(), "names": problem.names,
               "info": {k: v for k, v in info.items() if isinstance(v, (int, float, list))},
               "channels": [CHANNELS[c][0] for c in problem.channels],
               "t": data.t.tolist(), "y": problem.y.tolist(), "fine_t": data.fine_t.tolist(),
               "fit_fine": problem.predict(theta, t=data.fine_t).tolist(),
               "truth_fine": data.fine_state[:, problem.observed_state].tolist()}
    if draws is not None:  # 90% intervals and whether they cover the truth
        vals = problem.values(draws)
        row["ci90"] = {k: [float(vals[k].quantile(0.05)), float(vals[k].quantile(0.95))] for k in PARAMS}
        row["covered"] = {k: row["ci90"][k][0] <= row["truth"][k] <= row["ci90"][k][1] for k in PARAMS}
    path = job_file(method, name, kind, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row))
    print(method, name, kind, seed, {k: round(v, 3) for k, v in row["rel_err"].items()}, flush=True)


def report() -> None:
    rows = [json.loads(p.read_text()) for p in current(OUT)]
    for method in sorted({r["method"] for r in rows}):
        for kind in KINDS:
            mine = [r for r in rows if r["method"] == method and r["kind"] == kind]
            if not mine:
                continue
            print(f"\n{method}, {kind} cells: median relative error over seeds\n")
            print("| configuration | seeds | α | n | β | α₀ | fit/truth MSE | unknowns | wall [min] |")
            print("|---|---|---|---|---|---|---|---|---|")
            for name in PANELS:
                rs = [r for r in mine if r["panel"] == name]
                if rs:
                    med = lambda f: statistics.median(f(r) for r in rs)
                    errs = " | ".join(f"{med(lambda r: r['rel_err'][k]):.1%}" for k in PARAMS)
                    print(f"| {name} | {len(rs)} | {errs} | {med(lambda r: r['loss_fit'] / r['loss_truth']):.2f} "
                          f"| {rs[0]['unknowns']} | {med(lambda r: r['wall']) / 60:.0f} |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="gradient", help="comma-separated: gradient, laplace, abc, magi, pinn, fides")
    ap.add_argument("--seeds", type=int, default=0, help="main configurations x both kinds x this many seeds")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--one", nargs=4, metavar=("METHOD", "CONFIG", "KIND", "SEED"), help="run a single fit")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        return report()
    if args.one:
        setup_torch(args.threads)
        method, name, kind, seed = args.one
        return run_one(method, name, kind, int(seed))
    jobs = [["--threads", str(args.threads), "--one", m, n, k, str(s)]
            for n, k, s in grid(args.seeds) for m in args.method.split(",") if not job_file(m, n, k, s).exists()]
    run_pool(__file__, jobs, args.workers, OUT / "log.txt")


if __name__ == "__main__":
    main()
