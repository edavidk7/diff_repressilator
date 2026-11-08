"""How many epochs does the fit need? One benchmark cell, 32 restarts, linear curriculum.

    python experiments/epoch_budget.py 300 [linear|ramp|ramp90|ramp90_coupled|stages]     one run on CELL; run_results/epoch_budget/runs.json
    python experiments/epoch_budget.py --cells 600 ramp,stages        main configurations x kinds x seeds 0-1
    python experiments/epoch_budget.py --grid 300,400 32,64 ramp90_coupled   epochs x restarts on those cells
    python experiments/epoch_budget.py --cells-report 600 ramp,stages

Curricula (fit.fit_gradient): stages 25/50/100%; linear 10% -> 100% over training; ramp hold 10% for the
first 10% of epochs, rise to 100% at 75%; ramp90 the same, reaching 100% at 90%; ramp90_coupled:
ramp90 with lr constant while the window grows, then a cosine anneal over the final full-record epochs.
Cell: 3 reporters, Gillespie, seed 0 (10 h). Reference: the benchmark's fit of the same cell (64 restarts,
1500 epochs, staged curriculum). Reports the fit (MSE / MSE at the true parameters), the NRMSE to the hidden
trajectory, alpha / n / beta errors, how many restarts end within 5% of the best, and wall time.
"""

import json
import sys
import time

import torch

from common import RESULTS, cell, run_pool, setup_torch, slug, true_theta
from panel_benchmark import V1_SETTINGS, job_file

OUT = RESULTS / "epoch_budget"
CELL = ("3 reporters", "ssa", 0)
RESTARTS = 32


def run(epochs: int, curriculum: str = "linear", which=CELL, restarts: int = RESTARTS) -> dict:
    from differentiable_cell.fit import fit_gradient

    problem, data = cell(*which)
    truth = true_theta(problem, data).clamp(problem.lo, problem.hi)
    t0 = time.perf_counter()
    variants = {"stages": {}, "linear": {"curriculum": "linear"}, "ramp": {"curriculum": "ramp", "ramp": (0.1, 0.75)},
                "ramp90": {"curriculum": "ramp", "ramp": (0.1, 0.9)},
                "ramp90_coupled": {"curriculum": "ramp", "ramp": (0.1, 0.9), "schedule": "coupled"}}
    settings = {k: v for k, v in V1_SETTINGS.items() if k not in ("restarts", "epochs")} | variants[curriculum]
    thetas, final = fit_gradient(problem, restarts=restarts, epochs=epochs, seed=which[2], return_all=True,
                                 **settings)
    wall = time.perf_counter() - t0
    best = int(final.argmin())
    theta = thetas[best]
    with torch.no_grad():
        fine = problem.predict(theta, t=data.fine_t)
        hidden = data.fine_state[:, problem.observed_state]
        mse_truth = float(problem.loss(truth))
    fit, true = problem.values(theta), problem.values(truth)
    return {"epochs": epochs, "curriculum": curriculum, "restarts": restarts, "wall_min": wall / 60,
            "fit": float(final[best]) / mse_truth,
            "nrmse_hidden": float((((fine - hidden) / problem.range) ** 2).mean().sqrt()),
            "rel_err": {k: abs(float(fit[k]) - float(true[k])) / float(true[k]) for k in ("alpha", "n", "beta")},
            "near_best": int((final <= 1.05 * final.min()).sum()),
            "restart_fits": (final / mse_truth).tolist()}


def reference(which=CELL) -> dict:
    r = json.loads(job_file("gradient_v1", *which).read_text())  # the first benchmark's fit (64 x 1500, staged)
    hidden, fine = torch.tensor(r["truth_fine"]), torch.tensor(r["fit_fine"])
    rng = torch.tensor(r["y"]).max(0).values - torch.tensor(r["y"]).min(0).values
    return {"fit": r["loss_fit"] / r["loss_truth"], "nrmse_hidden": float((((fine - hidden) / rng) ** 2).mean().sqrt()),
            "rel_err": {k: r["rel_err"][k] for k in ("alpha", "n", "beta")}, "wall_min": r["wall"] / 60}


CELLS = [(n, k, s) for n in ("GFP only", "3 reporters", "TetR fusion + GFP") for k in ("ode", "ssa") for s in (0, 1)]


def cell_file(epochs: int, curriculum: str, which, restarts: int = RESTARTS):
    folder = f"{curriculum}_e{epochs}" + ("" if restarts == RESTARTS else f"_r{restarts}")
    return OUT / folder / f"{slug(which[0])}__{which[1]}__seed{which[2]}.json"


def cells_report(epochs: int, curricula: list[str]) -> None:
    import statistics as st

    print(f"{RESTARTS} restarts, {epochs} epochs; fit relative to the benchmark's 64 x 1500 staged fit of the same cell")
    print(f"  {'cell':32s} " + " ".join(f"{c:>14s}" for c in curricula))
    gaps = {c: [] for c in curricula}
    for which in CELLS:
        ref = reference(which)["fit"]
        row = []
        for c in curricula:
            f = cell_file(epochs, c, which)
            if f.exists():
                g = json.loads(f.read_text())["fit"] / ref - 1
                gaps[c].append(g)
                row.append(f"{g:+14.1%}")
            else:
                row.append(f"{'-':>14s}")
        print(f"  {which[0] + ', ' + which[1] + ', seed ' + str(which[2]):32s} " + " ".join(row))
    for c in curricula:
        if gaps[c]:
            v = gaps[c]
            print(f"  {c}: median {st.median(v):+.1%}, worst {max(v):+.1%}, within 2% on {sum(g <= 0.02 for g in v)}/{len(v)}")
    walls = {c: [json.loads(cell_file(epochs, c, w).read_text())["wall_min"] for w in CELLS if cell_file(epochs, c, w).exists()]
             for c in curricula}
    print("  wall [min]: " + ", ".join(f"{c} {st.median(v):.1f}" for c, v in walls.items() if v))


def grid(epochs: list[int], restarts: list[int], curriculum: str) -> None:
    """Run every (epochs, restarts) on CELLS, then a table against the first benchmark's fits."""
    import statistics as st

    jobs = [["--one", str(e), curriculum, *map(str, w), str(r)] for r in restarts for e in epochs for w in CELLS
            if not cell_file(e, curriculum, w, r).exists()]
    run_pool(__file__, jobs, 6, OUT / "log.txt")
    print(f"{curriculum}; gap = fit / first benchmark's fit (64 x 1500, staged) - 1, over {len(CELLS)} cells")
    print("restarts epochs | median gap | worst gap | within 2% | ODE at optimum (worst) | near best | min/fit")
    for r in restarts:
        for e in epochs:
            rows = [(w, json.loads(cell_file(e, curriculum, w, r).read_text())) for w in CELLS]
            gaps = [row["fit"] / reference(w)["fit"] - 1 for w, row in rows]
            ode = [row["fit"] for w, row in rows if w[1] == "ode"]
            print(f"{r:8d} {e:6d} | {st.median(gaps):+9.1%} | {max(gaps):+8.1%} | {sum(g <= 0.02 for g in gaps):4d}/{len(gaps)} | "
                  f"{sum(f <= 1 for f in ode)}/{len(ode)} ({max(ode):.3f}) | {st.median(row['near_best'] for _, row in rows):5.1f}/{r} | "
                  f"{st.median(row['wall_min'] for _, row in rows):.1f}")


def main() -> None:
    if sys.argv[1] == "--grid":
        return grid([int(x) for x in sys.argv[2].split(",")], [int(x) for x in sys.argv[3].split(",")], sys.argv[4])
    if sys.argv[1] in ("--cells", "--cells-report"):
        epochs, curricula = int(sys.argv[2]), sys.argv[3].split(",")
        if sys.argv[1] == "--cells-report":
            return cells_report(epochs, curricula)
        jobs = [["--one", str(epochs), c, *map(str, w)] for w in CELLS for c in curricula if not cell_file(epochs, c, w).exists()]
        run_pool(__file__, jobs, 6, OUT / "log.txt")
        return cells_report(epochs, curricula)
    if sys.argv[1] == "--one":
        setup_torch(1)
        epochs, c, name, kind, seed = int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5], int(sys.argv[6])
        restarts = int(sys.argv[7]) if len(sys.argv) > 7 else RESTARTS
        row = run(epochs, c, (name, kind, seed), restarts) | {"cell": [name, kind, seed]}
        path = cell_file(epochs, c, (name, kind, seed), restarts)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(row))
        return
    setup_torch(0)
    epochs, curriculum = int(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "linear"
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "runs.json"
    runs = json.loads(path.read_text()) if path.exists() else []
    row = run(epochs, curriculum)
    runs = [r for r in runs if (r["epochs"], r.get("curriculum", "linear")) != (epochs, curriculum)] + [row]
    path.write_text(json.dumps(runs, indent=1))
    ref = reference()
    fmt = lambda r: (f"fit {r['fit']:.4f}  hidden NRMSE {r['nrmse_hidden']:.4f}  "
                     + " ".join(f"{k} {v:.1%}" for k, v in r["rel_err"].items()) + f"  ({r['wall_min']:.1f} min)")
    print(f"reference (64 restarts, 1500 epochs, staged): {fmt(ref)}")
    for r in sorted(runs, key=lambda r: (r.get("curriculum", "linear"), r["epochs"])):
        print(f"{r['epochs']:5d} epochs, {r['restarts']} restarts, {r.get('curriculum', 'linear'):6s}:  {fmt(r)}  "
              f"near best {r['near_best']}/{r['restarts']}")


if __name__ == "__main__":
    main()
