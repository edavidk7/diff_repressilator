"""Run the estimator benchmark: cases x methods -> one JSON per result.

    python experiments/benchmark.py --suite ode_panels --workers 3

Each case runs in its own subprocess (so nested sampling can open its own
process pool), with its methods run one after another.  Results are written
as soon as each method finishes, to

    run_results/benchmark_<suite>/<case_id>/<method>.json

and anything already on disk is skipped, so an interrupted run resumes
where it stopped.  ``--list`` prints the cases of a suite without running.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

ODE_METHODS = ["grad_ode", "nuts", "abc_ode", "nested", "magi", "pinn"]
STOCH_METHODS = ["grad_ode", "nuts", "abc_ode", "abc_cle", "grad_cle", "nested", "magi", "pinn"]


def suites() -> dict[str, list[dict]]:
    """Every suite as a list of case specs (kwargs for ``make_case`` + methods)."""
    from differentiable_cell.inference.datasets import paper_truth, sample_truth

    g = torch.Generator().manual_seed(2026)
    sampled = [sample_truth(g) for _ in range(2)]
    truths = {"box1": paper_truth(), "rand0": sampled[0], "rand1": sampled[1]}
    out: dict[str, list[dict]] = {}

    out["validate"] = [dict(truth="box1", panel="all", regime="ode", noise=0.01, seed=0,
                            methods=ODE_METHODS)]
    out["ode_panels"] = [
        dict(truth=tr, panel=pn, regime="ode", noise=0.01, seed=0, methods=ODE_METHODS)
        for tr in ("box1", "rand0", "rand1")
        for pn in ("tetR+gfp", "gfp", "3proteins", "cI+gfp", "all")
    ]
    out["stochastic"] = [
        dict(truth="box1", panel=pn, regime=rg, noise=0.01, seed=0, omega=40.0,
             n_cells=200, methods=STOCH_METHODS)
        for rg in ("ensemble", "single_cell") for pn in ("tetR+gfp", "3proteins")
    ]
    # Small system, one period: where the differentiable Gillespie is affordable.
    out["dga_small"] = [
        dict(truth="box1", panel="tetR+gfp", regime="ensemble", noise=0.01, seed=0, omega=2.0,
             n_cells=200, horizon=45.0, n_frames=27,
             methods=["grad_dga", "grad_cle", "grad_ode", "abc_cle"])
    ]
    out["noise"] = [
        dict(truth="box1", panel="tetR+gfp", regime="ode", noise=nz, seed=0, methods=ODE_METHODS)
        for nz in (0.03, 0.1)
    ]
    # Everything above in priority order, for one long unattended run:
    # Box 1 across panels, the stochastic regimes, a second truth, noise, a third.
    by = lambda tr: [c for c in out["ode_panels"] if c["truth"] == tr]
    out["overnight"] = by("box1") + out["stochastic"] + by("rand0") + out["noise"] + by("rand1")
    out["nested_slice"] = [dict(c, methods=["nested_slice"]) for c in
                           out["validate"] + by("box1")[:3]]

    # v2: every configuration over 5 seeds, one pass per method, cheapest
    # first.  All passes write into run_results/benchmark_v2 (see ROOTS).
    seeds = range(5)
    ode = [dict(truth=tr, panel=pn, regime="ode", noise=0.01, seed=s)
           for s in seeds for tr in ("box1", "rand0", "rand1")
           for pn in ("tetR+gfp", "gfp", "3proteins", "cI+gfp", "all")]
    noisy = [dict(truth="box1", panel="tetR+gfp", regime="ode", noise=nz, seed=s)
             for s in seeds for nz in (0.03, 0.1)]
    stoch = [dict(truth="box1", panel=pn, regime=rg, noise=0.01, seed=s, omega=40.0, n_cells=200)
             for s in seeds for rg in ("ensemble", "single_cell") for pn in ("tetR+gfp", "3proteins")]
    everything = ode + noisy + stoch
    for method in ("grad_ode", "magi", "abc_ode", "pinn"):
        out[f"v2_{method}"] = [dict(c, methods=[method]) for c in everything]
    for method in ("grad_cle", "abc_cle"):
        out[f"v2_{method}"] = [dict(c, methods=[method]) for c in stoch]
    # The two expensive methods (~35 min and 1-3 h per case) get the Box 1
    # truth only, seeds 0-1: every Box 1 configuration, replicated twice.
    expensive = [c for c in everything if c["truth"] == "box1" and c["seed"] < 2]
    out["v2_nuts"] = [dict(c, methods=["grad_ode", "nuts"]) for c in expensive]
    # Nested sampling dropped from v2 (1-3 h per case; its failure mode is
    # documented from v1 and it works in its own paper's low-dimensional
    # setting, see docs/guide.md 6.5).  The pass is kept, empty, so the
    # running driver script moves straight past it.
    out["v2_nested"] = []
    for cases in out.values():
        for c in cases:
            c["truth_values"] = truths[c["truth"]]
    return out


def case_id(spec: dict) -> str:
    parts = [spec["truth"], spec["panel"], spec["regime"], f"noise{spec['noise']:g}",
             f"seed{spec['seed']}"]
    if spec["regime"] != "ode":
        parts.append(f"omega{spec.get('omega', 40.0):g}")
    if "horizon" in spec:
        parts.append(f"T{spec['horizon']:g}")
    return "_".join(parts)


# --------------------------------------------------------------------------
# One case (runs inside a subprocess)
# --------------------------------------------------------------------------


def run_method(name: str, problem, seed: int, cache: dict, threads: int):
    from differentiable_cell.inference import (
        abc_smc, grad_ode, grad_stochastic, magi, nested, nuts, pinn)

    if name == "grad_ode":
        res = grad_ode.fit(problem, seed=seed)
        cache["grad_ode"] = res
        return res
    if name == "nuts":
        return nuts.fit(problem, map_result=cache.get("grad_ode"), seed=seed, chains=4,
                        max_seconds=1800)
    if name == "abc_ode":
        return abc_smc.fit(problem, simulator="ode", max_sims=3_000_000, max_seconds=1200,
                           min_acceptance=1e-4, seed=seed)
    if name == "abc_cle":
        return abc_smc.fit(problem, simulator="cle", n_particles=500, batch=500,
                           max_sims=20_000_000, max_seconds=1200, min_acceptance=1e-3, seed=seed)
    if name == "nested":
        return nested.fit(problem, nlive=300, max_calls=1_000_000, workers=max(threads, 2),
                          seed=seed)
    if name == "nested_unif":
        # MultiNest-faithful: uniform sampling within the multi-ellipsoid
        # bound (Pullen & Morris used MultiNest), Delta log Z = 0.5.
        return nested.fit(problem, nlive=400, sample="unif", max_calls=2_000_000,
                          workers=max(threads, 2), seed=seed)
    if name == "nested_slice":
        # Second chance for nested sampling: random-slice proposals (dynesty's
        # recommendation for ~10-20 dimensions) and more live points, after
        # random-walk runs collapsed onto a secondary mode on the easy case.
        return nested.fit(problem, nlive=500, sample="rslice", max_calls=2_000_000,
                          workers=max(threads, 2), seed=seed)
    if name == "magi":
        return magi.fit(problem, seed=seed, max_seconds=600)
    if name == "pinn":
        return pinn.fit(problem, seed=seed, max_seconds=400)
    if name == "grad_cle":
        return grad_stochastic.fit(problem, kind="cle", seed=seed, max_seconds=1200)
    if name == "grad_dga":
        # DGA gradients grow ~exponentially with the number of events (see
        # experiments/dga_gradient_growth.py), so it sees only the first 4
        # frames (~5 model units); Adam normalises the step size.
        return grad_stochastic.fit(problem, kind="dga", restarts=4, epochs=150, n_cells=8,
                                   eval_cells=8, n_obs_times=4, seed=seed,
                                   max_seconds=3 * 3600)
    raise ValueError(name)


def run_case(spec: dict, out_dir: Path, threads: int) -> None:
    from differentiable_cell.inference.base import Result
    from differentiable_cell.inference.datasets import make_case
    from differentiable_cell.inference.scoring import score

    torch.set_num_threads(threads)
    torch.set_default_dtype(torch.float64)
    out_dir.mkdir(parents=True, exist_ok=True)
    case_file = out_dir / "case.pt"
    if case_file.exists():
        case = torch.load(case_file, weights_only=False)
    else:
        kw = {k: v for k, v in spec.items() if k not in ("truth", "truth_values", "methods")}
        t0 = time.time()
        case = make_case(spec["truth_values"], **kw)
        case.meta["generation_seconds"] = time.time() - t0
        torch.save(case, case_file)
        (out_dir / "case.json").write_text(json.dumps(
            {"spec": {k: v for k, v in spec.items() if k != "methods"}, "truth": case.truth,
             "meta": case.meta, "free_params": case.problem.free_params,
             "observed": case.problem.observed}, indent=2))
    cache: dict = {}
    for name in spec["methods"]:
        path = out_dir / f"{name}.json"
        if path.exists():
            if name == "grad_ode":  # NUTS starts from its MAP: reload it
                row = json.loads(path.read_text())
                if "theta" in row:
                    cache["grad_ode"] = Result(method="grad_ode+laplace",
                                               theta=torch.tensor(row["theta"]), samples=None,
                                               n_sims=row["n_sims"], wall=row["wall"])
            continue
        print(f"[{out_dir.name}] {name} ...", flush=True)
        t0 = time.time()
        try:
            res = run_method(name, case.problem, spec["seed"], cache, threads)
            row = score(case, res)
            row["info"] = res.info
            row["theta"] = res.theta.tolist()
            row["names"] = case.problem.names
            if res.samples is not None:
                torch.save(res.samples, out_dir / f"{name}_samples.pt")
        except Exception as exc:  # a failing method is a result, not a crash
            row = {"method": name, "error": repr(exc), "traceback": traceback.format_exc(),
                   "wall": time.time() - t0, **case.meta}
        path.write_text(json.dumps(row, indent=2, default=float))
        print(f"[{out_dir.name}] {name} done in {time.time() - t0:.0f}s", flush=True)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--suite", required=True)
    ap.add_argument("--workers", type=int, default=3, help="cases run in parallel")
    ap.add_argument("--threads", type=int, default=3, help="torch threads per case")
    ap.add_argument("--only", nargs="*", help="restrict to these methods")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--case_index", type=int, help=argparse.SUPPRESS)
    args = ap.parse_args()

    cases = suites()[args.suite]
    if args.only:
        for c in cases:
            c["methods"] = [m for m in c["methods"] if m in args.only]
    root = REPO / "run_results" / f"benchmark_{'v2' if args.suite.startswith('v2_') else args.suite}"

    if args.case_index is not None:
        spec = cases[args.case_index]
        run_case(spec, root / case_id(spec), args.threads)
        return
    if args.list:
        for i, c in enumerate(cases):
            print(i, case_id(c), c["methods"])
        return

    root.mkdir(parents=True, exist_ok=True)
    pending = list(range(len(cases)))
    running: list[subprocess.Popen] = []
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    log = open(root / "orchestrator.log", "a")
    while pending or running:
        running = [p for p in running if p.poll() is None]
        while pending and len(running) < args.workers:
            i = pending.pop(0)
            cmd = [sys.executable, __file__, "--suite", args.suite, "--threads",
                   str(args.threads), "--case_index", str(i)]
            if args.only:
                cmd += ["--only", *args.only]
            running.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env))
            print(f"started case {i}: {case_id(cases[i])}", flush=True)
        time.sleep(5)
    print("suite done", flush=True)


if __name__ == "__main__":
    main()
