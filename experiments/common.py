"""Shared by the experiment scripts: reporter configurations, one cell's fitting problem, a job pool."""

import math
import subprocess
import sys
import time
from pathlib import Path

import torch

from differentiable_cell import reporters as R
from differentiable_cell.fit import Problem
from differentiable_cell.reporter_sim import CHANNELS, box1_truth, generate

RESULTS = Path(__file__).resolve().parent.parent / "run_results"
FIGURES = RESULTS / "figures"

# Reporter configurations: observed channels (names from reporter_sim.CHANNELS).
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
MAIN_PANELS = ("GFP only", "3 reporters", "TetR fusion + GFP")  # every method ran on these
KINDS = ("ode", "ssa")  # deterministic or exact-Gillespie cell


def current(folder):
    """Result JSONs one level down, skipping archived earlier versions (folders ending in _v1)."""
    return [p for p in sorted(folder.glob("*/*.json")) if not p.parent.name.endswith("_v1")]


def slug(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text)


def panel(spec: str) -> tuple[list[int], list[int]]:
    """Comma-separated channel names -> (channel indices, reporters built)."""
    names = [c[0] for c in CHANNELS]
    channels = [names.index(n) for n in spec.split(",")]
    return channels, [c for c in channels if c < 3]


def cell(name: str, kind: str, seed: int):
    """One simulated cell of configuration ``name`` and its fitting problem: ``(problem, data)``."""
    channels, present = panel(PANELS[name])
    data = generate(box1_truth(), present, kind=kind, noise=True, seed=seed)
    problem = Problem(t=data.t, y=data.y[:, channels], channels=channels, present=present, tau=data.truth["tau"])
    return problem, data


def true_theta(problem: Problem, data) -> torch.Tensor:
    """The true unknowns (log values, in ``problem.names`` order): parameters and the cell's start."""
    tr, s0 = data.truth, data.fine_state[0]
    true = {"alpha": tr["alpha"], "alpha_0": tr["alpha_0"], "n": tr["n"], "beta": tr["beta"]}
    for i, s in enumerate(["m_lacI", "m_tetR", "m_cI", "p_LacI", "p_TetR", "p_CI"]):
        true[f"{s}(0)"] = float(s0[i])
    for j in problem.present:
        r = R.REPORTERS[j]
        true.update({f"a_{r}": tr["a"][j], f"a0_{r}": tr["a_0"][j], f"r_{r}(0)": float(s0[R.R_IDX[j]]),
                     f"D_{r}(0)": float(s0[R.D_IDX[j]]), f"F_{r}(0)": float(s0[R.F_IDX[j]])})
    return torch.tensor([math.log(max(true[n], 1e-12)) for n in problem.names])


def run_pool(script: str, jobs: list[list[str]], workers: int, log: Path) -> None:
    """Run ``python script <job args>`` for every job, ``workers`` at a time, output to ``log``."""
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"{len(jobs)} jobs", flush=True)
    pending, running = list(jobs), []
    with open(log, "a") as out:
        while pending or running:
            running = [p for p in running if p.poll() is None]
            while pending and len(running) < workers:
                running.append(subprocess.Popen([sys.executable, script, *pending.pop(0)],
                                                stdout=out, stderr=subprocess.STDOUT))
            time.sleep(5)
    print("done", flush=True)


def setup_torch(threads: int) -> None:
    """``threads`` <= 0: torch's default, all cores."""
    if threads > 0:
        torch.set_num_threads(threads)
    torch.set_default_dtype(torch.float64)
