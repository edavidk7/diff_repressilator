"""Time per epoch of the fit against the number of restarts, on the CPU and on a GPU.

    python experiments/gpu_scaling.py [--restarts 64,256,1024,4096,16384] [--epochs 20]

One cell (3 reporters, ODE, seed 0), whole-record epochs (the costliest of the schedule); prints seconds per epoch
and restarts per second for each device, and the full-record MSE of the best restart as a check.
"""

import argparse
import time

import torch

from common import cell, setup_torch
from differentiable_cell.fit import fit_gradient


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--restarts", default="64,256,1024,4096,16384")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    setup_torch(args.threads)
    problem, _ = cell("3 reporters", "ode", 0)
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for device in devices:
        fit_gradient(problem, restarts=8, epochs=2, curriculum="none", device=device)  # warm-up
        for n in map(int, args.restarts.split(",")):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _, final = fit_gradient(problem, restarts=n, epochs=args.epochs, curriculum="none", device=device)
            if device == "cuda":
                torch.cuda.synchronize()
            per_epoch = (time.perf_counter() - t0) / args.epochs
            print(f"{device:5s} restarts {n:6d}: {per_epoch:7.3f} s per epoch, {n / per_epoch:9.0f} restarts per second, "
                  f"best MSE {float(final.min()):.4g}", flush=True)


if __name__ == "__main__":
    main()
