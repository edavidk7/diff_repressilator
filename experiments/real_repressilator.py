"""Our method on real cells: the original repressilator in a mother machine (Potvin-Trottier et al. 2016).

    python experiments/real_repressilator.py --workers 6     download, extract traces, fit (resumes)
    python experiments/real_repressilator.py --report        table and figures in run_results/real_repressilator/

Data: Supplementary Video 2 of the paper (strain NDL332: the Elowitz repressilator with a P_Ltet-GFP-AAV
reporter, 5-min frames, IPTG pulse at frames 580-600). It is a compressed 8-bit movie, not raw images, so
intensities are proportional at best. Each of the 9 trenches holds a mother cell at its closed (top) end;
its trace is the mean intensity of a fixed box there minus the dark gap beside the trench.

Fit: the benchmark's "GFP only" problem, unchanged (the reporter strength and leak absorb the unknown scale
and background), on 10-h windows before the pulse. Checks: fit error against the frame-to-frame noise, a
fitted sinusoid with free period, phase, amplitude and offset as baseline, and a 2-h forecast past each window.
"""

import argparse
import json
import math
import subprocess
import urllib.request

import numpy as np
import torch

from common import RESULTS, run_pool, setup_torch

OUT = RESULTS / "real_repressilator"
URL = ("https://static-content.springer.com/esm/art%3A10.1038%2Fnature19841/MediaObjects/"
       "41586_2016_BFnature19841_MOESM72_ESM.mov")
TRENCH_X = [20, 74, 127, 181, 234, 287, 341, 394, 447]  # trench centres (px)
MOTHER_ROWS = slice(28, 41)  # the mother cell at the closed end
WINDOW, FORECAST = 120, 24  # frames: 10 h fitted, 2 h forecast
STARTS = [0, 120, 240]  # window starts (frames); the IPTG pulse is at 580


def traces() -> np.ndarray:
    """Background-subtracted mother-cell intensity, ``(9 trenches, frames)``; cached."""
    cached = OUT / "traces.npy"
    if cached.exists():
        return np.load(cached)
    OUT.mkdir(parents=True, exist_ok=True)
    movie = OUT / "movie.mov"
    if not movie.exists():
        req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
        movie.write_bytes(urllib.request.urlopen(req).read())
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(movie), "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                         capture_output=True, check=True).stdout
    w, h = 468, 146
    frames = np.frombuffer(raw, np.uint8).reshape(-1, h, w).astype(float)
    box = lambda x: frames[:, MOTHER_ROWS, x - 3:x + 4].mean((1, 2))
    out = np.array([box(x) - box(x + 27 if x + 27 < w else x - 27) for x in TRENCH_X])
    np.save(cached, out)
    return out


def sinusoid_fit(t: np.ndarray, y: np.ndarray):
    """Least-squares offset + amplitude * sin(2 pi t / T + phase), T on a grid: returns a predictor."""
    best = (math.inf, None)
    for T in np.linspace(10, 120, 221):  # model time units (~0.5 to 6 h)
        X = np.stack([np.ones_like(t), np.sin(2 * np.pi * t / T), np.cos(2 * np.pi * t / T)], 1)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        err = float(((X @ coef - y) ** 2).mean())
        if err < best[0]:
            best = (err, (T, coef))
    T, coef = best[1]
    return lambda s: coef[0] + coef[1] * np.sin(2 * np.pi * s / T) + coef[2] * np.cos(2 * np.pi * s / T), T


def run_one(trench: int, start: int) -> None:
    from differentiable_cell.fit import Problem, fit_gradient
    from differentiable_cell.reporter_sim import DT_FRAME
    from panel_benchmark import GRADIENT_SETTINGS

    y_all = traces()[trench]
    t = torch.arange(WINDOW + 1, dtype=torch.float64) * DT_FRAME
    y = torch.tensor(y_all[start:start + WINDOW + 1]).unsqueeze(-1)
    problem = Problem(t=t, y=y, channels=[1], present=[1], tau=[0.75] * 3)  # GFP reporter of TetR
    theta, _ = fit_gradient(problem, seed=trench * 1000 + start, **GRADIENT_SETTINGS)
    t_ext = torch.arange(WINDOW + FORECAST + 1, dtype=torch.float64) * DT_FRAME
    with torch.no_grad():
        pred = problem.predict(theta, t=t_ext)[:, 0].numpy()
    y_ext = y_all[start:start + WINDOW + FORECAST + 1]
    rng = float(y.max() - y.min())
    sine, T_sine = sinusoid_fit(t.numpy(), y[:, 0].numpy())
    rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)) / rng)
    fit_part, fc = slice(0, WINDOW + 1), slice(WINDOW + 1, None)
    vals = problem.values(theta)
    row = {"trench": trench, "start": start, "theta": theta.tolist(), "names": problem.names,
           "estimate": {k: float(vals[k]) for k in ("alpha", "alpha_0", "n", "beta")},
           "nrmse_fit": rmse(pred[fit_part], y_ext[fit_part]), "nrmse_sine_fit": rmse(sine(t.numpy()), y[:, 0].numpy()),
           "nrmse_forecast": rmse(pred[fc], y_ext[fc]), "nrmse_sine_forecast": rmse(sine(t_ext.numpy()[fc]), y_ext[fc]),
           "nrmse_mean_forecast": rmse(np.full(FORECAST, float(y.mean())), y_ext[fc]),
           "noise": float(np.std(np.diff(y[:, 0].numpy())) / math.sqrt(2) / rng),  # frame-to-frame noise
           "period_sine_h": T_sine * DT_FRAME / 12, "y": y_ext.tolist(), "pred": pred.tolist()}
    (OUT / f"fit_trench{trench}_start{start}.json").write_text(json.dumps(row))
    print(trench, start, round(row["nrmse_fit"], 3), round(row["nrmse_sine_fit"], 3), flush=True)


def report() -> None:
    rows = sorted((json.loads(p.read_text()) for p in OUT.glob("fit_*.json")), key=lambda r: (r["trench"], r["start"]))
    med = lambda k: float(np.median([r[k] for r in rows]))
    print(f"{len(rows)} fits. Median NRMSE (range-normalised):")
    for k in ("noise", "nrmse_fit", "nrmse_sine_fit", "nrmse_forecast", "nrmse_sine_forecast", "nrmse_mean_forecast"):
        print(f"  {k:22s} {med(k):.3f}")
    print(f"  fit beats sinusoid on {sum(r['nrmse_fit'] < r['nrmse_sine_fit'] for r in rows)}/{len(rows)}, "
          f"forecast beats it on {sum(r['nrmse_forecast'] < r['nrmse_sine_forecast'] for r in rows)}/{len(rows)}")
    for k in ("alpha", "alpha_0", "n", "beta"):
        v = np.array([r["estimate"][k] for r in rows])
        print(f"  {k:8s} median {np.median(v):8.3g}  IQR {np.percentile(v, 25):8.3g} - {np.percentile(v, 75):8.3g}")
    plot(rows)


def plot(rows) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    trenches = sorted({r["trench"] for r in rows})
    fig, axes = plt.subplots(len(trenches), 1, figsize=(14, 1.6 * len(trenches)), sharex=True, squeeze=False)
    for ax, tr in zip(axes[:, 0], trenches):
        for r in (r for r in rows if r["trench"] == tr):
            h = (r["start"] + np.arange(len(r["y"]))) * 5 / 60
            ax.scatter(h, r["y"], s=3, color="#3f7f2a", alpha=0.6, linewidths=0)
            ax.plot(h[:WINDOW + 1], r["pred"][:WINDOW + 1], color="#2a78d6", lw=1.3)
            ax.plot(h[WINDOW:], r["pred"][WINDOW:], color="#2a78d6", lw=1.3, ls="--")
        ax.set_ylabel(f"cell {tr}", fontsize=8)
        ax.grid(color="#e0e0e0", lw=0.5)
    axes[-1, 0].set_xlabel("time [h]  (dots: mother-cell GFP; line: fit; dashed: 2-h forecast)")
    fig.tight_layout()
    fig.savefig(OUT / "fits.png", dpi=110)
    print(OUT / "fits.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--one", nargs=2, type=int, metavar=("TRENCH", "START"))
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        return report()
    if args.one:
        setup_torch(args.threads)
        return run_one(*args.one)
    traces()
    jobs = [["--threads", str(args.threads), "--one", str(tr), str(s)] for s in STARTS for tr in range(len(TRENCH_X))
            if not (OUT / f"fit_trench{tr}_start{s}.json").exists()]
    run_pool(__file__, jobs, args.workers, OUT / "log.txt")


if __name__ == "__main__":
    main()
