"""Our method on the improved repressilator in a mother machine (Potvin-Trottier et al. 2016, LPT117).

    python experiments/real_improved.py --workers 6     download, extract traces, fit (resumes)
    python experiments/real_improved.py --report        table and run_results/real_improved/fits.png

Data: Supplementary Video 6 (no protein degradation, TetR titration sponge, three reporters: mKate2 red for
CI, Venus green for TetR, CFP blue for LacI; 8-min frames). A compressed RGB movie; each trench's mother
cell is a fixed box at the closed (top) end minus the gap beside the trench, per colour channel.
Model: circuits.REPRESSILATOR_DILUTION (all proteins diluted at rate beta; reporter maturation and sponge
strength free). 24-h windows, the benchmark's fit; checks as in real_repressilator.py (noise, a per-channel
fitted sinusoid, a 4-h forecast, consistency across cells).
"""

import argparse
import json
import math
import subprocess
import urllib.request

import numpy as np
import torch

from common import RESULTS, run_pool, setup_torch
from real_repressilator import sinusoid_fit

OUT = RESULTS / "real_improved"
URL = ("https://static-content.springer.com/esm/art%3A10.1038%2Fnature19841/MediaObjects/"
       "41586_2016_BFnature19841_MOESM76_ESM.mov")
TRENCH_X = [66, 172, 279, 386, 490, 702, 808, 916]
MOTHER_ROWS = slice(58, 82)
WINDOW, FORECAST = 180, 30  # frames of 8 min: 24 h fitted, 4 h forecast
# (trench, window start): cells that oscillate through the window and its forecast
JOBS = [(tr, s) for tr in (0, 1, 4, 5, 6) for s in (0, 180)] + [(2, 0)]
RGB_TO_STATE = [14, 13, 12]  # red mKate2 -> F_cI, green Venus -> F_tet, blue CFP -> F_lac


def traces() -> np.ndarray:
    """Background-subtracted mother-cell intensity, ``(trenches, frames, RGB)``; cached."""
    cached = OUT / "traces.npy"
    if cached.exists():
        return np.load(cached)
    OUT.mkdir(parents=True, exist_ok=True)
    movie = OUT / "movie.mov"
    if not movie.exists():
        req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
        movie.write_bytes(urllib.request.urlopen(req).read())
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(movie), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    w, h = 932, 364
    frames = np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3).astype(float)
    box = lambda x: frames[:, MOTHER_ROWS, x - 5:x + 6].mean((1, 2))
    out = np.array([box(x) - box(x + 53 if x + 53 < w - 6 else x - 53) for x in TRENCH_X])
    np.save(cached, out)
    return out


def run_one(trench: int, start: int) -> None:
    from dataclasses import replace

    from differentiable_cell.circuits import REPRESSILATOR_DILUTION, CircuitProblem
    from differentiable_cell.fit import fit_gradient
    from panel_benchmark import GRADIENT_SETTINGS

    c = replace(REPRESSILATOR_DILUTION, observed=RGB_TO_STATE)
    y_all = traces()[trench]
    t = torch.arange(WINDOW + 1, dtype=torch.float64) * c.dt_frame
    y = torch.tensor(y_all[start:start + WINDOW + 1])
    problem = CircuitProblem(c, t, y)
    theta, _ = fit_gradient(problem, seed=trench * 1000 + start, **GRADIENT_SETTINGS)
    t_ext = torch.arange(WINDOW + FORECAST + 1, dtype=torch.float64) * c.dt_frame
    with torch.no_grad():
        pred = problem.predict(theta, t=t_ext).numpy()
    y_ext = y_all[start:start + WINDOW + FORECAST + 1]
    rng = (y.max(0).values - y.min(0).values).numpy()
    fit_part, fc = slice(0, WINDOW + 1), slice(WINDOW + 1, None)
    nrmse = lambda a, b: float(np.sqrt(np.mean(((a - b) / rng) ** 2)))
    sines = [sinusoid_fit(t.numpy(), y[:, j].numpy())[0] for j in range(3)]
    sine = lambda s: np.stack([f(s) for f in sines], -1)
    vals = problem.values(theta)
    period = lambda x: float(np.median(np.diff(np.nonzero((x[:-1] < x.mean()) & (x[1:] >= x.mean()))[0]))) * c.frame_min / 60
    row = {"trench": trench, "start": start, "theta": theta.tolist(), "names": problem.names,
           "estimate": {k: float(vals[k]) for k in c.params},
           "nrmse_fit": nrmse(pred[fit_part], y_ext[fit_part]), "nrmse_sine_fit": nrmse(sine(t.numpy()), y.numpy()),
           "nrmse_forecast": nrmse(pred[fc], y_ext[fc]), "nrmse_sine_forecast": nrmse(sine(t_ext.numpy()[fc]), y_ext[fc]),
           "nrmse_mean_forecast": nrmse(np.broadcast_to(y.numpy().mean(0), y_ext[fc].shape), y_ext[fc]),
           "noise": float(np.mean(np.std(np.diff(y.numpy(), axis=0), axis=0) / math.sqrt(2) / rng)),
           "period_data_h": period(y[:, 0].numpy()), "period_fit_h": period(pred[fit_part, 0]),
           "y": y_ext.tolist(), "pred": pred.tolist()}
    (OUT / f"fit_trench{trench}_start{start}.json").write_text(json.dumps(row))
    print(trench, start, round(row["nrmse_fit"], 3), round(row["nrmse_sine_fit"], 3), flush=True)


def report() -> None:
    rows = sorted((json.loads(p.read_text()) for p in OUT.glob("fit_*.json")), key=lambda r: (r["trench"], r["start"]))
    med = lambda k: float(np.median([r[k] for r in rows]))
    print(f"{len(rows)} fits. Median NRMSE (range-normalised, mean over the 3 channels):")
    for k in ("noise", "nrmse_fit", "nrmse_sine_fit", "nrmse_forecast", "nrmse_sine_forecast", "nrmse_mean_forecast"):
        print(f"  {k:22s} {med(k):.3f}")
    print(f"  fit beats sinusoids on {sum(r['nrmse_fit'] < r['nrmse_sine_fit'] for r in rows)}/{len(rows)}, "
          f"forecast beats them on {sum(r['nrmse_forecast'] < r['nrmse_sine_forecast'] for r in rows)}/{len(rows)}, "
          f"beats the mean on {sum(r['nrmse_forecast'] < r['nrmse_mean_forecast'] for r in rows)}/{len(rows)}")
    print(f"  period data {med('period_data_h'):.2f} h, fit {med('period_fit_h'):.2f} h")
    for k in rows[0]["estimate"]:
        v = np.array([r["estimate"][k] for r in rows])
        print(f"  {k:8s} median {np.median(v):8.3g}  IQR {np.percentile(v, 25):8.3g} - {np.percentile(v, 75):8.3g}")
    plot(rows)


def plot(rows) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = ["#c0392b", "#27ae60", "#2e6fd8"]
    trenches = sorted({r["trench"] for r in rows})
    fig, axes = plt.subplots(len(trenches), 1, figsize=(14, 2.0 * len(trenches)), sharex=True, squeeze=False)
    for ax, tr in zip(axes[:, 0], trenches):
        for r in (r for r in rows if r["trench"] == tr):
            h = (r["start"] + np.arange(len(r["y"]))) * 8 / 60
            y, pred = np.array(r["y"]), np.array(r["pred"])
            for j, col in enumerate(colours):
                ax.scatter(h, y[:, j], s=2, color=col, alpha=0.5, linewidths=0)
                ax.plot(h[:WINDOW + 1], pred[:WINDOW + 1, j], color=col, lw=1.2)
                ax.plot(h[WINDOW:], pred[WINDOW:, j], color=col, lw=1.2, ls="--")
        ax.set_ylabel(f"cell {tr}", fontsize=8)
        ax.grid(color="#e0e0e0", lw=0.5)
    axes[-1, 0].set_xlabel("time [h]  (dots: mKate2 / Venus / CFP of the mother cell; line: fit; dashed: 4-h forecast)")
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
    jobs = [["--threads", str(args.threads), "--one", str(tr), str(s)] for tr, s in JOBS
            if not (OUT / f"fit_trench{tr}_start{s}.json").exists()]
    run_pool(__file__, jobs, args.workers, OUT / "log.txt")


if __name__ == "__main__":
    main()
