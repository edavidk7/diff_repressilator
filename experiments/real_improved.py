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
import time
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
FULL, FULL_FRAMES = [(tr, 0) for tr in (0, 1, 4, 5, 6)], 406  # one fit of each whole record (54 h), no forecast
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


def job_file(method: str, trench: int, start: int, frames: int = WINDOW):
    name = f"fit_trench{trench}_start{start}" + ("" if frames == WINDOW else f"_frames{frames}") + ".json"
    return OUT / name if method == "gradient" else OUT / method / name


def run_one(method: str, trench: int, start: int, frames: int = WINDOW) -> None:
    """Fit ``frames`` frames from ``start``; forecast the next 4 h where the record allows."""
    from dataclasses import replace

    from differentiable_cell.circuits import REPRESSILATOR_DILUTION, CircuitProblem
    from panel_benchmark import estimate

    c = replace(REPRESSILATOR_DILUTION, observed=RGB_TO_STATE)
    y_all = traces()[trench]
    forecast = min(FORECAST, len(y_all) - 1 - start - frames)
    t = torch.arange(frames + 1, dtype=torch.float64) * c.dt_frame
    y = torch.tensor(y_all[start:start + frames + 1])
    problem = CircuitProblem(c, t, y)
    t0 = time.perf_counter()
    theta = estimate(method, problem, trench * 1000 + start)[0]
    wall = time.perf_counter() - t0
    t_ext = torch.arange(frames + forecast + 1, dtype=torch.float64) * c.dt_frame
    with torch.no_grad():
        pred = problem.predict(theta, t=t_ext).numpy()
    y_ext = y_all[start:start + frames + forecast + 1]
    rng = (y.max(0).values - y.min(0).values).numpy()
    fit_part, fc = slice(0, frames + 1), slice(frames + 1, None)
    nrmse = lambda a, b: float(np.sqrt(np.mean(((a - b) / rng) ** 2))) if len(a) else math.nan
    sines = [sinusoid_fit(t.numpy(), y[:, j].numpy())[0] for j in range(3)]
    sine = lambda s: np.stack([f(s) for f in sines], -1)
    vals = problem.values(theta)

    def period(x):
        if not np.isfinite(x).all():
            return math.nan
        ups = np.nonzero((x[:-1] < x.mean()) & (x[1:] >= x.mean()))[0]
        return float(np.median(np.diff(ups))) * c.frame_min / 60 if len(ups) > 1 else math.nan

    row = {"method": method, "trench": trench, "start": start, "frames": frames, "wall": wall,
           "theta": theta.tolist(), "names": problem.names, "estimate": {k: float(vals[k]) for k in c.params},
           "nrmse_fit": nrmse(pred[fit_part], y_ext[fit_part]), "nrmse_sine_fit": nrmse(sine(t.numpy()), y.numpy()),
           "nrmse_forecast": nrmse(pred[fc], y_ext[fc]), "nrmse_sine_forecast": nrmse(sine(t_ext.numpy()[fc]), y_ext[fc]),
           "nrmse_mean_forecast": nrmse(np.broadcast_to(y.numpy().mean(0), y_ext[fc].shape), y_ext[fc]),
           "noise": float(np.mean(np.std(np.diff(y.numpy(), axis=0), axis=0) / math.sqrt(2) / rng)),
           "period_data_h": period(y[:, 0].numpy()), "period_fit_h": period(pred[fit_part, 0]),
           "y": y_ext.tolist(), "pred": pred.tolist()}
    path = job_file(method, trench, start, frames)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row))
    print(method, trench, start, frames, round(row["nrmse_fit"], 3), round(row["nrmse_sine_fit"], 3), flush=True)


METHODS = ["gradient", "abc", "magi", "pinn"]


def load(method: str, frames: int = WINDOW) -> list[dict]:
    cells = JOBS if frames == WINDOW else FULL
    rows = [json.loads(job_file(method, tr, st, frames).read_text()) for tr, st in cells
            if job_file(method, tr, st, frames).exists()]
    return sorted(rows, key=lambda r: (r["trench"], r["start"]))


def report() -> None:
    keys = ("noise", "nrmse_fit", "nrmse_sine_fit", "nrmse_forecast", "nrmse_sine_forecast", "nrmse_mean_forecast")
    print("24-h windows, median NRMSE (range-normalised, mean over the 3 channels)")
    print(f"  {'method':9s} {'n':>3s} " + " ".join(f"{k:>20s}" for k in keys) + "   beats sine (fit / forecast)")
    for m in METHODS:
        rows = load(m)
        if rows:
            med = lambda k: float(np.nanmedian([r[k] for r in rows]))
            wins = (sum(r["nrmse_fit"] < r["nrmse_sine_fit"] for r in rows), sum(r["nrmse_forecast"] < r["nrmse_sine_forecast"] for r in rows))
            print(f"  {m:9s} {len(rows):3d} " + " ".join(f"{med(k):20.3f}" for k in keys) + f"   {wins[0]}/{len(rows)} / {wins[1]}/{len(rows)}")
            print(f"            period data {med('period_data_h'):.2f} h, fit {med('period_fit_h'):.2f} h; "
                  + ", ".join(f"{k} {np.median([r['estimate'][k] for r in rows]):.3g}" for k in ("alpha", "n", "beta", "tau_tet")))
            plot(rows, f"fits_{m}.png" if m != "gradient" else "fits.png")
    for m in ("gradient",):
        full = load(m, FULL_FRAMES)
        if not full:
            continue
        print(f"\nwhole record ({FULL_FRAMES} frames, {FULL_FRAMES * 8 / 60:.0f} h), {m}: "
              f"NRMSE {np.median([r['nrmse_fit'] for r in full]):.3f} vs sinusoids {np.median([r['nrmse_sine_fit'] for r in full]):.3f}, "
              f"period fit {np.median([r['period_fit_h'] for r in full]):.2f} h (data {np.median([r['period_data_h'] for r in full]):.2f} h), "
              f"beta {np.median([r['estimate']['beta'] for r in full]):.3g}")
        plot(full, "fits_whole_record.png" if m == "gradient" else f"fits_whole_record_{m}.png")


def plot(rows, name: str = "fits.png") -> None:
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
                n = r.get("frames", WINDOW)
                if np.isfinite(pred[:, j]).all():
                    ax.plot(h[:n + 1], pred[:n + 1, j], color=col, lw=1.2)
                    ax.plot(h[n:], pred[n:, j], color=col, lw=1.2, ls="--")
        ax.set_ylabel(f"cell {tr}", fontsize=8)
        ax.grid(color="#e0e0e0", lw=0.5)
    axes[-1, 0].set_xlabel("time [h]  (dots: mKate2 / Venus / CFP of the mother cell; line: fit; dashed: 4-h forecast)")
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=110)
    print(OUT / name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="gradient", help="comma-separated: gradient, abc, magi, pinn")
    ap.add_argument("--whole-record", action="store_true", help="also fit each full record once (gradient methods)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--one", nargs=4, metavar=("METHOD", "TRENCH", "START", "FRAMES"))
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        return report()
    if args.one:
        setup_torch(args.threads)
        return run_one(args.one[0], *map(int, args.one[1:]))
    traces()
    todo = [(m, tr, s, WINDOW) for tr, s in JOBS for m in args.method.split(",")]
    if args.whole_record:
        todo = [(m, tr, s, FULL_FRAMES) for tr, s in FULL for m in args.method.split(",")
                if m.startswith("gradient")] + todo  # longest first
    jobs = [["--threads", str(args.threads), "--one", m, str(tr), str(s), str(n)] for m, tr, s, n in todo
            if not job_file(m, tr, s, n).exists()]
    run_pool(__file__, jobs, args.workers, OUT / "log.txt")


if __name__ == "__main__":
    main()
