"""How well does each estimate reproduce the time series? From the saved fits; no new runs.

    python experiments/fit_quality.py      -> run_results/fit_quality/{figures, summary.json}

Main configurations, seeds 0-12, cells every method finished. Per cell and method, from the ODE integrated
at the estimate (30-s grid) against the frames and the hidden noise-free trajectory, all errors divided by
the channel's range:
  nrmse_frames   RMSE to the frames (sqrt of the benchmark loss; at the truth on ODE cells: camera noise)
  nrmse_hidden   RMSE to the hidden trajectory;  hourly: the same in 1-h windows
  period_err     relative period error (autocorrelation);  amp_ratio: amplitude fit / truth (2-98%)
  late_amp_ratio amp_ratio over the last 2 h; below 0.25 the oscillation is "damped"
  nrmse_aligned  nrmse_hidden after the best constant time shift (within half a period)
Paired tests of the differentiable ODE against each method on nrmse_hidden (wins, Wilcoxon, bootstrap 95%
interval of the median ratio). Adds the results of fit_controls.py and the restart spread when present.
"""

import json
import math
import random
import statistics as st
from collections import defaultdict

import matplotlib
import numpy as np
from scipy.stats import wilcoxon

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import MAIN_PANELS, RESULTS, current

IN, CTRL, OUT = RESULTS / "panel_benchmark", RESULTS / "fit_controls", RESULTS / "fit_quality"
HOURS_PER_UNIT = 2.0 / math.log(2) / 60
SEEDS = 13  # the seeds every method finished on both kinds of cell
METHODS = ["gradient", "abc", "magi", "pinn"]
LABEL = {"gradient": "diff. ODE", "abc": "ABC-SMC", "magi": "MAGI", "pinn": "inverse PINN"}
COLOUR = {"gradient": "#2a78d6", "abc": "#eb6834", "magi": "#e87ba4", "pinn": "#4a3aa7"}
KINDS = {"ode": "ODE cells", "ssa": "Gillespie cells"}
CH_LABEL = {"lac": "mJuniper (P$_{Llac}$)", "tet": "GFP (P$_{Ltet}$)", "cI": "mScarlet-I3 (λP$_R$)",
            "TetR-fusion": "TetR fusion"}
CH_COLOUR = {"lac": "#1baf7a", "tet": "#3f7f2a", "cI": "#b8323f", "TetR-fusion": "#c98a00"}


# ---------------------------------------------------------------- metrics

def period(x: np.ndarray, dt: float) -> float:
    """First autocorrelation peak after its first zero crossing; nan if there is none."""
    x = x - x.mean()
    if not np.isfinite(x).all() or x.std() < 1e-12:
        return math.nan
    ac = np.correlate(x, x, "full")[len(x) - 1:]
    neg = np.nonzero(ac < 0)[0]
    if not len(neg):
        return math.nan
    seg = ac[neg[0]: len(ac) * 3 // 4]
    return (neg[0] + int(seg.argmax())) * dt if len(seg) and seg.max() > 0 else math.nan


def shifted_rmse(fit, true, rng, max_shift: int) -> float:
    """Smallest RMSE over constant shifts of at most ``max_shift`` samples."""
    return min(float(np.sqrt(np.mean(((fit[max(s, 0): len(fit) + min(s, 0)]
                                       - true[max(-s, 0): len(true) + min(-s, 0)]) / rng) ** 2)))
               for s in range(-max_shift, max_shift + 1))


def metrics(r: dict) -> dict:
    y, fit, true = np.array(r["y"]), np.array(r["fit_fine"]), np.array(r["truth_fine"])
    ft = np.array(r["fine_t"]) * HOURS_PER_UNIT
    out = {"ok": bool(np.isfinite(fit).all() and math.isfinite(r["loss_fit"])),
           "nrmse_frames": math.sqrt(r["loss_fit"]), "nrmse_truth_frames": math.sqrt(r["loss_truth"])}
    if not out["ok"]:
        return out | dict.fromkeys(["nrmse_frames", "nrmse_hidden", "period_err", "amp_ratio", "late_amp_ratio",
                                    "nrmse_aligned"], math.nan) | {"hourly": [math.nan] * 10}
    rng, dt = y.max(0) - y.min(0), ft[1] - ft[0]
    res = (fit - true) / rng
    per_hour = int(round(1 / dt))
    out["nrmse_hidden"] = float(np.sqrt(np.mean(res**2)))
    out["hourly"] = [float(np.sqrt(np.mean(res[h * per_hour:(h + 1) * per_hour] ** 2))) for h in range(10)]
    p_true = [period(true[:, c], dt) for c in range(true.shape[1])]
    errs = [abs(period(fit[:, c], dt) - p) / p for c, p in enumerate(p_true)]
    errs = [e for e in errs if math.isfinite(e)]
    out["period_err"] = float(np.median(errs)) if errs else math.inf  # inf: the fit does not oscillate
    amp = lambda x: np.percentile(x, 98, axis=0) - np.percentile(x, 2, axis=0)
    late = ft >= ft[-1] - 2.0
    out["amp_ratio"] = float(np.median(amp(fit) / amp(true)))
    out["late_amp_ratio"] = float(np.median(amp(fit[late]) / amp(true[late])))
    T = np.nanmedian(p_true) if np.isfinite(p_true).any() else math.nan
    shift = int(0.5 * T / dt) if math.isfinite(T) else per_hour
    out["nrmse_aligned"] = float(np.sqrt(np.mean([shifted_rmse(fit[:, c], true[:, c], rng[c], shift) ** 2
                                                   for c in range(fit.shape[1])])))
    return out


# ---------------------------------------------------------------- data and statistics

def load():
    """rows[(kind, config, seed)][method] for cells every method finished, and their metrics."""
    raw = defaultdict(dict)
    for p in current(IN):
        r = json.loads(p.read_text())
        if r["method"] in METHODS and r["panel"] in MAIN_PANELS and r["seed"] < SEEDS:
            raw[(r["kind"], r["panel"], r["seed"])][r["method"]] = r
    rows = {k: v for k, v in raw.items() if all(m in v for m in METHODS)}
    return rows, {k: {m: metrics(v[m]) for m in METHODS} for k, v in rows.items()}


def cells(mets, kind, config=None):
    return sorted(k for k in mets if k[0] == kind and config in (None, k[1]))


def med(vals):
    vals = [v for v in vals if not math.isnan(v)]
    return st.median(vals) if vals else math.nan


def paired(mets, kind, key="nrmse_hidden", n_boot=4000):
    """Differentiable ODE against each other method on matched cells."""
    out = {}
    for m in METHODS[1:]:
        pairs = [(mets[k]["gradient"][key], mets[k][m][key]) for k in cells(mets, kind)]
        pairs = [(a, b) for a, b in pairs if math.isfinite(a) and math.isfinite(b)]
        ratios = [b / a for a, b in pairs]
        rnd = random.Random(0)
        boots = sorted(st.median(rnd.choices(ratios, k=len(ratios))) for _ in range(n_boot))
        out[m] = {"n": len(pairs), "wins": sum(a < b for a, b in pairs),
                  "p": float(wilcoxon(*zip(*pairs)).pvalue), "median_ratio": st.median(ratios),
                  "ci": [boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)]]}
    return out


def load_controls():
    refined = defaultdict(dict)
    for p in current(CTRL / "refine"):
        r = json.loads(p.read_text())
        if r["seed"] < SEEDS and not r.get("failed"):
            refined[(r["kind"], r["panel"], r["seed"])][r["method"]] = r
    abc = {}
    for d in [CTRL / "abc", *sorted((CTRL / "abc_budget").glob("p*_sims*"))]:
        for p in d.glob("*.json"):
            r = json.loads(p.read_text())
            abc[(r.get("particles", 500), r.get("max_sims", 300_000), r["kind"], r["panel"], r["seed"])] = r
    return refined, abc


def controls_summary(refined, kind):
    out = {}
    for m in METHODS:
        rs = [v[m] for k, v in refined.items() if k[0] == kind and m in v]
        out[m] = {key: med([r[key] / r["loss_truth"] for r in rs]) for key in ("loss_saved", "loss_start_only", "loss_local")}
        out[m] |= {"n": len(rs), "local_reaches_optimum": sum(r["loss_local"] <= 1.05 * r["loss_truth"] for r in rs),
                   "local_rel_err": {p: med([r["rel_err_local"][p] for r in rs]) for p in ("alpha", "n", "beta")}}
    return out


def best_of_k(mse_ratio: list, k: int, draws: int = 2000) -> float:
    """Median over random subsets of the best of ``k`` restarts."""
    rnd = random.Random(k)
    return st.median(min(rnd.sample(mse_ratio, k)) for _ in range(draws))


# ---------------------------------------------------------------- figures

def style(ax):
    ax.grid(color="#d9d9d9", lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def boxes(ax, data, positions, colours, width=0.18):
    """A box per list of values, the values as dots."""
    for vals, x, c in zip(data, positions, colours):
        vals = [v for v in vals if math.isfinite(v)]
        if not vals:
            continue
        bp = ax.boxplot([vals], positions=[x], widths=width, patch_artist=True, showfliers=False,
                        medianprops=dict(color="white", lw=1.6), whiskerprops=dict(color=c), capprops=dict(color=c))
        bp["boxes"][0].set(facecolor=c, edgecolor=c)
        jitter = np.random.default_rng(len(vals)).uniform(-width / 4, width / 4, len(vals))
        ax.scatter(x + jitter, vals, s=7, color="white", edgecolors="black", linewidths=0.4, zorder=3)


def method_legend(fig, extra=(), **kw):
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOUR[m]) for m in METHODS] + [h for h, _ in extra]
    fig.legend(handles, [LABEL[m] for m in METHODS] + [l for _, l in extra], loc="upper center",
               ncol=len(handles), frameon=False, **kw)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200)
    plt.close(fig)


def fig_overview(mets):
    """NRMSE to the frames and to the hidden trajectory, per configuration."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.6), sharey="row")
    pos = lambda g: [g + (i - 1.5) * 0.2 for i in range(len(METHODS))]
    for col, kind in enumerate(KINDS):
        for row, key in enumerate(["nrmse_frames", "nrmse_hidden"]):
            ax = axes[row, col]
            for g, config in enumerate(MAIN_PANELS):
                ks = cells(mets, kind, config)
                boxes(ax, [[mets[k][m][key] for k in ks] for m in METHODS], pos(g), [COLOUR[m] for m in METHODS])
                if key == "nrmse_frames" and kind == "ode":  # the camera noise
                    ax.hlines(med([mets[k]["gradient"]["nrmse_truth_frames"] for k in ks]), g - 0.45, g + 0.45,
                              color="black", lw=0.9, ls="--")
            ax.set_yscale("log")
            ax.set_xticks(range(len(MAIN_PANELS)))
            ax.set_xticklabels(MAIN_PANELS if row == 1 else [""] * len(MAIN_PANELS))
            style(ax)
            if row == 0:
                ax.set_title(f"{KINDS[kind]} (n = {len(cells(mets, kind))})", loc="left", fontsize=12)
    axes[0, 0].set_ylabel("NRMSE to the frames")
    axes[1, 0].set_ylabel("NRMSE to the hidden trajectory")
    method_legend(fig, [(plt.Line2D([], [], color="black", ls="--", lw=0.9), "camera noise (true parameters)")],
                  bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "fit_overview")


def fig_gallery(rows, mets, kind, seed):
    """One cell per configuration (the seed is fixed in advance), one column per method."""
    fig, axes = plt.subplots(len(MAIN_PANELS), len(METHODS), figsize=(14, 7.2), sharex=True)
    for i, config in enumerate(MAIN_PANELS):
        for j, m in enumerate(METHODS):
            ax, r = axes[i, j], rows[(kind, config, seed)][m]
            ft, t = np.array(r["fine_t"]) * HOURS_PER_UNIT, np.array(r["t"]) * HOURS_PER_UNIT
            y, fit, true = np.array(r["y"]), np.array(r["fit_fine"]), np.array(r["truth_fine"])
            for c, ch in enumerate(r["channels"]):
                ax.plot(ft, true[:, c], color="black", lw=0.9, alpha=0.35)
                ax.scatter(t, y[:, c], s=3, color=CH_COLOUR[ch], alpha=0.6, linewidths=0)
                if np.isfinite(fit[:, c]).all():
                    ax.plot(ft, fit[:, c], color=CH_COLOUR[ch], lw=1.4, label=CH_LABEL[ch])
            ax.text(0.99, 0.97, f"NRMSE {mets[(kind, config, seed)][m]['nrmse_hidden']:.3f}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=8, bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1))
            style(ax)
            ax.tick_params(labelsize=8)
            if i == 0:
                ax.set_title(LABEL[m], fontsize=12, color=COLOUR[m], fontweight="bold")
            if j == 0:
                ax.set_ylabel(f"{config}\n[K$_M$ units]", fontsize=9)
            if i == len(MAIN_PANELS) - 1:
                ax.set_xlabel("time [h]")
            if j == len(METHODS) - 1:
                ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    save(fig, f"fit_gallery_{kind}_seed{seed}")


def fig_over_time(mets):
    """Median (band: interquartile range) of the hourly NRMSE to the hidden trajectory."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    hours = np.arange(10) + 0.5
    for ax, kind in zip(axes, KINDS):
        for m in METHODS:
            H = np.array([mets[k][m]["hourly"] for k in cells(mets, kind)])
            q25, q50, q75 = np.percentile(H[np.isfinite(H).all(1)], [25, 50, 75], axis=0)
            ax.plot(hours, q50, color=COLOUR[m], lw=2, marker="o", ms=3, label=LABEL[m])
            ax.fill_between(hours, q25, q75, color=COLOUR[m], alpha=0.15, lw=0)
        ax.set_yscale("log")
        ax.set_xlabel("time [h]")
        ax.set_title(KINDS[kind], loc="left")
        style(ax)
    axes[0].set_ylabel("NRMSE to hidden trajectory\n(1-h windows)")
    axes[1].legend(frameon=False, loc="lower right", fontsize=9)
    fig.tight_layout()
    save(fig, "fit_over_time")


def fig_timing_shape(mets):
    """Period error, amplitude ratio, and the error left after the best time shift."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    specs = [("period_err", "relative period error", None), ("amp_ratio", "amplitude ratio (fit / truth)", 1.0),
             ("nrmse_aligned", "NRMSE after best time shift", None)]
    for ax, (key, label, ref) in zip(axes, specs):
        for g, kind in enumerate(KINDS):
            data = [[max(mets[k][m][key], 1e-4) if math.isfinite(mets[k][m][key]) else math.nan
                     for k in cells(mets, kind)] for m in METHODS]  # below the grid resolution: drawn at 1e-4
            boxes(ax, data, [g + (i - 1.5) * 0.2 for i in range(len(METHODS))], [COLOUR[m] for m in METHODS])
        if ref:
            ax.axhline(ref, color="black", lw=0.8, ls="--")
        ax.set_yscale("log")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(list(KINDS.values()))
        ax.set_title(label, loc="left", fontsize=11)
        style(ax)
    method_legend(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "fit_timing_shape")


def fig_controls(refined, abc):
    """Fit of each estimate as reported, after a start-only re-fit and after local refinement."""
    stages = [("loss_saved", "as reported", 1.0), ("loss_start_only", "start re-fitted", 0.65),
              ("loss_local", "locally refined", 0.35)]
    gold = "#c9a227"
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)
    for ax, kind in zip(axes, KINDS):
        for i, m in enumerate(METHODS):
            rs = [v[m] for k, v in refined.items() if k[0] == kind and m in v]
            for s, (key, _, shade) in enumerate(stages):
                colour = tuple(1 - shade * (1 - x) for x in matplotlib.colors.to_rgb(COLOUR[m]))
                boxes(ax, [[r[key] / r["loss_truth"] for r in rs]], [i + (s - 1.5) * 0.21], [colour])
        best = [r["loss_best_particle"] / r["loss_truth"] for k, r in abc.items() if k[:3] == (500, 300_000, kind)]
        boxes(ax, [best], [METHODS.index("abc") + 1.5 * 0.21], [gold])
        ax.axhline(1, color="black", lw=0.8, ls="--")
        ax.set_yscale("log")
        ax.set_xticks(range(len(METHODS)))
        ax.set_xticklabels([LABEL[m] for m in METHODS])
        ax.set_title(KINDS[kind], loc="left")
        style(ax)
    axes[0].set_ylabel("MSE / MSE at true parameters")
    handles = [plt.Rectangle((0, 0), 1, 1, color=(1 - 0.7 * s,) * 3) for _, _, s in stages]
    fig.legend(handles + [plt.Rectangle((0, 0), 1, 1, color=gold)], [l for _, l, _ in stages] + ["ABC best particle"],
               loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "fit_controls")


def fig_budget(abc) -> dict:
    """Fit against trajectory solves: best of k restarts of backprop (k x epochs forward + backward solves,
    from the restart spread) and ABC-SMC per particle count and budget. Median over cells, bars: range."""
    ks, out = [1, 2, 4, 8, 16, 32, 64], {}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)
    for ax, kind in zip(axes, KINDS):
        spread = [json.loads(p.read_text()) for p in (RESULTS / "restart_spread").glob(f"*__{kind}__seed*.json")]
        epochs = spread[0].get("epochs", 1500)
        ks = [k for k in ks if k <= len(spread[0]["mse_ratio"])]
        per_k = [[best_of_k(r["mse_ratio"], k) for r in spread] for k in ks]
        meds = [st.median(v) for v in per_k]
        ax.errorbar([k * epochs for k in ks], meds, yerr=[np.subtract(meds, [min(v) for v in per_k]),
                                                         np.subtract([max(v) for v in per_k], meds)],
                    color=COLOUR["gradient"], marker="o", ms=4, lw=1.8, capsize=2)
        for k, y in zip(ks, meds):
            ax.annotate(f"k={k}", (k * epochs, y), textcoords="offset points", xytext=(0, -12), ha="center",
                        fontsize=7, color=COLOUR["gradient"])
        out[kind] = {"gradient_best_of_k": dict(zip(ks, meds))}
        for P, S in sorted({k[:2] for k in abc}):
            rs = [r for k, r in abc.items() if k[:3] == (P, S, kind)]
            for key, marker, fill, dx in [("loss_median", "s", "white", 0.95), ("loss_best_particle", "D", None, 1.05)]:
                v = [r[key] / r["loss_truth"] for r in rs]
                m = st.median(v)
                ax.errorbar([S * (0.82 if P == 64 else 1.22) * dx], [m], yerr=[[m - min(v)], [max(v) - m]],
                            color=COLOUR["abc"], marker=marker, ms=6, mfc=fill or COLOUR["abc"], capsize=2, ls="none",
                            alpha=1.0 if P == 64 else 0.45)
                out[kind][f"abc_p{P}_sims{S}_{key}"] = m
            out[kind][f"abc_p{P}_sims{S}_n"] = len(rs)
        ax.axhline(1, color="black", lw=0.8, ls="--")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("trajectory solves per cell")
        ax.set_title(KINDS[kind], loc="left")
        style(ax)
    axes[0].set_ylabel("MSE / MSE at true parameters")
    handles = [plt.Line2D([], [], color=COLOUR["gradient"], marker="o", ms=4),
               plt.Line2D([], [], color=COLOUR["abc"], marker="s", mfc="white", ls="none"),
               plt.Line2D([], [], color=COLOUR["abc"], marker="D", ls="none"),
               plt.Rectangle((0, 0), 1, 1, color=COLOUR["abc"]), plt.Rectangle((0, 0), 1, 1, color=COLOUR["abc"], alpha=0.45)]
    fig.legend(handles, ["diff. ODE, best of k restarts", "ABC-SMC, weighted median", "ABC-SMC, best particle",
                         "ABC 64 particles", "ABC 500 particles"], loc="upper center", ncol=5, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "fit_budget")
    return out


# ---------------------------------------------------------------- report

def main():
    rows, mets = load()
    summary = {"n_cells": {k: len(cells(mets, k)) for k in KINDS}, "per_method": {}, "paired": {}}
    for kind in KINDS:
        ks = cells(mets, kind)
        summary["per_method"][kind] = {m: {key: med([mets[k][m][key] for k in ks]) for key in
                                           ["nrmse_frames", "nrmse_truth_frames", "nrmse_hidden", "period_err",
                                            "amp_ratio", "nrmse_aligned"]}
                                       | {"failed": sum(not mets[k][m]["ok"] for k in ks),
                                          "damped": sum(mets[k][m]["late_amp_ratio"] < 0.25 for k in ks),
                                          "hourly_first_last": [med([mets[k][m]["hourly"][h] for k in ks]) for h in (0, -1)]}
                                       for m in METHODS}
        summary["paired"][kind] = paired(mets, kind)
    refined, abc = load_controls()
    if refined:
        summary["controls"] = {kind: controls_summary(refined, kind) for kind in KINDS}
        fig_controls(refined, abc)
    if abc:
        fits = {"median_fit": lambda r: r["loss_median"], "best_particle_fit": lambda r: r["loss_best_particle"],
                "particle_median_fit": lambda r: r["loss_particle_quantiles"]["0.5"]}
        summary["abc_particles"] = {kind: {key: med([f(r) / r["loss_truth"] for k, r in abc.items()
                                                     if k[:3] == (500, 300_000, kind)]) for key, f in fits.items()}
                                    for kind in KINDS}
        summary["budget"] = fig_budget(abc)
    fig_overview(mets)
    fig_over_time(mets)
    fig_timing_shape(mets)
    for kind in KINDS:
        fig_gallery(rows, mets, kind, seed=0)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps(summary, indent=1, default=lambda v: round(v, 4) if isinstance(v, float) else v))


if __name__ == "__main__":
    main()
