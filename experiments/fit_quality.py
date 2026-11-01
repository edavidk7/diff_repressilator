"""How well does each estimate reproduce the time series? (from saved fits, no new runs)

    python experiments/fit_quality.py           -> run_results/fit_quality/{*.png,*.pdf,summary.json,tables.md}

Every saved benchmark fit (``run_results/panel_benchmark``) holds the frames,
the ODE integrated at the method's estimate (30-s resolution) and the cell's
hidden noise-free fluorescence. On the three configurations every method ran
on, for matched cells (a cell counts only if every method has it), this
computes per cell and method:

* ``nrmse_frames``  - RMSE to the frames, divided by the channel's range: the
                      square root of the benchmark's loss. At the true
                      parameters on ODE cells this is the camera noise.
* ``nrmse_hidden``  - the same against the hidden noise-free trajectory on the
                      30-s grid: how close the estimate is to what the cell did.
* ``hourly``        - ``nrmse_hidden`` in 1-h windows: does the error grow?
* ``period_err``    - relative error of the oscillation period (autocorrelation
                      of the trajectory), fit against hidden truth.
* ``amp_ratio``     - peak-to-trough amplitude of the fit / of the truth.
* ``late_amp_ratio`` - the same over the last 2 h; below 0.25 the fitted
                      oscillation has (nearly) died out ("damped").
* ``nrmse_aligned`` - ``nrmse_hidden`` after the best constant time shift
                      (within half a period): what is left once timing is
                      forgiven, i.e. errors in shape and amplitude.

Statistics: paired comparisons of the differentiable ODE against each method
on ``nrmse_hidden`` (wins and a two-sided Wilcoxon signed-rank test), and a
bootstrap 95% interval of the median ratio. Results of
``experiments/fit_controls.py`` are added when they exist.
"""

import json
import math
import random
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import wilcoxon

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent / "run_results"
IN, CTRL, OUT = ROOT / "panel_benchmark", ROOT / "fit_controls", ROOT / "fit_quality"
HOURS_PER_UNIT = 2.0 / math.log(2) / 60
METHODS = ["gradient", "abc", "magi", "pinn"]
LABEL = {"gradient": "diff. ODE", "abc": "ABC-SMC", "magi": "MAGI", "pinn": "inverse PINN"}
COLOUR = {"gradient": "#2a78d6", "abc": "#eb6834", "magi": "#e87ba4", "pinn": "#4a3aa7"}
PANELS = ["GFP only", "3 reporters", "TetR fusion + GFP"]
SEEDS = 13  # as in plot_results.results_main: the seeds complete for every method on both kinds of cell
KINDS = {"ode": "ODE cells", "ssa": "Gillespie cells"}
CH_LABEL = {"lac": "mJuniper (P$_{Llac}$)", "tet": "GFP (P$_{Ltet}$)", "cI": "mScarlet-I3 (λP$_R$)",
            "TetR-fusion": "TetR fusion"}
CH_COLOUR = {"lac": "#1baf7a", "tet": "#3f7f2a", "cI": "#b8323f", "TetR-fusion": "#c98a00"}


# ---------------------------------------------------------------------------- metrics


def period(x: np.ndarray, dt: float) -> float:
    """Period from the first autocorrelation peak after its first zero crossing."""
    x = x - x.mean()
    if not np.isfinite(x).all() or x.std() < 1e-12:
        return math.nan
    ac = np.correlate(x, x, "full")[len(x) - 1:]
    ac = ac / ac[0]
    neg = np.nonzero(ac < 0)[0]
    if not len(neg):
        return math.nan
    start = neg[0]
    seg = ac[start: len(ac) * 3 // 4]
    if not len(seg) or seg.max() <= 0:
        return math.nan
    return (start + int(seg.argmax())) * dt


def shifted_rmse(fit: np.ndarray, true: np.ndarray, rng: np.ndarray, max_shift: int) -> float:
    """Smallest range-normalised RMSE over constant shifts |s| <= max_shift samples."""
    best = math.inf
    for s in range(-max_shift, max_shift + 1):
        a = fit[max(s, 0): len(fit) + min(s, 0)]
        b = true[max(-s, 0): len(true) + min(-s, 0)]
        best = min(best, float(np.sqrt(np.mean(((a - b) / rng) ** 2))))
    return best


def metrics(r: dict) -> dict:
    y, fit, true = np.array(r["y"]), np.array(r["fit_fine"]), np.array(r["truth_fine"])
    ft = np.array(r["fine_t"]) * HOURS_PER_UNIT
    rng = y.max(0) - y.min(0)
    ok = np.isfinite(fit).all() and math.isfinite(r["loss_fit"])
    out = {"ok": bool(ok), "nrmse_frames": math.sqrt(r["loss_fit"]) if ok else math.nan,
           "nrmse_truth_frames": math.sqrt(r["loss_truth"])}
    if not ok:
        return out | {"nrmse_hidden": math.nan, "hourly": [math.nan] * 10, "period_err": math.nan,
                      "amp_ratio": math.nan, "late_amp_ratio": math.nan, "nrmse_aligned": math.nan}
    res = (fit - true) / rng
    out["nrmse_hidden"] = float(np.sqrt(np.mean(res**2)))
    per_hour = int(round(1 / (ft[1] - ft[0])))
    out["hourly"] = [float(np.sqrt(np.mean(res[h * per_hour:(h + 1) * per_hour] ** 2))) for h in range(10)]
    dt = ft[1] - ft[0]
    p_true = [period(true[:, c], dt) for c in range(true.shape[1])]
    p_fit = [period(fit[:, c], dt) for c in range(fit.shape[1])]
    errs = [abs(f - t) / t for f, t in zip(p_fit, p_true) if math.isfinite(f) and math.isfinite(t)]
    out["period_err"] = float(np.median(errs)) if errs else math.inf  # inf: the fit does not oscillate
    out["period_true_h"], out["period_fit_h"] = p_true[0], p_fit[0]
    amp = lambda x: np.percentile(x, 98, axis=0) - np.percentile(x, 2, axis=0)
    out["amp_ratio"] = float(np.median(amp(fit) / amp(true)))
    late = ft >= ft[-1] - 2.0  # last 2 h: has the fitted oscillation died out?
    out["late_amp_ratio"] = float(np.median(amp(fit[late]) / amp(true[late])))
    T = np.nanmedian(p_true)
    max_shift = int(0.5 * T / dt) if math.isfinite(T) else per_hour
    out["nrmse_aligned"] = float(np.sqrt(np.mean([shifted_rmse(fit[:, c], true[:, c], rng[c], max_shift) ** 2
                                                   for c in range(fit.shape[1])])))
    return out


# ---------------------------------------------------------------------------- data


def load() -> tuple[dict, dict]:
    """rows[(kind, panel, seed)][method] for matched cells, and their metrics."""
    raw = defaultdict(dict)
    for p in IN.glob("*/*.json"):
        r = json.loads(p.read_text())
        if r["method"] in METHODS and r["panel"] in PANELS and r["seed"] < SEEDS:
            raw[(r["kind"], r["panel"], r["seed"])][r["method"]] = r
    rows = {k: v for k, v in raw.items() if all(m in v for m in METHODS)}
    mets = {k: {m: metrics(v[m]) for m in METHODS} for k, v in rows.items()}
    return rows, mets


def cells(mets, kind, panel=None):
    return sorted(k for k in mets if k[0] == kind and (panel is None or k[1] == panel))


def med(vals):
    vals = [v for v in vals if not math.isnan(v)]
    return st.median(vals) if vals else math.nan


def boot_ratio(a, b, n=4000, seed=0):
    """Median of b/a with a percentile bootstrap 95% interval (paired cells)."""
    pairs = [(x, y) for x, y in zip(a, b) if math.isfinite(x) and math.isfinite(y)]
    ratios = [y / x for x, y in pairs]
    rnd = random.Random(seed)
    boots = sorted(st.median(rnd.choices(ratios, k=len(ratios))) for _ in range(n))
    return st.median(ratios), boots[int(0.025 * n)], boots[int(0.975 * n)]


def paired(mets, kind, key):
    ks = cells(mets, kind)
    out = {}
    for m in METHODS[1:]:
        a = [mets[k]["gradient"][key] for k in ks]
        b = [mets[k][m][key] for k in ks]
        pairs = [(x, y) for x, y in zip(a, b) if math.isfinite(x) and math.isfinite(y)]
        wins = sum(x < y for x, y in pairs)
        p = float(wilcoxon([x for x, _ in pairs], [y for _, y in pairs]).pvalue) if len(pairs) > 5 else math.nan
        ratio = boot_ratio(a, b)
        out[m] = {"n": len(pairs), "failed": len(ks) - len(pairs), "wins": wins, "p": p,
                  "median_ratio": ratio[0], "ci": ratio[1:]}
    return out


# ---------------------------------------------------------------------------- figures


def style(ax):
    ax.grid(color="#d9d9d9", lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def boxes(ax, data, positions, colours, width=0.18):
    for vals, x, c in zip(data, positions, colours):
        vals = [v for v in vals if math.isfinite(v)]
        if not vals:
            continue
        bp = ax.boxplot([vals], positions=[x], widths=width, patch_artist=True, showfliers=False,
                        medianprops=dict(color="white", lw=1.6), whiskerprops=dict(color=c),
                        capprops=dict(color=c))
        bp["boxes"][0].set(facecolor=c, edgecolor=c)
        jitter = np.random.default_rng(len(vals)).uniform(-width / 4, width / 4, len(vals))
        ax.scatter(x + jitter, vals, s=7, color="white", edgecolors="black", linewidths=0.4, zorder=3)


def fig_overview(mets):
    """NRMSE to frames and to the hidden trajectory, per configuration, both kinds of cell."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.6), sharey="row")
    for col, kind in enumerate(KINDS):
        for row, key in enumerate(["nrmse_frames", "nrmse_hidden"]):
            ax = axes[row, col]
            for g, panel in enumerate(PANELS):
                ks = cells(mets, kind, panel)
                pos = [g + (i - 1.5) * 0.2 for i in range(len(METHODS))]
                boxes(ax, [[mets[k][m][key] for k in ks] for m in METHODS], pos, [COLOUR[m] for m in METHODS])
                if key == "nrmse_frames" and kind == "ode":
                    floor = med([mets[k]["gradient"]["nrmse_truth_frames"] for k in ks])
                    ax.hlines(floor, g - 0.45, g + 0.45, color="black", lw=0.9, ls="--")
            ax.set_yscale("log")
            ax.set_xticks(range(len(PANELS)))
            ax.set_xticklabels(PANELS if row == 1 else [""] * len(PANELS))
            style(ax)
            if row == 0:
                ax.set_title(f"{KINDS[kind]} (n = {len(cells(mets, kind))})", loc="left", fontsize=12)
        axes[0, 0].set_ylabel("NRMSE to the frames")
        axes[1, 0].set_ylabel("NRMSE to the hidden trajectory")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOUR[m]) for m in METHODS]
    handles.append(plt.Line2D([], [], color="black", ls="--", lw=0.9))
    fig.legend(handles, [LABEL[m] for m in METHODS] + ["camera noise (true parameters)"], loc="upper center",
               ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "fit_overview")


def fig_gallery(rows, mets, kind, seed):
    """One cell per configuration (prespecified seed), one column per method."""
    fig, axes = plt.subplots(len(PANELS), len(METHODS), figsize=(14, 7.2), sharex=True)
    for i, panel in enumerate(PANELS):
        for j, m in enumerate(METHODS):
            ax, r = axes[i, j], rows[(kind, panel, seed)][m]
            ft = np.array(r["fine_t"]) * HOURS_PER_UNIT
            t = np.array(r["t"]) * HOURS_PER_UNIT
            y, fit, true = np.array(r["y"]), np.array(r["fit_fine"]), np.array(r["truth_fine"])
            for c, ch in enumerate(r["channels"]):
                ax.plot(ft, true[:, c], color="black", lw=0.9, alpha=0.35)
                ax.scatter(t, y[:, c], s=3, color=CH_COLOUR[ch], alpha=0.6, linewidths=0)
                if np.isfinite(fit[:, c]).all():
                    ax.plot(ft, fit[:, c], color=CH_COLOUR[ch], lw=1.4, label=CH_LABEL[ch])
            mt = mets[(kind, panel, seed)][m]
            ax.text(0.99, 0.97, f"NRMSE {mt['nrmse_hidden']:.3f}", transform=ax.transAxes, ha="right",
                    va="top", fontsize=8, bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1))
            style(ax)
            ax.tick_params(labelsize=8)
            if i == 0:
                ax.set_title(LABEL[m], fontsize=12, color=COLOUR[m], fontweight="bold")
            if j == 0:
                ax.set_ylabel(f"{panel}\n[K$_M$ units]", fontsize=9)
            if i == len(PANELS) - 1:
                ax.set_xlabel("time [h]")
            if j == len(METHODS) - 1:
                ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    save(fig, f"fit_gallery_{kind}_seed{seed}")


def fig_over_time(mets):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    hours = np.arange(10) + 0.5
    for ax, kind in zip(axes, KINDS):
        ks = cells(mets, kind)
        for m in METHODS:
            H = np.array([mets[k][m]["hourly"] for k in ks])
            H = H[np.isfinite(H).all(1)]
            q25, q50, q75 = np.percentile(H, [25, 50, 75], axis=0)
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
    specs = [("period_err", "relative period error", True, None),
             ("amp_ratio", "amplitude ratio (fit / truth)", True, 1.0),
             ("nrmse_aligned", "NRMSE after best time shift", True, None)]
    for ax, (key, label, log, ref) in zip(axes, specs):
        for g, kind in enumerate(KINDS):
            ks = cells(mets, kind)
            pos = [g + (i - 1.5) * 0.2 for i in range(len(METHODS))]
            data = [[max(mets[k][m][key], 1e-4) if math.isfinite(mets[k][m][key]) else math.nan for k in ks]
                    for m in METHODS]
            boxes(ax, data, pos, [COLOUR[m] for m in METHODS])
        if ref:
            ax.axhline(ref, color="black", lw=0.8, ls="--")
        ax.set_yscale("log" if log else "linear")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(list(KINDS.values()))
        ax.set_title(label, loc="left", fontsize=11)
        style(ax)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOUR[m]) for m in METHODS]
    fig.legend(handles, [LABEL[m] for m in METHODS], loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "fit_timing_shape")


def load_controls():
    ref = defaultdict(dict)
    for p in (CTRL / "refine").glob("*/*.json"):
        r = json.loads(p.read_text())
        if r["seed"] < SEEDS:
            ref[(r["kind"], r["panel"], r["seed"])][r["method"]] = r
    abc = {}
    for p in (CTRL / "abc").glob("*.json"):
        r = json.loads(p.read_text())
        abc[(r["kind"], r["panel"], r["seed"])] = r
    return ref, abc


def fig_controls(ref, abc):
    """Fit (MSE / MSE at truth) of the saved estimate, start-only re-fit, local refinement."""
    stages = [("loss_saved", "as reported"), ("loss_start_only", "start re-fitted"),
              ("loss_local", "locally refined")]
    shades = [1.0, 0.65, 0.35]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)
    for ax, kind in zip(axes, KINDS):
        ks = [k for k in ref if k[0] == kind]
        for i, m in enumerate(METHODS):
            for s, ((key, _), shade) in enumerate(zip(stages, shades)):
                vals = [ref[k][m][key] / ref[k][m]["loss_truth"] for k in ks
                        if m in ref[k] and not ref[k][m].get("failed") and math.isfinite(ref[k][m].get(key, math.nan))]
                c = matplotlib.colors.to_rgb(COLOUR[m])
                c = tuple(1 - shade * (1 - x) for x in c)
                boxes(ax, [vals], [i + (s - 1.5) * 0.21], [c], width=0.18)
        if kind in [k[0] for k in abc]:
            vals = [r["loss_best_particle"] / r["loss_truth"] for k, r in abc.items() if k[0] == kind]
            boxes(ax, [vals], [METHODS.index("abc") + 1.5 * 0.21], ["#c9a227"], width=0.18)
        ax.axhline(1, color="black", lw=0.8, ls="--")
        ax.set_yscale("log")
        ax.set_xticks(range(len(METHODS)))
        ax.set_xticklabels([LABEL[m] for m in METHODS])
        ax.set_title(f"{KINDS[kind]}", loc="left")
        style(ax)
    axes[0].set_ylabel("MSE / MSE at true parameters")
    grey = [tuple(1 - s * (1 - 0.3) for _ in range(3)) for s in shades]
    handles = [plt.Rectangle((0, 0), 1, 1, color=g) for g in grey] + [plt.Rectangle((0, 0), 1, 1, color="#c9a227")]
    fig.legend(handles, [l for _, l in stages] + ["ABC best particle"], loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "fit_controls")


def fig_budget():
    """Fit against compute: best of k restarts of the differentiable ODE (from
    ``restart_spread``, 1500 epochs; k restarts = k x 1500 forward + backward solves) and
    ABC-SMC at each population size and simulation budget (weighted median and best
    particle). Median over cells, bars: range."""
    import random as _r
    from math import comb  # noqa: F401
    budget_root = CTRL / "abc_budget"
    arms = {(500, 300_000): CTRL / "abc"} | {
        tuple(int(x) for x in d.name[1:].split("_sims")): d for d in sorted(budget_root.glob("p*_sims*"))}
    out = {}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)
    for ax, kind in zip(axes, KINDS):
        spread = [json.loads(p.read_text()) for p in (ROOT / "restart_spread").glob(f"*__{kind}__seed*.json")]
        ks = [1, 2, 4, 8, 16, 32, 64]
        meds, lo, hi = [], [], []
        for k in ks:
            per_cell = []
            for r in spread:
                rnd = _r.Random(k)
                per_cell.append(st.median(min(rnd.sample(r["mse_ratio"], k)) for _ in range(2000)))
            meds.append(st.median(per_cell)); lo.append(min(per_cell)); hi.append(max(per_cell))
        cost = [k * 1500 for k in ks]
        ax.errorbar(cost, meds, yerr=[np.subtract(meds, lo), np.subtract(hi, meds)], color=COLOUR["gradient"],
                    marker="o", ms=4, lw=1.8, capsize=2, label="diff. ODE, best of k restarts")
        for x, k, y in zip(cost, ks, meds):
            ax.annotate(f"k={k}", (x, y), textcoords="offset points", xytext=(0, -12), ha="center", fontsize=7,
                        color=COLOUR["gradient"])
        out[kind] = {"gradient_best_of_k": dict(zip(ks, meds))}
        for (P, S), d in sorted(arms.items()):
            rows = [json.loads(p.read_text()) for p in d.glob(f"*__{kind}__seed*.json")]
            if not rows:
                continue
            shift = 0.82 if P == 64 else 1.22  # the two population sizes side by side
            for key, marker, fill, dx in [("loss_median", "s", "white", 0.95), ("loss_best_particle", "D", None, 1.05)]:
                v = [r[key] / r["loss_truth"] for r in rows]
                m = st.median(v)
                ax.errorbar([S * shift * dx], [m], yerr=[[m - min(v)], [max(v) - m]], color=COLOUR["abc"],
                            marker=marker, ms=6, mfc=fill or COLOUR["abc"], capsize=2, ls="none",
                            alpha=1.0 if P == 64 else 0.45)
                out[kind][f"abc_p{P}_sims{S}_{key}"] = m
            out[kind][f"abc_p{P}_sims{S}_n"] = len(rows)
        ax.axhline(1, color="black", lw=0.8, ls="--")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("trajectory solves per cell")
        ax.set_title(KINDS[kind], loc="left")
        style(ax)
    axes[0].set_ylabel("MSE / MSE at true parameters")
    handles = [plt.Line2D([], [], color=COLOUR["gradient"], marker="o", ms=4),
               plt.Line2D([], [], color=COLOUR["abc"], marker="s", mfc="white", ls="none"),
               plt.Line2D([], [], color=COLOUR["abc"], marker="D", ls="none"),
               plt.Rectangle((0, 0), 1, 1, color=COLOUR["abc"]),
               plt.Rectangle((0, 0), 1, 1, color=COLOUR["abc"], alpha=0.45)]
    fig.legend(handles, ["diff. ODE, best of k restarts", "ABC-SMC, weighted median", "ABC-SMC, best particle",
                         "ABC 64 particles", "ABC 500 particles"],
               loc="upper center", ncol=5, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "fit_budget")
    return out


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------- report


def main():
    rows, mets = load()
    summary = {"n_cells": {k: len(cells(mets, k)) for k in KINDS}, "per_method": {}, "paired": {}, "per_panel": {}}
    for kind in KINDS:
        ks = cells(mets, kind)
        summary["per_method"][kind] = {
            m: {key: med([mets[k][m][key] for k in ks]) for key in
                ["nrmse_frames", "nrmse_truth_frames", "nrmse_hidden", "period_err", "amp_ratio", "nrmse_aligned"]}
            | {"failed": sum(not mets[k][m]["ok"] for k in ks),
               "no_oscillation": sum(mets[k][m]["period_err"] == math.inf for k in ks),
               "damped": sum(mets[k][m]["late_amp_ratio"] < 0.25 for k in ks),
               "hourly_first_last": [med([mets[k][m]["hourly"][0] for k in ks]),
                                     med([mets[k][m]["hourly"][-1] for k in ks])]}
            for m in METHODS}
        summary["paired"][kind] = {key: paired(mets, kind, key) for key in ["nrmse_hidden", "nrmse_frames"]}
        summary["per_panel"][kind] = {p: {m: med([mets[k][m]["nrmse_hidden"] for k in cells(mets, kind, p)])
                                          for m in METHODS} | {"n": len(cells(mets, kind, p))} for p in PANELS}
    ref, abc = load_controls()
    if ref:
        summary["controls"] = {}
        for kind in KINDS:
            ks = [k for k in ref if k[0] == kind]
            summary["controls"][kind] = {
                m: {key: med([ref[k][m][key] / ref[k][m]["loss_truth"] for k in ks
                              if m in ref[k] and not ref[k][m].get("failed")])
                    for key in ["loss_saved", "loss_start_only", "loss_local"]}
                | {"n": sum(m in ref[k] for k in ks),
                   "local_reaches_optimum": sum(ref[k][m]["loss_local"] <= 1.05 * ref[k][m]["loss_truth"]
                                                for k in ks if m in ref[k] and not ref[k][m].get("failed")),
                   "local_rel_err": {p: med([ref[k][m]["rel_err_local"][p] for k in ks
                                              if m in ref[k] and not ref[k][m].get("failed")])
                                     for p in ["alpha", "n", "beta"]}}
                for m in METHODS}
        fig_controls(ref, abc)
    if abc:
        summary["abc_particles"] = {
            kind: {"n": sum(k[0] == kind for k in abc),
                   "median_fit": med([r["loss_median"] / r["loss_truth"] for k, r in abc.items() if k[0] == kind]),
                   "best_particle_fit": med([r["loss_best_particle"] / r["loss_truth"] for k, r in abc.items() if k[0] == kind]),
                   "particle_median_fit": med([r["loss_particle_quantiles"]["0.5"] / r["loss_truth"]
                                               for k, r in abc.items() if k[0] == kind]),
                   "reproduces_saved": all(abs(r["loss_median"] - r["loss_median_saved"]) < 1e-9 * max(1, r["loss_median"])
                                           for r in abc.values())}
            for kind in KINDS}
    if (CTRL / "abc").exists():
        summary["budget"] = fig_budget()
    fig_overview(mets)
    fig_over_time(mets)
    fig_timing_shape(mets)
    for kind in KINDS:
        fig_gallery(rows, mets, kind, seed=0)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps(summary, indent=1, default=lambda v: round(v, 4) if isinstance(v, float) else v))


if __name__ == "__main__":
    main()
