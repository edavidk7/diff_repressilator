"""Figures from run_results/panel_benchmark.

    python experiments/plot_results.py         -> run_results/figures/

* ``timeseries/<method>/<panel>__<kind>.png`` - one row per seed, one column
  per observed channel: data frames (dots), the fit (line) and the hidden
  truth (grey), with that seed's errors in the row title.
* ``errors_<kind>.png`` - percentage error of alpha, n, beta, alpha_0 across
  seeds: one box per method for every panel.
* ``errors_summary_{ode,gillespie}.{png,pdf,svg}`` - the same pooled over seeds and the
  panels every method ran on, one box per method (poster style, 200 dpi).
* ``errors_summary_both.{png,pdf,svg}`` - both of those side by side in one figure.
* ``results_main.{png,pdf,svg}`` - poster results: alpha/n/beta error per method on ODE and
  Gillespie cells (shared axis) and the fit to the data, main panels, matched cells only.
"""

import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

PERCENT = FuncFormatter(lambda v, _: f"{v:g}%")

ROOT = Path(__file__).resolve().parent.parent / "run_results"
IN, OUT = ROOT / "panel_benchmark", ROOT / "figures"
MINUTES_PER_UNIT = 2.0 / 0.6931471805599453  # model time unit = mRNA lifetime

METHOD_COLOUR = {"gradient": "#2a78d6", "laplace": "#1baf7a", "abc": "#eb6834", "magi": "#e87ba4",
                 "pinn": "#4a3aa7", "dga": "#6b6b6b"}
# The results figure's methods, in slot order. DGA has no fits (its gradients explode, see
# dga_gradient_growth.py): its slots carry a cross instead of a box.
METHOD_ORDER = ["gradient", "abc", "magi", "pinn", "dga"]
NOT_RUN = {"dga"}
METHOD_LABEL = {"gradient": "diff. ODE (ours)", "laplace": "backprop + Laplace", "abc": "ABC-SMC",
                "magi": "MAGI", "pinn": "PINN", "dga": "DGA"}
CHANNEL_COLOUR = {"lac": "#1baf7a", "tet": "#008300", "cI": "#e34948",
                  "LacI-fusion": "#4a3aa7", "TetR-fusion": "#eda100", "CI-fusion": "#e87ba4"}
PARAMS = [("alpha", "α"), ("n", "n"), ("beta", "β"), ("alpha_0", "α₀")]
PANEL_ORDER = ["GFP only", "2 reporters", "3 reporters", "TetR fusion", "TetR fusion + GFP",
               "TetR fusion + cI reporter", "TetR fusion + 3 reporters", "3 fusions"]


def style(ax):
    ax.grid(color="#e6e6e1", lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def timeseries(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["method"], r["panel"], r["kind"])].append(r)
    for (method, name, kind), rs in groups.items():
        rs.sort(key=lambda r: r["seed"])
        n_ch = len(rs[0]["channels"])
        fig, axes = plt.subplots(len(rs), n_ch, figsize=(4.2 * n_ch + 0.5, 2.6 * len(rs) + 0.6),
                                 squeeze=False, sharex=True)
        for i, r in enumerate(rs):
            t = [x * MINUTES_PER_UNIT for x in r["t"]]
            ft = [x * MINUTES_PER_UNIT for x in r["fine_t"]]
            for c, ch in enumerate(r["channels"]):
                ax = axes[i, c]
                ax.plot(ft, [row[c] for row in r["truth_fine"]], color="#9a9a93", lw=1.2, label="truth (hidden)")
                ax.plot(ft, [row[c] for row in r["fit_fine"]], color=CHANNEL_COLOUR[ch], lw=1.8, label="fit")
                ax.scatter(t, [row[c] for row in r["y"]], s=7, color=CHANNEL_COLOUR[ch], alpha=0.55,
                           linewidths=0, label="data (5-min frames)")
                style(ax)
                if i == 0:
                    ax.set_title(ch, loc="left", fontsize=10)
                if c == 0:
                    err = ", ".join(f"{sym} {r['rel_err'][k]:.0%}" for k, sym in PARAMS[:3])
                    ax.set_ylabel(f"seed {r['seed']}\n{err}\n[K_M units]", fontsize=8)
                if i == len(rs) - 1:
                    ax.set_xlabel("time [min]")
        axes[0, -1].legend(frameon=False, fontsize=7, loc="upper right")
        fig.suptitle(f"{METHOD_LABEL[method]} - {name} - {'ODE' if kind == 'ode' else 'Gillespie'} cell",
                     x=0.01, ha="left", fontsize=11)
        fig.tight_layout()
        path = OUT / "timeseries" / method / f"{''.join(ch if ch.isalnum() else '_' for ch in name)}__{kind}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=110)
        plt.close(fig)


def errors_by_panel(rows, kind):
    methods = [m for m in METHOD_COLOUR if any(r["method"] == m for r in rows)]
    panels = [p for p in PANEL_ORDER if any(r["panel"] == p for r in rows)]
    fig, axes = plt.subplots(len(PARAMS), 1, figsize=(max(9, 1.3 * len(panels) + 3), 11), sharex=True)
    width = 0.8 / len(methods)
    for ax, (key, sym) in zip(axes, PARAMS):
        for m_i, m in enumerate(methods):
            data, pos = [], []
            for p_i, p in enumerate(panels):
                vals = [100 * r["rel_err"][key] for r in rows
                        if r["method"] == m and r["panel"] == p and r["kind"] == kind]
                if vals:
                    data.append(vals)
                    pos.append(p_i - 0.4 + (m_i + 0.5) * width)
            if data:
                bp = ax.boxplot(data, positions=pos, widths=width * 0.8, patch_artist=True,
                                medianprops=dict(color="white", lw=1.5), showfliers=True)
                for box in bp["boxes"]:
                    box.set(facecolor=METHOD_COLOUR[m], edgecolor=METHOD_COLOUR[m])
                for part in ("whiskers", "caps"):
                    for line in bp[part]:
                        line.set(color=METHOD_COLOUR[m])
                for x, vals in zip(pos, data):  # every seed as a dot: boxes of 3 hide the values
                    ax.scatter([x] * len(vals), vals, s=16, color="white", edgecolors=METHOD_COLOUR[m],
                               linewidths=1.2, zorder=3)
                ax.plot([], [], color=METHOD_COLOUR[m], lw=8, label=METHOD_LABEL[m])
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(PERCENT)
        ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
        ax.set_ylabel(f"{sym}: error")
        ax.axhline(10, color="#9a9a93", lw=0.8, ls="--")
        style(ax)
    axes[-1].set_xticks(range(len(panels)))
    axes[-1].set_xticklabels(panels, rotation=25, ha="right", fontsize=9)
    axes[0].legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    axes[0].set_title(f"Parameter error across seeds - {'ODE' if kind == 'ode' else 'Gillespie'} cell "
                      "(dashed: 10%)", loc="left")
    fig.tight_layout()
    fig.savefig(OUT / f"errors_{kind}.png", dpi=120)
    plt.close(fig)


def summary_rows(rows, kind):
    """Methods run on ``kind`` cells (no Laplace: same estimate as backprop) and their rows,
    pooled only over the panels every one of them ran on, so each box sees the same cells."""
    methods = [m for m in METHOD_COLOUR if m != "laplace" and any(r["method"] == m and r["kind"] == kind for r in rows)]
    panels = set.intersection(*[{r["panel"] for r in rows if r["method"] == m and r["kind"] == kind} for m in methods])
    return methods, [r for r in rows if r["kind"] == kind and r["panel"] in panels and r["method"] in methods]


def error_row(axes, rows, methods, labels=True, titles=True, params=PARAMS):
    """One box per method for each parameter, one axis per parameter (poster style)."""
    for ax, (key, sym) in zip(axes, params):
        data = [[100 * r["rel_err"][key] for r in rows if r["method"] == m] for m in methods]
        bp = ax.boxplot(data, patch_artist=True, widths=0.5, medianprops=dict(color="white", lw=1.5),
                        whiskerprops=dict(color="black"), capprops=dict(color="black"),
                        flierprops=dict(markeredgecolor="black"))
        for box, m in zip(bp["boxes"], methods):
            box.set(facecolor=METHOD_COLOUR[m], edgecolor=METHOD_COLOUR[m])
        for x, vals in enumerate(data, start=1):
            ax.scatter([x] * len(vals), vals, s=14, color="white", edgecolors="black", linewidths=0.6, zorder=3)
        ax.set_xticks(range(1, len(methods) + 1))
        if labels:
            ax.set_xticklabels([METHOD_LABEL[m] for m in methods], rotation=45, ha="right",
                               rotation_mode="anchor", fontsize=12)
        else:
            ax.set_xticklabels([])
        ax.tick_params(axis="y", labelsize=12)
        ax.set_xlim(0.5, len(methods) + 0.5)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(PERCENT)
        ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
        if titles:
            ax.set_title(sym, loc="left", fontsize=15)
        ax.axhline(10, color="black", lw=0.8, ls="--")
        ax.grid(color="black", lw=0.4, alpha=0.35)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)


def save(fig, name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200, transparent=True)
    plt.close(fig)


def errors_summary(rows, kind):
    """Poster figure for one kind of cell: error pooled over seeds and the common panels."""
    methods, rows = summary_rows(rows, kind)
    fig, axes = plt.subplots(1, len(PARAMS), figsize=(8.0, 4.8))
    error_row(axes, rows, methods)
    cell = "ODE" if kind == "ode" else "Gillespie"
    how = "ODE" if kind == "ode" else "Gillespie's algorithm"
    fig.suptitle(f"Percentual parameter estimation errors,\ncells simulated through {how}",
                 x=0.01, ha="left", fontsize=15)
    fig.tight_layout(w_pad=0.3)
    save(fig, f"errors_summary_{cell.lower()}")


def errors_summary_both(rows):
    """Both kinds of cell in one poster figure, side by side: ODE cells left, Gillespie
    cells right, each a 2 x 2 grid of parameters.

    Both use the same method slots so they line up; a method not (yet) run on a kind of
    cell leaves its slot empty.
    """
    per_kind = {kind: summary_rows(rows, kind) for kind in ("ode", "ssa")}
    methods = [m for m in METHOD_COLOUR if any(m in per_kind[k][0] for k in per_kind)]
    fig = plt.figure(figsize=(11.0, 7.4), layout="constrained")
    fig.suptitle("Percentual parameter estimation errors", x=0.01, ha="left", fontsize=15)
    for sub, (kind, label) in zip(fig.subfigures(1, 2, wspace=0.04),
                                  [("ode", "ODE cells"), ("ssa", "Gillespie cells")]):
        sub.suptitle(label, fontsize=14, fontweight="bold")
        axes = sub.subplots(2, 2)
        error_row(axes[0], per_kind[kind][1], methods, labels=False, params=PARAMS[:2])
        error_row(axes[1], per_kind[kind][1], methods, params=PARAMS[2:])
    save(fig, "errors_summary_both")


MAIN_PANELS = ["GFP only", "3 reporters", "TetR fusion + GFP"]  # the panels every method runs on


def matched_cells(rows, kind):
    """Methods run on ``kind`` cells of the main panels, and their rows restricted to the
    cells (panel, seed) that every one of those methods has finished."""
    rows = [r for r in rows if r["kind"] == kind and r["panel"] in MAIN_PANELS and r["method"] != "laplace"]
    methods = [m for m in METHOD_COLOUR if any(r["method"] == m for r in rows)]
    cells = set.intersection(*[{(r["panel"], r["seed"]) for r in rows if r["method"] == m} for m in methods])
    return methods, [r for r in rows if (r["panel"], r["seed"]) in cells], cells


def grouped(ax, groups, methods, values):
    """Boxes grouped at x = 0, 1, ...: one per method in each group, seeds as dots."""
    width = 0.8 / len(METHOD_ORDER)
    for g in range(len(groups)):
        for m in methods:
            x = g + (METHOD_ORDER.index(m) - (len(METHOD_ORDER) - 1) / 2) * width
            if m in NOT_RUN:  # no result: a gray band over the whole slot with a big cross on it
                ax.axvspan(x - width * 0.5, x + width * 0.5, color="#e3e3e3", zorder=0)
                ax.plot([x], [0.5], transform=ax.get_xaxis_transform(), marker="x", ms=13, mew=3,
                        color=METHOD_COLOUR[m])
                continue
            vals = [v for v in values(g, m) if math.isfinite(v)]  # failed fits (NaN) are reported, not drawn
            if not vals:
                continue
            bp = ax.boxplot([vals], positions=[x], widths=width * 0.8, patch_artist=True, showfliers=False,
                            medianprops=dict(color="white", lw=1.8), whiskerprops=dict(color="black"),
                            capprops=dict(color="black"))
            bp["boxes"][0].set(facecolor=METHOD_COLOUR[m], edgecolor=METHOD_COLOUR[m])
            ax.scatter([x] * len(vals), vals, s=12, color="white", edgecolors="black", linewidths=0.5, zorder=3)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, fontsize=16)
    ax.set_xlim(-0.5, len(groups) - 0.5)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax.tick_params(axis="y", labelsize=13)
    ax.grid(axis="y", color="black", lw=0.4, alpha=0.35)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def results_main(rows):
    """Poster results figure: parameter error on ODE and on Gillespie cells (shared axis),
    plus how well each estimate reproduces the data. Main panels only, matched cells."""
    per_kind = {kind: matched_cells(rows, kind) for kind in ("ode", "ssa")}
    # Same seeds on both sides: those with every configuration finished for both kinds of cell.
    seeds = {s for _, s in set.union(*[c for _, _, c in per_kind.values()])}
    even = {s for s in seeds if all((p, s) in per_kind[k][2] for k in per_kind for p in MAIN_PANELS)}
    per_kind = {k: (ms, [r for r in rs if r["seed"] in even], {c for c in cells if c[1] in even})
                for k, (ms, rs, cells) in per_kind.items()}
    params = [p for p in PARAMS if p[0] != "alpha_0"]  # alpha_0: unidentifiable for every method
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.6), gridspec_kw={"width_ratios": [3, 3, 2.2]})
    for ax, kind, label in [(axes[0], "ode", "ODE cells"), (axes[1], "ssa", "Gillespie cells")]:
        methods, kind_rows, cells = per_kind[kind]
        grouped(ax, [sym for _, sym in params], methods + sorted(NOT_RUN),
                lambda g, m: [100 * r["rel_err"][params[g][0]] for r in kind_rows if r["method"] == m])
        ax.yaxis.set_major_formatter(PERCENT)
        ax.axhline(10, color="black", lw=0.8, ls="--")
        ax.set_title(f"{label} (n = {len(cells)})", loc="left", fontsize=16)
    axes[0].set_ylabel("relative error", fontsize=15)
    lo = min(axes[0].get_ylim()[0], axes[1].get_ylim()[0])
    hi = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
    for ax in axes[:2]:
        ax.set_ylim(lo, hi)

    kinds = [("ode", "ODE"), ("ssa", "Gillespie")]
    grouped(axes[2], [k for _, k in kinds], METHOD_ORDER,
            lambda g, m: [r["loss_fit"] / r["loss_truth"] for r in per_kind[kinds[g][0]][1] if r["method"] == m])
    axes[2].axhline(1, color="black", lw=0.8, ls="--")
    axes[2].set_title("Fit to the data", loc="left", fontsize=16)
    axes[2].set_ylabel("MSE / MSE at true parameters", fontsize=15)
    axes[2].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))

    for kind in per_kind:
        for m in per_kind[kind][0]:
            failed = [r for r in per_kind[kind][1] if r["method"] == m and not math.isfinite(r["loss_fit"])]
            if failed:
                print(f"results_main: {METHOD_LABEL[m]} failed (NaN) on {len(failed)} {kind} cells, not drawn")
    fig.suptitle(f"Relative parameter estimation errors ({len(MAIN_PANELS)} reporter configurations × "
                 f"{len(even)} random seeds)", x=0.01, ha="left", fontsize=18)
    handles = [plt.Rectangle((0, 0), 1, 1, color=METHOD_COLOUR[m]) for m in METHOD_ORDER]
    fig.legend(handles, [METHOD_LABEL[m] for m in METHOD_ORDER], loc="upper left", bbox_to_anchor=(0.01, 0.94),
               ncol=len(METHOD_ORDER), frameon=False, fontsize=15, handlelength=1.2)
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=1.5)
    save(fig, "results_main")


def main():
    rows = [json.loads(p.read_text()) for p in IN.glob("*/*.json")]
    OUT.mkdir(parents=True, exist_ok=True)
    timeseries(rows)
    for kind in ("ode", "ssa"):
        if any(r["kind"] == kind for r in rows):
            errors_by_panel(rows, kind)
    for kind in ("ode", "ssa"):
        errors_summary(rows, kind)
    errors_summary_both(rows)
    results_main(rows)
    print(f"{len(rows)} fits -> {OUT}")


if __name__ == "__main__":
    main()
