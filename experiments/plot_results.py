"""Figures from the benchmark fits.

    python experiments/plot_results.py      -> run_results/figures/

results_main          parameter errors on ODE and Gillespie cells, and the fit ratio (poster and paper)
errors_<kind>         errors per configuration, one box per method
timeseries/<method>/  per configuration and kind: frames, fit and hidden truth for every seed
"""

import json
import math
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import style  # noqa: F401  (applies the figure style)
from matplotlib.ticker import FuncFormatter

from common import FIGURES, MAIN_PANELS, PANELS, RESULTS, current

PERCENT = FuncFormatter(lambda v, _: f"{v:g}%")

IN, OUT = RESULTS / "panel_benchmark", FIGURES
MINUTES_PER_UNIT = 2.0 / 0.6931471805599453  # model time unit = mRNA lifetime

METHOD_COLOUR = style.METHOD_COLOUR
# The results figure's methods, in slot order. DGA has no fits (its gradients explode, see
# dga_gradient_growth.py): its slots carry a cross instead of a box.
METHOD_ORDER = ["gradient", "abc", "magi", "pinn", "dga"]
NOT_RUN = {"dga"}
METHOD_LABEL = {"gradient": "diff. ODE (ours)", "laplace": "backprop + Laplace", "abc": "ABC-SMC",
                "magi": "MAGI", "pinn": "PINN", "dga": "DGA"}
CHANNEL_COLOUR = style.CHANNEL_COLOUR
PARAMS = [("alpha", "α"), ("n", "n"), ("beta", "β"), ("alpha_0", "α₀")]
PANEL_ORDER = list(PANELS)


def tidy(ax):
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
                ax.plot(ft, [row[c] for row in r["truth_fine"]], color=style.PALETTE["grey"], lw=1.2, label="truth (hidden)")
                ax.plot(ft, [row[c] for row in r["fit_fine"]], color=CHANNEL_COLOUR[ch], lw=1.8, label="fit")
                ax.scatter(t, [row[c] for row in r["y"]], s=7, color=CHANNEL_COLOUR[ch], alpha=0.55,
                           linewidths=0, label="data (5-min frames)")
                tidy(ax)
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
        fig.savefig(path)
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
        ax.axhline(10, color=style.PALETTE["grey"], lw=0.8, ls="--")
        tidy(ax)
    axes[-1].set_xticks(range(len(panels)))
    axes[-1].set_xticklabels(panels, rotation=25, ha="right", fontsize=9)
    axes[0].legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    axes[0].set_title(f"Parameter error across seeds - {'ODE' if kind == 'ode' else 'Gillespie'} cell "
                      "(dashed: 10%)", loc="left")
    fig.tight_layout()
    fig.savefig(OUT / f"errors_{kind}.png")
    plt.close(fig)


def save(fig, name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)




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
                ax.axvspan(x - width * 0.5, x + width * 0.5, color=style.PALETTE["neutral"], zorder=0)
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
    rows = [json.loads(p.read_text()) for p in current(IN)]
    OUT.mkdir(parents=True, exist_ok=True)
    timeseries(rows)
    for kind in ("ode", "ssa"):
        if any(r["kind"] == kind for r in rows):
            errors_by_panel(rows, kind)
    results_main(rows)
    print(f"{len(rows)} fits -> {OUT}")


if __name__ == "__main__":
    main()
