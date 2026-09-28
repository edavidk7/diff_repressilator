"""Tables and figures from a benchmark directory.

    python experiments/benchmark_report.py run_results/benchmark_ode_panels [more dirs...]

Writes ``report.md`` plus figures next to the first directory given.  The
tables carry every number the figures show, so the figures are never the
only way to read a result.
"""

import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CANONICAL = ("alpha", "n", "beta")
# Fixed method -> hue, validated categorical order (never re-cycled).
COLOURS = {
    "grad_ode": "#2a78d6", "nuts": "#eb6834", "abc_ode": "#1baf7a", "nested": "#eda100",
    "magi": "#e87ba4", "pinn": "#008300", "grad_cle": "#4a3aa7", "abc_cle": "#e34948",
    "grad_dga": "#6b6b6b", "nested_slice": "#a35f00", "nested_unif": "#a35f00",
}
MARKERS = {"grad_ode": "o", "nuts": "s", "abc_ode": "^", "nested": "D", "magi": "v",
           "pinn": "P", "grad_cle": "X", "abc_cle": "<", "grad_dga": ">",
           "nested_slice": "d", "nested_unif": "d"}
ORDER = list(COLOURS)


def load(dirs: list[Path]) -> list[dict]:
    rows = []
    for d in dirs:
        for case_dir in sorted(p for p in d.iterdir() if p.is_dir()):
            for f in sorted(case_dir.glob("*.json")):
                if f.name == "case.json":
                    continue
                row = json.loads(f.read_text())
                row["method_key"] = f.stem
                row["case"] = case_dir.name
                rows.append(row)
    return rows


def fmt(x, digits=2):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "–"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if abs(x) >= 100 or (abs(x) < 0.01 and x != 0):
        return f"{x:.2e}"
    return f"{x:.{digits}f}"


def case_table(rows: list[dict]) -> str:
    by_case = defaultdict(list)
    for r in rows:
        by_case[r["case"]].append(r)
    out = []
    for case, rs in by_case.items():
        out.append(f"\n### {case}\n")
        out.append("| method | " + " | ".join(f"{p} rel.err" for p in CANONICAL)
                   + " | 90% CI covers (α/n/β) | NRMSE obs | NRMSE hidden | NRMSE forecast | wall [s] | sims |")
        out.append("|---" * (len(CANONICAL) + 7) + "|")
        for r in sorted(rs, key=lambda r: ORDER.index(r["method_key"]) if r["method_key"] in ORDER else 99):
            if "error" in r:
                out.append(f"| {r['method_key']} | failed: `{r['error'][:60]}` |" + " |" * (len(CANONICAL) + 5))
                continue
            p = r["params"]
            cov = "/".join(fmt(p[k].get("covered")) if "covered" in p[k] else "–" for k in CANONICAL)
            out.append(f"| {r['method_key']} | " + " | ".join(fmt(p[k]["rel_err"]) for k in CANONICAL)
                       + f" | {cov} | {fmt(r['nrmse_observed'])} | {fmt(r['nrmse_hidden'])} | "
                       f"{fmt(r['nrmse_forecast'])} | {fmt(r['wall'], 0)} | {r['n_sims']} |")
    return "\n".join(out)


def summary_table(rows: list[dict]) -> str:
    """Per method: median log-error over all cases, coverage rate, median wall."""
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if "error" in r:
            agg[r["method_key"]]["failed"].append(1)
            continue
        for k in CANONICAL:
            agg[r["method_key"]][k].append(r["params"][k]["log_err"])
            if "covered" in r["params"][k]:
                agg[r["method_key"]]["cov"].append(float(r["params"][k]["covered"]))
                agg[r["method_key"]]["width"].append(r["params"][k]["ci_width_log"])
        rhat = r.get("info", {}).get("max_rhat_params")
        if rhat is not None:
            agg[r["method_key"]]["unconverged"].append(float(rhat > 1.1 or rhat != rhat))
        agg[r["method_key"]]["forecast"].append(r["nrmse_forecast"])
        agg[r["method_key"]]["wall"].append(r["wall"])
    out = ["| method | cases | median |log err| α | n | β | CI coverage (target 0.90) | median CI width (log) | median forecast NRMSE | median wall [s] | failed | R-hat > 1.1 |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in sorted(agg, key=lambda m: ORDER.index(m) if m in ORDER else 99):
        a = agg[m]
        med = lambda v: statistics.median(v) if v else None
        cov = (sum(a["cov"]) / len(a["cov"])) if a["cov"] else None
        out.append(f"| {m} | {len(a['wall'])} | " + " | ".join(fmt(med(a[k])) for k in CANONICAL)
                   + f" | {fmt(cov)} | {fmt(med(a['width']))} | {fmt(med(a['forecast']))} | {fmt(med(a['wall']), 0)} | {len(a['failed'])} | "
                   + (f"{int(sum(a['unconverged']))}/{len(a['unconverged'])}" if a["unconverged"] else "–") + " |")
    return "\n".join(out)


def figure_errors(rows: list[dict], path: Path) -> None:
    """Small multiples: one panel per canonical parameter; x = case, y = log10 rel. error."""
    ok = [r for r in rows if "error" not in r]
    cases = sorted({r["case"] for r in ok})
    methods = [m for m in ORDER if any(r["method_key"] == m for r in ok)]
    fig, axes = plt.subplots(len(CANONICAL), 1, figsize=(max(8, 0.9 * len(cases) + 4), 8.5),
                             sharex=True)
    width = 0.8 / max(len(methods), 1)
    for ax, k in zip(axes, CANONICAL):
        for j, m in enumerate(methods):
            xs, ys = [], []
            for r in ok:
                if r["method_key"] == m:
                    xs.append(cases.index(r["case"]) - 0.4 + (j + 0.5) * width)
                    ys.append(max(r["params"][k]["rel_err"], 1e-4))
            ax.scatter(xs, ys, s=42, color=COLOURS[m], marker=MARKERS[m], label=m,
                       edgecolors="white", linewidths=1.2, zorder=3)
        ax.set_yscale("log")
        ax.set_ylabel(f"{k}: relative error")
        ax.axhline(0.1, color="#9a9a93", lw=0.8, ls="--", zorder=1)
        ax.grid(axis="y", color="#e6e6e1", lw=0.6, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[-1].set_xticks(range(len(cases)))
    axes[-1].set_xticklabels([c.replace("_seed0", "").replace("_ode", "") for c in cases],
                             rotation=35, ha="right", fontsize=8)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    axes[0].set_title("Parameter recovery by method (dashed: 10% error)", loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def figure_cost(rows: list[dict], path: Path) -> None:
    """Wall time vs mean log-error over the canonical parameters, one dot per result."""
    ok = [r for r in rows if "error" not in r]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for m in [m for m in ORDER if any(r["method_key"] == m for r in ok)]:
        rs = [r for r in ok if r["method_key"] == m]
        ax.scatter([r["wall"] for r in rs],
                   [statistics.mean(r["params"][k]["log_err"] for k in CANONICAL) for r in rs],
                   s=42, color=COLOURS[m], marker=MARKERS[m], label=m, edgecolors="white",
                   linewidths=1.2, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("wall-clock seconds")
    ax.set_ylabel("mean |log error| over α, n, β")
    ax.grid(color="#e6e6e1", lw=0.6, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.set_title("Cost vs accuracy (lower left is better)", loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    dirs = [Path(a) for a in sys.argv[1:]]
    rows = load(dirs)
    out_dir = dirs[0]
    figure_errors(rows, out_dir / "errors.png")
    figure_cost(rows, out_dir / "cost.png")
    text = [f"# Benchmark report: {', '.join(d.name for d in dirs)}\n",
            "Relative error of the point estimate; `|log err|` = |log(est/truth)|. "
            "Coverage: whether the central 90% posterior interval contains the truth "
            "(methods with samples only) - read it together with the interval width: "
            "a near-prior-wide interval covers trivially. NRMSE: ODE at the estimate vs the noise-free "
            "truth, per-species range-normalised; *forecast* is the unseen next half-record.\n",
            *[part for regime in ("ode", "ensemble", "single_cell")
              if any(r.get("regime") == regime for r in rows)
              for part in (f"## Summary: {regime} data\n",
                           summary_table([r for r in rows if r.get("regime") == regime]), "")],
            "\n![errors](errors.png)\n\n![cost](cost.png)\n",
            "## Per case", case_table(rows)]
    (out_dir / "report.md").write_text("\n".join(text))
    print(f"wrote {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
