#!/usr/bin/env bash
# The remaining methods on the same 48 cases, then the report and figures. Resumable.
set -u
cd "$(dirname "$0")/.."
for method in pinn magi nuts; do
  echo "=== $method $(date)"
  caffeinate -is uv run python experiments/panel_benchmark.py --method "$method" --workers 3 --threads 3
done
uv run python experiments/panel_benchmark.py --report
uv run python experiments/plot_results.py
echo "=== all done $(date)"
