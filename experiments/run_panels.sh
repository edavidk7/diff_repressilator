#!/usr/bin/env bash
# Full panel benchmark: backprop, then ABC-SMC, then the figures. Resumable.
set -u
cd "$(dirname "$0")/.."
for method in gradient abc; do
  echo "=== $method $(date)"
  caffeinate -is uv run python experiments/panel_benchmark.py --method "$method" --workers 3 --threads 3
done
uv run python experiments/panel_benchmark.py --report
uv run python experiments/plot_results.py
echo "=== all done $(date)"
