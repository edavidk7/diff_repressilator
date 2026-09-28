#!/usr/bin/env bash
# PINN, MAGI and backprop+Laplace on the reduced grid (3 panels x ODE x 3 seeds), then figures.
set -u
cd "$(dirname "$0")/.."
for method in pinn magi laplace; do
  echo "=== $method $(date)"
  caffeinate -is uv run python experiments/panel_benchmark.py --method "$method" --reduced --workers 3 --threads 3
done
uv run python experiments/panel_benchmark.py --report
uv run python experiments/plot_results.py
echo "=== all done $(date)"
