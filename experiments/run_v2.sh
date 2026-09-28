#!/usr/bin/env bash
# The multi-seed benchmark: one pass per method, cheapest first.  Resumable:
# each pass skips results already on disk.  Keeps the machine awake while it runs.
#   ./experiments/run_v2.sh [workers] [threads]
set -u
cd "$(dirname "$0")/.."
W=${1:-3}; T=${2:-2}
for pass in grad_ode magi abc_ode pinn grad_cle abc_cle nuts nested; do
  echo "=== pass $pass $(date)"
  caffeinate -is uv run python experiments/benchmark.py --suite "v2_$pass" --workers "$W" --threads "$T"
done
echo "=== all passes done $(date)"
