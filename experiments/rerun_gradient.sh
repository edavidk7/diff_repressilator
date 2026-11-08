#!/usr/bin/env bash
# Rerun every result that depends on the backprop fit with the current defaults (the first version is in *_v1).
# Resumable: each step skips fits already on disk.
set -euo pipefail
cd "$(dirname "$0")/.."
W=${1:-8}
run() { uv run python "$@"; }

# the benchmark: exactly the cells of the first version (gradient and Laplace)
run - "$W" <<'EOF'
import sys
sys.path.insert(0, "experiments")
from common import RESULTS, run_pool
from panel_benchmark import job_file
import json
jobs = []
for method in ("gradient", "laplace"):
    for p in sorted((RESULTS / "panel_benchmark" / f"{method}_v1").glob("*.json")):
        r = json.loads(p.read_text())
        if not job_file(method, r["panel"], r["kind"], r["seed"]).exists():
            jobs.append(["--threads", "1", "--one", method, r["panel"], r["kind"], str(r["seed"])])
run_pool("experiments/panel_benchmark.py", jobs, int(sys.argv[1]), RESULTS / "panel_benchmark" / "log.txt")
EOF
run experiments/circuits_benchmark.py --method gradient --workers "$W"
run experiments/real_repressilator.py --method gradient --workers "$W"
run experiments/real_improved.py --method gradient --whole-record --workers "$W"
run experiments/restart_spread.py --workers 6 --threads 1
run experiments/fit_controls.py refine --workers "$W"
echo "all done"
