#!/usr/bin/env bash
# Rerun every result that depends on the backprop fit with the current defaults (the first version is in *_v1).
#   experiments/rerun_gradient.sh [workers] [archive tag]
# With a tag (e.g. v4), the current results are first moved aside under that tag, once. Resumable: each step skips
# fits already on disk.
set -euo pipefail
cd "$(dirname "$0")/.."
W=${1:-8}
TAG=${2:-}
run() { uv run python "$@"; }
export PYTHONWARNINGS=ignore

archive() {  # archive <folder holding the current results> <glob> <destination>
    if [ -n "$TAG" ] && [ ! -e "$3" ] && ls $1/$2 >/dev/null 2>&1; then mkdir -p "$3" && mv $1/$2 "$3"/; fi
}
R=run_results
archive $R/panel_benchmark/gradient "*.json" $R/panel_benchmark/gradient_$TAG
archive $R/panel_benchmark/laplace "*.json" $R/panel_benchmark/laplace_$TAG
archive $R/circuits/gradient "*.json" $R/circuits/gradient_$TAG
archive $R/real_repressilator "fit_*.json" $R/real_repressilator/gradient_$TAG
archive $R/real_improved "fit_*.json" $R/real_improved/gradient_$TAG
archive $R/restart_spread "*.json" $R/restart_spread/$TAG
archive $R/fit_controls/refine/gradient "*.json" $R/fit_controls/refine/gradient_$TAG

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
