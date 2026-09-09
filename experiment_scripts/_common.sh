# Shared helpers for the fixed experiment scripts.
#
# Each experiment script takes exactly one argument - the directory of a clean
# run produced by experiments/run_clean.py - and hard-codes every other
# setting, so an experiment is reproduced by running the same script rather
# than by remembering a command line.
#
# Not executable on its own; sourced by the numbered scripts.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PARTIAL="$REPO_ROOT/experiments/run_partial.py"

# Describe the experiment and check the one argument.
#   setup "<one-line description>" "<expected runtime>" "$@"
setup() {
    local description="$1" runtime="$2"
    shift 2
    SCRIPT_NAME="$(basename "$0" .sh)"

    if [ $# -ne 1 ] || [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
        cat >&2 <<USAGE
usage: $(basename "$0") <clean-run-directory>

  $description

  Expected runtime: $runtime
  The argument is a directory written by experiments/run_clean.py, e.g.
    run_results/20260909-144046_ode_dopri5_steps3000_dt0.05_float64_seed0
USAGE
        exit 2
    fi

    CLEAN="$1"
    if [ ! -f "$CLEAN/trajectory.pt" ]; then
        echo "error: $CLEAN does not look like a clean run (no trajectory.pt)" >&2
        exit 2
    fi

    echo "============================================================"
    echo " $SCRIPT_NAME"
    echo " $description"
    echo " clean run: $CLEAN"
    echo " expected runtime: $runtime"
    echo "============================================================"
    START_TIME=$SECONDS
}

# Run one fit, echoing the exact arguments so the log records them.
fit() {
    echo
    echo "+ run_partial.py $CLEAN $*"
    python "$PARTIAL" "$CLEAN" "$@"
}

# Print total elapsed time.  Registered by each script via `trap finish EXIT`.
finish() {
    local elapsed=$((SECONDS - START_TIME))
    echo
    echo "------------------------------------------------------------"
    printf ' %s finished in %dm %02ds\n' "$SCRIPT_NAME" $((elapsed / 60)) $((elapsed % 60))
    echo "------------------------------------------------------------"
}
