#!/usr/bin/env bash
# Run every experiment in order, one at a time.
#
# Sequentially, deliberately: back-propagating through the solver keeps the
# whole tape, so two long fits at once can exhaust memory and the OS will kill
# them mid-run.  Pass --adjoint in an individual script instead if you are
# memory-bound - it trades roughly 6x the time for roughly 6x less memory.
#
# Total runtime is a few hours.  A failing experiment is reported and the rest
# continue, since one bad fit should not cost the whole sweep.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
setup "every experiment, sequentially" "~3.5 hours" "$@"
trap finish EXIT

failed=()
for script in "$(dirname "${BASH_SOURCE[0]}")"/[0-9][0-9]_*.sh; do
    echo
    echo "############################################################"
    echo "# $(basename "$script")"
    echo "############################################################"
    if ! bash "$script" "$CLEAN"; then
        echo "!! $(basename "$script") FAILED" >&2
        failed+=("$(basename "$script")")
    fi
done

echo
if [ ${#failed[@]} -eq 0 ]; then
    echo "all experiments completed"
else
    echo "failed: ${failed[*]}"
    exit 1
fi
