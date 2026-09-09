#!/usr/bin/env bash
# Nothing is hidden: the fit should start and stay at zero loss.
#
# This is the 1:1 test.  If the clean run's log is a faithful record of its own
# forward problem, then re-posing that problem with nothing free must reproduce
# the trajectory exactly.  Any non-zero loss here means the log and the model
# have drifted apart, and every other experiment in this directory is void.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
setup "1:1 check - nothing dropped, loss must be exactly 0" "~10 s" "$@"
trap finish EXIT

fit --epochs 1 --no-plot --tag sanity
