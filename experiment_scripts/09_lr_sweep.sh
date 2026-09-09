#!/usr/bin/env bash
# Why the learning rate is the difference between exact recovery and failure.
#
# Same three-parameter problem at three rates.  At 0.2 an early step in log
# space drives n into its prior floor and momentum holds it there; at 0.03 the
# optimiser cannot cross the four decades of the alpha prior within the budget;
# at 0.1 it recovers all three parameters to machine precision.
#
# Check the at_prior_bound field in metrics.json for each: a value sitting on
# its box is a fit that ran out of room, not a converged one.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
setup "the same 3-parameter fit at lr 0.03, 0.1 and 0.2" "~30 min" "$@"
trap finish EXIT

for lr in 0.03 0.1 0.2; do
    fit --drop_params alpha beta n --restarts 2 \
        --epochs 600 --lr_max "$lr" --lr_min 0.002 --curriculum 1,2 \
        --tag "lr-$lr"
done
