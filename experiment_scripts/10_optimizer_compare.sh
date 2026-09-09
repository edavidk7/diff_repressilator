#!/usr/bin/env bash
# Does the choice of optimiser matter on this problem?
#
# Same single-parameter fit under four torch.optim classes.  Parameter decay is
# disabled automatically in every case - decay toward zero is meaningless for
# physical rate constants, and in log space it would drag every one of them
# toward 1.
#
# LBFGS is deliberately absent: it diverges from a cold start here, because a
# quasi-Newton model of the landscape is worthless 35x away from the optimum.
# It would need warm-starting from an Adam solution to be useful.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
setup "Adam vs AdamW vs RMSprop vs SGD on the same fit" "~16 min" "$@"
trap finish EXIT

for optimiser in Adam AdamW RMSprop SGD; do
    fit --drop_params alpha --optim "$optimiser" \
        --epochs 400 --lr_max 0.2 --lr_min 0.001 --curriculum 1,2 \
        --tag "optim-$optimiser"
done
