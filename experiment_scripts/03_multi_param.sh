#!/usr/bin/env bash
# Recover three coupled parameters - alpha, beta, n - with three restarts.
#
# The validated configuration.  lr 0.1 matters: at 0.2 an early step drives n
# into its prior floor and the optimiser's momentum holds it there, while at
# 0.03 it cannot travel far enough in the budget.  At 0.1 all three restarts
# recover all three parameters to machine precision.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
setup "recover alpha, beta and n (3 restarts)" "~23 min" "$@"
trap finish EXIT

fit --drop_params alpha beta n --restarts 3 \
    --epochs 800 --lr_max 0.1 --lr_min 0.002 --curriculum 1,2 \
    --tag multi-param
