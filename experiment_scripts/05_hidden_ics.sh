#!/usr/bin/env bash
# Hide the cI arm entirely: fit alpha plus the initial values of m_cI and p_CI.
#
# A dropped species contributes one unknown, its value at t = 0, and the ODE
# supplies the rest of its trajectory.  These initial conditions are only
# weakly identifiable: moving m_cI(0) from 0 to 50 changes the loss by ~1.7e-3,
# which can be smaller than the fit's own residual, so they are the last thing
# to be pinned down and they need alpha to be right first.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
setup "recover alpha plus the hidden initial values of m_cI and p_CI" "~15 min" "$@"
trap finish EXIT

fit --drop_params alpha --drop_state m_cI p_CI \
    --epochs 1200 --lr_max 0.2 --lr_min 0.0005 --curriculum 1,2,3 \
    --tag hidden-ics
