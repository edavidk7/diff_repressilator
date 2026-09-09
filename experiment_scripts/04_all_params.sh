#!/usr/bin/env bash
# Recover all six kinetic parameters at once, with the full trajectory observed.
#
# The upper bound on what parameter estimation can do here: every species is
# measured and only the parameters are unknown.  alpha_GFP and beta_GFP shape
# only the reporter, so watch their restart spread in restarts.csv - it is the
# first place a trade-off between them would show.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
setup "recover all six parameters (3 restarts)" "~25 min" "$@"
trap finish EXIT

fit --drop_params alpha alpha_0 n beta alpha_GFP beta_GFP --restarts 3 \
    --epochs 800 --lr_max 0.1 --lr_min 0.002 --curriculum 1,2 \
    --tag all-params
