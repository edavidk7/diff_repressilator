#!/usr/bin/env bash
# Recover one parameter, alpha, from a broad random initialisation.
#
# The easiest real inverse problem here, and the reference point for
# everything else: measured recovery is ~1% from an initialisation that can be
# 35x wrong.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
setup "recover alpha alone" "~4 min" "$@"
trap finish EXIT

fit --drop_params alpha \
    --epochs 400 --lr_max 0.2 --lr_min 0.001 --curriculum 1,2 \
    --tag single-param
