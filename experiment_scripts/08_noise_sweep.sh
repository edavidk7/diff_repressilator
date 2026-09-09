#!/usr/bin/env bash
# How gracefully does recovery degrade as the measurement gets noisier?
#
# Noise is added as a fraction of each species' range, so sigma = 0.05 is 5% of
# the full swing of that species.  The paper's central finding is that the real
# repressilator is noisy; this is the observation-level analogue, and it asks
# whether the estimate degrades smoothly or collapses.
#
# Measured previously: alpha recovers to ~1% clean and ~12% at sigma = 0.05.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
setup "alpha recovery at sigma = 0, 0.02, 0.05, 0.10" "~16 min" "$@"
trap finish EXIT

for sigma in 0 0.02 0.05 0.10; do
    fit --drop_params alpha --noise "$sigma" \
        --epochs 400 --lr_max 0.2 --lr_min 0.001 --curriculum 1,2 \
        --tag "noise-$sigma"
done
