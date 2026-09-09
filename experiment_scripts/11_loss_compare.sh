#!/usr/bin/env bash
# Does the loss function matter, on clean data and on noisy data?
#
# On clean data the three should be near-indistinguishable.  The comparison
# only becomes interesting under noise, where MAE and Huber are less swayed by
# the largest residuals than MSE is - and in this system the largest residuals
# sit on the sharp mRNA peaks.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
setup "mse vs mae vs huber, clean and at sigma = 0.05" "~24 min" "$@"
trap finish EXIT

for loss in mse mae huber; do
    for sigma in 0 0.05; do
        fit --drop_params alpha --loss "$loss" --noise "$sigma" \
            --epochs 400 --lr_max 0.2 --lr_min 0.001 --curriculum 1,2 \
            --tag "loss-$loss-noise$sigma"
    done
done
