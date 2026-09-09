#!/usr/bin/env bash
# The honest experimental setup: only GFP is observed.
#
# In the paper, GFP fluorescence is the single measurable quantity - everything
# else in the network is invisible.  This poses that problem: one observed
# curve, seven hidden species (each contributing its initial value) and two
# free parameters.  GFP is also the worst possible probe of the fast dynamics,
# since GFP-AAV's ~90 min half-life low-pass filters the underlying repressor
# cycle, so the fit sees a smoothed, delayed shadow of the oscillation.
#
# The question this answers is not "can we recover the parameters" but "how
# much of the network is knowable from what an experimentalist can actually
# measure".  Read restarts.csv, not just the best fit: agreement across
# restarts is the evidence of identifiability, and disagreement is the result.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
setup "GFP-only observation: recover alpha and beta with 7 hidden species" "~25 min" "$@"
trap finish EXIT

fit --drop_params alpha beta \
    --drop_state m_lacI m_tetR m_cI p_LacI p_TetR p_CI m_gfp \
    --restarts 3 \
    --epochs 800 --lr_max 0.1 --lr_min 0.002 --curriculum 1,2 \
    --tag gfp-only
