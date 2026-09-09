#!/usr/bin/env bash
# Are the two reporter parameters separable from GFP alone?
#
# alpha_GFP sets how much reporter mRNA is made and beta_GFP how fast the
# protein decays.  Both shape the same single observed curve - one its
# amplitude, the other its smoothing and lag - so they are the most likely pair
# in this model to trade off against each other.  With only GFP observed there
# may be a ridge of (alpha_GFP, beta_GFP) combinations that fit equally well.
#
# The diagnostic is the spread across restarts in restarts.csv and the
# spread_over_restarts field in metrics.json.  A tight best-fit with a wide
# restart spread means a ridge, not a measurement.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
setup "reporter identifiability: alpha_GFP vs beta_GFP from GFP alone" "~25 min" "$@"
trap finish EXIT

fit --drop_params alpha_GFP beta_GFP \
    --drop_state m_lacI m_tetR m_cI p_LacI p_TetR p_CI m_gfp \
    --restarts 4 \
    --epochs 600 --lr_max 0.1 --lr_min 0.002 --curriculum 1,2 \
    --tag reporter-tradeoff
