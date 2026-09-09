#!/usr/bin/env bash
# The measurement-design question, under realistic measurement conditions.
#
# Four reporter configurations an experimentalist could actually build, each
# fitting the same three core kinetic parameters (alpha, beta, n), each with
# every unobserved species hidden - so each hidden species also costs one
# unknown initial value.
#
#   gfp            p_GFP                     the original experiment
#   tetr-gfp       p_TetR + p_GFP            one repressor-YFP fusion + reporter
#   3fusions       p_LacI, p_TetR, p_CI      three repressor-FP fusions
#   3fusions-gfp   those + p_GFP             fusions plus the reporter
#
# Conditions, unlike the idealised runs in 12_reporter_configs.sh:
#   --sample_hz 0.00333   one frame every 5 min - the paper's imaging cadence,
#                         86 frames over 7.2 h.  Every other result in this
#                         repository used the raw solver grid, one frame every
#                         8.66 s, which is 35x denser than any real time-lapse
#                         of this system and ~20x finer than the 2.9 min mRNA
#                         lifetime.  Sampling faster than the system's own
#                         timescales adds no dynamical information, only noise
#                         averaging, so sparser - not faster - is the direction
#                         that tests the claim in the title.
#   --noise 0.01          1% of each species' range, mild but non-zero
#
# Caveat on the cadence axis: --noise applies a FIXED sigma per frame, so in
# this model denser sampling always helps, as sigma/sqrt(N).  A real detector
# splits a fixed photon budget across frames, where sigma per frame grows as
# sqrt(1/exposure) and the two effects very nearly cancel.  Cadence comparisons
# from this script therefore flatter fast imaging; the configuration comparison
# at a fixed cadence, which is what this script is for, is unaffected.
#
# The verdict is in restarts.csv, not the best fit. Restarts agreeing means the
# configuration determines the parameter; restarts disagreeing at comparable
# loss means it does not, however good the best number looks.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
setup "reporter configurations at a realistic 5 min cadence, 1% noise" "~45 min" "$@"
trap finish EXIT

fit_config() {
    local tag="$1"; shift
    fit --drop_params alpha beta n --drop_state "$@" \
        --sample_hz 0.00333 --noise 0.01 --restarts 2 \
        --epochs 500 --lr_max 0.1 --lr_min 0.002 --curriculum 1,2 \
        --tag "practical-$tag"
}

fit_config gfp          m_lacI m_tetR m_cI p_LacI p_TetR p_CI m_gfp
fit_config tetr-gfp     m_lacI m_tetR m_cI p_LacI p_CI m_gfp
fit_config 3fusions     m_lacI m_tetR m_cI m_gfp p_GFP
fit_config 3fusions-gfp m_lacI m_tetR m_cI m_gfp

# Controls: the two configurations that recovered exactly on the dense grid.
# They establish whether those results survive a realistic cadence, or were an
# artefact of sampling 35x more often than anyone would.
fit --drop_params alpha beta n \
    --sample_hz 0.00333 --noise 0.01 --restarts 2 \
    --epochs 500 --lr_max 0.1 --lr_min 0.002 --curriculum 1,2 \
    --tag practical-full-abn

fit --drop_params alpha alpha_0 n beta alpha_GFP beta_GFP \
    --sample_hz 0.00333 --noise 0.01 --restarts 2 \
    --epochs 500 --lr_max 0.1 --lr_min 0.002 --curriculum 1,2 \
    --tag practical-full-six
