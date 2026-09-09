#!/usr/bin/env bash
# Which set of reporters makes the core kinetics measurable? (idealised)
#
# Noise-free and at full solver resolution. For the realistic version - 5 min
# imaging cadence and 1% noise - see 13_practical_configs.sh.
#
# GFP alone does not: restarts disagree on alpha by more than an order of
# magnitude at equal loss.  This asks what else an experimentalist could build
# and whether it helps.  Each configuration fits the same three core
# parameters - alpha, beta, n - and hides every species it does not observe,
# so each hidden species also costs one unknown initial value.
#
#   p_GFP                      the paper: one reporter off P_Ltet01
#   p_TetR + p_GFP             one repressor-YFP fusion plus the reporter
#   p_LacI + p_TetR + p_CI     three repressor fusions (CFP/YFP/mCherry)
#   all three fusions + GFP    fusions plus the reporter
#   m_lacI + m_tetR + m_cI     the three mRNAs, by smFISH or MS2 tagging
#
# The answer is in restarts.csv, not the best fit: two restarts agreeing is
# evidence the experiment determines the parameter; two restarts disagreeing
# at equal loss means it does not, however good the best number looks.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
setup "which reporter configuration makes alpha, beta and n measurable" "~80 min" "$@"
trap finish EXIT

fit_config() {
    local tag="$1"; shift
    fit --drop_params alpha beta n --drop_state "$@" --restarts 2 \
        --epochs 800 --lr_max 0.1 --lr_min 0.002 --curriculum 1,2 \
        --tag "reporters-$tag"
}

fit_config gfp        m_lacI m_tetR m_cI p_LacI p_TetR p_CI m_gfp
fit_config tetr-gfp   m_lacI m_tetR m_cI p_LacI p_CI m_gfp
fit_config 3fusions   m_lacI m_tetR m_cI m_gfp p_GFP
fit_config 3fusions-gfp m_lacI m_tetR m_cI m_gfp
fit_config 3mrna      p_LacI p_TetR p_CI m_gfp p_GFP
