import math

import torch

from conftest import paper_params, start_state
from differentiable_cell.helpers import TAU_M_MIN
from differentiable_cell.model import Repressilator
from differentiable_cell.run import FIELDS, PARAM_FIELDS
from differentiable_cell.units import (
    STOICHIOMETRY, count_scales, counts_to_state, propensities, state_to_counts,
)


def test_alpha_216_from_paper_kinetics():
    # 0.5 transcripts/s x 20 proteins/transcript, protein half-life 10 min,
    # in units of K_M = 40 monomers.
    copies = 0.5 * 60 * 20 / (math.log(2) / 10.0)
    assert abs(copies / 40.0 - 216.4) < 0.1


def test_mrna_scale_matches_transcripts_per_cell(params):
    # Fully induced gene: m = alpha in model units should be ~0.5/s x 173 s.
    transcripts = float(count_scales(params)[FIELDS.index("m_tetR")]) * 216.0
    assert abs(transcripts - 0.5 * 60 * TAU_M_MIN) / transcripts < 0.01


def test_roundtrip(params):
    state = start_state(3)
    state.m_cI += 7.0
    back = counts_to_state(state_to_counts(state, params), params, state.batch_size)
    for name in FIELDS:
        torch.testing.assert_close(getattr(back, name), getattr(state, name))


def test_network_drift_is_the_ode():
    params = paper_params(n=2.3, alpha_0=0.5)
    y = torch.rand(5, 8, dtype=torch.float64) * 30
    scales = count_scales(params)
    drift = propensities(y * scales, params, omega=40.0) @ STOICHIOMETRY.double()
    ode = torch.stack(
        Repressilator().forward_tensor(
            **{f: y[:, i] for i, f in enumerate(FIELDS)},
            **{p: getattr(params, p) for p in PARAM_FIELDS},
        ),
        dim=-1,
    )
    torch.testing.assert_close(drift / scales, ode)


def test_propensities_non_negative(params):
    x = torch.randn(100, 8, dtype=torch.float64) * 50
    assert (propensities(x, params) >= 0).all()
