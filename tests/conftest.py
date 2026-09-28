"""Shared fixtures: the paper's parameters and the repo's standard start state."""

import pytest
import torch

from differentiable_cell.data import RepressilatorParams, RepressilatorState


def paper_params(**overrides: float) -> RepressilatorParams:
    values = dict(
        alpha=216.0, alpha_0=0.216, n=2.0, beta=0.2, alpha_GFP=216.0, beta_GFP=2.0 / 90.0
    )
    values.update(overrides)
    return RepressilatorParams(
        **{k: torch.tensor(v, dtype=torch.float64) for k, v in values.items()},
        batch_size=(),
    )


def start_state(n_cells: int | None = None) -> RepressilatorState:
    values = dict(m_lacI=0.0, m_tetR=0.0, m_cI=0.0, p_LacI=5.0, p_TetR=0.0,
                  p_CI=15.0, m_gfp=0.0, p_GFP=0.0)
    shape = () if n_cells is None else (n_cells,)
    return RepressilatorState(
        **{k: torch.full(shape, v, dtype=torch.float64) for k, v in values.items()},
        batch_size=shape,
    )


@pytest.fixture
def params() -> RepressilatorParams:
    return paper_params()


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: takes more than a few seconds")
