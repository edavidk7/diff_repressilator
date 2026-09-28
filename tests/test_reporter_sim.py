"""The three-reporter model and its data generation."""


import pytest
import torch

from differentiable_cell import reporters as R
from differentiable_cell.helpers import estimate_period
from differentiable_cell.reporter_sim import (
    CHANNEL_STATE, DT_FRAME, Camera, box1_truth, generate, mask, params, simulate_ode, simulate_ssa, start_state,
)

torch.set_default_dtype(torch.float64)
ALL = [0, 1, 2]


def test_reaction_network_drift_is_the_ode():
    p, m = params(box1_truth()), mask(ALL)
    y = torch.rand(15) * 50
    scales = R.count_scales(p, 40.0)
    drift = R.propensities((y * scales).unsqueeze(0), p, m, 40.0) @ R.stoichiometry()
    torch.testing.assert_close(drift[0] / scales, R.rhs(y, p, m))


def test_titration_only_through_present_reporters():
    p = params(box1_truth())
    y = torch.rand(15) * 50
    none = R.rhs(y, p, mask([]))
    only_tet = R.rhs(y, p, mask([1]))
    # TetR represses cI (index 2): shifted when the tet reporter is present.
    assert none[2] != only_tet[2]
    # LacI represses tetR (index 1), CI represses lacI (index 0): unaffected.
    torch.testing.assert_close(none[[0, 1]], only_tet[[0, 1]])


def test_equal_titration_is_a_rescaled_circuit():
    """K = 1 + tau on every repressor == alpha/K, alpha_0/K with p, m scaled by 1/K."""
    truth = box1_truth()
    p, m = params(truth), mask(ALL)
    K = 1 + truth["tau"][0]
    y0 = start_state(p, m, 0.0)
    t = torch.linspace(0, 80, 81)
    titrated = simulate_ode(p, m, y0, t)[:, :6]
    scaled = params(dict(truth, alpha=truth["alpha"] / K, alpha_0=truth["alpha_0"] / K, tau=[0.0] * 3))
    plain = simulate_ode(scaled, mask([]), torch.cat([y0[:6] / K, y0[6:]]), t)[:, :6]
    torch.testing.assert_close(titrated, plain * K, rtol=1e-5, atol=1e-6)


def test_reporter_steady_state_and_maturation():
    """At a fixed point, F = k / (k + delta) * r; r = a h + a_0."""
    truth = dict(box1_truth(), n=1.0)  # n = 1: no oscillation, a stable fixed point
    p, m = params(truth), mask(ALL)
    y = start_state(p, m, 0.0, burn_in=2000.0)
    assert R.rhs(y, p, m).abs().max() < 1e-6
    k, delta = R.K_MAT, R.DELTA
    torch.testing.assert_close(y[R.F_IDX], k / (k + delta) * y[R.R_IDX], rtol=1e-6, atol=1e-8)


def test_oscillates_with_expected_period():
    p, m = params(box1_truth()), mask(ALL)
    t = torch.linspace(0, 300, 3001)
    traj = simulate_ode(p, m, start_state(p, m, 0.0), t)
    period = estimate_period(t, traj[:, 4])
    assert period is not None and 30 < period < 60, period  # ~2 h at 2.885 min/unit


def test_camera_noise_switch_and_statistics():
    cam = Camera()
    F = torch.full((20000, 6), 50.0)  # K_M units
    mean = cam.read(F, 40.0, noise=False)
    torch.testing.assert_close(mean[0], 50.0 * cam.au_per_km(40.0) + cam.offset)
    y = cam.read(F, 40.0, noise=True, generator=torch.Generator().manual_seed(0))
    expected_var = cam.gain * (mean[0] - cam.offset) + cam.read_noise**2
    assert torch.allclose(y.mean(0), mean[0], rtol=2e-3)
    assert torch.allclose(y.var(0), expected_var, rtol=0.05)


def test_generate_frames_are_a_subsample():
    d = generate(box1_truth(), ALL, kind="ode", noise=False, hours=2.0, seed=3)
    assert d.t.numel() == 25 and abs(float(d.t[1] - d.t[0]) - DT_FRAME) < 1e-12
    cam = Camera()
    frames = d.fine_state[::10][:, CHANNEL_STATE]
    torch.testing.assert_close(d.y_au, frames * cam.au_per_km(40.0) + cam.offset)
    torch.testing.assert_close(d.y, frames)  # calibrated back to model units


def test_seed_sets_the_start_phase():
    a = generate(box1_truth(), ALL, noise=False, hours=1.0, seed=0)
    b = generate(box1_truth(), ALL, noise=False, hours=1.0, seed=1)
    c = generate(box1_truth(), ALL, noise=False, hours=1.0, seed=0)
    assert a.meta["phase"] != b.meta["phase"]
    torch.testing.assert_close(a.y, c.y)


@pytest.mark.slow
def test_gillespie_mean_follows_ode_at_large_system_size():
    """At omega = 400 the ensemble mean of exact cells tracks the ODE."""
    p, m = params(box1_truth()), mask(ALL)
    y0 = start_state(p, m, 0.0)
    t = torch.linspace(0, 4, 5)
    ode = simulate_ode(p, m, y0, t)
    cells = torch.stack([simulate_ssa(p, m, y0, t, 400.0, torch.Generator().manual_seed(s), settle=0)
                         for s in range(12)])
    rel = (cells.mean(0)[-1] - ode[-1]).abs() / ode[-1].abs().clamp_min(1.0)
    assert (rel[R.F_IDX] < 0.03).all() and (rel[R.P_IDX] < 0.1).all(), rel
