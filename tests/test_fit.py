"""The inverse problem: parametrisation, loss at the truth, and the estimators' plumbing."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
from common import cell, true_theta  # noqa: E402
from differentiable_cell.abc import weighted_median  # noqa: E402
from differentiable_cell.fit import fit_gradient  # noqa: E402

torch.set_default_dtype(torch.float64)


def test_z_round_trip_and_box():
    problem, _ = cell("TetR fusion + GFP", "ode", 0)
    theta = problem.sample_prior(5, torch.Generator().manual_seed(0))
    torch.testing.assert_close(problem.from_z(problem.to_z(theta)), theta)
    z = torch.randn(5, len(problem.names)) * 10
    assert ((problem.from_z(z) >= problem.lo) & (problem.from_z(z) <= problem.hi)).all()


def test_truth_fits_to_the_camera_noise():
    problem, data = cell("3 reporters", "ode", 0)
    truth = true_theta(problem, data)
    assert ((truth >= problem.lo) & (truth <= problem.hi)).all()
    # RK4 at the truth reproduces dopri5 data up to camera noise (a few % of the range)
    assert 1e-4 < float(problem.loss(truth)) < 3e-3


def test_gradient_fit_improves_on_its_starts():
    problem, _ = cell("GFP only", "ode", 0)
    starts = problem.loss(problem.sample_prior(4, torch.Generator().manual_seed(0)))
    _, final = fit_gradient(problem, restarts=4, epochs=30, seed=0)
    assert float(final.min()) < float(torch.nan_to_num(starts, nan=float("inf")).min())


def test_weighted_median():
    particles = torch.tensor([[0.0, 5.0], [1.0, 6.0], [2.0, 7.0]])
    torch.testing.assert_close(weighted_median(particles, torch.tensor([0.1, 0.1, 0.8])), torch.tensor([2.0, 7.0]))
