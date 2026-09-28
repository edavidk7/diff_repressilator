"""Estimator plumbing and the pieces that can be checked exactly.

Recovery quality is not asserted here (it is what the benchmark measures);
these tests pin down the parts whose correctness is binary: kernels, the
fast integrator, the Laplace covariance, and that each method runs end to
end and improves on its starting point.
"""

import math

import pytest
import torch

from differentiable_cell.inference import abc_smc, grad_ode, magi, pinn
from differentiable_cell.inference.base import gauss_newton_covariance
from differentiable_cell.inference.datasets import make_case, paper_truth
from differentiable_cell.inference.fast_ode import _rhs as fast_rhs
from differentiable_cell.inference.scoring import score
from differentiable_cell.model import Repressilator
from differentiable_cell.run import FIELDS

torch.set_default_dtype(torch.float64)


@pytest.fixture(scope="module")
def short_case():
    # ~1 period, fully observed, 1% noise: cheap for every method.
    return make_case(paper_truth(), "all", noise=0.01, horizon=45.0, n_frames=27, seed=0)


def test_fast_rhs_matches_reference_model():
    y = torch.rand(6, 8) * 40
    p = dict(alpha=torch.tensor(300.0), alpha_0=torch.tensor(0.3), n=torch.tensor(2.4),
             beta=torch.tensor(0.3), alpha_GFP=torch.tensor(150.0), beta_GFP=torch.tensor(0.05))
    ref = torch.stack(Repressilator().forward_tensor(
        **{f: y[:, i] for i, f in enumerate(FIELDS)}, **p), -1)
    lane = lambda s, g: torch.stack([s, s, s, g])
    fast = fast_rhs(y, lane(p["alpha"], p["alpha_GFP"]), p["alpha_0"].unsqueeze(-1),
                    p["n"].unsqueeze(-1), lane(p["beta"], p["beta_GFP"]))
    torch.testing.assert_close(fast, ref)


def test_problem_simulation_matches_clean_truth(short_case):
    p = short_case.problem
    pred = p.simulate(short_case.truth_theta, all_species=True)
    err = (pred - short_case.clean).abs().max(0).values / short_case.clean.abs().max(0).values
    assert (err < 2e-3).all(), err


def test_theta_transforms_roundtrip(short_case):
    p = short_case.problem
    th = p.sample_prior(10, torch.Generator().manual_seed(0))
    torch.testing.assert_close(p.from_z(p.to_z(th)), th)


def test_matern_derivatives_match_autograd():
    t = torch.linspace(0, 10, 7)
    var, ls = 2.0, 1.7
    c, dc, ddc = magi.matern52(t, var, ls, jitter=0.0)

    def k(s, u):
        a = math.sqrt(5) / ls
        r = (s - u).abs()
        return var * (1 + a * r + a**2 * r**2 / 3) * torch.exp(-a * r)

    for i in (1, 3):
        for j in (0, 2, 5):
            s = t[i].clone().requires_grad_(True)
            u = t[j].clone().requires_grad_(True)
            val = k(s, u)
            (ds,) = torch.autograd.grad(val, s, create_graph=True)
            (dsdu,) = torch.autograd.grad(ds, u)
            assert abs(float(ds) - float(dc[i, j])) < 1e-9
            assert abs(float(dsdu) - float(ddc[i, j])) < 1e-9


def test_laplace_is_exact_for_a_linear_gaussian_model(short_case):
    """For a linear model the Gauss-Newton covariance is the exact posterior one."""
    from dataclasses import replace
    p = short_case.problem
    design = torch.randn(p.t.numel() * len(p.observed), p.dim, generator=torch.Generator().manual_seed(1))

    class Linear(type(p)):
        def simulate(self, theta, n_obs_times=None, all_species=False, compiled=True):
            return (design @ theta).reshape(p.t.numel(), len(p.observed))

    lin = Linear(**{f: getattr(p, f) for f in p.__dataclass_fields__})
    cov = gauss_newton_covariance(lin, torch.zeros(p.dim))
    lo, hi = p.bounds.unbind(-1)
    precision = design.T @ (design / (p.sigma.repeat(p.t.numel()) ** 2).unsqueeze(-1))
    expected = torch.linalg.inv(precision + torch.diag(12 / (hi - lo) ** 2))
    torch.testing.assert_close(cov, expected, rtol=1e-6, atol=1e-10)


def test_grad_ode_improves_likelihood_and_scores(short_case):
    p = short_case.problem
    res = grad_ode.fit(p, restarts=2, epochs=150, seed=0)
    prior_ll = p.log_likelihood(p.sample_prior(2, torch.Generator().manual_seed(0)))
    assert res.info["best_loglik"] > float(prior_ll.max())
    assert res.samples is not None and torch.isfinite(res.samples).all()
    row = score(short_case, res)
    assert set(row["params"]) == set(p.free_params)
    assert math.isfinite(row["nrmse_forecast"])


def test_abc_tolerance_decreases(short_case):
    res = abc_smc.fit(short_case.problem, n_particles=200, batch=400, max_sims=6000, seed=0)
    eps = [h["eps"] for h in res.info["populations"]]
    assert len(eps) >= 3 and all(a > b for a, b in zip(eps[1:], eps[2:]))


def test_magi_map_runs(short_case):
    res = magi.fit(short_case.problem, restarts=1, map_iters=300, n_samples=0)
    assert torch.isfinite(res.theta).all()


def test_pinn_runs(short_case):
    res = pinn.fit(short_case.problem, restarts=1, iterations=50, width=16, depth=2)
    assert torch.isfinite(res.theta).all()
