import torch

from conftest import paper_params, start_state
from differentiable_cell.langevin import run_langevin
from differentiable_cell.model import Repressilator
from differentiable_cell.run import FIELDS, run_torchdiffeq


def _stack(traj):
    return torch.stack([getattr(traj, f) for f in FIELDS], dim=-1)


def test_non_negative_at_small_system_size(params):
    t = torch.linspace(0, 60, 121, dtype=torch.float64)
    out = _stack(run_langevin(start_state(32), params, t, omega=2.0,
                              generator=torch.Generator().manual_seed(0)))
    assert torch.isfinite(out).all() and (out >= 0).all()


def test_large_system_mean_follows_ode(params):
    t = torch.linspace(0, 30, 61, dtype=torch.float64)
    ode = _stack(run_torchdiffeq(Repressilator(), start_state(), params, t))
    cle = _stack(run_langevin(start_state(64), params, t, omega=4000.0,
                              generator=torch.Generator().manual_seed(0)))
    err = (cle.mean(1) - ode).abs().max(0).values / ode.abs().max(0).values
    assert (err < 0.05).all(), err


def test_noise_variance_scales_inversely_with_system_size(params):
    t = torch.linspace(0, 3, 7, dtype=torch.float64)
    var = []
    for omega in (400.0, 4000.0):
        out = run_langevin(start_state(2000), params, t, omega=omega,
                           generator=torch.Generator().manual_seed(1))
        var.append(float(out.p_TetR[-1].var()))
    assert 7.0 < var[0] / var[1] < 13.0, var


def test_pathwise_gradient_matches_finite_difference():
    t = torch.linspace(0, 5, 11, dtype=torch.float64)
    noise = torch.randn(100, 4, 16, generator=torch.Generator().manual_seed(2),
                        dtype=torch.float64)

    def loss(beta: torch.Tensor) -> torch.Tensor:
        p = paper_params()
        p.beta = beta
        return run_langevin(start_state(4), p, t, omega=400.0, noise=noise).p_TetR.sum()

    beta = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    (grad,) = torch.autograd.grad(loss(beta), beta)
    h = 1e-6
    fd = (loss(torch.tensor(0.2 + h, dtype=torch.float64))
          - loss(torch.tensor(0.2 - h, dtype=torch.float64))) / (2 * h)
    assert torch.isfinite(grad)
    assert abs(float(grad) - float(fd)) < 1e-4 * abs(float(fd)), (grad, fd)
