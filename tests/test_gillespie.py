import torch

from differentiable_cell import reporters as R
from differentiable_cell.gillespie import ssa_dga, ssa_exact
from differentiable_cell.reporter_sim import box1_truth, mask, params, start_state

BIRTH_DEATH = torch.tensor([[1.0], [-1.0]], dtype=torch.float64)


def birth_death(k, gamma):
    return lambda x: torch.stack(torch.broadcast_tensors(k, gamma * x[:, 0]), dim=-1)


def test_exact_birth_death_is_poisson():
    x0 = torch.zeros(4000, 1, dtype=torch.float64)
    t = torch.tensor([0.0, 10.0], dtype=torch.float64)
    x = ssa_exact(birth_death(torch.tensor(10.0), 1.0), BIRTH_DEATH, x0, t,
                  generator=torch.Generator().manual_seed(0))[-1, :, 0]
    assert (x == x.round()).all()
    # Poisson(10): mean = var = 10; standard errors ~0.05 and ~0.25.
    assert abs(float(x.mean()) - 10.0) < 0.2
    assert abs(float(x.var()) - 10.0) < 1.0


def _dga_moments(a, b, k=10.0, cells=2000, seed=0):
    x0 = torch.zeros(cells, 1, dtype=torch.float64)
    t = torch.tensor([0.0, 10.0], dtype=torch.float64)
    x = ssa_dga(birth_death(torch.tensor(k, dtype=torch.float64), 1.0), BIRTH_DEATH,
                x0, t, a=a, b=b, c=0.0, generator=torch.Generator().manual_seed(seed))
    return float(x[-1].mean()), float(x[-1].var())


def test_dga_birth_death_moments_close_to_exact():
    mean, var = _dga_moments(1 / 200, 1 / 20)
    assert abs(mean - 10.0) < 1.0 and abs(var - 10.0) < 2.5, (mean, var)


def test_dga_bias_shrinks_with_sharper_smoothing():
    # The mean is nearly unbiased here (lost births and deaths cancel); the
    # smoothing shows up as a variance deficit that closes as a, b -> 0.
    loose = abs(_dga_moments(1 / 10, 1 / 3)[1] - 10.0)
    sharp = abs(_dga_moments(1 / 200, 1 / 20)[1] - 10.0)
    assert loose > 1.5 and sharp < 0.5 * loose, (loose, sharp)


def test_dga_gradient_of_mean_matches_analytic():
    # E[x(T)] = k/g (1 - exp(-g T)) from x(0) = 0.
    k = torch.tensor(10.0, dtype=torch.float64, requires_grad=True)
    g = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    x0 = torch.zeros(2000, 1, dtype=torch.float64)
    t = torch.tensor([0.0, 2.0], dtype=torch.float64)
    x = ssa_dga(birth_death(k, g), BIRTH_DEATH, x0, t, c=0.05,
                generator=torch.Generator().manual_seed(3))
    dk, dg = torch.autograd.grad(x[-1].mean(), (k, g))
    e = torch.exp(torch.tensor(-2.0))
    true_dk = float(1 - e)
    true_dg = float(-10.0 * (1 - e) + 10.0 * 2.0 * e)
    assert abs(float(dk) - true_dk) < 0.15 * abs(true_dk), (dk, true_dk)
    assert abs(float(dg) - true_dg) < 0.15 * abs(true_dg), (dg, true_dg)


def test_dga_repressilator_runs_and_differentiates():
    p0, m = params(box1_truth()), mask([0, 1, 2])
    alpha = p0["alpha"].clone().requires_grad_(True)
    p = dict(p0, alpha=alpha)
    x0 = (start_state(p0, m, 0.0) * R.count_scales(p0, 2.0)).expand(4, R.N_STATE).clone()
    x = ssa_dga(lambda x: R.propensities(x, p, m, 2.0), R.stoichiometry(), x0, torch.linspace(0, 0.5, 6),
                generator=torch.Generator().manual_seed(0), max_events=2_000_000)
    (grad,) = torch.autograd.grad(x[-1].mean(), alpha)
    assert torch.isfinite(x).all() and torch.isfinite(grad) and grad != 0
