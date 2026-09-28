"""ABC-SMC (Toni et al., 2009): likelihood-free, no gradients.

Particles are drawn from the prior, simulated, and kept if the simulation is
within a tolerance ``eps`` of the data. Each later population resamples the
previous one by weight, perturbs it, and uses a smaller ``eps``:

    distance  = sqrt(range-normalised MSE)   (the same measure the gradient fit minimises)
    kernel    = uniform random walk per coordinate, width from the previous population
    eps_t     = median distance of population t-1
    weight    = prior / sum_j w_j K(theta | theta_j)   (prior is uniform on the box)

Toni et al. tune the kernel and the tolerances by hand; here both adapt, so
nothing is tuned on the true answer.
"""

import math

import torch

from differentiable_cell.fit import Problem


def distance(problem: Problem, theta: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        d = problem.loss(theta).sqrt()
    return torch.nan_to_num(d, nan=math.inf)


def proposals(theta, w, half_width, lo, hi, batch, g):
    """``batch`` perturbed particles inside the prior box.

    Outside the box the prior is zero, so such proposals are rejected before
    simulating. In ~15 dimensions only a small fraction of perturbations land
    inside, so draw many and keep the first ``batch`` (a redraw costs nothing).
    """
    kept, have = [], 0
    while have < batch:
        parents = theta[torch.multinomial(w, 20 * batch, replacement=True, generator=g)]
        cand = parents + half_width * (2 * torch.rand(parents.shape, generator=g, dtype=torch.float64) - 1)
        cand = cand[((cand >= lo) & (cand <= hi)).all(-1)]
        kept.append(cand)
        have += len(cand)
    return torch.cat(kept)[:batch]


def fit_abc(problem: Problem, particles: int = 500, batch: int = 500, max_sims: int = 200_000,
            min_acceptance: float = 1e-3, seed: int = 0):
    """Run populations until the simulation budget or the acceptance floor is hit.

    Returns the final particles ``(particles, dim)``, their weights, and one
    ``(eps, acceptance)`` record per population.
    """
    g = torch.Generator().manual_seed(seed)
    lo, hi = problem.lo, problem.hi

    theta = problem.sample_prior(particles, g)
    d = distance(problem, theta)
    w = torch.full((particles,), 1.0 / particles, dtype=torch.float64)
    sims, history = particles, [(math.inf, 1.0)]

    while sims < max_sims:
        eps = float(d.median())
        mean = (w[:, None] * theta).sum(0)
        half_width = (6.0 * (w[:, None] * (theta - mean) ** 2).sum(0)).sqrt()  # U(-h,h) has var h^2/3 = 2 var

        kept, kept_d, tried = [], [], 0
        while sum(len(k) for k in kept) < particles and sims < max_sims:
            cand = proposals(theta, w, half_width, lo, hi, batch, g)
            dc = distance(problem, cand)
            sims += len(cand)
            tried += len(cand)
            kept.append(cand[dc <= eps])
            kept_d.append(dc[dc <= eps])
        new_theta = torch.cat(kept)[:particles]
        if len(new_theta) < particles:
            break  # budget ran out mid-population: keep the last complete one

        inside = ((new_theta[:, None, :] - theta[None, :, :]).abs() <= half_width).all(-1)
        kernel = inside.double() / (2 * half_width).prod()
        new_w = 1.0 / (kernel * w[None, :]).sum(-1)
        theta, d, w = new_theta, torch.cat(kept_d)[:particles], new_w / new_w.sum()
        acceptance = particles / max(tried, 1)
        history.append((eps, acceptance))
        if acceptance < min_acceptance:
            break
    return theta, w, history
