"""Baseline: ABC-SMC (Toni et al. 2009, steps S1-S3 of their algorithm).

As in the paper: sample by weight, perturb with a uniform random-walk kernel, reject outside the prior, simulate,
accept within eps, weight = prior / sum_j w_j K. Our choices where the paper leaves them open: eps_t is the median
distance of the previous population (their S1 fixes a schedule in advance) and the kernel's variance is twice the
previous population's (Beaumont et al. 2009). The prior is uniform on the log box.
"""

import math

import torch

from differentiable_cell.fit import Problem


def distance(problem: Problem, theta: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        d = problem.distance(theta) if hasattr(problem, "distance") else problem.loss(theta).sqrt()
    return torch.nan_to_num(d, nan=math.inf)


def summary_distance(problem: Problem):
    """A distance on summary statistics of each channel, for oscillating data (a trajectory distance punishes a
    small period error as much as no oscillation): mean, sd, log period of the spectral peak, and the Fourier
    coefficient at the data's peak frequency (amplitude and phase), relative to the data's."""
    G = problem.t.numel()

    def spectrum(y):  # (..., G, k) range-normalised -> rfft without the mean, (..., G // 2 + 1, k)
        return torch.fft.rfft(y - y.mean(-2, keepdim=True), dim=-2)

    y = problem.y / problem.range
    F = spectrum(y)
    peak = F[1:].abs().argmax(0) + 1  # the data's peak frequency index, per channel
    obs = (y.mean(0), y.std(0), torch.log(G / peak.double()), F.gather(0, peak[None])[0])

    def d(theta):
        with torch.no_grad():
            s = problem.predict(theta) / problem.range  # (B, G, k)
            Fs = spectrum(s)
            mine = Fs[:, 1:].abs().argmax(1) + 1
            at_peak = Fs.gather(1, peak.expand(len(s), 1, -1))[:, 0]
            terms = torch.stack([(s.mean(1) - obs[0]) ** 2, (s.std(1) - obs[1]) ** 2,
                                 (torch.log(G / mine.double()) - obs[2]) ** 2,
                                 ((at_peak - obs[3]).abs() / obs[3].abs()) ** 2], -1)
        return torch.nan_to_num(terms.mean((-1, -2)).sqrt(), nan=math.inf)

    return d


def proposals(theta, w, half_width, lo, hi, batch, g):
    """S2.1: ``batch`` particles sampled by weight and perturbed; those with zero prior are redrawn."""
    kept, have = [], 0
    while have < batch:
        parents = theta[torch.multinomial(w, 20 * batch, replacement=True, generator=g)]
        cand = parents + half_width * (2 * torch.rand(parents.shape, generator=g, dtype=torch.float64) - 1)
        cand = cand[((cand >= lo) & (cand <= hi)).all(-1)]
        kept.append(cand)
        have += len(cand)
    return torch.cat(kept)[:batch]


def fit_abc(problem: Problem, particles: int = 500, batch: int = 500, max_sims: int = 300_000,
            min_acceptance: float = 1e-3, seed: int = 0, distance=distance):
    """Returns the last complete population, its weights and ``(eps, acceptance)`` per population."""
    g = torch.Generator().manual_seed(seed)
    lo, hi = problem.lo, problem.hi

    dist = distance if distance is not globals()["distance"] else (lambda th: distance(problem, th))
    theta = problem.sample_prior(particles, g)  # population 0: the prior (S2.1, t = 0)
    d = dist(theta)
    w = torch.full((particles,), 1.0 / particles, dtype=torch.float64)
    sims, history = particles, [(math.inf, 1.0)]

    while sims < max_sims:
        eps = float(d.median())  # S1, adaptive
        mean = (w[:, None] * theta).sum(0)
        half_width = (6.0 * (w[:, None] * (theta - mean) ** 2).sum(0)).sqrt()  # U(-h,h) has var h^2/3 = 2 var

        kept, kept_d, tried = [], [], 0
        while sum(len(k) for k in kept) < particles and sims < max_sims:
            cand = proposals(theta, w, half_width, lo, hi, batch, g)
            dc = dist(cand)
            sims += len(cand)
            tried += len(cand)
            kept.append(cand[dc <= eps])  # S2.1: accept if d <= eps
            kept_d.append(dc[dc <= eps])
        new_theta = torch.cat(kept)[:particles]
        if len(new_theta) < particles:
            break  # budget ran out mid-population: keep the last complete one

        # S2.2: w_i = prior(theta_i) / sum_j w_j K(theta_i | theta_j), prior constant on the box
        inside = ((new_theta[:, None, :] - theta[None, :, :]).abs() <= half_width).all(-1)
        kernel = inside.double() / (2 * half_width).prod()
        new_w = 1.0 / (kernel * w[None, :]).sum(-1)
        theta, d, w = new_theta, torch.cat(kept_d)[:particles], new_w / new_w.sum()  # S3
        acceptance = particles / max(tried, 1)
        history.append((eps, acceptance))
        if acceptance < min_acceptance:
            break
    return theta, w, history


def weighted_median(particles: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Per-coordinate weighted median of the particles: the point estimate."""
    out = []
    for i in range(particles.shape[1]):
        order = particles[:, i].argsort()
        cdf = weights[order].cumsum(0)
        out.append(particles[order[int((cdf >= 0.5).nonzero()[0])], i])
    return torch.stack(out)
