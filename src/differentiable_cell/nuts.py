"""No-U-Turn sampler (Hoffman & Gelman 2014, Algorithm 6 with dual averaging), identity metric."""

import math
import time
from typing import Callable

import torch

LogProb = Callable[[torch.Tensor], tuple[float, torch.Tensor]]


def _leapfrog(x, r, grad, eps, logp_grad: LogProb):
    r = r + 0.5 * eps * grad
    x = x + eps * r
    lp, grad = logp_grad(x)
    r = r + 0.5 * eps * grad
    return x, r, lp, grad


def _build_tree(x, r, grad, log_u, v, j, eps, joint0, logp_grad, rng):
    """Recursive doubling; returns the tree's edges, proposal and statistics."""
    if j == 0:
        x1, r1, lp1, g1 = _leapfrog(x, r, grad, v * eps, logp_grad)
        joint = lp1 - 0.5 * float(r1 @ r1)
        if not math.isfinite(joint):
            joint = -math.inf
        n1 = int(log_u <= joint)
        s1 = int(log_u < joint + 1000.0)
        accept = min(1.0, math.exp(min(0.0, joint - joint0))) if math.isfinite(joint) else 0.0
        return x1, r1, g1, x1, r1, g1, x1, lp1, g1, n1, s1, accept, 1
    (xm, rm, gm, xp, rp, gp, x1, lp1, g1, n1, s1, a1, na1) = _build_tree(
        x, r, grad, log_u, v, j - 1, eps, joint0, logp_grad, rng)
    if s1 == 1:
        if v == -1:
            xm, rm, gm, _, _, _, x2, lp2, g2, n2, s2, a2, na2 = _build_tree(
                xm, rm, gm, log_u, v, j - 1, eps, joint0, logp_grad, rng)
        else:
            _, _, _, xp, rp, gp, x2, lp2, g2, n2, s2, a2, na2 = _build_tree(
                xp, rp, gp, log_u, v, j - 1, eps, joint0, logp_grad, rng)
        if n1 + n2 > 0 and float(torch.rand((), generator=rng)) < n2 / (n1 + n2):
            x1, lp1, g1 = x2, lp2, g2
        a1 += a2
        na1 += na2
        dx = xp - xm
        s1 = s2 * int(float(dx @ rm) >= 0) * int(float(dx @ rp) >= 0)
        n1 += n2
    return xm, rm, gm, xp, rp, gp, x1, lp1, g1, n1, s1, a1, na1


def nuts(
    logp_grad: LogProb,
    x0: torch.Tensor,
    n_warmup: int = 300,
    n_samples: int = 500,
    target_accept: float = 0.8,
    max_depth: int = 6,
    seed: int = 0,
    max_seconds: float | None = None,
) -> tuple[torch.Tensor, dict]:
    """Draw ``n_samples`` after ``n_warmup`` step-size adaptation iterations from ``x0``.

    ``logp_grad(x)`` returns (log density, gradient); ``max_seconds`` stops early, keeping what was drawn.
    Returns ``(samples (N, d), diagnostics)``.
    """
    rng = torch.Generator().manual_seed(seed)
    d = x0.numel()
    x = x0.clone()
    lp, grad = logp_grad(x)

    # Heuristic initial step size (Algorithm 4).
    eps = 0.5
    r = torch.randn(d, generator=rng, dtype=x.dtype)
    _, r1, lp1, _ = _leapfrog(x, r, grad, eps, logp_grad)
    ratio = (lp1 - 0.5 * float(r1 @ r1)) - (lp - 0.5 * float(r @ r))
    a = 1 if (math.isfinite(ratio) and ratio > math.log(0.5)) else -1
    for _ in range(50):
        _, r1, lp1, _ = _leapfrog(x, r, grad, eps, logp_grad)
        ratio = (lp1 - 0.5 * float(r1 @ r1)) - (lp - 0.5 * float(r @ r))
        if not math.isfinite(ratio):
            ratio = -math.inf
        if a * ratio <= -a * math.log(2):
            break
        eps *= 2.0**a

    mu, log_eps_bar, h_bar = math.log(10 * eps), 0.0, 0.0
    gamma, t0, kappa = 0.05, 10.0, 0.75
    samples, depths, accepts = [], [], []
    start = time.perf_counter()
    for m in range(1, n_warmup + n_samples + 1):
        r0 = torch.randn(d, generator=rng, dtype=x.dtype)
        joint0 = lp - 0.5 * float(r0 @ r0)
        log_u = joint0 + math.log(float(torch.rand((), generator=rng)) + 1e-300)
        xm = xp = x
        rm = rp = r0
        gm = gp = grad
        j, n, s = 0, 1, 1
        acc, n_acc = 0.0, 1
        while s == 1 and j < max_depth:
            v = 1 if float(torch.rand((), generator=rng)) < 0.5 else -1
            if v == -1:
                xm, rm, gm, _, _, _, x1, lp1, g1, n1, s1, acc, n_acc = _build_tree(
                    xm, rm, gm, log_u, v, j, eps, joint0, logp_grad, rng)
            else:
                _, _, _, xp, rp, gp, x1, lp1, g1, n1, s1, acc, n_acc = _build_tree(
                    xp, rp, gp, log_u, v, j, eps, joint0, logp_grad, rng)
            if s1 == 1 and float(torch.rand((), generator=rng)) < min(1.0, n1 / n):
                x, lp, grad = x1, lp1, g1
            n += n1
            dx = xp - xm
            s = s1 * int(float(dx @ rm) >= 0) * int(float(dx @ rp) >= 0)
            j += 1
        stat = acc / max(n_acc, 1)
        if m <= n_warmup:
            h_bar = (1 - 1 / (m + t0)) * h_bar + (target_accept - stat) / (m + t0)
            log_eps = mu - math.sqrt(m) / gamma * h_bar
            eta = m**-kappa
            log_eps_bar = eta * log_eps + (1 - eta) * log_eps_bar
            eps = math.exp(log_eps)
            if m == n_warmup:
                eps = math.exp(log_eps_bar)
        else:
            samples.append(x.clone())
            depths.append(j)
            accepts.append(stat)
        if max_seconds is not None and time.perf_counter() - start > max_seconds:
            break
    out = torch.stack(samples) if samples else x.unsqueeze(0)
    return out, {
        "step_size": eps,
        "mean_tree_depth": sum(depths) / max(len(depths), 1),
        "mean_accept": sum(accepts) / max(len(accepts), 1),
        "n_kept": len(samples),
    }
