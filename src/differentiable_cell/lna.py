"""Stochastic-likelihood baseline: linear noise approximation filtered with an extended Kalman filter.

Between frames the mean follows the ODE and the covariance dSigma/dt = A Sigma + Sigma A^T + D (A: Jacobian of
the drift, D: LNA diffusion of the reaction network); each frame updates both and restarts the next interval
from the update. Komorowski et al. 2009 (BMC Bioinformatics 10:343), Burton et al. 2021 (J. R. Soc. Interface
18:20210393), Swallow, Rand & Minas (arXiv 2205.05955). Same unknowns and first-order recipe as fit_gradient.
"""

import math
import time

import torch

from differentiable_cell import reporters as R
from differentiable_cell.fit import RK4_STEP, Problem

JITTER = 1e-10

# Each species has one first-order loss and one source: a Hill term (mRNAs) or a linear term in one other species.
_SOURCE = list(range(R.N_STATE))  # the species the source depends on
_HILL = [0] * R.N_STATE  # for mRNAs: which repressor (0..2) sets the Hill term
for _i in range(3):
    _SOURCE[R.M_IDX[_i]], _HILL[R.M_IDX[_i]] = R.P_IDX[R.GENE_REPRESSOR[_i]], R.GENE_REPRESSOR[_i]
    _SOURCE[R.P_IDX[_i]] = R.M_IDX[_i]
    _SOURCE[R.R_IDX[_i]], _HILL[R.R_IDX[_i]] = R.P_IDX[R.REPORTER_REPRESSOR[_i]], R.REPORTER_REPRESSOR[_i]
    _SOURCE[R.D_IDX[_i]], _SOURCE[R.F_IDX[_i]] = R.R_IDX[_i], R.D_IDX[_i]
_SOURCE, _HILL = torch.tensor(_SOURCE), torch.tensor(_HILL)
_MATURATION = torch.zeros(3, R.N_STATE, R.N_STATE, dtype=torch.float64)  # the only reaction moving two species
for _i in range(3):
    _MATURATION[_i, R.D_IDX[_i], R.F_IDX[_i]] = _MATURATION[_i, R.F_IDX[_i], R.D_IDX[_i]] = -1.0
_MATURATION = _MATURATION.flatten(1)


def lna_terms(p: dict, present: torch.Tensor, omega: float):
    """``x (B, 15) -> (drift, Jacobian diagonal, Jacobian off-diagonal, diffusion D (B, 15, 15))`` for batched
    parameters; the Jacobian row of species i is ``diag_i`` at i and ``off_i`` at ``_SOURCE[i]``. Written out by
    hand to stay batched and cheap; equal to reporters.rhs, its autograd Jacobian and
    D = sum_r (s_r / scale)(s_r / scale)^T a_r of the reaction network (tested)."""
    beta, B = p["beta"].unsqueeze(-1), p["beta"].shape[0]
    k, delta = R.K_MAT.to(beta), R.DELTA.to(beta)
    K, n = 1.0 + p["tau"] * present, p["n"].unsqueeze(-1)
    zeros = lambda: torch.zeros(B, R.N_STATE, dtype=beta.dtype)
    scale = torch.full((B, R.N_STATE), float(omega), dtype=beta.dtype)  # reporters.count_scales, batched over beta
    scale[:, R.M_IDX] = omega * beta / R.EFF
    scale[:, R.R_IDX] = omega * delta / R.EFF
    diag, linear, strength, leak = zeros(), zeros(), zeros(), zeros()
    diag[:, R.M_IDX], diag[:, R.P_IDX], diag[:, R.R_IDX] = -1.0, -beta, -1.0
    diag[:, R.D_IDX], diag[:, R.F_IDX] = -(k + delta), -delta
    linear[:, R.P_IDX], linear[:, R.D_IDX], linear[:, R.F_IDX] = beta, delta, k
    strength[:, R.M_IDX], strength[:, R.R_IDX] = p["alpha"].unsqueeze(-1), p["a"]
    leak[:, R.M_IDX], leak[:, R.R_IDX] = p["alpha_0"].unsqueeze(-1), p["a_0"]

    def terms(x):
        u = x[:, R.P_IDX].clamp_min(1e-12) / K
        un = u**n
        h = 1.0 / (1.0 + un)  # reporters.repression
        dh = -n * un / u / K * h * h  # its derivative in the protein
        made = strength * h.index_select(1, _HILL) + leak  # transcription
        drift = diag * x + linear * x.index_select(1, _SOURCE) + made
        off = linear + strength * dh.index_select(1, _HILL)
        # diffusion: (production + loss) / scale on the diagonal; maturation also couples D and F
        xc = x.clamp_min(0.0)
        D = torch.diag_embed((made + linear * xc.index_select(1, _SOURCE) - diag * xc) / scale)
        D = D + ((xc[:, R.D_IDX] * k / omega) @ _MATURATION).view(B, R.N_STATE, R.N_STATE)
        return drift, diag, off, D

    return terms, scale


def jacobian(diag: torch.Tensor, off: torch.Tensor) -> torch.Tensor:
    """The dense Jacobian ``(B, 15, 15)`` from the two entries per row (for tests)."""
    A = torch.diag_embed(diag)
    A[:, range(R.N_STATE), _SOURCE] = off
    return A


def noise_variance(y: torch.Tensor) -> torch.Tensor:
    """Observation-noise variance per channel from the frames alone: second differences of white noise have
    variance 6 sigma^2, and the slow signal mostly cancels in them."""
    d2 = y[2:] - 2.0 * y[1:-1] + y[:-2]
    return d2.pow(2).mean(0) / 6.0


def lna_loglik(problem: Problem, theta: torch.Tensor, omega: float = 40.0, obs_var: torch.Tensor | None = None,
               n_frames: int | None = None, step: float = RK4_STEP) -> torch.Tensor:
    """Log-likelihood of the first ``n_frames`` frames under the LNA / extended Kalman filter, per leading index
    of ``theta``; -inf where the filter broke down."""
    single = theta.dim() == 1
    theta = theta.unsqueeze(0) if single else theta
    p, x = problem.params_and_start(theta)
    present, obs = problem.mask, problem.observed_state
    terms, scale = lna_terms(p, present, omega)
    noise = torch.diag(noise_variance(problem.y) if obs_var is None else torch.as_tensor(obs_var, dtype=x.dtype))
    eye = torch.eye(R.N_STATE, dtype=x.dtype)
    Sig = torch.diag_embed(x / scale) + JITTER * eye  # start covariance: Poisson counts, uncorrelated

    def derivative(x, Sig):
        drift, diag, off, D = terms(x)
        AS = diag.unsqueeze(-1) * Sig + off.unsqueeze(-1) * Sig.index_select(1, _SOURCE)  # A Sigma, two entries per row of A
        return drift, AS + AS.mT + D

    t = problem.t if n_frames is None else problem.t[:n_frames]
    loglik = torch.zeros(theta.shape[0], dtype=x.dtype)
    ok = torch.ones(theta.shape[0], dtype=torch.bool)
    for g in range(t.numel()):
        if g > 0:  # predict: RK4 on the mean and the covariance together
            steps = max(1, math.ceil(float(t[g] - t[g - 1]) / step - 1e-9))
            dt = (t[g] - t[g - 1]) / steps
            for _ in range(steps):
                k1 = derivative(x, Sig)
                k2 = derivative(x + 0.5 * dt * k1[0], Sig + 0.5 * dt * k1[1])
                k3 = derivative(x + 0.5 * dt * k2[0], Sig + 0.5 * dt * k2[1])
                k4 = derivative(x + dt * k3[0], Sig + dt * k3[1])
                x = x + dt / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
                Sig = Sig + dt / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        # update on the frame: y = x[obs] + Gaussian noise
        v = (problem.y[g] - x[:, obs]).unsqueeze(-1)  # innovation (B, k, 1)
        C = Sig[:, :, obs]  # Sigma H^T (B, 15, k)
        S = C[:, obs, :] + noise
        L, bad = torch.linalg.cholesky_ex(0.5 * (S + S.mT))
        ok = ok & (bad == 0)
        loglik = loglik - 0.5 * ((v * torch.cholesky_solve(v, L)).sum((-1, -2)) + len(obs) * math.log(2 * math.pi)) \
            - torch.log(L.diagonal(dim1=-2, dim2=-1)).sum(-1)
        gain = torch.cholesky_solve(C.mT, L).mT  # Sigma H^T S^-1
        x = x + (gain @ v).squeeze(-1)
        Sig = Sig - gain @ C.mT - C @ gain.mT + gain @ S @ gain.mT  # Joseph form for a selection H
        Sig = 0.5 * (Sig + Sig.mT) + JITTER * eye
    loglik = torch.where(ok & torch.isfinite(loglik), loglik, torch.full_like(loglik, -math.inf))
    return loglik[0] if single else loglik


def fit_lna(problem: Problem, seed: int = 0, restarts: int = 64, epochs: int = 500, omega: float = 40.0,
            lr: float = 0.05, lr_final: float = 2.5e-4, anneal: float = 0.1, obs_var: torch.Tensor | None = None,
            step: float = RK4_STEP, return_all: bool = False):
    """Maximise the LNA likelihood from ``restarts`` prior draws as one batch (Adam on z; ``lr`` held, then
    cosine-annealed over the last ``anneal`` of the epochs). Returns the most likely restart and plain numbers."""
    started = time.time()
    g = torch.Generator().manual_seed(seed)
    z = problem.to_z(problem.sample_prior(restarts, g)).requires_grad_(True)
    obs_var = noise_variance(problem.y) if obs_var is None else torch.as_tensor(obs_var, dtype=problem.y.dtype)
    opt = torch.optim.Adam([z], lr=lr)
    full, r = round((1.0 - anneal) * epochs), lr_final / lr
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: 1.0 if e < full else
                                              r + (1 - r) * 0.5 * (1 + math.cos(math.pi * (e - full) / max(epochs - full, 1))))
    for _ in range(epochs):
        opt.zero_grad()
        nll = -lna_loglik(problem, problem.from_z(z), omega, obs_var, step=step)
        # summing keeps the restarts' gradients independent; a non-finite restart stops moving
        torch.where(torch.isfinite(nll), nll, torch.zeros_like(nll)).sum().backward()
        z.grad = torch.nan_to_num(z.grad, nan=0.0, posinf=0.0, neginf=0.0)
        opt.step()
        sched.step()
    with torch.no_grad():
        theta = problem.from_z(z)
        loglik = lna_loglik(problem, theta, omega, obs_var, step=step)
        best = int(loglik.argmax())
        mse = float(torch.nan_to_num(problem.loss(theta[best]), nan=math.inf))
    info = {"loglik": float(loglik[best]), "loglik_all": loglik.tolist(), "mse": mse, "omega": float(omega),
            "obs_var": obs_var.tolist(), "restarts": restarts, "epochs": epochs, "seconds": time.time() - started}
    return (theta, info) if return_all else (theta[best], info)
