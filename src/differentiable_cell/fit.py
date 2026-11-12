"""Our estimator: backpropagation through the ODE solver (paper Sec. "Method").

``Problem`` is the repressilator inverse problem, ``fit_gradient`` the estimator. Unknowns ``theta`` are logs on a
box: alpha, alpha_0, n, beta, each built reporter's (a, a_0), and the starting value of every species.
"""

import copy
import math
from dataclasses import dataclass

import torch
from torchdiffeq import odeint

from differentiable_cell import reporters as R
from differentiable_cell.reporter_sim import CHANNEL_STATE, mask

CIRCUIT_BOX = {
    "alpha": (20.0, 1000.0),  # must oscillate; 1000 = 40k monomers per cell
    "alpha_0": (1e-3, 1.0),  # a leak of order K_M stops the oscillation
    "n": (1.0, 4.0),  # two to four operator sites
    "beta": (0.1, 1.0),  # ssrA-tagged repressors: protein half-life 2-20 min
}
CIRCUIT_START_BOX = (1e-3, 1000.0)
CIRCUIT_SPECIES = ["m_lacI", "m_tetR", "m_cI", "p_LacI", "p_TetR", "p_CI"]
RK4_STEP = 0.35  # ~5 steps per frame; the data come from tight dopri5, so solver error is not fitted


def solve(f, y0, t):
    return odeint(f, y0, t, method="rk4", options={"step_size": RK4_STEP})


@dataclass
class Problem:
    t: torch.Tensor  # frame times (G,)
    y: torch.Tensor  # calibrated frames of the observed channels, K_M units (G, k)
    channels: list[int]  # observed channels, indices into reporter_sim.CHANNELS
    present: list[int]  # reporters built (they titrate and have parameters)
    tau: list[float]  # known titration, per reporter

    def __post_init__(self):
        self.mask = mask(self.present)
        self.observed_state = [CHANNEL_STATE[c] for c in self.channels]
        self.tau_t = torch.tensor(self.tau, dtype=torch.float64)
        self.range = (self.y.max(0).values - self.y.min(0).values).clamp_min(1e-9)

        def first_frame_box(state, default):
            if state not in self.observed_state:
                return default
            c = self.observed_state.index(state)
            first, rng = float(self.y[0, c]), float(self.range[c])
            return max(first - 0.1 * rng, 1e-3), first + 0.1 * rng

        def reading_range(j):
            c = self.observed_state.index(R.F_IDX[j])
            return max(float(self.y[:, c].min()), 1e-3), float(self.y[:, c].max())

        boxes = list(CIRCUIT_BOX.values())
        for j in self.present:
            low, high = reading_range(j)
            boxes += [(0.8 * high, 10 * high),  # a: F <= a when fully derepressed
                      (1e-4 * high, low)]  # a_0: F >= a_0 when fully repressed
        boxes += [first_frame_box(i, CIRCUIT_START_BOX) for i in range(6)]
        for j in self.present:
            low, high = reading_range(j)
            boxes += [(1e-4 * high, 10 * high),  # r(0): between ~a_0 and ~a
                      (5e-6 * high, 0.5 * high),  # D(0): dark FP is ~5% of r
                      first_frame_box(R.F_IDX[j], (0.1 * low, 3 * high))]  # F(0)
        self.lo = torch.tensor([math.log(lo) for lo, _ in boxes], dtype=torch.float64)
        self.hi = torch.tensor([math.log(hi) for _, hi in boxes], dtype=torch.float64)

    @property
    def names(self) -> list[str]:
        reporters = [R.REPORTERS[j] for j in self.present]
        return (list(CIRCUIT_BOX) + [f"{k}_{r}" for r in reporters for k in ("a", "a0")]
                + [f"{s}(0)" for s in CIRCUIT_SPECIES]
                + [f"{k}_{r}(0)" for r in reporters for k in ("r", "D", "F")])

    # theta (log values) = lo + sigmoid(z) (hi - lo): z is unconstrained, the box exact.
    def from_z(self, z):
        return self.lo + torch.sigmoid(z) * (self.hi - self.lo)

    def to_z(self, theta):
        u = ((theta - self.lo) / (self.hi - self.lo)).clamp(1e-9, 1 - 1e-9)
        return torch.log(u) - torch.log1p(-u)

    def sample_prior(self, n, generator=None):
        return self.lo + torch.rand(n, len(self.lo), generator=generator, dtype=torch.float64) * (self.hi - self.lo)

    def values(self, theta) -> dict[str, torch.Tensor]:
        return {name: theta[..., i].exp() for i, name in enumerate(self.names)}

    def params_and_start(self, theta):
        """Model parameters and the 15-species start ``(..., 15)``; absent reporters stay at zero."""
        v = self.values(theta)
        one, zero = torch.ones_like(v["alpha"]), torch.zeros_like(v["alpha"])

        def per_reporter(key, absent):
            return torch.stack([v[f"{key}_{r}"] if j in self.present else absent
                                for j, r in enumerate(R.REPORTERS)], dim=-1)

        p = {"alpha": v["alpha"], "alpha_0": v["alpha_0"], "n": v["n"], "beta": v["beta"],
             "a": per_reporter("a", one), "a_0": per_reporter("a0", zero),
             "tau": self.tau_t.expand(*theta.shape[:-1], 3)}
        start = [v[f"{s}(0)"] for s in CIRCUIT_SPECIES]
        for j, r in enumerate(R.REPORTERS):
            start += [v[f"{k}_{r}(0)"] for k in ("r", "D", "F")] if j in self.present else [zero] * 3
        return p, torch.stack(start, dim=-1)

    def predict(self, theta, n_frames=None, t=None):
        """y_hat = h(x(t_g)): observed channels ``(..., G, k)`` at the first ``n_frames`` frames, or at ``t``."""
        p, y0 = self.params_and_start(theta)
        if t is None:
            t = self.t if n_frames is None else self.t[:n_frames]
        traj = solve(lambda _t, y: R.rhs(y, p, self.mask), y0, t)  # (G, ..., 15)
        return traj[..., self.observed_state].movedim(0, -2)

    def loss(self, theta, n_frames=None, kind: str = "mse"):
        """Eq. (5) on the first ``n_frames`` frames; ``kind`` is the penalty rho ("huber": delta = 0.05)."""
        y = self.y if n_frames is None else self.y[:n_frames]
        r = (self.predict(theta, n_frames) - y) / self.range  # range-normalised residual
        if kind == "mse":
            per_point = r**2
        elif kind == "l1":
            per_point = r.abs()
        elif kind == "huber":
            per_point = torch.nn.functional.huber_loss(r, torch.zeros_like(r), reduction="none", delta=0.05)
        else:
            raise ValueError(kind)
        return per_point.mean((-1, -2))


def _cast(obj, dtype, device=None):
    """A shallow copy of a problem with its tensors (and those of nested objects) on ``device``, floating ones in
    ``dtype``."""
    out = copy.copy(obj)
    for k, v in vars(obj).items():
        if torch.is_tensor(v):
            setattr(out, k, v.to(device=device, dtype=dtype if v.is_floating_point() else None))
        elif hasattr(v, "__dict__") and not callable(v) and any(torch.is_tensor(x) for x in vars(v).values()):
            setattr(out, k, _cast(v, dtype, device))
    return out


def fit_gradient(problem: Problem, restarts: int = 64, epochs: int = 500, lr: float | None = None,
                 lr_final: float | None = None, stages=(0.25, 0.5, 1.0), curriculum: str = "ramp",
                 start_fraction: float = 0.1, ramp=(0.1, 0.9), seed: int = 0, loss: str = "mse",
                 optimizer: str = "adam", schedule: str = "coupled", return_all: bool = False,
                 history: list | None = None, history_every: int = 25, box: str = "unit",
                 dtype=torch.float32, device=None):
    """Minimise Eq. (5) from ``restarts`` prior draws as one batch; returns the best restart's theta and all MSEs.

    ``curriculum``: how the loss window G' grows ("ramp", ours; "linear"; "stages"; "none" = ablation).
    ``schedule``: "coupled" (ours) holds ``lr`` until the window is whole, then anneals; "cosine" / "exp" decay throughout.
    ``loss``: the penalty rho of Eq. (5): "mse" (ours), "huber" or "l1".
    ``box``: "unit" (ours) optimises each unknown's position in its box, in [0, 1], and clips to it; "sigmoid" maps
    an unconstrained z onto the box; "clamp" clips the log values. ``lr`` defaults to 0.0125 for "unit" (the sigmoid's
    step at the box centre) and 0.05 otherwise, ``lr_final`` to ``lr`` / 200. The fit runs in ``dtype`` on ``device``
    (e.g. "cuda"); the best restart is chosen, and theta returned, in double precision on the CPU.
    """
    g = torch.Generator().manual_seed(seed)
    lr = lr if lr is not None else (0.0125 if box == "unit" else 0.05)
    lr_final = lr_final if lr_final is not None else lr / 200
    exact, start = problem, problem.sample_prior(restarts, g)  # the prior draws, in double precision
    problem = _cast(problem, dtype, device)
    start = start.to(device)
    if box == "clamp":  # z is theta itself
        to_theta, z = (lambda v: v), start.to(dtype).requires_grad_(True)
    elif box == "unit":  # z is the position in the box, in [0, 1]
        to_theta = lambda v: problem.lo + v * (problem.hi - problem.lo)
        z = ((start - exact.lo.to(device)) / (exact.hi - exact.lo).to(device)).to(dtype).requires_grad_(True)
    else:
        to_theta, z = problem.from_z, exact.to_z(start.cpu()).to(device=device, dtype=dtype).requires_grad_(True)
    opt = {"adam": lambda: torch.optim.Adam([z], lr=lr),
           "rmsprop": lambda: torch.optim.RMSprop([z], lr=lr),
           "sgd": lambda: torch.optim.SGD([z], lr=lr, momentum=0.9, nesterov=True)}[optimizer]()
    G = problem.t.numel()
    # --- window G' of Eq. (5), per epoch
    if curriculum in ("linear", "ramp"):
        a, b = (0.0, 1.0) if curriculum == "linear" else ramp
        progress = lambda e: min(max((e / max(epochs - 1, 1) - a) / (b - a), 0.0), 1.0)
        window = [max(4, math.ceil((start_fraction + (1 - start_fraction) * progress(e)) * G)) for e in range(epochs)]
    elif curriculum == "none":  # the whole record from the first epoch (ablation)
        window = [G] * epochs
    elif curriculum == "stages":
        steps = [max(4, math.ceil(f * G)) for f in stages]
        window = [steps[e * len(steps) // epochs] for e in range(epochs)]
    else:
        raise ValueError(f"curriculum {curriculum!r}")
    # --- learning rate
    if schedule == "exp":
        sched = torch.optim.lr_scheduler.ExponentialLR(opt, (lr_final / lr) ** (1.0 / max(epochs - 1, 1)))
    elif schedule == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr_final)
    elif schedule == "coupled":
        # anneal once the window is whole; without a curriculum, at ramp[1] as in the default
        full = round(ramp[1] * epochs) if curriculum == "none" else next(e for e in range(epochs) if window[e] >= G)
        r = lr_final / lr
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: 1.0 if e < full else
                                                  r + (1 - r) * 0.5 * (1 + math.cos(math.pi * (e - full) / max(epochs - full, 1))))
    else:
        raise ValueError(f"schedule {schedule!r}")

    def full_mse():
        with torch.no_grad():
            theta = to_theta(z)
            return torch.nan_to_num(problem.loss(theta), nan=math.inf), theta.clone()

    for epoch in range(epochs):
        if history is not None and epoch % history_every == 0:
            history.append((epoch, *full_mse()))
        opt.zero_grad()
        per_restart = problem.loss(to_theta(z), window[epoch], kind=loss)
        # summing keeps the restarts' gradients independent; a non-finite restart stops moving
        torch.where(torch.isfinite(per_restart), per_restart, torch.zeros_like(per_restart)).sum().backward()
        z.grad = torch.nan_to_num(z.grad, nan=0.0, posinf=0.0, neginf=0.0)
        opt.step()
        if box == "clamp":
            with torch.no_grad():
                z.copy_(torch.maximum(torch.minimum(z, problem.hi), problem.lo))
        elif box == "unit":
            with torch.no_grad():
                z.clamp_(0.0, 1.0)
        sched.step()
    final, theta = full_mse()
    if history is not None:
        history.append((epochs, final, theta))
    with torch.no_grad():  # the estimate is the restart with the lowest full-record MSE, scored in double precision
        theta = torch.maximum(torch.minimum(theta.double().cpu(), exact.hi), exact.lo)
        final = torch.nan_to_num(exact.loss(theta), nan=math.inf)
    if return_all:
        return theta, final
    return theta[int(final.argmin())], final


def fit_laplace(problem: Problem, theta_map: torch.Tensor, n_draws: int = 4000, seed: int = 0):
    """Laplace approximation at the fit: Gaussian draws in z from the inverse Hessian of the log posterior
    (eigenvalues floored at the prior's curvature, 0.25)."""
    with torch.no_grad():
        noise_var = ((problem.predict(theta_map) - problem.y) / problem.range).pow(2).mean(0)
    z_map = problem.to_z(theta_map).detach()

    def neg_log_post(z):
        r = (problem.predict(problem.from_z(z)) - problem.y) / problem.range
        log_prior = (torch.nn.functional.logsigmoid(z) + torch.nn.functional.logsigmoid(-z)).sum()
        return 0.5 * (r**2 / noise_var).sum() - log_prior

    hess = torch.nan_to_num(torch.autograd.functional.hessian(neg_log_post, z_map))
    try:
        evals, evecs = torch.linalg.eigh(0.5 * (hess + hess.T))
    except torch.linalg.LinAlgError:  # badly conditioned (unknowns at the box edge): LAPACK's slower, robust driver
        import scipy.linalg

        w, v = scipy.linalg.eigh((0.5 * (hess + hess.T)).numpy(), driver="ev")
        evals, evecs = torch.from_numpy(w), torch.from_numpy(v)
    chol = evecs / evals.clamp_min(0.25).sqrt()
    eps = torch.randn(n_draws, len(z_map), generator=torch.Generator().manual_seed(seed), dtype=z_map.dtype)
    return problem.from_z(z_map + eps @ chol.T), {"negative_hessian_eigs": int((evals < 0).sum())}


def fit_cms(problem: Problem, restarts: int = 64, epochs: int = 500, windows: int = 5, final_fraction: float = 0.1,
            continuity: float = 1.0, lr: float = 0.05, lr_final: float = 2.5e-4, seed: int = 0, loss: str = "huber",
            return_all: bool = False):
    """Curriculum multiple shooting (Persson et al. 2026), not used in the paper: ``windows`` windows with their
    own starts and a continuity penalty merge stage by stage into single shooting (the last ``final_fraction``)."""
    G = problem.t.numel()
    assert (G - 1) % windows == 0, "frames - 1 must be divisible by the number of windows"
    seg = (G - 1) // windows
    start = [i for i, n in enumerate(problem.names) if n.endswith("(0)")]
    # window starts: the problem's start box, observed species re-centred on that window's first frame
    lo_u = problem.lo[start].expand(windows - 1, -1).clone()
    hi_u = problem.hi[start].expand(windows - 1, -1).clone()
    state_of = _start_states(problem)
    for c, state in enumerate(problem.observed_state):
        k = state_of.index(state)
        for w in range(1, windows):
            y = float(problem.y[w * seg, c])
            rng = float(problem.range[c])
            lo_u[w - 1, k], hi_u[w - 1, k] = math.log(max(y - 0.1 * rng, 1e-3)), math.log(y + 0.1 * rng)
    g = torch.Generator().manual_seed(seed)
    z = problem.to_z(problem.sample_prior(restarts, g)).requires_grad_(True)
    u0 = lo_u + torch.rand(restarts, windows - 1, len(start), generator=g, dtype=torch.float64) * (hi_u - lo_u)
    zu = torch.logit(((u0 - lo_u) / (hi_u - lo_u)).clamp(1e-9, 1 - 1e-9)).requires_grad_(True)
    opt = torch.optim.Adam([z, zu], lr=lr)
    stage_epochs = [round((1 - final_fraction) * epochs / (windows - 1))] * (windows - 1)
    stage_epochs.append(epochs - sum(stage_epochs))
    full = epochs - stage_epochs[-1]
    r = lr_final / lr
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: 1.0 if e < full else
                                              r + (1 - r) * 0.5 * (1 + math.cos(math.pi * (e - full) / max(epochs - full, 1))))
    t_rel = problem.t - problem.t[0]
    for s, n_ep in enumerate(stage_epochs, start=1):
        k, length = windows - s + 1, s * seg + 1  # windows at this stage and frames per window
        for _ in range(n_ep):
            theta = problem.from_z(z)
            if k == 1:
                per = problem.loss(theta, kind=loss)
            else:
                starts = torch.cat([theta[:, None, start], (lo_u + torch.sigmoid(zu) * (hi_u - lo_u))[:, :k - 1]], 1)
                theta_w = theta[:, None, :].expand(-1, k, -1).clone()
                theta_w[..., start] = starts
                p, y0 = problem.params_and_start(theta_w)
                traj = solve(lambda _t, y: R.rhs(y, p, problem.mask), y0, t_rel[:length])  # (L, B, k, 15)
                pred = traj[..., problem.observed_state].movedim(0, -2)  # (B, k, L, C)
                y = torch.stack([problem.y[w * seg: w * seg + length] for w in range(k)])  # (k, L, C)
                res = (pred - y) / problem.range
                data = torch.nn.functional.huber_loss(res, torch.zeros_like(res), reduction="none", delta=0.05)
                log_x = torch.log(traj.clamp_min(1e-9))
                gap = log_x[seg:, :, :-1] - log_x[: length - seg, :, 1:]  # window w at its frames >= seg vs window w+1
                per = data.mean((-1, -2, -3)) + continuity * gap.pow(2).mean((0, 2, 3))
            opt.zero_grad()
            torch.where(torch.isfinite(per), per, torch.zeros_like(per)).sum().backward()
            for v in (z, zu):
                if v.grad is not None:
                    v.grad = torch.nan_to_num(v.grad, nan=0.0, posinf=0.0, neginf=0.0)
            opt.step()
            sched.step()
    with torch.no_grad():
        theta = problem.from_z(z)
        final = torch.nan_to_num(problem.loss(theta), nan=math.inf)
    if return_all:
        return theta, final
    return theta[int(final.argmin())], final


def _start_states(problem: Problem) -> list[int]:
    """State index of each start coordinate, in ``problem.names`` order."""
    out = list(range(6))
    for j in problem.present:
        out += [R.R_IDX[j], R.D_IDX[j], R.F_IDX[j]]
    return out
