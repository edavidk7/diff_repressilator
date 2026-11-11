"""Baseline: truncated sequential neural posterior estimation (TSNPE, Deistler et al. 2022) with the `sbi` toolbox
(Tejero-Cantero et al. 2020). A neural spline flow on a 1-D CNN embedding of the frames is trained by maximum
likelihood (the NPE loss of Greenberg et al. 2019 without atoms: truncated proposals need no correction); each round
simulates from the prior restricted to the 1 - eps highest-density region of the current posterior at the observed
cell. The simulator is the ODE plus Gaussian noise, in range-normalised units; the prior is uniform on the log box.
"""

import tempfile
import time

import torch
from sbi.inference import NPE
from sbi.neural_nets import posterior_nn
from sbi.utils import BoxUniform
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from differentiable_cell.fit import Problem


def noise_sd(problem: Problem) -> torch.Tensor:
    """Per-channel noise sd as a fraction of range, from second differences (var = 6 sd^2 for white noise)."""
    return (problem.y.diff(n=2, dim=0).std(0) / 6**0.5 / problem.range).clamp_min(1e-3)


class Conv(nn.Module):
    """Strided 1-D convolution as unfold + matmul (much faster than nn.Conv1d on CPU here)."""

    def __init__(self, inputs: int, outputs: int, kernel: int, stride: int):
        super().__init__()
        self.kernel, self.stride, self.linear = kernel, stride, nn.Linear(inputs * kernel, outputs)

    def forward(self, x):  # (B, C, L) -> (B, O, L')
        patches = x.unfold(-1, self.kernel, self.stride).permute(0, 2, 1, 3).flatten(2)
        return self.linear(patches).transpose(1, 2)


class Embedding(nn.Module):
    """1-D CNN over time: (B, channels, frames) -> (B, features)."""

    def __init__(self, channels: int, frames: int, features: int, kernel: int = 5, stride: int = 2):
        super().__init__()
        layers, width, length = [], channels, frames
        for out in (32, 64, 64):
            layers += [Conv(width, out, kernel, stride), nn.ReLU()]
            width, length = out, (length - kernel) // stride + 1
        self.net = nn.Sequential(*layers, nn.Flatten(), nn.Linear(width * length, 128), nn.ReLU(),
                                 nn.Linear(128, features))

    def forward(self, x):
        # sbi repeats the observation once per theta when sampling or scoring: embed it once.
        if not self.training and len(x) > 1 and bool((x == x[:1]).all()):
            return self.net(x[:1]).expand(len(x), -1)
        return self.net(x)


def log_density(posterior, theta, batch: int = 20_000):
    with torch.no_grad():
        return torch.cat([posterior.log_prob(c, norm_posterior=False) for c in theta.split(batch)])


def density_threshold(posterior, epsilon: float, n: int) -> float:
    """Log density bounding the 1 - epsilon highest-density region of the posterior (Deistler et al., eq. 3)."""
    log_q = log_density(posterior, posterior.sample((n,), show_progress_bars=False))
    return float(log_q.sort().values[int(epsilon * n)])


def sample_restricted_prior(prior, posterior, threshold: float, n: int, rejection: bool, candidates: int,
                            batch: int = 20_000):
    """Draws from the prior restricted to {log q > threshold}: by rejection from the prior, or, when that region
    is too small a part of the box, by resampling posterior draws with weights prior / q (SIR)."""
    kept, have = [], 0
    while have < n:
        if rejection:
            theta = prior.sample((batch,))
            theta = theta[log_density(posterior, theta) > threshold]
        else:
            groups = min(n - have, batch // candidates)
            cand = posterior.sample((groups * candidates,), show_progress_bars=False)
            log_q = log_density(posterior, cand)
            log_w = torch.where(log_q > threshold, -log_q, torch.tensor(-torch.inf)).reshape(groups, candidates)
            ok = torch.isfinite(log_w).any(-1)  # groups with a candidate inside the region
            pick = torch.multinomial(log_w[ok].softmax(-1), 1)[:, 0]
            theta = cand.reshape(groups, candidates, -1)[ok, pick]
        kept.append(theta)
        have += len(theta)
    return torch.cat(kept)[:n]


def simulate(problem: Problem, theta: torch.Tensor, sd: torch.Tensor, batch: int = 2000):
    """Noisy range-normalised frames (n, k, G) of the finite simulations, and the mask of those kept."""
    out = []
    with torch.no_grad():
        for chunk in theta.double().split(batch):
            y = problem.predict(chunk) / problem.range
            out.append((y + sd * torch.randn_like(y)).transpose(1, 2).float())
    x = torch.cat(out)
    keep = torch.isfinite(x).all((1, 2))
    return x[keep], keep


def _fit_snpe(problem: Problem, seed: int = 0, rounds: int = 10, sims_per_round: int = 30_000,
             epsilon: float = 1e-4, support_samples: int = 100_000, min_prior_acceptance: float = 0.05,
             sir_candidates: int = 32, max_epochs: int = 15, stop_after_epochs: int = 5, batch_size: int = 1000,
             learning_rate: float = 5e-4, flow: str = "nsf", features: int = 48, hidden: int = 64, transforms: int = 5,
             n_draws: int = 4000, threads: int | None = None, verbose: bool = False):
    """Returns ``(theta, info)``: the posterior draw of lowest ``problem.loss`` among ``n_draws``, and a dict with
    the simulation count, timings, the posterior median (``theta_median``) and the draws (``draws``).
    The budget is ``rounds * sims_per_round`` simulations; the ``n_draws`` read-out integrations are counted apart.
    """
    if threads is not None:
        torch.set_num_threads(threads)
    torch.manual_seed(seed)  # sbi draws from the global generator
    start = time.time()
    lo, hi = problem.lo.float(), problem.hi.float()
    prior = BoxUniform(lo, hi)
    sd = noise_sd(problem)
    x_o = (problem.y / problem.range).T.float()[None]  # (1, k, G)
    k, frames = x_o.shape[1:]

    net = posterior_nn(flow, embedding_net=Embedding(k, frames, features), hidden_features=hidden,
                       num_transforms=transforms, z_score_x="independent")
    writer = SummaryWriter(tempfile.mkdtemp(prefix="sbi-"))  # sbi would otherwise log into ./sbi-logs
    inference = NPE(prior=prior, density_estimator=net, show_progress_bars=False, summary_writer=writer)

    sims, dropped, log = 0, 0, []
    posterior, threshold, sampler = None, 0.0, "none"
    seconds = {"simulate": 0.0, "train": 0.0, "propose": 0.0}
    for r in range(rounds):
        t0 = time.time()
        if r == 0:
            theta = prior.sample((sims_per_round,))
        else:
            theta = sample_restricted_prior(prior, posterior, threshold, sims_per_round, sampler == "rejection",
                                            sir_candidates)
        theta = theta.detach()
        t1 = time.time()
        x, keep = simulate(problem, theta, sd)
        sims += len(theta)
        dropped += int((~keep).sum())  # non-finite simulations are dropped
        t2 = time.time()

        # No proposal is passed: sbi then uses the maximum-likelihood loss, which is what TSNPE needs.
        inference.append_simulations(theta[keep], x, exclude_invalid_x=False)
        estimator = inference.train(force_first_round_loss=True, training_batch_size=batch_size,
                                    learning_rate=learning_rate, max_num_epochs=max_epochs,
                                    stop_after_epochs=stop_after_epochs)
        posterior = inference.build_posterior(estimator).set_default_x(x_o)
        t3 = time.time()

        acceptance, sampler = float("nan"), "none"
        if r < rounds - 1:
            threshold = density_threshold(posterior, epsilon, support_samples)
            acceptance = float((log_density(posterior, prior.sample((20_000,))) > threshold).float().mean())
            sampler = "rejection" if acceptance >= min_prior_acceptance else "sir"
        t4 = time.time()
        seconds["propose"] += (t1 - t0) + (t4 - t3)
        seconds["simulate"] += t2 - t1
        seconds["train"] += t3 - t2
        log.append({"round": r, "epochs": int(inference.summary["epochs_trained"][-1]),
                    "validation_loss": float(inference.summary["best_validation_loss"][-1]),
                    "prior_acceptance": acceptance, "next_sampler": sampler})
        if verbose:
            print(log[-1], f"{time.time() - start:.0f} s", flush=True)

    # Read-out: the draw that fits the frames best when integrated (the most favourable one), and the median.
    with torch.no_grad():
        draws = posterior.sample((n_draws,), show_progress_bars=False).double()
        draws = torch.minimum(torch.maximum(draws, problem.lo), problem.hi)
        loss = torch.cat([problem.loss(c) for c in draws.split(2000)])
        loss = torch.nan_to_num(loss, nan=float("inf"))
        theta = draws[loss.argmin()]
        median = draws.median(0).values
        median_loss = float(torch.nan_to_num(problem.loss(median[None]), nan=float("inf")))
    info = {"simulations": sims, "readout_simulations": n_draws + 1, "dropped_simulations": dropped,
            "rounds": rounds, "seconds": time.time() - start, "seconds_simulate": seconds["simulate"],
            "seconds_train": seconds["train"], "seconds_propose": seconds["propose"],
            "noise_sd": sd.tolist(), "loss": float(loss.min()), "loss_median": median_loss,
            "loss_draws_median": float(loss.median()), "theta_median": median.tolist(), "history": log,
            "draws": draws}
    return theta, info


def fit_snpe(problem: Problem, seed: int = 0, **kwargs):
    """``_fit_snpe`` with single precision as the default dtype (sbi's networks need it; the benchmark runs in double)."""
    dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        return _fit_snpe(problem, seed, **kwargs)
    finally:
        torch.set_default_dtype(dtype)
