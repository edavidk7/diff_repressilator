"""ABC-SMC (Toni et al., 2009), parameter-estimation variant.

Likelihood-free: particles are proposed, simulated, and kept when the
simulated data fall within a tolerance ``eps_t`` of the observed data; the
tolerance shrinks over populations.  Algorithm S1-S3 of the paper:

* population 0 samples the prior;
* later populations resample the previous one by weight and perturb with a
  component-wise uniform random walk ``K_t = sigma_t U(-1, 1)``;
* a particle's weight is ``prior(theta) / sum_j w_j K_t(theta | theta_j)``,
  times - for a stochastic simulator - the fraction of its ``B_t``
  replicate simulations that were accepted.

Two choices the paper makes by hand per problem are automated here, since a
benchmark must not tune on the truth:

* ``sigma_t`` per coordinate gives the uniform kernel the variance of twice
  the previous population's weighted variance (Beaumont et al., 2009);
* ``eps_t`` is a quantile (default the median) of the previous population's
  distances (Del Moral et al., 2012).

The simulator is the ODE (batched RK4) or, for stochastic data, the CLE
(:mod:`.stochastic`).  Distances are those of :func:`.stochastic.distance`,
which for ODE data is the RMS standardised residual.
"""


import torch

from differentiable_cell.inference import stochastic
from differentiable_cell.inference.base import Result, Timer
from differentiable_cell.inference.problem import Problem


def _distances(problem: Problem, theta: torch.Tensor, simulator: str, replicates: int,
               n_cells: int, generator: torch.Generator) -> torch.Tensor:
    """``(B, replicates)`` distances for a batch of particles."""
    if simulator == "ode":
        with torch.no_grad():
            pred = problem.simulate(theta)
        d = ((pred - problem.y).pow(2) / problem.obs_variance()).mean((-1, -2)).sqrt()
        return torch.nan_to_num(d, nan=torch.inf).unsqueeze(-1)
    out = []
    for _ in range(replicates):
        with torch.no_grad():
            sim = stochastic.simulate(problem, theta, n_cells, kind="cle", generator=generator)
            out.append(torch.nan_to_num(stochastic.distance(problem, sim), nan=torch.inf))
    return torch.stack(out, dim=-1)


def fit(
    problem: Problem,
    n_particles: int = 1000,
    max_sims: int = 2_000_000,
    max_seconds: float | None = None,
    quantile: float = 0.5,
    min_acceptance: float = 0.005,
    simulator: str = "ode",
    replicates: int = 1,
    n_cells: int | None = None,
    batch: int = 2000,
    seed: int = 0,
) -> Result:
    """Run ABC-SMC until the simulation budget or the acceptance floor is hit.

    Args:
        problem: the inverse problem.
        n_particles: population size ``N``.
        max_sims: budget in simulated trajectories (a CLE particle with
            ``n_cells`` cells and ``B_t`` replicates costs ``n_cells * B_t``).
        max_seconds: optional wall-clock budget.
        quantile: ``eps_t`` = this quantile of the previous distances.
        min_acceptance: stop once a population's acceptance rate (accepted per
            simulated proposal) falls below.
        simulator: ``"ode"`` or ``"cle"``.
        replicates: ``B_t``, simulations per particle (stochastic simulator).
        n_cells: cells per simulation for the CLE (default: the problem's
            ``n_cells`` for ensemble data, 8 for a single-cell summary).
        batch: proposals simulated at once.
        seed: random seed.
    """
    g = torch.Generator().manual_seed(seed)
    if n_cells is None:
        n_cells = min(problem.n_cells, 100) if problem.regime == "ensemble" else 8
    lo, hi = problem.bounds.unbind(-1)
    history = []
    stop_reason = "budget"

    def over_budget() -> bool:
        return (problem.n_sims - timer.sims0 >= max_sims) or (
            max_seconds is not None and timer.elapsed() > max_seconds)

    with Timer(problem) as timer:
        # Population 0: the prior, all accepted (eps_0 = infinity).
        theta = problem.sample_prior(n_particles, g)
        d = _distances(problem, theta, simulator, replicates, n_cells, g).mean(-1)
        weights = torch.full((n_particles,), 1.0 / n_particles, dtype=theta.dtype)
        eps = float("inf")
        history.append({"eps": eps, "acceptance": 1.0, "sims": problem.n_sims - timer.sims0})

        while not over_budget():
            finite = torch.isfinite(d)
            new_eps = float(torch.quantile(d[finite], quantile)) if finite.any() else eps
            if not new_eps < eps:
                stop_reason = "tolerance stalled"
                break
            mean = (weights.unsqueeze(-1) * theta).sum(0)
            var = (weights.unsqueeze(-1) * (theta - mean) ** 2).sum(0)
            half_width = (3.0 * 2.0 * var).sqrt().clamp_min(1e-9)  # U(-h,h) var = h^2/3

            accepted_theta, accepted_d, accepted_frac = [], [], []
            n_acc, n_prop = 0, 0
            while n_acc < n_particles and not over_budget():
                # Proposals outside the prior box have zero prior mass and are
                # rejected before simulating (they cost nothing but a redraw).
                pieces, have = [], 0
                for _ in range(1000):
                    idx = torch.multinomial(weights, batch, replacement=True, generator=g)
                    cand = theta[idx] + half_width * (
                        2 * torch.rand(batch, problem.dim, generator=g, dtype=theta.dtype) - 1)
                    cand = cand[((cand >= lo) & (cand <= hi)).all(-1)]
                    pieces.append(cand)
                    have += cand.shape[0]
                    if have >= batch:
                        break
                prop = torch.cat(pieces)[:batch]
                if prop.shape[0] == 0:
                    continue
                n_prop += prop.shape[0]
                dd = _distances(problem, prop, simulator, replicates, n_cells, g)
                frac = (dd <= new_eps).double().mean(-1)
                keep = frac > 0
                accepted_theta.append(prop[keep])
                accepted_d.append(dd[keep].mean(-1))
                accepted_frac.append(frac[keep])
                n_acc += int(keep.sum())
            if n_acc < n_particles:
                stop_reason = "budget (population incomplete)"
                break
            new_theta = torch.cat(accepted_theta)[:n_particles]
            new_d = torch.cat(accepted_d)[:n_particles]
            frac = torch.cat(accepted_frac)[:n_particles]
            # w_i ∝ prior / sum_j w_j K(theta_i | theta_j); prior uniform on the box.
            inside_kernel = ((new_theta.unsqueeze(1) - theta.unsqueeze(0)).abs()
                             <= half_width).all(-1).double()
            kernel_density = inside_kernel / (2 * half_width).prod()
            denom = (kernel_density * weights.unsqueeze(0)).sum(-1)
            new_w = frac / denom
            weights = new_w / new_w.sum()
            theta, d, eps = new_theta, new_d, new_eps
            acceptance = n_acc / max(n_prop, 1)
            history.append({"eps": eps, "acceptance": acceptance,
                            "sims": problem.n_sims - timer.sims0,
                            "ess": float(1.0 / (weights**2).sum())})
            if acceptance < min_acceptance:
                stop_reason = "acceptance floor"
                break

    idx = torch.multinomial(weights, 4000, replacement=True, generator=g)
    samples = theta[idx]
    return Result(
        method=f"abc_smc[{simulator}]",
        theta=samples.median(0).values,
        samples=samples,
        n_sims=timer.n_sims,
        wall=timer.wall,
        info={"populations": history, "final_eps": eps, "stop": stop_reason,
              "final_ess": float(1.0 / (weights**2).sum())},
    )
