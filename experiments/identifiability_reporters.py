"""Which parameter combinations do the reporter frames actually determine?

    python experiments/identifiability_reporters.py [channels]

channels: comma-separated names from reporter_sim.CHANNELS, e.g.
"lac,tet,cI" (the three reporters, default) or "TetR-fusion,tet". Reporter
plasmids built = the reporter channels chosen.

At the true parameters, the Fisher information of the frames about all
unknowns (log units) is J^T J / sigma^2, with J the sensitivity of every
predicted frame (divided by its channel's range, as in the loss) and sigma^2
the per-point noise variance in the same units. The starting state is a
nuisance: profiling it out (Schur complement) leaves the information about
the kinetic parameters alone. Its eigenvectors are the combinations; each
eigenvalue gives the 1-sigma uncertainty of its combination, 1/sqrt(lambda),
in log units (0.1 ~ 10%).
"""

import math
import sys

import torch

from differentiable_cell import reporters as R
from differentiable_cell.fit import Problem
from differentiable_cell.reporter_sim import CHANNELS, box1_truth, generate

torch.set_default_dtype(torch.float64)


def true_theta(problem: Problem, data) -> torch.Tensor:
    tr, s0 = data.truth, data.fine_state[0]
    true = {"alpha": tr["alpha"], "alpha_0": tr["alpha_0"], "n": tr["n"], "beta": tr["beta"]}
    for i, s in enumerate(["m_lacI", "m_tetR", "m_cI", "p_LacI", "p_TetR", "p_CI"]):
        true[f"{s}(0)"] = float(s0[i])
    for j in problem.present:
        r = R.REPORTERS[j]
        true.update({f"a_{r}": tr["a"][j], f"a0_{r}": tr["a_0"][j], f"r_{r}(0)": float(s0[R.R_IDX[j]]),
                     f"D_{r}(0)": float(s0[R.D_IDX[j]]), f"F_{r}(0)": float(s0[R.F_IDX[j]])})
    return torch.tensor([math.log(max(true[n], 1e-12)) for n in problem.names])


def profiled_fisher(problem: Problem, theta: torch.Tensor, noise_var: float):
    """Fisher information of the kinetic parameters with the starting state profiled out."""
    f = lambda th: (problem.predict(th) / problem.range).reshape(-1)
    J = torch.autograd.functional.jacobian(f, theta, vectorize=True, strategy="forward-mode")
    fisher = J.T @ J / noise_var
    kin = [i for i, n in enumerate(problem.names) if not n.endswith("(0)")]
    nui = [i for i, n in enumerate(problem.names) if n.endswith("(0)")]
    F_kk, F_kn, F_nn = fisher[kin][:, kin], fisher[kin][:, nui], fisher[nui][:, nui]
    schur = F_kk - F_kn @ torch.linalg.pinv(F_nn) @ F_kn.T
    return 0.5 * (schur + schur.T), [problem.names[i] for i in kin]


def describe(vec: torch.Tensor, names: list[str], top: int = 4) -> str:
    """A combination as its largest log-parameter components, e.g. '+0.71 log(alpha) -0.69 log(n)'."""
    order = vec.abs().argsort(descending=True)[:top]
    return "  ".join(f"{float(vec[i]):+.2f} log({names[i]})" for i in order if abs(vec[i]) > 0.05)


def panel(spec: str) -> tuple[list[int], list[int]]:
    """Channel names -> (channel indices, reporters built)."""
    names = [c[0] for c in CHANNELS]
    channels = [names.index(n) for n in spec.split(",")]
    return channels, [c for c in channels if c < 3]


def main() -> None:
    channels, present = panel(sys.argv[1] if len(sys.argv) > 1 else "lac,tet,cI")
    data = generate(box1_truth(), present, kind="ode", noise=True, seed=1)
    problem = Problem(t=data.t, y=data.y[:, channels], channels=channels, present=present,
                      tau=data.truth["tau"])
    theta = true_theta(problem, data)
    noise_var = float(problem.loss(theta))  # per-point variance of the residuals at the truth
    fisher, names = profiled_fisher(problem, theta, noise_var)

    evals, evecs = torch.linalg.eigh(fisher)
    print(f"channels {[CHANNELS[c][0] for c in channels]}, {data.t.numel()} frames, "
          f"noise variance {noise_var:.2e} (range units)\n")
    print("combination (eigenvector of the profiled Fisher)                      1-sigma (log units)")
    for k in reversed(range(len(evals))):
        sd = 1.0 / math.sqrt(max(float(evals[k]), 1e-300))
        print(f"  {describe(evecs[:, k], names):70s} {sd:10.3g}")

    # Per-parameter uncertainty needs the prior too: an unidentifiable direction
    # (alpha_0) would otherwise leak its unbounded variance into every marginal.
    # The log-uniform box has variance width^2 / 12 per coordinate.
    kin = [problem.names.index(n) for n in names]
    width = (problem.hi - problem.lo)[kin]
    cov = torch.linalg.inv(fisher + torch.diag(12.0 / width**2))
    print("\nsingle parameters (marginal 1-sigma with the prior box, log units ~ relative error)")
    for i, n in enumerate(names):
        print(f"  {n:10s} {math.sqrt(float(cov[i, i])):8.3g}   (prior box alone {float(width[i]) / math.sqrt(12):.3g})")


if __name__ == "__main__":
    main()
