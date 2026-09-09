"""Which parameters could a given experiment actually measure?

Fitting answers this one observation set at a time and takes tens of minutes
per answer.  The local (Fisher) version answers it for every candidate set at
once, in seconds, by linearising the observed trajectory around the true
parameters.

For parameters ``theta`` observed through species ``S`` at sampling times ``t``:

    J[t, s, i] = d x_s(t) / d log(theta_i)     (log, so errors come out relative)
    FIM        = J^T J / sigma^2               (summed over the observed s and t)
    Cov        = FIM^-1                        (Cramer-Rao lower bound)

``sqrt(Cov_ii)`` is then the best relative standard error any unbiased
estimator could achieve for ``theta_i`` from that experiment - a floor, not a
promise, since it ignores every nonlinearity and local minimum that makes real
fitting harder.  A parameter whose bound is above ~100% cannot be measured by
that experiment at all; a near-singular FIM means some *combination* of
parameters is unconstrained even when each looks fine alone.

Usage:
    python experiments/identifiability.py <clean-run-directory>
"""

import argparse
import math
from pathlib import Path

import torch

from differentiable_cell.data import RepressilatorState
from differentiable_cell.helpers import TAU_M_MIN
from differentiable_cell.model import Repressilator
from differentiable_cell.run import FIELDS, PARAM_FIELDS

from _common import integrate, load_run, params_from_dict, setup_logging

# Observation sets that correspond to experiments somebody could actually run.
CANDIDATES: dict[str, list[str]] = {
    "GFP only (the paper)": ["p_GFP"],
    "GFP + one fusion (TetR-YFP)": ["p_GFP", "p_TetR"],
    "two repressor fusions": ["p_LacI", "p_TetR"],
    "three repressor fusions": ["p_LacI", "p_TetR", "p_CI"],
    "three fusions + GFP": ["p_LacI", "p_TetR", "p_CI", "p_GFP"],
    "three mRNAs (smFISH / MS2)": ["m_lacI", "m_tetR", "m_cI"],
    "three mRNAs + three fusions": [
        "m_lacI", "m_tetR", "m_cI", "p_LacI", "p_TetR", "p_CI",
    ],
    "everything (not measurable)": list(FIELDS),
}


def jacobian(
    model: Repressilator,
    payload: dict,
    t: torch.Tensor,
    step: float = 1e-4,
) -> torch.Tensor:
    """d(trajectory) / d(log parameter), by central differences.

    Central differences rather than autograd because it needs only 2 solves
    per parameter and is immune to the solver's own gradient subtleties; the
    step is in log space so the result is already a relative sensitivity.
    """
    state = RepressilatorState(**payload["initial_state"], batch_size=())
    spec = payload["config"]["integration"]
    truth = payload["params"]

    columns = []
    for name in PARAM_FIELDS:
        shifted = []
        for sign in (+1, -1):
            values = dict(truth)
            values[name] = truth[name] * math.exp(sign * step)
            with torch.no_grad():
                traj = integrate(model, state, params_from_dict(values), t, spec)
            shifted.append(torch.stack([getattr(traj, f) for f in FIELDS], dim=-1))
        columns.append((shifted[0] - shifted[1]) / (2.0 * step))
    return torch.stack(columns, dim=-1)  # (time, species, parameter)


def analyse(
    J: torch.Tensor, observed: list[int], scales: torch.Tensor, sigma: float
) -> tuple[dict[str, float | None], float]:
    """Cramer-Rao relative standard errors, and the FIM condition number.

    A parameter the observed species do not respond to at all - alpha_GFP when
    no reporter is measured, say - would make the FIM singular and wrongly
    condemn every other parameter with it.  Those are separated out first and
    reported as having no effect, and the bound is computed on the rest.
    """
    # Normalise each species by its own range, so `sigma` means "this fraction
    # of the species' full swing" for every species alike.
    sensitivity = (J[:, observed, :] / scales[observed].unsqueeze(-1)).reshape(
        -1, len(PARAM_FIELDS)
    )
    norms = sensitivity.norm(dim=0)
    active = [
        i for i in range(len(PARAM_FIELDS))
        if norms[i] > 1e-10 * norms.max().clamp(min=1e-300)
    ]
    result: dict[str, float | None] = {p: None for p in PARAM_FIELDS}
    if not active:
        return result, float("inf")

    fim = sensitivity[:, active].T @ sensitivity[:, active] / (sigma**2)
    eigenvalues = torch.linalg.eigvalsh(fim)
    condition = float(eigenvalues[-1] / eigenvalues[0].clamp(min=1e-300))
    if eigenvalues[0] <= 0 or condition > 1e14:
        for i in active:
            result[PARAM_FIELDS[i]] = float("inf")
        return result, condition

    errors = torch.linalg.inv(fim).diagonal().sqrt()
    for slot, i in enumerate(active):
        result[PARAM_FIELDS[i]] = float(errors[slot])
    return result, condition


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--sigma", type=float, default=0.05,
        help="measurement noise, as a fraction of each species' range",
    )
    parser.add_argument(
        "--sample_min", type=float, default=5.0,
        help="sampling interval in minutes (time-lapse microscopy is ~5 min)",
    )
    args = parser.parse_args()
    setup_logging()

    payload = load_run(args.run_dir)
    torch.set_default_dtype(getattr(torch, payload["config"]["dtype"]))
    model = Repressilator()

    # Subsample to a realistic imaging cadence: information scales with the
    # number of frames, so using every solver output point would flatter every
    # experiment equally and unrealistically.
    full_t = payload["t"]
    stride = max(1, round(args.sample_min / TAU_M_MIN / float(full_t[1] - full_t[0])))
    t = full_t[::stride]

    clean = torch.stack([payload["trajectory"][f] for f in FIELDS], dim=-1)
    scales = (clean.max(0).values - clean.min(0).values).clamp(min=1e-8)
    J = jacobian(model, payload, t)

    print(
        f"\nSampling every {args.sample_min:g} min ({t.numel()} frames over "
        f"{float(t[-1]) * TAU_M_MIN:.0f} min), noise sigma = {args.sigma:.0%} "
        "of each species' range."
    )
    print(
        "Best achievable relative standard error per parameter "
        "(Cramer-Rao floor; '-' = unconstrained):\n"
    )
    header = "".join(f"{p:>11}" for p in PARAM_FIELDS)
    print(f"  {'experiment':<32}{header}   cond(FIM)")
    print("  " + "-" * (32 + 11 * len(PARAM_FIELDS) + 12))

    for label, species in CANDIDATES.items():
        observed = [FIELDS.index(s) for s in species]
        errors, condition = analyse(J, observed, scales, args.sigma)
        cells = ""
        for name in PARAM_FIELDS:
            e = errors[name]
            if e is None:
                cells += f"{'n/a':>11}"          # not reachable by this experiment
            elif not math.isfinite(e) or e >= 10:
                cells += f"{'>1000%':>11}"       # unconstrained
            else:
                cells += f"{e * 100:>10.2f}%"
        print(f"  {label:<32}{cells}   {condition:9.2e}")

    print(
        "\n  n/a    = the observed species do not respond to that parameter at "
        "all\n  >1000% = reachable in principle, but unconstrained in practice"
        "\n\nA floor above ~100% means the experiment cannot measure that "
        "parameter.\ncond(FIM) >> 1 means some *combination* is far less "
        "constrained than any\nsingle parameter suggests."
    )


if __name__ == "__main__":
    main()
