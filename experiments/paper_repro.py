"""Each dedicated method in its own paper's setting, next to backprop.

    python experiments/paper_repro.py toni       # ABC-SMC, Toni et al. 2009
    python experiments/paper_repro.py pullen     # nested sampling, Pullen & Morris 2014
    python experiments/paper_repro.py casajuana  # inverse PINN, Casajuana et al. 2026

The benchmark poses a harder problem than any of these papers (all eight
initial values unknown, 4-6 kinetic parameters, sparse frames).  These runs
separate "the method is weak in this setting" from "the implementation is
weak": if a method reproduces its own paper's headline result here, its poor
showing in the benchmark is about the setting.

Each setting follows its paper wherever the paper is specific, and records
where it had to choose (``assumptions`` in the output).  Priors stay
log-uniform on boxes approximating the papers' uniform priors (whose
negative lower bounds are meaningless for rate constants).  Results go to
``run_results/paper_repro/<paper>.json``.
"""

import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark import run_method  # noqa: E402

from differentiable_cell.data import RepressilatorParams, RepressilatorState
from differentiable_cell.inference.problem import Case, Problem
from differentiable_cell.inference.scoring import score
from differentiable_cell.model import Repressilator
from differentiable_cell.run import FIELDS, PARAM_FIELDS, run_torchdiffeq

OUT = Path(__file__).resolve().parent.parent / "run_results" / "paper_repro"


def _simulate_truth(params: dict, y0: dict, t: torch.Tensor) -> torch.Tensor:
    p = RepressilatorParams(**{k: torch.tensor(float(params[k])) for k in PARAM_FIELDS},
                            batch_size=())
    s = RepressilatorState(**{f: torch.tensor(float(y0[f])) for f in FIELDS}, batch_size=())
    with torch.no_grad():
        traj = run_torchdiffeq(Repressilator(), s, p, t)
    return torch.stack([getattr(traj, f) for f in FIELDS], -1)


def _case(params, y0, t, observed, free, known_ic, priors, sigma_abs=None, noise_frac=None,
          ic_prior=None, rk4_dt=0.05, seed=0, meta=None) -> Case:
    g = torch.Generator().manual_seed(seed)
    clean = _simulate_truth(params, y0, t)
    idx = [FIELDS.index(o) for o in observed]
    signal = clean[:, idx]
    if sigma_abs is not None:
        sigma = torch.full((len(observed),), float(sigma_abs))
    else:
        sigma = noise_frac * (signal.max(0).values - signal.min(0).values)
    y = signal + sigma * torch.randn(signal.shape, generator=g)
    problem = Problem(t=t, y=y, observed=list(observed), free_params=list(free),
                      fixed_params={k: v for k, v in params.items() if k not in free},
                      sigma=sigma, known_ic=dict(known_ic), priors=priors, ic_prior=ic_prior,
                      rk4_dt=rk4_dt)
    lo, hi = problem.bounds.unbind(-1)
    theta = torch.tensor([math.log(params[p]) for p in free]
                         + [math.log(max(y0[f], 1e-12)) for f in problem.ic_fields]).clamp(lo, hi)
    truth = dict(params)
    truth.update({f"{f}(0)": float(y0[f]) for f in FIELDS})
    return Case(problem=problem, truth=truth, truth_theta=theta, clean=clean,
                meta={"panel": "+".join(observed), "regime": "ode", "seed": seed, **(meta or {})})


def toni(seed: int = 0) -> tuple[Case, dict]:
    """Toni et al. 2009, section 3.2.1: deterministic repressilator, mRNA only."""
    params = dict(alpha=1000.0, alpha_0=1.0, n=2.0, beta=5.0, alpha_GFP=1.0, beta_GFP=0.1)
    y0 = dict(m_lacI=0.0, m_tetR=0.0, m_cI=0.0, p_LacI=2.0, p_TetR=1.0, p_CI=3.0,
              m_gfp=0.0, p_GFP=0.0)
    t = torch.linspace(0, 30, 21)
    case = _case(params, y0, t, observed=["m_lacI", "m_tetR", "m_cI"],
                 free=["alpha_0", "n", "beta", "alpha"], known_ic=y0,
                 priors={"alpha_0": (1e-2, 10.0), "n": (0.1, 10.0), "beta": (0.05, 20.0),
                         "alpha": (500.0, 2500.0)},
                 sigma_abs=5.0, rk4_dt=0.025, seed=seed, meta={"paper": "toni"})
    assumptions = {
        "time_grid": "21 points on [0, 30] (not stated in the main text)",
        "priors": "log-uniform on [1e-2,10], [0.1,10], [0.05,20], [500,2500] vs the paper's "
                  "U(-2,10), U(0,10), U(-5,20), U(500,2500)",
        "paper_result": "n inferred best (narrowest posterior), alpha barely inferable",
    }
    return case, assumptions


def pullen(seed: int = 0) -> tuple[Case, dict]:
    """Pullen & Morris 2014, Table 2: 26 noisy points of CI protein."""
    params = dict(alpha=125.0, alpha_0=0.0, n=2.0, beta=2.0, alpha_GFP=1.0, beta_GFP=0.1)
    y0 = dict(m_lacI=0.0, m_tetR=0.0, m_cI=0.0, p_LacI=5.0, p_TetR=0.0, p_CI=15.0,
              m_gfp=0.0, p_GFP=0.0)
    t = torch.linspace(0, 50, 26)
    case = _case(params, y0, t, observed=["p_CI"], free=["alpha", "beta"],
                 known_ic={"p_CI": 15.0, "m_gfp": 0.0, "p_GFP": 0.0},
                 priors={"alpha": (1.0, 1000.0), "beta": (0.01, 100.0)},
                 ic_prior=(1e-2, 50.0), noise_frac=0.10, rk4_dt=0.01, seed=seed,
                 meta={"paper": "pullen"})
    assumptions = {
        "time_units": "the paper's '2-minute intervals for 50 minutes' read as model time",
        "priors": "log-uniform alpha [1,1000], beta [0.01,100], ICs [0.01,50] vs the paper's "
                  "U(0,1000), U(0,100), U(0,50)",
        "paper_result": "alpha 128.47 +/- 5.88, beta 2.02 +/- 0.05 (truth 125, 2)",
    }
    return case, assumptions


# --------------------------------------------------------------------------
# Casajuana et al.: a different, protein-only model, so it gets its own code
# --------------------------------------------------------------------------


def _protein_rhs(x, beta, n):
    rep = x[..., [2, 0, 1]].clamp_min(0.0)
    return beta / (1 + rep**n) - x


def casajuana(noise: float = 0.01, seed: int = 0, iterations: int = 5000) -> dict:
    """Section 2 of the paper: 3-protein model, beta = 5, n = 3, 1000 points on [0, 20].

    Runs the paper's inverse PINN and, on the same data, Adam through an RK4
    solve of the same ODE (the backprop baseline).  Both estimate (beta, n).
    """
    from differentiable_cell.inference.pinn import _network

    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    beta_true, n_true = 5.0, 3.0
    x0 = torch.tensor([1.0, 1.0, 1.2])
    t = torch.linspace(0, 20, 1000)
    h = float(t[1] - t[0])

    def solve(beta, n, x_init, steps_per=1):
        xs, x = [x_init], x_init
        hh = h / steps_per
        for _ in range((t.numel() - 1) * steps_per):
            k1 = _protein_rhs(x, beta, n)
            k2 = _protein_rhs(x + 0.5 * hh * k1, beta, n)
            k3 = _protein_rhs(x + 0.5 * hh * k2, beta, n)
            k4 = _protein_rhs(x + hh * k3, beta, n)
            x = x + hh / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            xs.append(x)
        return torch.stack(xs)[::steps_per]

    with torch.no_grad():
        clean = solve(torch.tensor(beta_true), torch.tensor(n_true), x0, steps_per=4)
    amp = (clean.max(0).values - clean.min(0).values).mean()
    y = clean + noise * amp * torch.randn(clean.shape, generator=g)
    out = {"noise": noise, "seed": seed, "truth": {"beta": beta_true, "n": n_true}}

    # Inverse PINN as in the paper: 5 x 100 sine MLP, softplus output, Adam 1e-3,
    # residual + initial-condition + data terms with unit weights.
    t0 = time.perf_counter()
    net = _network(100, 5, 3)
    raw = torch.tensor([math.log(1.0), math.log(1.0)], requires_grad=True)  # init beta=n=1
    opt = torch.optim.Adam([{"params": net.parameters()}, {"params": [raw]}], lr=1e-3)
    # Raw time as the network input, as DeepXDE does.
    for _ in range(iterations):
        tc = float(t[-1]) * torch.rand(1000, generator=g)
        u, du = torch.func.jvp(lambda s: torch.nn.functional.softplus(net(s.unsqueeze(-1))),
                               (tc,), (torch.ones_like(tc),))
        beta, n = raw.exp()
        res = (du - _protein_rhs(u, beta, n)).pow(2).mean()
        u_obs = torch.nn.functional.softplus(net(t.unsqueeze(-1)))
        loss = res + (u_obs[0] - y[0]).pow(2).sum() + (u_obs - y).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    beta, n = (float(v) for v in raw.detach().exp())
    out["pinn"] = {"beta": beta, "n": n, "rel_err_beta": abs(beta - beta_true) / beta_true,
                   "rel_err_n": abs(n - n_true) / n_true, "wall": time.perf_counter() - t0}

    # Backprop baseline, same recipe as grad_ode: Adam (first order) through
    # RK4 from a batch of restarts drawn log-uniformly from a prior box
    # (beta in [0.5, 50], n in [1, 5]), fitting beta, n and x(0); the best
    # restart by loss wins.  A single start at beta = n = 1 (the PINN's
    # initial guess) sits at a stable steady state with a flat loss and
    # stalls ~1000x above the truth's loss.
    t0 = time.perf_counter()
    restarts, epochs = 16, 800
    lo = torch.tensor([math.log(0.5), math.log(1.0)])
    hi = torch.tensor([math.log(50.0), math.log(5.0)])
    z = (lo + torch.rand(restarts, 2, generator=g) * (hi - lo)).requires_grad_(True)
    x_init = y[0].clamp_min(1e-3).log().expand(restarts, 3).clone().requires_grad_(True)
    opt = torch.optim.Adam([z, x_init], lr=0.05)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: 0.01 ** (e / epochs))
    for _ in range(epochs):
        beta, n = z.exp().unbind(-1)
        pred = solve(beta.unsqueeze(-1), n.unsqueeze(-1), x_init.exp())  # (G, R, 3)
        per = (pred - y.unsqueeze(1)).pow(2).mean((0, 2))
        per = torch.nan_to_num(per, nan=1e6)
        opt.zero_grad()
        per.sum().backward()
        z.grad = torch.nan_to_num(z.grad)
        x_init.grad = torch.nan_to_num(x_init.grad)
        opt.step()
        sched.step()
    best = int(per.detach().argmin())
    beta, n = (float(v) for v in z.detach()[best].exp())
    out["grad_ode"] = {"beta": beta, "n": n, "rel_err_beta": abs(beta - beta_true) / beta_true,
                       "rel_err_n": abs(n - n_true) / n_true, "wall": time.perf_counter() - t0,
                       "restarts": restarts, "epochs": epochs}
    out["paper_result"] = "mean relative error of (beta, n): 8.3e-3 at 1% noise, 9.2e-3 at 10%"
    out["assumptions"] = {"iterations": iterations, "loss_weights": "all 1 (not stated)",
                          "init": "beta = n = 1 (the paper's initial-guess study varies it)"}
    return out


METHODS = {"toni": ["grad_ode", "nuts", "abc_ode", "nested", "magi", "pinn"],
           "pullen": ["grad_ode", "nuts", "abc_ode", "magi", "pinn"]}  # nested dropped


def main() -> None:
    which = sys.argv[1]
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(int(sys.argv[2]) if len(sys.argv) > 2 else 3)
    OUT.mkdir(parents=True, exist_ok=True)
    if which == "casajuana":
        rows = [casajuana(noise=nz, seed=s) for nz in (0.01, 0.05, 0.10) for s in (0, 1, 2)]
        (OUT / "casajuana.json").write_text(json.dumps(rows, indent=2))
        for r in rows:
            print(r["noise"], r["seed"], "pinn", round(r["pinn"]["rel_err_beta"], 4),
                  round(r["pinn"]["rel_err_n"], 4), "grad", round(r["grad_ode"]["rel_err_beta"], 4),
                  round(r["grad_ode"]["rel_err_n"], 4), flush=True)
        return
    case, assumptions = {"toni": toni, "pullen": pullen}[which]()
    path = OUT / f"{which}.json"
    results = json.loads(path.read_text()) if path.exists() else {"assumptions": assumptions}
    cache: dict = {}
    for m in METHODS[which]:
        if m in results:
            continue
        t0 = time.time()
        try:
            res = run_method(m, case.problem, 0, cache, 3)
            row = score(case, res)
            row["info"] = {k: v for k, v in res.info.items() if not isinstance(v, list) or len(v) < 50}
        except Exception as exc:
            import traceback
            row = {"error": repr(exc), "traceback": traceback.format_exc()}
        results[m] = row
        path.write_text(json.dumps(results, indent=2, default=float))
        p = row.get("params", {})
        print(which, m, round(time.time() - t0), {k: (round(v["estimate"], 3), round(v["rel_err"], 3),
                                                      v.get("covered")) for k, v in p.items()},
              flush=True)


if __name__ == "__main__":
    main()
