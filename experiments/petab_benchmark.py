"""Our method on PEtab benchmark problems (Hass et al. 2019; github.com/Benchmarking-Initiative/Benchmark-Models-PEtab).

    python experiments/petab_benchmark.py --check            each model against the benchmark's reference simulation
    python experiments/petab_benchmark.py --workers 4         fit (resumes)
    python experiments/petab_benchmark.py --report

Elowitz_Nature2000: the repressilator with a GFP reporter, one fluorescence time course (58 points), 21 parameters on
log10 scale in [1e-5, 1e3], log10-normal noise with sigma estimated. The objective is the benchmark's negative
log-likelihood; we report the best value found and the restarts within 0.1 of the best known value (the reference
parameters' NLL). The model is stiff (decay rates up to ln2 / 1e-5 per minute), so instead of RK4 we integrate with an
exponential (ETD2) scheme that is exact for the linear terms (decay and translation); everything else is fit_gradient's recipe.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from common import RESULTS, setup_torch

OUT = RESULTS / "petab"
SRC = Path("/tmp/petab/bm/problems")  # sparse clone of the benchmark collection (see the docstring)
LN10 = math.log(10.0)
SOLVER = "etd2"  # or "ros2"; set by --solver


# ------------------------------------------------------------------------------------------------ models
# Each model: parts(p) -> (A (..., S, S) linear terms, N(x, t) the rest), x0(p), observables(x, p) -> {id: (..., G)}.

ELOWITZ_SPECIES = ["X_protein", "Y_protein", "Z_protein", "GFP", "X_mRNA", "Y_mRNA", "Z_mRNA", "GFP_mRNA"]


def elowitz_parts(p):
    """As in the benchmark's SBML: A holds decay and translation, N the repressed transcription (GFP is produced
    directly, repressed by X)."""
    ln2 = math.log(2.0)
    A = torch.zeros(*p["KM"].shape, 8, 8, dtype=torch.float64)
    for i in range(3):  # proteins from their mRNA (index i + 4), decay
        A[..., i, i] = -ln2 / p["tau_prot"]
        A[..., i, i + 4] = p["eff"] / p["tau_mRNA"]
        A[..., i + 4, i + 4] = -ln2 / p["tau_mRNA"]
    A[..., 3, 3] = -ln2 / p["tau_prot_GFP"]
    A[..., 3, 7] = p["eff_GFP"] / p["tau_mRNA_GFP"]
    A[..., 7, 7] = -ln2 / p["tau_mRNA_GFP"]
    n, KM, lo, hi = p["n_Hill"], p["KM"], 60 * p["tps_repr"], 60 * p["tps_active"]

    def N(x, t):
        tx = lambda rep: lo + (hi - lo) * KM**n / (KM**n + rep.clamp_min(0.0) ** n)
        zero = torch.zeros_like(x[..., 0])
        return torch.stack([zero, zero, zero, tx(x[..., 0]), tx(x[..., 2]), tx(x[..., 0]), tx(x[..., 1]), zero], -1)

    return A, N


BOEHM_SPECIES = ["STAT5A", "STAT5B", "pApB", "pApA", "pBpB", "nucpApA", "nucpApB", "nucpBpB"]
CYT, NUC = 1.4, 0.45


def boehm_parts(p):
    """STAT5 phosphorylation by an Epo signal decaying in time (N), nuclear import and export (A); concentrations in
    the cytoplasm (1.4) and nucleus (0.45)."""
    A = torch.zeros(*p["k_phos"].shape, 8, 8, dtype=torch.float64)
    for cyt, nuc, k in [(3, 5, "k_imp_homo"), (2, 6, "k_imp_hetero"), (4, 7, "k_imp_homo")]:
        A[..., cyt, cyt] = -p[k]
        A[..., nuc, cyt] = CYT / NUC * p[k]
    for nuc, k, back in [(5, "k_exp_homo", [(0, 2.0)]), (6, "k_exp_hetero", [(0, 1.0), (1, 1.0)]), (7, "k_exp_homo", [(1, 2.0)])]:
        A[..., nuc, nuc] = -p[k]
        for cyt, stoich in back:
            A[..., cyt, nuc] = stoich * NUC / CYT * p[k]

    def N(x, t):
        epo = 1.25e-7 * torch.exp(-p["Epo_degradation_BaF3"] * t) * p["k_phos"]
        a, b = x[..., 0], x[..., 1]
        aa, ab, bb = epo * a**2, epo * a * b, epo * b**2
        zero = torch.zeros_like(a)
        return torch.stack([-2 * aa - ab, -ab - 2 * bb, ab, aa, bb, zero, zero, zero], -1)

    return A, N


def boehm_x0(p):
    ratio = p["ratio"] * torch.ones_like(p["k_phos"])
    zero = torch.zeros_like(ratio)
    return torch.stack([207.6 * ratio, 207.6 - 207.6 * ratio] + [zero] * 6, -1)


def boehm_obs(x, p):
    s = {k: x[..., i] for i, k in enumerate(BOEHM_SPECIES)}
    c = p["specC17"]
    A_, B_, AB, AA, BB = s["STAT5A"], s["STAT5B"], s["pApB"], s["pApA"], s["pBpB"]
    return {"pSTAT5A_rel": (100 * AB + 200 * AA * c) / (AB + A_ * c + 2 * AA * c),
            "pSTAT5B_rel": -(100 * AB - 200 * BB * (c - 1)) / ((B_ * (c - 1) - AB) + 2 * BB * (c - 1)),
            "rSTAT5A_rel": (100 * AB + 100 * A_ * c + 200 * AA * c)
                           / (2 * AB + A_ * c + 2 * AA * c - B_ * (c - 1) - 2 * BB * (c - 1))}


MODELS = {
    "Elowitz_Nature2000": dict(parts=elowitz_parts, x0=lambda p: torch.stack([p[f"init_{s}"] for s in ELOWITZ_SPECIES], -1),
                               obs=lambda x, p: {"fluorescence": p["background"] + p["scale"] * x[..., 3]}, h=0.5),
    "Boehm_JProteomeRes2014": dict(parts=boehm_parts, x0=boehm_x0, obs=boehm_obs, h=0.25),
}


def etd2(parts, x0, t_obs, p, h=0.5):
    """Exponential time differencing (ETD2RK, Cox & Matthews 2002) with the matrix exponential of the linear part:
    exact for the stiff linear terms, explicit for N. Fixed step ``h`` from t = 0, read out at ``t_obs``."""
    A, N = parts(p)
    d = A.shape[-1]
    M = torch.zeros(*A.shape[:-2], 3 * d, 3 * d, dtype=torch.float64)
    eye = torch.eye(d, dtype=torch.float64)
    M[..., :d, :d], M[..., :d, d:2 * d], M[..., d:2 * d, 2 * d:] = A * h, eye, eye
    E = torch.linalg.matrix_exp(M)  # blocks e^{Ah}, phi1(Ah) / h, phi2(Ah) / h
    eA, phi1, phi2 = E[..., :d, :d], E[..., :d, d:2 * d] * h, E[..., :d, 2 * d:] * h
    mv = lambda m, v: (m @ v.unsqueeze(-1)).squeeze(-1)
    steps = int(round(float(t_obs[-1]) / h))
    idx = torch.round(t_obs / h).long()
    x, out = x0, [x0]
    for i in range(steps):
        t = i * h
        n0 = N(x, t)
        a = mv(eA, x) + mv(phi1, n0)
        x = a + mv(phi2, N(a, t + h) - n0)
        out.append(x)
    return torch.stack(out)[idx]


def ros2(rhs, x0, t_obs, p, h=0.5):
    """Verwer's ROS2 (linearly implicit, L-stable, order 2) with the exact Jacobian from autograd; batched over
    leading dims and differentiable. Fixed step ``h`` from t = 0, read out at ``t_obs``."""
    gamma = 1 + 1 / math.sqrt(2)
    d = x0.shape[-1]
    eye = torch.eye(d, dtype=torch.float64)

    def jac(x, t):  # (..., d, d), one forward-mode pass per state direction
        cols = [torch.func.jvp(lambda v: rhs(t, v, p), (x,), (eye[j].expand_as(x),))[1] for j in range(d)]
        return torch.stack(cols, -1)

    steps = int(round(float(t_obs[-1]) / h))
    idx = torch.round(t_obs / h).long()
    x, out = x0, [x0]
    for i in range(steps):
        t = i * h
        W = eye - gamma * h * jac(x, t)
        k1 = torch.linalg.solve(W, rhs(t, x, p))
        k2 = torch.linalg.solve(W, rhs(t + h, x + h * k1, p) - 2 * k1)
        x = x + h * (1.5 * k1 + 0.5 * k2)
        out.append(x)
    return torch.stack(out)[idx]


class PetabProblem:
    """A benchmark's objective in the interface fit_gradient expects. theta = natural logs of the estimated
    parameters (all on log10 scale in these problems); the curriculum grows over the sorted measurement times."""

    def __init__(self, name: str):
        d = SRC / name / "v1"
        self.name, self.m = name, MODELS[name]
        rows = list(csv.DictReader(open(d / "parameters.tsv"), delimiter="\t"))
        self.fixed = {r["parameterId"]: float(r["nominalValue"]) for r in rows if r["estimate"] == "0"}
        est = [r for r in rows if r["estimate"] == "1"]
        assert all(r["parameterScale"] == "log10" for r in est)
        self.names = [r["parameterId"] for r in est]
        self.lo = torch.tensor([math.log(float(r["lowerBound"])) for r in est], dtype=torch.float64)
        self.hi = torch.tensor([math.log(float(r["upperBound"])) for r in est], dtype=torch.float64)
        self.nominal = torch.tensor([math.log(float(r["nominalValue"])) for r in est], dtype=torch.float64)
        self.obs = {r["observableId"]: r for r in csv.DictReader(open(d / "observables.tsv"), delimiter="\t")}
        self.meas = list(csv.DictReader(open(d / "measurements.tsv"), delimiter="\t"))
        self.t = torch.tensor(sorted({float(m["time"]) for m in self.meas}), dtype=torch.float64)
        self.time_idx = torch.tensor([int((self.t == float(m["time"])).nonzero()) for m in self.meas])
        self.y = torch.tensor([float(m["measurement"]) for m in self.meas], dtype=torch.float64)
        self.range = torch.ones(1, dtype=torch.float64)
        self.log10 = torch.tensor([self.obs[m["observableId"]]["observableTransformation"] == "log10" for m in self.meas])

    def from_z(self, z):
        return self.lo + torch.sigmoid(z) * (self.hi - self.lo)

    def to_z(self, theta):
        u = ((theta - self.lo) / (self.hi - self.lo)).clamp(1e-9, 1 - 1e-9)
        return torch.log(u) - torch.log1p(-u)

    def sample_prior(self, n, generator=None):
        return self.lo + torch.rand(n, len(self.lo), generator=generator, dtype=torch.float64) * (self.hi - self.lo)

    def values(self, theta):
        v = {k: theta[..., i].exp() for i, k in enumerate(self.names)}
        return v | {k: torch.full(theta.shape[:-1], c, dtype=torch.float64) for k, c in self.fixed.items()}

    def simulate(self, theta):
        """Observables at every measurement, ``(..., M)``, and the noise sd of each."""
        p = self.values(theta)
        if SOLVER == "ros2":
            x = ros2(self.rhs, self.m["x0"](p), self.t, p, self.m["h"])
        else:
            x = etd2(self.m["parts"], self.m["x0"](p), self.t, p, self.m["h"])  # (T, ..., S)
        return self.measured(self.m["obs"](x, p), p)

    def measured(self, obs, p):
        """Observables ``{id: (T, ...)}`` at the measurement times -> simulated values and noise sd, ``(..., M)``."""
        sim = torch.stack([obs[m["observableId"]][int(i)] for m, i in zip(self.meas, self.time_idx)], -1)
        sd = torch.stack([p[m["noiseParameters"]] if m["noiseParameters"] in p else p["sigma"] for m in self.meas], -1)
        return sim, sd

    def nll_of(self, sim, sd, keep=None):
        """The benchmark's negative log-likelihood of ``sim``, over the measurements in ``keep``."""
        keep = torch.ones(len(self.y), dtype=torch.bool) if keep is None else keep
        y, sim, sd = self.y[keep], sim[..., keep], sd[..., keep]
        log10 = self.log10[keep]
        r = torch.where(log10, torch.log10(y) - torch.log10(sim.clamp_min(1e-300)), y - sim) / sd
        norm = torch.where(log10, torch.log(2 * math.pi * (sd * y * LN10) ** 2), torch.log(2 * math.pi * sd**2))
        return (0.5 * norm + 0.5 * r**2).sum(-1)

    def nll(self, theta, n_frames=None):
        """The benchmark's negative log-likelihood, over the measurements in the first ``n_frames`` time points."""
        sim, sd = self.simulate(theta)
        return self.nll_of(sim, sd, self.time_idx < (n_frames or len(self.t)))

    def nll_radau(self, theta) -> float:
        """The NLL of one ``theta`` with scipy's Radau at tight tolerances: the referee for the fixed-step fits."""
        from scipy.integrate import solve_ivp

        with torch.no_grad():
            p = self.values(theta)
            f = lambda t, x: self.rhs(t, torch.from_numpy(x), p).numpy()
            t_eval = np.unique(np.concatenate([[0.0], self.t.numpy()]))
            sol = solve_ivp(f, (0.0, float(self.t[-1])), self.m["x0"](p).numpy(), method="Radau", t_eval=t_eval,
                            rtol=1e-10, atol=1e-12)
            if not sol.success:
                return math.inf
            x = torch.from_numpy(sol.y.T[len(t_eval) - len(self.t):])
            return float(self.nll_of(*self.measured(self.m["obs"](x, p), p)))

    def distance(self, theta):  # for ABC: its median tolerance only uses the ranking
        return self.nll(theta)

    def rhs(self, t, x, p):
        A, N = self.m["parts"](p)
        return (A @ x.unsqueeze(-1)).squeeze(-1) + N(x, t)

    def loss(self, theta, n_frames=None, kind="mse"):
        return self.nll(theta, n_frames) / len(self.y)  # per measurement, for the optimiser


# ------------------------------------------------------------------------------------------------ baselines
# ABC-SMC as on the circuits (distance = the NLL). MAGI and the PINN as in solver_free, adapted to PEtab: no
# species is observed directly, so every species is a curve ``x_s(t) = exp(c_s) u_s(t)`` with a learned log
# scale c_s, the data term is the benchmark NLL of the observables of the curves, and the curve at t = 0 is tied
# to the model's x0(p). Both return the point estimate (MAP / final network), scored by simulating it.


def _grid(pr, refine=2):
    """Uniform grid on [0, t_end] with ``refine`` points per measurement interval, plus the measurement times."""
    n = (len(pr.t) - 1) * refine + 1
    t = torch.cat([torch.linspace(0.0, float(pr.t[-1]), n, dtype=torch.float64), pr.t]).unique()
    return t, torch.searchsorted(t, pr.t)


def _start_penalty(pr, x0_curve, p, scale):
    return (((x0_curve - pr.m["x0"](p)) / scale) ** 2).sum()


def fit_pinn_petab(pr, seed=0, restarts=3, iterations=20000, lr=1e-3, n_colloc=512):
    from differentiable_cell.solver_free import _network

    S, t_end = len(pr.m["x0"](pr.values(pr.nominal))), float(pr.t[-1])
    best = None
    for r in range(restarts):
        g = torch.Generator().manual_seed(seed * 100 + r)
        torch.manual_seed(seed * 100 + r)
        net = _network(S)
        c = torch.zeros(S, dtype=torch.float64, requires_grad=True)  # log scale of each species
        z = pr.to_z(pr.sample_prior(1, g)[0]).clone().requires_grad_(True)
        opt = torch.optim.Adam([{"params": net.parameters()}, {"params": [z, c]}], lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iterations, eta_min=lr / 10)
        states = lambda tt: nn.functional.softplus(net((tt / t_end * 10).unsqueeze(-1))) * c.exp()
        for _ in range(iterations):
            p = pr.values(pr.from_z(z))
            tc = t_end * torch.rand(n_colloc, generator=g, dtype=torch.float64)
            u, du = torch.func.jvp(states, (tc,), (torch.ones_like(tc),))
            residual = ((du - pr.rhs(tc, u, p)) / c.exp()).pow(2).mean()
            x = states(torch.cat([torch.zeros(1, dtype=torch.float64), pr.t]))
            misfit = pr.nll_of(*pr.measured(pr.m["obs"](x[1:], p), p)) / len(pr.y)
            loss = residual + misfit + _start_penalty(pr, x[0], p, c.exp()) / S
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        final = float(loss)
        if math.isfinite(final) and (best is None or final < best[0]):
            best = (final, pr.from_z(z.detach()))
    return best[1]


def fit_magi_petab(pr, seed=0, restarts=3, map_iters=30000, lr=0.01, refine=2):
    from differentiable_cell.solver_free import matern52

    g = torch.Generator().manual_seed(seed)
    t, at_meas = _grid(pr, refine)
    S = len(pr.m["x0"](pr.values(pr.nominal)))
    ls = float(t[-1]) / 10  # one length scale for all (no species is observed to fit it on)
    c_, dc, ddc = matern52(t, 1.0, ls)
    Cinv = torch.linalg.inv(c_)
    M = dc @ Cinv
    kd = ddc - M @ dc.T
    Kinv = torch.linalg.inv(0.5 * (kd + kd.T))
    temperature = t.numel() * S / len(pr.y)

    def log_post(u, c, z):  # u: (S, n) standardised curves, c: log scales, z: parameters
        p = pr.values(pr.from_z(z))
        x = u * c.exp().unsqueeze(-1)
        f = pr.rhs(t, x.T.clamp_min(1e-12), p).T / c.exp().unsqueeze(-1)
        dev = f - u @ M.T
        gp = torch.einsum("sn,nm,sm->", u, Cinv, u) + torch.einsum("sn,nm,sm->", dev, Kinv, dev)
        data = pr.nll_of(*pr.measured(pr.m["obs"](x.T[at_meas], p), p))
        prior = (nn.functional.logsigmoid(z) + nn.functional.logsigmoid(-z)).sum()
        return -0.5 * gp / temperature - data - _start_penalty(pr, x[:, 0], p, 0.01 * c.exp()) + prior

    best = None
    for _ in range(restarts):
        u = (0.5 + 0.1 * torch.randn(S, t.numel(), generator=g, dtype=torch.float64)).requires_grad_(True)
        c = torch.zeros(S, dtype=torch.float64, requires_grad=True)
        z = pr.to_z(pr.sample_prior(1, g)[0]).clone().requires_grad_(True)
        opt = torch.optim.Adam([u, c, z], lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=map_iters, eta_min=lr / 50)
        for _ in range(map_iters):
            loss = -log_post(u, c, z)
            if not torch.isfinite(loss):
                break
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        with torch.no_grad():
            lp = float(log_post(u, c, z))
        if math.isfinite(lp) and (best is None or lp > best[0]):
            best = (lp, pr.from_z(z.detach()))
    return best[1]


def run_baseline(name: str, method: str, seed: int) -> None:
    import time

    from differentiable_cell.abc import fit_abc

    pr = PetabProblem(name)
    t0 = time.perf_counter()
    if method == "abc":
        particles, weights, _ = fit_abc(pr, seed=seed)
        with torch.no_grad():
            nll = torch.nan_to_num(pr.nll(particles), nan=math.inf)
        theta = particles[int(nll.argmin())]  # best particle: ABC's most favourable point estimate
    else:
        theta = {"magi": fit_magi_petab, "pinn": fit_pinn_petab}[method](pr, seed=seed)
    wall = time.perf_counter() - t0
    with torch.no_grad():
        known = float(pr.nll(pr.nominal))
    best = pr.nll_radau(theta)
    row = {"model": name, "method": method, "seed": seed, "best_known": known, "best": best,
           "theta_best": theta.tolist(), "wall": wall}
    baseline_file(name, method, seed).parent.mkdir(parents=True, exist_ok=True)
    baseline_file(name, method, seed).write_text(json.dumps(row))
    print(name, method, seed, "NLL", round(best, 3), "known", round(known, 3), f"{wall / 60:.1f} min", flush=True)


def baseline_file(name: str, method: str, seed: int) -> Path:
    return OUT / "baselines" / f"{name}__{method}_seed{seed}.json"


def check(name: str) -> None:
    """The model at the benchmark's nominal parameters against its reference simulation (simulations.tsv)."""
    pr = PetabProblem(name)
    ref = torch.tensor([float(r["simulation"]) for r in csv.DictReader(open(SRC / name / "v1/simulations.tsv"), delimiter="\t")])
    with torch.no_grad():
        sim, _ = pr.simulate(pr.nominal)
        print(f"{name}: max |ours - reference| {float((sim - ref).abs().max()):.2e} (data range "
              f"{float(pr.y.min()):.3g}-{float(pr.y.max()):.3g}); NLL at the nominal parameters {float(pr.nll(pr.nominal)):.4f}")


def job_file(name: str, seed: int, restarts: int, epochs: int, box: str = "sigmoid") -> Path:
    return OUT / f"{name}__seed{seed}_r{restarts}_e{epochs}{'' if box == 'sigmoid' else '_' + box}.json"


def run_one(name: str, seed: int, epochs: int, restarts: int, box: str = "sigmoid") -> None:
    import time

    from differentiable_cell.fit import fit_gradient

    pr = PetabProblem(name)
    with torch.no_grad():
        best_known = float(pr.nll(pr.nominal))
    t0 = time.perf_counter()
    thetas, _ = fit_gradient(pr, restarts=restarts, epochs=epochs, seed=seed, return_all=True, box=box,
                             dtype=torch.float64)  # stiff models: double precision
    wall = time.perf_counter() - t0
    with torch.no_grad():
        nll = torch.nan_to_num(pr.nll(thetas), nan=math.inf)
    finalists = nll.argsort()[:8].tolist()  # re-scored with Radau, so a fit cannot win by integration error
    radau = {i: pr.nll_radau(thetas[i]) for i in finalists}
    pick = min(radau, key=radau.get)
    row = {"model": name, "seed": seed, "restarts": restarts, "epochs": epochs, "box": box, "best_known": best_known,
           "nll": nll.tolist(), "best": radau[pick], "best_etd2": float(nll[pick]), "theta_best": thetas[pick].tolist(),
           "wall": wall}
    OUT.mkdir(parents=True, exist_ok=True)
    job_file(name, seed, restarts, epochs, box).write_text(json.dumps(row))
    print(name, seed, "best", round(row["best"], 3), "known", round(best_known, 3), flush=True)


def report() -> None:
    """Best NLL per run, every method, all refereed by Radau (also the best-known value)."""
    load = lambda paths: [json.loads(p.read_text()) for p in sorted(paths)]
    for name in MODELS:
        known = PetabProblem(name).nll_radau(PetabProblem(name).nominal)
        print(f"{name}: best known NLL {known:.3f}")
        groups = {f"ours ({box} box)": [r for r in load(OUT.glob(f"{name}__seed*.json")) if r.get("box", "sigmoid") == box]
                  for box in ("sigmoid", "clamp")}
        groups |= {m: load((OUT / "baselines").glob(f"{name}__{m}_*.json")) for m in ("abc", "magi", "pinn")}
        for label, rows in groups.items():
            if rows:
                b = sorted(r["best"] for r in rows)
                print(f"  {label:20s} {len(rows)} runs; NLL per run {', '.join(f'{v:.4g}' for v in b)}; within 0.1 of "
                      f"best known: {sum(v <= known + 0.1 for v in b)}/{len(b)}; {np.median([r['wall'] for r in rows]) / 60:.1f} min per run")


def trajectory_radau(pr, theta, t):
    """Observables ``{id: (len(t),)}`` on the grid ``t`` (starting at 0), with Radau."""
    from scipy.integrate import solve_ivp

    with torch.no_grad():
        p = pr.values(theta)
        f = lambda tt, x: pr.rhs(tt, torch.from_numpy(x), p).numpy()
        sol = solve_ivp(f, (0.0, float(t[-1])), pr.m["x0"](p).numpy(), method="Radau", t_eval=t, rtol=1e-10, atol=1e-12)
        if not sol.success or sol.y.shape[1] != len(t):
            return None
        return {k: v.numpy() for k, v in pr.m["obs"](torch.from_numpy(sol.y.T), p).items()}


def figure() -> None:
    """Data, best-known fit and each method's best run, per observable."""
    import matplotlib.pyplot as plt

    from style import METHOD_COLOUR, PALETTE

    load = lambda paths: [json.loads(p.read_text()) for p in sorted(paths)]
    panels = [(name, obs) for name in MODELS for obs in PetabProblem(name).obs]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.6))
    labels = {"gradient": "ours", "abc": "ABC-SMC", "magi": "MAGI", "pinn": "PINN"}
    for ax, (name, obs) in zip(axes, panels):
        pr = PetabProblem(name)
        t = np.linspace(0.0, float(pr.t[-1]), 400)
        runs = {"gradient": load(OUT.glob(f"{name}__seed*.json"))}
        runs |= {m: load((OUT / "baselines").glob(f"{name}__{m}_*.json")) for m in ("abc", "magi", "pinn")}
        mask = np.array([m["observableId"] == obs for m in pr.meas])
        y = pr.y.numpy()[mask]
        known = pr.nll_radau(pr.nominal)
        curve = trajectory_radau(pr, pr.nominal, t)
        ax.plot(t, curve[obs], color=PALETTE["ink"], ls="--", lw=1.4, label=f"best known ({known:.1f})", zorder=3)
        for m, rows in runs.items():
            best = min(rows, key=lambda r: r["best"])
            curve = trajectory_radau(pr, torch.tensor(best["theta_best"]), t)
            if curve is not None:
                ax.plot(t, curve[obs], color=METHOD_COLOUR[m], lw=2.0 if m == "gradient" else 1.4,
                        label=f"{labels[m]} ({best['best']:.1f})" if abs(best['best']) < 1e4 else f"{labels[m]} ({best['best']:.0e})", zorder=4 if m == "gradient" else 2)
        ax.scatter(pr.t.numpy()[pr.time_idx.numpy()[mask]], y, s=14, color=PALETTE["grey"], zorder=5, label="data")
        pad = 0.25 * (y.max() - y.min())
        ax.set_ylim(y.min() - pad, y.max() + pad)  # diverged fits leave the frame rather than squash it
        ax.set_title(f"{name.split('_')[0]}: {obs}", fontsize=11)
        ax.set_xlabel("time (min)")
        ax.legend(fontsize=7.5, loc="best", title="best run (NLL)", title_fontsize=7.5)
    fig.tight_layout()
    from common import FIGURES

    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"petab_fits.{ext}", bbox_inches="tight")
    print(FIGURES / "petab_fits.png")


def main() -> None:
    from common import run_pool

    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--figure", action="store_true")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--restarts", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--box", default="sigmoid", choices=["sigmoid", "clamp"])
    ap.add_argument("--method", default="gradient", choices=["gradient", "abc", "magi", "pinn"])
    ap.add_argument("--one", nargs=2, metavar=("MODEL", "SEED"))
    args = ap.parse_args()
    setup_torch(1)
    if args.check:
        for name in args.models.split(","):
            check(name)
        return
    if args.report:
        return report()
    if args.figure:
        return figure()
    if args.one is not None and args.method != "gradient":
        return run_baseline(args.one[0], args.method, int(args.one[1]))
    if args.one is not None:
        return run_one(args.one[0], int(args.one[1]), args.epochs, args.restarts, args.box)
    if args.method != "gradient":
        jobs = [["--one", m, str(s), "--method", args.method] for m in args.models.split(",") for s in range(args.seeds)
                if not baseline_file(m, args.method, s).exists()]
        return run_pool(__file__, jobs, args.workers, OUT / "baselines" / f"log_{args.method}.txt")
    jobs = [["--one", m, str(s), "--epochs", str(args.epochs), "--restarts", str(args.restarts), "--box", args.box]
            for m in args.models.split(",") for s in range(args.seeds)
            if not job_file(m, s, args.restarts, args.epochs, args.box).exists()]
    run_pool(__file__, jobs, args.workers, OUT / "log.txt")


if __name__ == "__main__":
    main()
