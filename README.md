# A differentiable simulation is all you need: repressilator identification from sparse reporters

David Korcak (dkorcak@ethz.ch) · MSc, Department of Computer Science, ETH Zürich

Inferring a gene circuit's parameters from sparse single-cell observations usually
relies on dedicated estimators, optionally with a simulation. Using the repressilator,
we show that **backpropagation through a differentiable simulation itself matches or
outperforms them**. What limits inference is the choice of reporters, which we assess
with an identifiability analysis.

<p align="center"><a href="docs/poster.pdf"><img src="assets/poster.png" width="520" alt="Poster"></a></p>

The poster is in [`docs/poster.pdf`](docs/poster.pdf). This README expands on it: the
model, the method, every result behind the poster's figures, and the side experiments
that shaped the final setup.

---

## Contents

1. [Circuit model](#1-circuit-model)
2. [Differentiable simulation](#2-differentiable-simulation)
3. [Parameter inference](#3-parameter-inference)
4. [Results](#4-results)
5. [Side experiments](#5-side-experiments)
6. [Limitations](#6-limitations)
7. [Repository layout](#7-repository-layout)
8. [Reproducing](#8-reproducing)
9. [References](#9-references)

---

## 1. Circuit model

<p align="center"><img src="assets/circuit_schematic.png" width="760" alt="Circuit schematic"></p>

- **Circuit:** the Elowitz–Leibler repressilator [1] on pSC101. LacI represses *tetR*,
  TetR represses *cI* and CI represses *lacI*. It uses the symmetric Box 1 model with
  α = 216, α₀ = 0.216, n = 2 and β = 0.2.
- **Reporters:** three transcriptional reporters on one p15A plasmid. Each copies one
  circuit promoter and drives a degradation-tagged fluorescent protein:
  - P_Llac → mJuniper-LVA
  - P_Ltet → GFP-AAV
  - λP_R → mScarlet-I3-LVA
- **Reporter chain:** each reporter's mRNA r makes a dark protein D, which matures into
  the fluorescent form F.
  - Maturation rates k come from FPbase: 1.7, 4.1 and 2.0 min.
  - Tag half-lives δ come from Andersen et al. (1998): LVA 40 min, AAV 60 min.
- **Titration:** the reporters' operator sites titrate their repressor. This is modelled
  as a fixed effective shift of the repression threshold, K = 1 + τ with τ = 0.75, for
  every promoter that repressor controls.
- **Fusions (optional):** a repressor can itself be fused to a fluorescent protein and
  read out directly. The fusion is idealised: it does not perturb the circuit.
- **State:** 15 species (3 mRNAs, 3 repressors, and r, D and F for each reporter).
  Time is measured in mRNA lifetimes (2.885 min) and protein in units of K_M.

For one repressor pair and its reporter:

$$
\frac{dm_{\text{tetR}}}{dt} = -m_{\text{tetR}} + \frac{\alpha}{1 + (p_{\text{LacI}}/K_{\text{LacI}})^{n}} + \alpha_0,
\qquad
\frac{dp_{\text{TetR}}}{dt} = -\beta\,(p_{\text{TetR}} - m_{\text{tetR}})
$$

$$
\frac{dr_{\text{GFP}}}{dt} = -r_{\text{GFP}} + \frac{a}{1 + (p_{\text{TetR}}/K_{\text{TetR}})^{n}} + a_0,
\qquad
\frac{dD_{\text{GFP}}}{dt} = \delta\, r_{\text{GFP}} - (k + \delta)\, D_{\text{GFP}},
\qquad
\frac{dF_{\text{GFP}}}{dt} = k\, D_{\text{GFP}} - \delta\, F_{\text{GFP}}
$$

The meaning of each parameter:

| symbol | meaning |
|---|---|
| α, α₀ | maximal (derepressed) and leaky transcription rate |
| n | Hill coefficient of repression |
| β | protein-to-mRNA decay-rate ratio |
| a, a₀ | reporter strength and leak |

**Cells.** One cell is simulated either as the ODE (torchdiffeq `dopri5`, rtol 10⁻⁸) or
as an exact Gillespie process with Ω = K_M = 40 molecules, using 30 reactions and count
scales set so the reaction network's drift equals the ODE exactly (tested).

**Imaging.** Cells are simulated at full resolution (30 s) and imaged every 5 min for
10 h. The camera model has Poisson photon noise with a per-fluorophore brightness from
FPbase, gain 2, offset 100 and read noise 5, and is calibrated back to model units.

<table><tr>
<td><img src="assets/poster_sim_ode_dotted.png" alt="ODE cell"></td>
<td><img src="assets/poster_sim_gillespie_dotted.png" alt="Gillespie cell"></td>
</tr><tr>
<td align="center"><sub>ODE cell: idealised.</sub></td>
<td align="center"><sub>Gillespie cell: phase drift and cycle-to-cycle amplitude variation.</sub></td>
</tr></table>

Code: [`reporters.py`](src/differentiable_cell/reporters.py) (model and reaction network),
[`reporter_sim.py`](src/differentiable_cell/reporter_sim.py) (cells, camera, frames),
[`gillespie.py`](src/differentiable_cell/gillespie.py) (exact SSA and the differentiable
Gillespie).

## 2. Differentiable simulation

The right-hand side is a smooth function of both the state and the parameters, so
gradients of any loss on the trajectory flow back through the solver steps.

The simulation is written in PyTorch and integrated with torchdiffeq [2]. Fitting
backpropagates through time (BPTT) over a fixed-step RK4 solver, with a step of 0.35
model units (about 1 min, roughly 5 steps per frame). The data are generated with
tight-tolerance `dopri5`, so the fit never sees the solver it is judged by.

## 3. Parameter inference

<p align="center"><img src="assets/method_schematic.png" width="900" alt="Method schematic"></p>

- **Trajectory shooting.** The unknowns are the circuit parameters (α, α₀, n, β), each
  built reporter's strength and leak (a, a₀), and the cell's **full starting state**.
  That is 10 + 5R unknowns for R reporters (10–25 in total). Everything after t = 0 is
  determined by the ODE. Maturation, tag decay and titration are fixed.
- **Bounded, randomised initialisation.** Every unknown is log-uniform on a physically
  motivated box:
  - α 20–1000
  - α₀ 10⁻³–1
  - n 1–4
  - β 0.1–1

  A measured species' starting value is boxed to its first frame ± 10% of the channel's
  range, which pins where in its cycle the cell starts. Hidden species are free, and
  optimisation runs in logit coordinates, so no iterate leaves its box.
- **Loss.** Huber loss (δ = 0.05) on range-normalised residuals
  r = (F̂ − y) / range(y) at every frame, reduced by the arithmetic mean:

  $$\mathcal{L} = \frac{1}{G\cdot C}\sum_{i=1}^{G\cdot C} H_\delta(r_i)$$
- **Optimisation.**
  - Adam, 1500 epochs, cosine learning rate 0.05 → 2.5·10⁻⁴.
  - Curriculum: the loss sees the first 25%, then 50%, then 100% of the frames. A
    small period error grows into a phase error over a long window, which flattens the
    loss.
  - First-order optimisers only.
- **Restarts.** 64 prior draws are optimised in parallel as one batch; the restart with
  the lowest plain MSE over the full record is the estimate.
- **Uncertainty (optional).** A Laplace approximation from the Hessian at the optimum.
  It gives 90% intervals that covered the truth in 23 of 27 cases (9 cells × α, n, β). The Hessian is used
  only for the error bars, never for the fit.

Code: [`fit.py`](src/differentiable_cell/fit.py) (`Problem`, `fit_gradient`),
[`nuts.py`](src/differentiable_cell/nuts.py) (`fit_laplace`).

## 4. Results

### 4.1 Against dedicated estimators

The compared methods:
- **ABC-SMC** [3]: likelihood-free; accepts simulations within a shrinking tolerance.
  It uses 500 particles and up to 300k simulations, and its estimate is the weighted
  posterior median.
- **MAGI** [4]: MAnifold-constrained Gaussian process Inference. It fits Gaussian-process
  trajectories constrained to satisfy the ODE, with no solver, followed by a MAP and
  Hessian-preconditioned NUTS.
- **Inverse PINN** [5]: a 5×100 sine-activated network fitted to the data and the ODE
  residual.
- **Differentiable Gillespie (DGA)** [6]: did not qualify (see §4.3).

All methods see the same cells, the same frames and the same prior boxes. There are
three reporter configurations: GFP only; mJuniper, GFP and mScarlet-I3; TetR fusion +
GFP. Each is run with **13 random seeds** on both ODE and Gillespie cells (39 cells per
method and cell type).

<p align="center"><img src="assets/results_main.png" alt="Main results"></p>

The fit panel shows the MSE of the ODE integrated at each estimate, relative to that at
the true parameters. On ODE cells, 1× is the best fit the data allow.

Medians:

| | α | n | β | fit (× MSE at truth) |
|---|---|---|---|---|
| **ODE cells:** diff. ODE (ours) | **18%** | **3.4%** | **1.6%** | **0.94×** (all 39 in 0.87–1.0×) |
| ABC-SMC | 61% | 15% | 54% | 219× |
| MAGI | 34% | 12% | 17% | 64× |
| inverse PINN | 50% | 28% | 16% | 278× |
| **Gillespie cells:** diff. ODE (ours) | 73% | 8.5% | **11%** | **0.28×** |
| ABC-SMC | 53% | 16% | 22% | 1.5× |
| MAGI (37 of 39: 2 NaN failures) | 64% | **5.8%** | 23% | 1.04× |
| inverse PINN | 58% | 23% | 16% | 2.2× |

- **ODE cells: ours is best on every parameter**, and it is the only method at the
  optimum. The others' estimates reproduce the data 64–278× worse. The solver-free
  methods (MAGI, PINN) can satisfy their own ODE penalty with trajectories that are not
  true ODE solutions. ABC-SMC runs out of budget in 10–25 dimensions.
- **Gillespie cells: the picture is mixed.**
  - Ours gives the best β and by far the closest fit.
  - MAGI's n is better.
  - No method recovers α: cycle-to-cycle amplitude noise swamps the one parameter that
    sets amplitude, and per-cell α errors range from 4% to 341%.
  - Values below 1× are expected here: the deterministic model at the true parameters
    cannot follow a stochastic cell.
- **Seeds matter.** The first 3 seeds made us look better on Gillespie cells than 13
  seeds do (α 52% vs 73%, n 5.5% vs 8.5%). The ODE-cell result was stable.

### 4.2 What the difference looks like

<p align="center"><img src="assets/fit_comparison.png" alt="Fit comparison"></p>

One Gillespie cell with two reporters (seed 1). It is the median case for ABC-SMC of the
three seeds available for this configuration, not the most flattering contrast. Both
methods started from the same prior draws (ABC's first 64 particles are exactly our 64
starts).

- Ours follows each cycle's timing and amplitude, at 0.53× the MSE at the truth.
- ABC-SMC settles on small, fast oscillations, at 2.6×.

### 4.3 Differentiable Gillespie does not qualify

<p align="center"><img src="assets/dga_gradients.png" width="620" alt="DGA gradient growth"></p>

The figure shows |d mean state / d log α| over 8 cells at the paper's smoothing
(1/a = 200, 1/b = 20) on the three-reporter model:

| simulated window | 1.4 min | 2.9 min | 5.8 min | 14 min | 29 min | 58 min |
|---|---|---|---|---|---|---|
| DGA, Ω = 2 | 8·10¹ | 1·10⁴ | 1·10⁹ | 4·10²⁴ | 6·10³⁸ | **2·10⁸³** |
| DGA, Ω = 5 | 8·10² | 5·10⁸ | 4·10¹⁴ | 1·10⁴⁴ | 4·10⁶⁹ | **2·10¹²⁰** |
| ODE, same quantity | 0.1 | 0.3 | 0.6 | 0.6 | 2.1 | 0.5 |

Each event's smoothed reaction choice amplifies perturbations, and the feedback loop
compounds them over ~10⁴ events per period (a period is ~120 min).

This does not contradict Rijal & Mehta [6]. They fit steady-state *moments* averaged
over thousands of cells of a promoter without feedback, and they flag non-stationary
time series as untested. A single oscillating cell is exactly that regime.

### 4.4 Identifiability: the reporters set the limit

<p align="center"><img src="assets/identifiability.png" alt="Identifiability"></p>

The plot shows the best achievable 1σ uncertainty of each parameter per configuration.
It comes from the Fisher information of one cell's frames at the true parameters, with
the unknown starting state profiled out (Schur complement) and the prior box added.

- **Transcriptional reporters** show *when* promoters switch, which gives n and β. They
  do not show *how hard* the repressors swing: α is only known relative to the unknown
  reporter strength a.
- **A fused repressor** shows its absolute level and brings α to about 10%.
- **The leak α₀** is unidentifiable in every configuration (43–192%; not shown).

Our achieved errors (ODE cells, median over the available seeds) sit at or near these
bounds. **The fit extracts what the data contain**, and the remaining error is set by
what is measured:

| configuration | seeds | achieved α / n / β | achievable 1σ α / n / β |
|---|---|---|---|
| GFP only | 14 | 37% / 6.4% / 5.3% | 69% / 11.6% / 4.6% |
| 2 reporters (GFP + mScarlet-I3) | 3 | 10% / 2.8% / 0.6% | 30% / 5.7% / 2.1% |
| 3 reporters | 14 | 30% / 2.9% / 1.6% | 39% / 7.0% / 2.0% |
| TetR fusion | 3 | 11% / 3.4% / 0.3% | 11% / 3.6% / 0.9% |
| TetR fusion + GFP | 13 | 9% / 2.0% / 0.7% | 9% / 3.5% / 1.5% |
| TetR fusion + mScarlet-I3 | 3 | 7% / 0.4% / 0.5% | 8% / 1.7% / 0.7% |
| TetR fusion + 3 reporters | 3 | 6% / 2.1% / 0.9% | 11% / 3.4% / 0.9% |
| 3 fusions | 3 | 2% / 1.4% / 0.6% | 9% / 3.0% / 0.7% |

### Conclusion

On the repressilator, plain backpropagation through a differentiable ODE matches or
outperforms dedicated estimators. Accuracy is limited by the information the reporters
reveal: transcriptional reporters constrain the circuit's timing but not its absolute
transcription rate. Identifiability analysis can therefore guide reporter design before
a physical experiment, both to maximise the chance of parameter recovery and to choose
a suitable inference method.

## 5. Side experiments

These shaped the final settings; none of them changes the conclusions.

**Loss and optimiser tuning** ([`tuning.py`](experiments/tuning.py)). This used 6
datasets (3 configurations × 2 seeds, ODE cells) at a reduced budget of 32 restarts ×
1000 epochs. Variants were judged on fit quality, never on errors against the truth.

| variant | median fit / truth MSE | reached optimum | α / n / β |
|---|---|---|---|
| MSE + exponential lr (baseline) | 1.097 | 3/6 | 42 / 9.1 / 3.9% |
| Huber + exponential lr | 0.968 | 5/6 | 11 / 7.5 / 2.4% |
| MSE + cosine lr | 0.981 | 5/6 | 40 / 9.2 / 3.6% |
| **Huber + cosine lr (used)** | **0.940** | **6/6** | 33 / 13 / 2.9% |
| RMSprop | 1.228 | 0/6 | 31 / 19 / 11% |
| SGD + Nesterov | 5.70 | 0/6 | 81 / 47 / 36% |

Huber mainly helps the *optimisation*: it caps the gradient from large early residuals.
It is not a better noise model for camera noise, where MSE is the likelihood. Once a fit
is at the optimum, the parameter errors are set by the data (§4.4), not by the loss.

**How the 64 restarts spread** ([`restart_spread.py`](experiments/restart_spread.py)).
This recorded every restart's full-record loss every 25 epochs.

<p align="center"><img src="assets/restart_curves.png" width="820" alt="Restart curves"></p>

- About half the restarts reach the optimum on ODE cells: 24–32 of 64. The rest sit on
  flat plateaus (local minima) from about epoch 500. Restarts, not epochs, protect
  against them.
- **Restarts at the same optimum disagree on α.** On reporter-only configurations, α
  ranges from 2% to 65% error at equal fit. That is the flat valley of §4.4, seen
  directly. On the fusion configuration they agree (5–31%).
- The best fit settles within ~100 epochs of the curriculum reaching the full record
  (epoch ~1100). The last ~400 epochs change the MSE by less than 0.3%.
- **The schedule cannot simply be shortened:**
  - 600 epochs still reaches the optimum, but only 3–10 of 64 restarts get there
    (instead of 24–32). The early stages decide how many escape bad basins.
  - Annealing the final learning rate to 10⁻⁵ instead of 2.5·10⁻⁴ changes nothing at
    either length.
  - The setup therefore stays at 1500 epochs.

**Not in the final benchmark.**
- **NUTS through the ODE** works but costs about 1 s per gradient, so hours per cell. It
  was replaced by the Laplace approximation.
- **Nested sampling** (Pullen & Morris 2014) belonged to an earlier version of the
  problem and was dropped.

## 6. Limitations

- **Synthetic data with a known model structure.** The same equations generate and fit
  the ODE cells. Gillespie cells are misspecified for every method.
- **One parameter set** (Box 1).
- **Configurations and seeds:**
  - 13 seeds on the three benchmark configurations. The 20-seed run was stopped at the
    last seed complete for every method on both cell types.
  - 3 seeds on the other five configurations.
- **Idealised model details:** fusions do not perturb the circuit, and titration is a
  fixed effective shift.
- **ABC-SMC is budget-limited** (300k simulations). A larger budget would help it; we
  did not measure how much.
- **Time is not a claim.** Methods ran on fixed budgets, not until convergence:
  - ours ~10 min per cell (64 restarts)
  - ABC ~4 min
  - MAGI ~10 min
  - PINN ~12 min

## 7. Repository layout

The current pipeline, used for the poster:

```
src/differentiable_cell/
  reporters.py        15-species model: ODE right-hand side, reaction network, constants
  reporter_sim.py     one cell: ODE or Gillespie, camera model, 5-min frames (generate)
  gillespie.py        exact SSA and differentiable Gillespie (DGA)
  fit.py              the inverse problem (Problem) and backprop through the ODE (fit_gradient)
  abc.py              ABC-SMC
  solver_free.py      MAGI and inverse PINN
  nuts.py             NUTS and the Laplace approximation
experiments/
  panel_benchmark.py        the benchmark: configurations x cell type x seeds x method
  plot_results.py           results_main (Fig. 4) and the other error figures
  plot_fit_comparison.py    ODE vs ABC-SMC on one cell (Fig. 5)
  identifiability_reporters.py, plot_identifiability.py   Fisher analysis (Fig. 6)
  poster_sim_plots.py       example ODE and Gillespie cells (Figs. 2, 3)
  dga_gradient_growth.py    DGA gradient explosion
  restart_spread.py         restart and schedule study
  tuning.py                 loss / optimiser / schedule tuning
  show_reporter_data.py     quick look at one generated cell
  run_panels.sh, run_reduced.sh, run_remaining.sh   earlier benchmark drivers
tests/test_reporter_sim.py  model and simulator tests
assets/                     figures used in this README
docs/poster.pdf             the poster
```

**Legacy.** This is the first version of the project: an 8-species model with GFP only
and no reporter chain. It is kept for reference and not used by the results above.
- `src/differentiable_cell/{data,model,run,helpers,units,langevin}.py` and
  `src/differentiable_cell/inference/`
- `experiments/{run_clean,run_partial,benchmark,benchmark_report,paper_repro,identifiability,_common}.py`
  and `experiments/run_v2.sh`
- `experiment_scripts/`
- `tests/{test_units,test_langevin,test_estimators}.py`. `tests/test_gillespie.py`
  checks the SSA and DGA in `gillespie.py` on the legacy model.
- `docs/poster_summary.md`: working notes from poster day. Its numbers are 3-seed
  numbers, superseded by this README.

All outputs go to `run_results/`, which is git-ignored.

## 8. Reproducing

```bash
uv sync
uv run pytest tests -q                                   # all tests

# the benchmark (resumable; one JSON per fit in run_results/panel_benchmark/<method>/)
uv run python experiments/panel_benchmark.py --method gradient,abc,magi,pinn --seeds 20 --workers 5 --threads 2
uv run python experiments/panel_benchmark.py --method gradient,abc --workers 3   # all 8 configurations, 3 seeds

# figures (run_results/figures/)
uv run python experiments/plot_results.py                # results_main: seeds complete for every method on both cell types
uv run python experiments/plot_fit_comparison.py
uv run python experiments/plot_identifiability.py        # add --replot to restyle without recomputing
uv run python experiments/poster_sim_plots.py
uv run python experiments/dga_gradient_growth.py

# side experiments
uv run python experiments/tuning.py --workers 3 && uv run python experiments/tuning.py --report
uv run python experiments/restart_spread.py --workers 3 && uv run python experiments/restart_spread.py --report
uv run python experiments/restart_spread.py --workers 3 --epochs 600 --lr-final 1e-5
```

A full benchmark fit takes about 10 minutes per cell and method on a laptop. The 20-seed
grid is about 480 fits, so budget a day or more.

## 9. References

1. Elowitz MB, Leibler S. A synthetic oscillatory network of transcriptional regulators.
   *Nature* 403, 335–338 (2000).
2. Chen RTQ, Rubanova Y, Bettencourt J, Duvenaud D. Neural ordinary differential
   equations. *NeurIPS* 31 (2018).
3. Toni T, Welch D, Strelkowa N, Ipsen A, Stumpf MPH. Approximate Bayesian computation
   scheme for parameter inference and model selection in dynamical systems.
   *J. R. Soc. Interface* 6, 187–202 (2009).
4. Yang S, Wong SWK, Kou SC. Inference of dynamic systems from noisy and sparse data via
   manifold-constrained Gaussian processes. *PNAS* 118, e2020397118 (2021).
5. Casajuana B, Casals-Franch R, López García de Lomana A, Martí-Puig P, Villà-Freixa J.
   Physics-informed neural networks for parameter recovery in the repressilator
   oscillatory model. *bioRxiv* 10.64898/2026.05.12.724679 (2026).
6. Rijal K, Mehta P. A differentiable Gillespie algorithm for simulating chemical
   kinetics, parameter estimation, and designing synthetic biological circuits.
   *arXiv* 2407.04865 (2025).

Parameter sources: FPbase (Lambert, *Nat. Methods* 16, 277–278, 2019) for maturation
rates and brightness; Andersen et al., *Appl. Environ. Microbiol.* 64, 2240–2246 (1998)
for degradation-tag half-lives.
