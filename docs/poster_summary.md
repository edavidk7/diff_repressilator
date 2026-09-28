# Poster summary: differentiable simulation for single-cell repressilator inference

Numbers below are measured (Box 1 parameters, 3 seeds, one cell of 10 h at
5-min frames); `run_results/` holds every fit and figure. MAGI, PINN and
Laplace ran on a reduced grid (3 panels x ODE cells x 3 seeds).

## Question

Does a dedicated inference method beat plain backpropagation through the ODE
solver (which comes for free with a differentiable simulator) at recovering the
repressilator's parameters from one cell's fluorescence? And which reporters
must be built for that to work?

## Model (figure: `figures/schematic.png`, `reporter_data.png`)

- Repressilator (Elowitz & Leibler 2000, Box 1): α = 216, α₀ = 0.216, n = 2, β = 0.2.
- Three transcriptional reporters on one p15A plasmid: P_Llac → mJuniper-LVA,
  P_Ltet → GFP-AAV (Elowitz's), P_R → mScarlet-I3-LVA; each mRNA → dark → mature FP
  (FPbase maturation, Andersen 1998 tag half-lives); reporter operator sites
  titrate their repressor (effective shift, τ = 0.75, known from the design).
- Optional fusion channels: the repressor itself carries an FP (idealised).
- Data: one cell, deterministic (ODE) or exact Gillespie (Ω = K_M = 40), camera
  model (photon + read noise), calibrated to model units, 5-min frames, 10 h.
- Inference unknowns: α, α₀, n, β, each reporter's strength and leak, and the
  cell's full starting state (10-25 unknowns); observed species' starts are
  pinned by the first frame. Everything after t = 0 is the forward simulation.

## Methods

| method | uses the solver? | uses gradients? |
|---|---|---|
| backprop through ODE (Adam, Huber loss, cosine lr, 64 restarts) | yes | yes |
| backprop + Laplace (Hessian at the fit) | yes | yes |
| ABC-SMC (Toni et al. 2009; 300k simulations) | yes | no |
| MAGI (Yang, Wong & Kou 2021; Wong 2024) | no (GP + ODE residual) | yes |
| inverse PINN (Casajuana et al. 2026) | no (network + ODE residual) | yes |

NUTS through the ODE was tried and is infeasible here (~1 s per gradient, a
chain needs ~10^4 gradients: hours per case).

## Results

### 1. Backprop through the ODE recovers what the data allow (ODE cell)
Median error over 3 seeds (`figures/errors_ode.png`); every fit reached the
likelihood optimum (fit MSE / truth MSE 0.90-0.96):

| panel | α | n | β | data allow (1σ, α / n / β) |
|---|---|---|---|---|
| GFP only | 42% | 12% | 15% | 69% / 12% / 4.6% |
| 2 reporters | 10% | 2.8% | 0.6% | 30% / 5.7% / 2.1% |
| 3 reporters | 25% | 6.1% | 1.7% | 39% / 7.0% / 2.0% |
| TetR fusion | 11% | 3.4% | 0.3% | 11% / 3.6% / 0.9% |
| TetR fusion + GFP | 8.8% | 0.4% | 0.5% | 9.1% / 3.5% / 1.5% |
| TetR fusion + cI reporter | 7.3% | 0.4% | 0.5% | 7.7% / 1.7% / 0.7% |
| TetR fusion + 3 reporters | 5.6% | 2.1% | 0.9% | 11% / 3.4% / 0.9% |
| 3 fusions | 2.2% | 1.4% | 0.6% | 9.2% / 3.0% / 0.7% |

α₀ (leakiness) is not identifiable from any panel (19-96% error; 43-192% allowed).

### 2. Reporter design: n and β from reporters, α needs a fusion
Transcriptional reporters show when promoters switch (period, order → n, β),
but not how hard repressors swing: α is only known relative to the unknown
reporter strength. A fused repressor shows its absolute level and pins α
(`figures/identifiability.png`).

### 3. Against the dedicated methods
- ABC-SMC, same 48 cases: α 30-70%, β up to 90% on reporter panels; its fits
  stay 50-450x above the achievable MSE - in 10-25 unknowns 300k simulations
  are not enough (`figures/errors_summary.png`).
- All five methods on 3 panels x 3 seeds, ODE cells (median error; "fit" = MSE of the
  ODE run at the estimate relative to the true parameters' MSE; <= 1 = at the optimum):

| method | GFP only α / n / β | 3 reporters α / n / β | TetR fusion + GFP α / n / β | fit | 90% CI covers truth |
|---|---|---|---|---|---|
| backprop | 42 / 12 / 15% | 25 / 6.1 / 1.7% | 9 / 0.4 / 0.5% | 0.90-0.96 | - |
| backprop + Laplace | same | same | same | same | 23/27 (85%) |
| ABC-SMC | 55 / 18 / 62% | 70 / 15 / 90% | 36 / 2.0 / 20% | 227-252x | - |
| MAGI | 24 / 32 / 24% | 43 / 10 / 17% | 23 / 8.8 / 6.5% | 58-89x | 0/27 |
| PINN | 89 / 19 / 28% | 71 / 11 / 18% | 39 / 20 / 0.7% | 102-369x | - |

  Only backprop reaches the optimum, and its free Laplace intervals are close
  to calibrated. MAGI's posterior is confidently wrong. The two medians where
  another method beats backprop (MAGI α on GFP only, PINN β on the fusion
  panel) are within what the data can determine (GFP-only α: +-69%) and not
  significant at 3 seeds. The solver-free methods' estimates do not reproduce
  the data when the ODE is integrated at them (58-369x the achievable MSE).

### 4. Stochastic (Gillespie) cells
Backprop (misspecified: deterministic model, noisy cell) keeps n and β mostly
at 1-14%; α degrades to 18-165% (`figures/errors_ssa.png`; time series show the
fit following the rhythm but not the cycle-to-cycle amplitude).

### 5. Differentiable Gillespie is not usable over a period
|d mean state / d log α| over 8 cells at the paper's smoothing (1/a = 200,
1/b = 20), three-reporter model (`figures/dga_gradients.png`):

| window | 1.4 min | 2.9 min | 5.8 min | 14 min | 29 min | 58 min |
|---|---|---|---|---|---|---|
| DGA, Ω = 2 | 8e1 | 1e4 | 1e9 | 4e24 | 6e38 | 2e83 |
| DGA, Ω = 5 | 8e2 | 5e8 | 4e14 | 1e44 | 4e69 | 2e120 |
| ODE (same quantity) | 0.1 | 0.3 | 0.6 | 0.6 | 2.1 | 0.5 |

Each event's smoothed reaction selection amplifies perturbations, and a
feedback loop compounds them over ~10^4 events per period; a period is ~120 min.

## Caveats

- One parameter set (Box 1), 3 seeds per configuration.
- The model is known exactly (same equations generate and fit the ODE data);
  Gillespie cells are misspecified for every method here.
- Fusions are idealised (they do not perturb the circuit).
- ABC is budget-limited; MAGI and PINN run on a reduced grid.
- Tuning (Huber + cosine) was chosen on fit quality over 6 datasets, not on
  errors against the truth.
