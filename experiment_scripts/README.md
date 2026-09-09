# Fixed experiments

Each script runs one experiment with hyperparameters that are fixed in the
file, and takes exactly one argument: the directory of a clean run produced by
`experiments/run_clean.py`.

```bash
python experiments/run_clean.py --ode --steps 3000 --dt 0.05
./experiment_scripts/03_multi_param.sh run_results/<the-directory-it-printed>
```

Results land in `run_results/<timestamp>_partial_..._<tag>/`, one directory per
fit, containing `fit.pt`, `config.json`, `metrics.json`, `restarts.csv` and
`fit.png`.

| script | question | runtime |
|---|---|---|
| `01_sanity.sh` | is the clean run's log 1:1 with its own forward problem? | ~10 s |
| `02_single_param.sh` | can one parameter be recovered from a broad prior? | ~4 min |
| `03_multi_param.sh` | can three coupled parameters be recovered? | ~23 min |
| `04_all_params.sh` | can all six be recovered with everything observed? | ~25 min |
| `05_hidden_ics.sh` | can a hidden species' initial value be recovered? | ~15 min |
| `06_gfp_only.sh` | what is knowable from the one measurable species? | ~25 min |
| `07_gfp_reporter_tradeoff.sh` | are `alpha_GFP` and `beta_GFP` separable? | ~25 min |
| `08_noise_sweep.sh` | how does recovery degrade with measurement noise? | ~16 min |
| `09_lr_sweep.sh` | why the learning rate decides success here | ~30 min |
| `10_optimizer_compare.sh` | does the optimiser choice matter? | ~16 min |
| `11_loss_compare.sh` | does the loss choice matter, clean and noisy? | ~24 min |
| `run_all.sh` | all of the above, sequentially | ~3.5 h |

Run `01_sanity.sh` first. If it does not report zero loss and zero error, the
clean run and the model have drifted apart and nothing else here is meaningful.

## Reading a result

`metrics.json` is the summary. Three fields decide whether a number is a
measurement or an artefact:

- **`at_prior_bound`** - a parameter sitting on its prior box has run out of
  room, not converged. Widen `PRIORS` if the bound is wrong, or lower the
  learning rate: a large early step drives a parameter into the wall and
  momentum keeps it there.
- **`spread_over_restarts`** - agreement across restarts is the evidence of
  identifiability. A tight best fit with a wide spread means the data admits
  many answers.
- **`full_horizon_normalised_mse`** - the fit is scored on a growing window,
  so this is what separates a fit that matched its training window from one
  that got the period right over the whole record.

`solver_failures` counts steps that were rejected as unintegrable and retried
at a smaller learning rate; a few are normal, many mean the fit is wandering
somewhere the model cannot be solved.

## Why the horizon grows

Fitting by simulation over many periods destroys the gradient: a small period
error accumulates into a phase flip and the loss saturates at the value it
would take for two unrelated phases. Measured on `beta` at 11 periods, the
loss is 815 at 0.19 and 855 at 0.21 with the truth at 0.20 - almost no signal.
At 2 periods the same landscape is cleanly curved (104 / 0 / 101). So every
script fits one period first and then widens.

Hidden initial conditions pull the other way: the influence of a hidden
species' starting value on the observed ones grows with the window, from
~1e-7 at one time unit to ~2e-3 at two periods, because it reaches the
observables only indirectly, by shifting the phase of the whole limit cycle.
The curriculum serves both.
