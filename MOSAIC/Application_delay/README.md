# Application 1: transcription with heterogeneous delays

<p align="center"><img src="../../Figures/RNA_oscillations.png" width="700"></p>

A Hes1-like negative feedback loop in which transcription initiation (`0 -> N`),
elongation (`N -> M`), translation (`M -> M + P`) and degradation are simulated as
individual reactions. Elongation has a gamma-distributed duration and the initiation
rate depends on the protein level, so the inter-event times are system dependent.
The panels above show (B) the accuracy of MOSAIC as a function of system size,
(C, D) oscillations with energy-limited and unconstrained elongation and (E, F) the
runtime scaling of MOSAIC against DelaySSA and the standard Gillespie algorithm (SG).
The nMGA curve of panel E is not produced by these scripts (no nMGA implementation is
included). The accuracy panel is computed for the β values listed in
`Model_benchmark_accuracy.py` (0.01 to 100); its precomputed data in `Saved_pop/` were
regenerated after the population-recording fix described in the main README, so the
values differ from the published panel B.

| Script | Produces | Precomputed data | Flag to recompute |
|---|---|---|---|
| `Model_benchmark_accuracy.py` | RMSD vs. mean system size (panel B) and example trajectories | `Saved_pop/{mosaic,delay1,delay2}_<beta>_<Nsim>.npy` | `recompute_accuracy = True` |
| `Model_benchmark_cost.py` | runtime per simulation and per accepted event vs. system size (panels E, F) | `Simulation_results/time_*.npy`, `Simulation_results/Nreac_*.npy` | `recompute_time_complexity = True` |

Run from this folder, for example:

```bash
python Model_benchmark_cost.py
```

Both scripts use `lib_MOSAIC_with_delay.py`, a snapshot of the core library that
additionally implements the Anderson (modified next-reaction) scheme used for
DelaySSA and the Hes1 rate functions (`RNA_rate_function`,
`RNA_delay_rate_function`). The model follows the Hes1 delayed feedback loop of Monk (2003); references and
parameter sources are listed in the docstrings at the top of each script.
Regenerating all precomputed data takes several hours.
