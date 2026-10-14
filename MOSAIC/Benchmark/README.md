# General scalability benchmarks

Runtime comparison of the rejection method (MOSAIC) against the standard Gillespie
algorithm (SG), DelaySSA and the next-reaction method on simple one- and two-species
systems with gamma-distributed waiting times. These are the benchmarks referred to in
the method section of the paper.

| Script | System | Compares | Precomputed results |
|---|---|---|---|
| `Benchmark_scaling.py` | `A -> A` (gamma, α = 6) for N = 8 … 65 536 reactants | MOSAIC, SG (exponential), DelaySSA (precomputed delays) | `Simulation_results/time_*.npy`, `Simulation_results/Nreac_*.npy` |
| `Benchmark_vs_DSSA_NRM.py` | `A ⇌ B` (gamma, α = 6) | MOSAIC, DelaySSA, next-reaction method, SG | none (runs live) |

Run from this folder:

```bash
python Benchmark_scaling.py
```

`Benchmark_scaling.py` loads the precomputed timings by default. Set
`recompute_time_complexity = True` in `main()` to rerun the measurements
(expect about an hour on a laptop for the largest systems). Both scripts use the core
library `../MOSAIC.py`.
