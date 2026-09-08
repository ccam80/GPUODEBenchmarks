# Cubie ↔ DiffEqGPU benchmark

Algorithms: `tsit5` ↔ `GPUTsit5()`, `vern7` ↔ `GPUVern7()`, `rosenbrock23_sciml` ↔ `GPURosenbrock23()`, `kvaerno3` ↔ `GPUKvaerno3()`, and `kvaerno5` ↔ `GPUKvaerno5()`.

`diffeqgpu_ode_inventory.csv` contains the eight specialized DiffEqGPU GPU ODE algorithms and their Cubie mappings.

All timed samples are synchronized end-to-end solves, including final host transfer. Each point has an untimed warmup. Fixed performance uses dyadic `dt=2^-10`; full numerical equivalence uses N=1024 with `dt=2^-1..2^-13` and tolerances `1e-2..1e-6`; full work-precision uses N=32768 with `dt=2^-4..2^-13`, tolerances `1e-2..1e-8`, and 20 repeats by default. The raw sample CSVs remain the authoritative timing record.

Every phase, including performance, records finite and failed trajectory counts. A point without a complete all-finite validity metric is excluded from timing summaries, speedups, and plots. Workers continue after point failures to preserve successful evidence, then return nonzero so the launcher marks the suite incomplete.

Analytic Lorenz Jacobian and time-gradient functions are supplied to every Julia problem, including explicit runs, and implicit Julia constructors disable autodiff. A failed point is appended to the failure ledger and later points continue.

## Performance speedups

`julia_over_cubie_speedup > 1` means Cubie was faster. Percentile timing statistics for both frameworks are in `timing_summary.csv`.

| algorithm | mode | cubie_tier | n | cubie_median_ms | julia_median_ms | julia_over_cubie_speedup |
|---|---|---|---|---|---|---|
| kvaerno3 | adaptive | default | 32 | 200.41 | 8.2806 | 0.041319 |
| kvaerno3 | adaptive | default | 8 | 108.45 | 8.0275 | 0.074021 |
| kvaerno3 | adaptive | default | 32 | 199.44 | 8.2251 | 0.041241 |
| kvaerno3 | adaptive | default | 8 | 104.84 | 8.8009 | 0.083946 |
| kvaerno3 | adaptive | pi | 32 | 159.56 | 8.2806 | 0.051897 |
| kvaerno3 | adaptive | pi | 8 | 91.612 | 8.0275 | 0.087624 |
| kvaerno3 | adaptive | pi | 32 | 153.38 | 8.2251 | 0.053626 |
| kvaerno3 | adaptive | pi | 8 | 90.041 | 8.8009 | 0.097743 |
| kvaerno3 | fixed | fixed | 32 | 4.5548 | 3.1236 | 0.68579 |
| kvaerno3 | fixed | fixed | 8 | 4.0771 | 3.8872 | 0.95344 |
| kvaerno3 | fixed | fixed | 32 | 2.7229 | 2.9182 | 1.0717 |
| kvaerno3 | fixed | fixed | 8 | 3.043 | 3.373 | 1.1084 |
| kvaerno5 | adaptive | default | 32 | 277.91 | 0.87805 | 0.0031594 |
| kvaerno5 | adaptive | default | 8 | 107.99 | 0.91339 | 0.008458 |
| kvaerno5 | adaptive | default | 32 | 278.83 | 0.83415 | 0.0029917 |
| kvaerno5 | adaptive | default | 8 | 104.77 | 0.83694 | 0.0079887 |
| kvaerno5 | adaptive | pi | 32 | 106.11 | 0.87805 | 0.0082747 |
| kvaerno5 | adaptive | pi | 8 | 39.91 | 0.91339 | 0.022886 |
| kvaerno5 | adaptive | pi | 32 | 109.02 | 0.83415 | 0.0076514 |
| kvaerno5 | adaptive | pi | 8 | 40.966 | 0.83694 | 0.02043 |
| kvaerno5 | fixed | fixed | 32 | 9.0944 | 5.9868 | 0.65829 |
| kvaerno5 | fixed | fixed | 8 | 9.3098 | 5.9574 | 0.6399 |
| kvaerno5 | fixed | fixed | 32 | 7.0445 | 5.952 | 0.84491 |
| kvaerno5 | fixed | fixed | 8 | 7.0481 | 5.893 | 0.83612 |
| rosenbrock23_sciml | adaptive | default | 32 | 6.083 | 4.0817 | 0.671 |
| rosenbrock23_sciml | adaptive | default | 8 | 6.7119 | 3.2328 | 0.48166 |
| rosenbrock23_sciml | adaptive | default | 32 | 5.2817 | 3.5821 | 0.67821 |
| rosenbrock23_sciml | adaptive | default | 8 | 5.9211 | 3.1657 | 0.53465 |
| rosenbrock23_sciml | adaptive | pi | 32 | 8.0345 | 4.0817 | 0.50801 |
| rosenbrock23_sciml | adaptive | pi | 8 | 6.6801 | 3.2328 | 0.48395 |
| rosenbrock23_sciml | adaptive | pi | 32 | 7.3908 | 3.5821 | 0.48468 |
| rosenbrock23_sciml | adaptive | pi | 8 | 5.9708 | 3.1657 | 0.5302 |
| rosenbrock23_sciml | fixed | fixed | 32 | 4.4836 | 1.2063 | 0.26905 |
| rosenbrock23_sciml | fixed | fixed | 8 | 3.6295 | 2.4184 | 0.66631 |
| rosenbrock23_sciml | fixed | fixed | 32 | 1.6188 | 1.1051 | 0.68265 |
| rosenbrock23_sciml | fixed | fixed | 8 | 1.7044 | 1.3306 | 0.78068 |
| tsit5 | adaptive | default | 32 | 4.24 | 0.33739 | 0.079572 |
| tsit5 | adaptive | default | 8 | 4.729 | 0.40157 | 0.084918 |
| tsit5 | adaptive | default | 32 | 0.67576 | 0.29795 | 0.44091 |
| tsit5 | adaptive | default | 8 | 0.76014 | 0.35211 | 0.46322 |
| tsit5 | adaptive | pi | 32 | 5.487 | 0.33739 | 0.061488 |
| tsit5 | adaptive | pi | 8 | 5.0174 | 0.40157 | 0.080036 |
| tsit5 | adaptive | pi | 32 | 1.5326 | 0.29795 | 0.1944 |
| tsit5 | adaptive | pi | 8 | 0.71645 | 0.35211 | 0.49146 |
| tsit5 | fixed | fixed | 32 | 3.6116 | 0.49024 | 0.13574 |
| tsit5 | fixed | fixed | 8 | 5.0453 | 0.59806 | 0.11854 |
| tsit5 | fixed | fixed | 32 | 0.80681 | 0.4493 | 0.55689 |
| tsit5 | fixed | fixed | 8 | 0.84409 | 0.51565 | 0.6109 |
| vern7 | adaptive | default | 32 | 4.108 | 0.59833 | 0.14565 |
| vern7 | adaptive | default | 8 | 5.0817 | 0.63041 | 0.12405 |
| vern7 | adaptive | default | 32 | 0.6989 | 0.57004 | 0.81563 |
| vern7 | adaptive | default | 8 | 1.2296 | 0.55483 | 0.45122 |
| vern7 | adaptive | pi | 32 | 4.8593 | 0.59833 | 0.12313 |
| vern7 | adaptive | pi | 8 | 5.7422 | 0.63041 | 0.10978 |
| vern7 | adaptive | pi | 32 | 0.7967 | 0.57004 | 0.7155 |
| vern7 | adaptive | pi | 8 | 1.3945 | 0.55483 | 0.39786 |
| vern7 | fixed | fixed | 32 | 4.8188 | 0.68841 | 0.14286 |
| vern7 | fixed | fixed | 8 | 6.0037 | 0.80115 | 0.13344 |
| vern7 | fixed | fixed | 32 | 1.0225 | 0.64103 | 0.6269 |
| vern7 | fixed | fixed | 8 | 2.1695 | 1.3137 | 0.60554 |

## Numerical equivalence

Per-trajectory Float32 finals are retained beneath `finals/`. Mutual metrics are elementwise at t=1.

_No successful rows._

## Observed fixed-step convergence order

_No successful rows._

## Work-precision

`work_precision.csv` joins every work-point timing distribution (min/p05/median/p95/max) to golden RMSE. `plots/work_precision.png` plots median runtime on the x-axis against golden RMSE on the y-axis; it is an error-work plot, not another setting sweep.

## Failures and non-finite results

0 point failures were recorded. See the framework failure CSVs for full messages. Non-finite trajectory counts are retained per successful point in the metric CSVs.

_No successful rows._

## Artifacts

Plots: `plots/performance_scaling.png`, `plots/numerical_equivalence.png`, and `plots/work_precision.png`. Raw and derived CSVs in this directory are algorithm-, mode-, tier-, N-, and setting-keyed.
