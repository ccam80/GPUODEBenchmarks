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
| tsit5 | adaptive | default | 128 | 4.2481 | 0.36747 | 0.086502 |
| tsit5 | adaptive | default | 131072 | 4.0784 | 4.4375 | 1.088 |
| tsit5 | adaptive | default | 2048 | 4.3254 | 0.4617 | 0.10674 |
| tsit5 | adaptive | default | 2097152 | 32.082 | 74.625 | 2.326 |
| tsit5 | adaptive | default | 32 | 3.392 | 0.304 | 0.089623 |
| tsit5 | adaptive | default | 32768 | 3.9296 | 1.4383 | 0.36601 |
| tsit5 | adaptive | default | 33554432 | 506.35 | 1396.9 | 2.7588 |
| tsit5 | adaptive | default | 512 | 3.2277 | 0.40669 | 0.126 |
| tsit5 | adaptive | default | 524288 | 9.04 | 18.91 | 2.0918 |
| tsit5 | adaptive | default | 8 | 4.3055 | 0.30848 | 0.071649 |
| tsit5 | adaptive | default | 8192 | 3.8664 | 0.69973 | 0.18098 |
| tsit5 | adaptive | default | 8388608 | 130.21 | 350.4 | 2.6911 |
| tsit5 | adaptive | pi | 128 | 4.0279 | 0.36747 | 0.091231 |
| tsit5 | adaptive | pi | 131072 | 4.3722 | 4.4375 | 1.0149 |
| tsit5 | adaptive | pi | 2048 | 3.1907 | 0.4617 | 0.1447 |
| tsit5 | adaptive | pi | 2097152 | 28.599 | 74.625 | 2.6093 |
| tsit5 | adaptive | pi | 32 | 3.2338 | 0.304 | 0.094007 |
| tsit5 | adaptive | pi | 32768 | 3.7389 | 1.4383 | 0.38468 |
| tsit5 | adaptive | pi | 33554432 | 441.24 | 1396.9 | 3.1659 |
| tsit5 | adaptive | pi | 512 | 4.3608 | 0.40669 | 0.093261 |
| tsit5 | adaptive | pi | 524288 | 8.1599 | 18.91 | 2.3174 |
| tsit5 | adaptive | pi | 8 | 4.2273 | 0.30848 | 0.072973 |
| tsit5 | adaptive | pi | 8192 | 3.3682 | 0.69973 | 0.20775 |
| tsit5 | adaptive | pi | 8388608 | 115.64 | 350.4 | 3.0301 |
| tsit5 | fixed | fixed | 128 | 3.6378 | 0.46707 | 0.12839 |
| tsit5 | fixed | fixed | 131072 | 8.0648 | 9.8983 | 1.2274 |
| tsit5 | fixed | fixed | 2048 | 3.5711 | 0.47858 | 0.13401 |
| tsit5 | fixed | fixed | 2097152 | 117.69 | 163.97 | 1.3933 |
| tsit5 | fixed | fixed | 32 | 3.6123 | 0.46239 | 0.128 |
| tsit5 | fixed | fixed | 32768 | 3.979 | 2.6168 | 0.65767 |
| tsit5 | fixed | fixed | 33554432 | 1802.5 | 2765.4 | 1.5342 |
| tsit5 | fixed | fixed | 512 | 4.451 | 0.46916 | 0.10541 |
| tsit5 | fixed | fixed | 524288 | 29.685 | 42.374 | 1.4274 |
| tsit5 | fixed | fixed | 8 | 3.7048 | 0.46072 | 0.12436 |
| tsit5 | fixed | fixed | 8192 | 3.8267 | 0.74303 | 0.19417 |
| tsit5 | fixed | fixed | 8388608 | 467.54 | 700.36 | 1.498 |

## Numerical equivalence

Per-trajectory Float32 finals are retained beneath `finals/`. Mutual metrics are elementwise at t=1.

_No successful rows._

## Observed fixed-step convergence order

| framework | algorithm | tier | observed_order | usable_intervals |
|---|---|---|---|---|
| julia | tsit5 | fixed | 5.3935 | 4 |

## Work-precision

`work_precision.csv` joins every work-point timing distribution (min/p05/median/p95/max) to golden RMSE. `plots/work_precision.png` plots median runtime on the x-axis against golden RMSE on the y-axis; it is an error-work plot, not another setting sweep.

## Failures and non-finite results

8 point failures were recorded. See the framework failure CSVs for full messages. Non-finite trajectory counts are retained per successful point in the metric CSVs.

| framework | algorithm | phase | mode | tier | setting_kind | setting | error_type | message |
|---|---|---|---|---|---|---|---|---|
| julia | tsit5 | performance | fixed | fixed | dt | 0.0009765625 | OutOfGPUMemoryError | Out of GPU memory trying to allocate 3.000 GiB Effective GPU memory usage: 99.50% (7.544 GiB/7.582 GiB) Memory pool usage: 5.375 GiB (7.000 GiB reserved)  |
| julia | tsit5 | performance | adaptive | julia | tol | 1.0e-8 | OutOfGPUMemoryError | Out of GPU memory trying to allocate 3.500 GiB Effective GPU memory usage: 99.50% (7.544 GiB/7.582 GiB) Memory pool usage: 4.375 GiB (7.000 GiB reserved)  |
| julia | tsit5 | performance | fixed | fixed | dt | 0.0009765625 | OutOfGPUMemoryError | Out of GPU memory trying to allocate 14.000 GiB Effective GPU memory usage: 65.29% (4.951 GiB/7.582 GiB) Memory pool usage: 4.375 GiB (4.406 GiB reserved)  |
| julia | tsit5 | performance | adaptive | julia | tol | 1.0e-8 | OutOfGPUMemoryError | Out of GPU memory trying to allocate 14.000 GiB Effective GPU memory usage: 65.29% (4.951 GiB/7.582 GiB) Memory pool usage: 4.375 GiB (4.406 GiB reserved)  |
| julia | tsit5 | numerical | fixed | fixed | dt | 0.5 | ErrorException | non-finite result: 35/1024 trajectories valid |
| julia | tsit5 | numerical | fixed | fixed | dt | 0.25 | ErrorException | non-finite result: 289/1024 trajectories valid |
| julia | vern7 | performance | fixed | fixed | dt | 0.0009765625 | OutOfGPUMemoryError | Out of GPU memory trying to allocate 3.000 GiB Effective GPU memory usage: 99.50% (7.544 GiB/7.582 GiB) Memory pool usage: 5.375 GiB (7.000 GiB reserved)  |
| julia | vern7 | performance | adaptive | julia | tol | 1.0e-8 | OutOfGPUMemoryError | Out of GPU memory trying to allocate 3.500 GiB Effective GPU memory usage: 99.50% (7.544 GiB/7.582 GiB) Memory pool usage: 4.375 GiB (7.000 GiB reserved)  |

## Artifacts

Plots: `plots/performance_scaling.png`, `plots/numerical_equivalence.png`, and `plots/work_precision.png`. Raw and derived CSVs in this directory are algorithm-, mode-, tier-, N-, and setting-keyed.
