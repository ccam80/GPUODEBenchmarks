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
| tsit5 | adaptive | default | 128 | 0.73326 | 0.34932 | 0.47639 |
| tsit5 | adaptive | default | 131072 | 2.3072 | 4.3747 | 1.8962 |
| tsit5 | adaptive | default | 2048 | 0.75029 | 0.48254 | 0.64313 |
| tsit5 | adaptive | default | 2097152 | 27.993 | 72.82 | 2.6014 |
| tsit5 | adaptive | default | 32 | 0.76923 | 0.3315 | 0.43095 |
| tsit5 | adaptive | default | 32768 | 1.1027 | 1.4413 | 1.307 |
| tsit5 | adaptive | default | 33554432 | 435.07 | 1128.9 | 2.5948 |
| tsit5 | adaptive | default | 512 | 0.7462 | 0.3825 | 0.5126 |
| tsit5 | adaptive | default | 524288 | 7.4322 | 17.918 | 2.4108 |
| tsit5 | adaptive | default | 8 | 0.76126 | 0.33104 | 0.43486 |
| tsit5 | adaptive | default | 8192 | 0.79814 | 0.67824 | 0.84977 |
| tsit5 | adaptive | default | 8388608 | 110.45 | 289.87 | 2.6245 |
| tsit5 | adaptive | default | 128 | 0.4381 | 0.31202 | 0.71223 |
| tsit5 | adaptive | default | 131072 | 1.5458 | 3.4836 | 2.2536 |
| tsit5 | adaptive | default | 2048 | 0.44253 | 0.42313 | 0.95616 |
| tsit5 | adaptive | default | 2097152 | 18.51 | 49.429 | 2.6705 |
| tsit5 | adaptive | default | 32 | 0.47378 | 0.29406 | 0.62066 |
| tsit5 | adaptive | default | 32768 | 0.70363 | 1.1635 | 1.6536 |
| tsit5 | adaptive | default | 33554432 | 284.27 | 755.04 | 2.6561 |
| tsit5 | adaptive | default | 512 | 0.44456 | 0.34175 | 0.76874 |
| tsit5 | adaptive | default | 524288 | 4.9315 | 12.647 | 2.5645 |
| tsit5 | adaptive | default | 8 | 0.46344 | 0.30378 | 0.65549 |
| tsit5 | adaptive | default | 8192 | 0.47384 | 0.53497 | 1.129 |
| tsit5 | adaptive | default | 8388608 | 72.914 | 196.4 | 2.6936 |
| tsit5 | adaptive | pi | 128 | 0.70114 | 0.34932 | 0.49821 |
| tsit5 | adaptive | pi | 131072 | 2.0605 | 4.3747 | 2.1232 |
| tsit5 | adaptive | pi | 2048 | 0.71116 | 0.48254 | 0.67853 |
| tsit5 | adaptive | pi | 2097152 | 24.362 | 72.82 | 2.9891 |
| tsit5 | adaptive | pi | 32 | 0.73054 | 0.3315 | 0.45378 |
| tsit5 | adaptive | pi | 32768 | 1.021 | 1.4413 | 1.4116 |
| tsit5 | adaptive | pi | 33554432 | 378.94 | 1128.9 | 2.9791 |
| tsit5 | adaptive | pi | 512 | 0.73103 | 0.3825 | 0.52324 |
| tsit5 | adaptive | pi | 524288 | 6.5181 | 17.918 | 2.7489 |
| tsit5 | adaptive | pi | 8 | 0.72373 | 0.33104 | 0.45741 |
| tsit5 | adaptive | pi | 8192 | 0.75817 | 0.67824 | 0.89457 |
| tsit5 | adaptive | pi | 8388608 | 96.132 | 289.87 | 3.0153 |
| tsit5 | adaptive | pi | 128 | 0.40611 | 0.31202 | 0.76832 |
| tsit5 | adaptive | pi | 131072 | 1.2987 | 3.4836 | 2.6823 |
| tsit5 | adaptive | pi | 2048 | 0.40543 | 0.42313 | 1.0437 |
| tsit5 | adaptive | pi | 2097152 | 14.916 | 49.429 | 3.3139 |
| tsit5 | adaptive | pi | 32 | 0.43136 | 0.29406 | 0.68171 |
| tsit5 | adaptive | pi | 32768 | 0.61833 | 1.1635 | 1.8817 |
| tsit5 | adaptive | pi | 33554432 | 228.3 | 755.04 | 3.3073 |
| tsit5 | adaptive | pi | 512 | 0.4151 | 0.34175 | 0.82328 |
| tsit5 | adaptive | pi | 524288 | 4.0181 | 12.647 | 3.1475 |
| tsit5 | adaptive | pi | 8 | 0.42826 | 0.30378 | 0.70934 |
| tsit5 | adaptive | pi | 8192 | 0.4335 | 0.53497 | 1.2341 |
| tsit5 | adaptive | pi | 8388608 | 58.582 | 196.4 | 3.3525 |
| tsit5 | fixed | fixed | 128 | 0.90638 | 0.4426 | 0.48832 |
| tsit5 | fixed | fixed | 131072 | 7.4188 | 10.547 | 1.4217 |
| tsit5 | fixed | fixed | 2048 | 0.91868 | 0.50934 | 0.55442 |
| tsit5 | fixed | fixed | 2097152 | 108.69 | 157.31 | 1.4473 |
| tsit5 | fixed | fixed | 32 | 0.90427 | 0.49405 | 0.54635 |
| tsit5 | fixed | fixed | 32768 | 2.4006 | 2.6282 | 1.0948 |
| tsit5 | fixed | fixed | 33554432 | 1732.8 | 2499.4 | 1.4424 |
| tsit5 | fixed | fixed | 512 | 0.9047 | 0.44268 | 0.48931 |
| tsit5 | fixed | fixed | 524288 | 27.672 | 41.163 | 1.4875 |
| tsit5 | fixed | fixed | 8 | 0.90902 | 0.49176 | 0.54098 |
| tsit5 | fixed | fixed | 8192 | 1.0956 | 0.75114 | 0.68562 |
| tsit5 | fixed | fixed | 8388608 | 433.58 | 581.78 | 1.3418 |
| tsit5 | fixed | fixed | 128 | 0.60733 | 0.4048 | 0.66652 |
| tsit5 | fixed | fixed | 131072 | 6.6584 | 8.933 | 1.3416 |
| tsit5 | fixed | fixed | 2048 | 0.60935 | 0.43035 | 0.70625 |
| tsit5 | fixed | fixed | 2097152 | 99.204 | 133.94 | 1.3502 |
| tsit5 | fixed | fixed | 32 | 0.60629 | 0.44657 | 0.73656 |
| tsit5 | fixed | fixed | 32768 | 1.995 | 2.333 | 1.1694 |
| tsit5 | fixed | fixed | 33554432 | 1582.1 | 2125.1 | 1.3433 |
| tsit5 | fixed | fixed | 512 | 0.60678 | 0.40202 | 0.66255 |
| tsit5 | fixed | fixed | 524288 | 25.177 | 35.222 | 1.399 |
| tsit5 | fixed | fixed | 8 | 0.60703 | 0.44542 | 0.73377 |
| tsit5 | fixed | fixed | 8192 | 0.76822 | 0.6586 | 0.85731 |
| tsit5 | fixed | fixed | 8388608 | 395.96 | 531.68 | 1.3428 |

## Numerical equivalence

Per-trajectory Float32 finals are retained beneath `finals/`. Mutual metrics are elementwise at t=1.

_No successful rows._

## Observed fixed-step convergence order

| framework | algorithm | tier | observed_order | usable_intervals |
|---|---|---|---|---|
| julia | kvaerno3 | fixed | 2.7862 | 9 |
| julia | kvaerno5 | fixed | 3.3923 | 6 |
| julia | rosenbrock23_sciml | fixed | 1.9427 | 8 |
| julia | tsit5 | fixed | 5.3935 | 4 |
| julia | vern7 | fixed | 2.1609 | 4 |

## Work-precision

`work_precision.csv` joins every work-point timing distribution (min/p05/median/p95/max) to golden RMSE. `plots/work_precision.png` plots median runtime on the x-axis against golden RMSE on the y-axis; it is an error-work plot, not another setting sweep.

## Failures and non-finite results

17 point failures were recorded. See the framework failure CSVs for full messages. Non-finite trajectory counts are retained per successful point in the metric CSVs.

| framework | algorithm | phase | mode | tier | setting_kind | setting | error_type | message |
|---|---|---|---|---|---|---|---|---|
| cubie | tsit5 | performance | fixed | fixed | dt | 0.0009765625 | OutOfMemoryError | Out of memory allocating 285,296,072 bytes (allocated so far: 7,381,975,040 bytes, limit set to: 18,446,744,073,709,551,615 bytes). |
| cubie | tsit5 | performance | adaptive | default | tol | 1e-08 | ValueError | Device-resident results require the batch to fit in a single chunk, but this run is split into 334 chunks. Reduce the batch size or use a host solve. |
| cubie | tsit5 | performance | adaptive | pi | tol | 1e-08 | ValueError | Device-resident results require the batch to fit in a single chunk, but this run is split into 438 chunks. Reduce the batch size or use a host solve. |
| julia | tsit5 | performance | fixed | fixed | dt | 0.0009765625 | OutOfGPUMemoryError | Out of GPU memory trying to allocate 3.500 GiB Effective GPU memory usage: 99.86% (7.571 GiB/7.582 GiB) Memory pool usage: 4.375 GiB (7.094 GiB reserved)  |
| julia | tsit5 | performance | adaptive | julia | tol | 1.0e-8 | OutOfGPUMemoryError | Out of GPU memory trying to allocate 3.500 GiB Effective GPU memory usage: 99.86% (7.571 GiB/7.582 GiB) Memory pool usage: 4.375 GiB (7.094 GiB reserved)  |
| julia | tsit5 | numerical | fixed | fixed | dt | 0.5 | ErrorException | non-finite result: 35/1024 trajectories valid |
| julia | tsit5 | numerical | fixed | fixed | dt | 0.25 | ErrorException | non-finite result: 289/1024 trajectories valid |
| julia | vern7 | performance | fixed | fixed | dt | 0.0009765625 | OutOfGPUMemoryError | Out of GPU memory trying to allocate 3.500 GiB Effective GPU memory usage: 99.86% (7.571 GiB/7.582 GiB) Memory pool usage: 4.375 GiB (7.094 GiB reserved)  |
| julia | vern7 | performance | adaptive | julia | tol | 1.0e-8 | OutOfGPUMemoryError | Out of GPU memory trying to allocate 3.500 GiB Effective GPU memory usage: 99.86% (7.571 GiB/7.582 GiB) Memory pool usage: 4.375 GiB (7.094 GiB reserved)  |
| julia | vern7 | numerical | fixed | fixed | dt | 0.5 | ErrorException | non-finite result: 9/1024 trajectories valid |
| julia | vern7 | numerical | fixed | fixed | dt | 0.25 | ErrorException | non-finite result: 708/1024 trajectories valid |
| julia | rosenbrock23_sciml | performance | fixed | fixed | dt | 0.0009765625 | OutOfGPUMemoryError | Out of GPU memory trying to allocate 3.500 GiB Effective GPU memory usage: 99.86% (7.571 GiB/7.582 GiB) Memory pool usage: 4.375 GiB (7.094 GiB reserved)  |
| julia | rosenbrock23_sciml | performance | adaptive | julia | tol | 1.0e-8 | OutOfGPUMemoryError | Out of GPU memory trying to allocate 3.500 GiB Effective GPU memory usage: 99.86% (7.571 GiB/7.582 GiB) Memory pool usage: 4.375 GiB (7.094 GiB reserved)  |
| julia | kvaerno3 | performance | fixed | fixed | dt | 0.0009765625 | OutOfGPUMemoryError | Out of GPU memory trying to allocate 3.500 GiB Effective GPU memory usage: 99.86% (7.571 GiB/7.582 GiB) Memory pool usage: 4.375 GiB (7.094 GiB reserved)  |
| julia | kvaerno3 | performance | adaptive | julia | tol | 1.0e-8 | OutOfGPUMemoryError | Out of GPU memory trying to allocate 3.500 GiB Effective GPU memory usage: 99.86% (7.571 GiB/7.582 GiB) Memory pool usage: 4.375 GiB (7.094 GiB reserved)  |
| julia | kvaerno5 | performance | fixed | fixed | dt | 0.0009765625 | OutOfGPUMemoryError | Out of GPU memory trying to allocate 3.500 GiB Effective GPU memory usage: 99.86% (7.571 GiB/7.582 GiB) Memory pool usage: 4.375 GiB (7.094 GiB reserved)  |
| julia | kvaerno5 | performance | adaptive | julia | tol | 1.0e-8 | OutOfGPUMemoryError | Out of GPU memory trying to allocate 3.500 GiB Effective GPU memory usage: 99.86% (7.571 GiB/7.582 GiB) Memory pool usage: 4.375 GiB (7.094 GiB reserved)  |

## Artifacts

Plots: `plots/performance_scaling.png`, `plots/numerical_equivalence.png`, and `plots/work_precision.png`. Raw and derived CSVs in this directory are algorithm-, mode-, tier-, N-, and setting-keyed.
