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
| kvaerno3 | adaptive | default | 128 | 149.59 | 8.0889 | 0.054076 |
| kvaerno3 | adaptive | default | 131072 | 1242.9 | 76.708 | 0.061717 |
| kvaerno3 | adaptive | default | 2048 | 159.06 | 9.2155 | 0.057938 |
| kvaerno3 | adaptive | default | 2097152 | 18403 | 1093.2 | 0.059401 |
| kvaerno3 | adaptive | default | 32 | 188.3 | 8.1837 | 0.043461 |
| kvaerno3 | adaptive | default | 32768 | 384.99 | 26.409 | 0.068595 |
| kvaerno3 | adaptive | default | 512 | 155.73 | 8.4667 | 0.054368 |
| kvaerno3 | adaptive | default | 524288 | 4667.7 | 283.42 | 0.06072 |
| kvaerno3 | adaptive | default | 8 | 99.321 | 7.8836 | 0.079376 |
| kvaerno3 | adaptive | default | 8192 | 187.22 | 11.13 | 0.059448 |
| kvaerno3 | adaptive | default | 8388608 | 73282 | 4363.3 | 0.059541 |
| kvaerno3 | adaptive | default | 128 | 150.91 | 8.0441 | 0.053305 |
| kvaerno3 | adaptive | default | 131072 | 1242.5 | 75.797 | 0.061005 |
| kvaerno3 | adaptive | default | 2048 | 158.62 | 9.1636 | 0.05777 |
| kvaerno3 | adaptive | default | 2097152 | 18395 | 1080.6 | 0.058746 |
| kvaerno3 | adaptive | default | 32 | 187.97 | 8.1392 | 0.043301 |
| kvaerno3 | adaptive | default | 32768 | 384.32 | 26.14 | 0.068017 |
| kvaerno3 | adaptive | default | 512 | 155.4 | 8.4255 | 0.054218 |
| kvaerno3 | adaptive | default | 524288 | 4665 | 278.44 | 0.059688 |
| kvaerno3 | adaptive | default | 8 | 98.982 | 7.8401 | 0.079208 |
| kvaerno3 | adaptive | default | 8192 | 186.85 | 11.028 | 0.059019 |
| kvaerno3 | adaptive | default | 8388608 | 73244 | 4267.8 | 0.058269 |
| kvaerno3 | adaptive | pi | 128 | 115.3 | 8.0889 | 0.070156 |
| kvaerno3 | adaptive | pi | 131072 | 891.55 | 76.708 | 0.086039 |
| kvaerno3 | adaptive | pi | 2048 | 119.81 | 9.2155 | 0.076917 |
| kvaerno3 | adaptive | pi | 2097152 | 13145 | 1093.2 | 0.083165 |
| kvaerno3 | adaptive | pi | 32 | 146.75 | 8.1837 | 0.055767 |
| kvaerno3 | adaptive | pi | 32768 | 282.74 | 26.409 | 0.093404 |
| kvaerno3 | adaptive | pi | 512 | 117.19 | 8.4667 | 0.072247 |
| kvaerno3 | adaptive | pi | 524288 | 3341.2 | 283.42 | 0.084826 |
| kvaerno3 | adaptive | pi | 8 | 87.098 | 7.8836 | 0.090515 |
| kvaerno3 | adaptive | pi | 8192 | 138.21 | 11.13 | 0.080531 |
| kvaerno3 | adaptive | pi | 8388608 | 52351 | 4363.3 | 0.083347 |
| kvaerno3 | adaptive | pi | 128 | 114.97 | 8.0441 | 0.069965 |
| kvaerno3 | adaptive | pi | 131072 | 889.93 | 75.797 | 0.085172 |
| kvaerno3 | adaptive | pi | 2048 | 119.46 | 9.1636 | 0.076711 |
| kvaerno3 | adaptive | pi | 2097152 | 13136 | 1080.6 | 0.082265 |
| kvaerno3 | adaptive | pi | 32 | 146.42 | 8.1392 | 0.055587 |
| kvaerno3 | adaptive | pi | 32768 | 282.07 | 26.14 | 0.092672 |
| kvaerno3 | adaptive | pi | 512 | 116.85 | 8.4255 | 0.072103 |
| kvaerno3 | adaptive | pi | 524288 | 3338.8 | 278.44 | 0.083396 |
| kvaerno3 | adaptive | pi | 8 | 86.749 | 7.8401 | 0.090377 |
| kvaerno3 | adaptive | pi | 8192 | 137.86 | 11.028 | 0.079991 |
| kvaerno3 | adaptive | pi | 8388608 | 52313 | 4267.8 | 0.081583 |
| kvaerno3 | fixed | fixed | 128 | 2.691 | 2.9606 | 1.1002 |
| kvaerno3 | fixed | fixed | 131072 | 25.956 | 23.793 | 0.91666 |
| kvaerno3 | fixed | fixed | 2048 | 2.6577 | 3.3108 | 1.2457 |
| kvaerno3 | fixed | fixed | 2097152 | 387.69 | 338.7 | 0.87364 |
| kvaerno3 | fixed | fixed | 32 | 2.6345 | 2.9494 | 1.1195 |
| kvaerno3 | fixed | fixed | 32768 | 7.8501 | 9.4425 | 1.2029 |
| kvaerno3 | fixed | fixed | 512 | 2.6675 | 3.2078 | 1.2025 |
| kvaerno3 | fixed | fixed | 524288 | 97.896 | 85.229 | 0.87061 |
| kvaerno3 | fixed | fixed | 8 | 2.6173 | 2.957 | 1.1298 |
| kvaerno3 | fixed | fixed | 8192 | 3.1874 | 3.7018 | 1.1614 |
| kvaerno3 | fixed | fixed | 8388608 | 1539.5 | 1292.7 | 0.83967 |
| kvaerno3 | fixed | fixed | 128 | 2.3839 | 2.9113 | 1.2212 |
| kvaerno3 | fixed | fixed | 131072 | 25.182 | 22.875 | 0.90839 |
| kvaerno3 | fixed | fixed | 2048 | 2.3399 | 3.257 | 1.3919 |
| kvaerno3 | fixed | fixed | 2097152 | 378.07 | 315.16 | 0.83361 |
| kvaerno3 | fixed | fixed | 32 | 2.3291 | 2.9073 | 1.2483 |
| kvaerno3 | fixed | fixed | 32768 | 7.434 | 9.1669 | 1.2331 |
| kvaerno3 | fixed | fixed | 512 | 2.3595 | 3.1576 | 1.3383 |
| kvaerno3 | fixed | fixed | 524288 | 95.374 | 81.958 | 0.85933 |
| kvaerno3 | fixed | fixed | 8 | 2.2812 | 2.9113 | 1.2762 |
| kvaerno3 | fixed | fixed | 8192 | 2.8485 | 3.6105 | 1.2675 |
| kvaerno3 | fixed | fixed | 8388608 | 1501.6 | 1243.2 | 0.82791 |
| kvaerno5 | adaptive | default | 128 | 232.37 | 0.91901 | 0.0039549 |
| kvaerno5 | adaptive | default | 131072 | 2147.6 | 14.443 | 0.006725 |
| kvaerno5 | adaptive | default | 2048 | 246.25 | 1.5384 | 0.0062474 |
| kvaerno5 | adaptive | default | 2097152 | 32346 | 208.73 | 0.0064529 |
| kvaerno5 | adaptive | default | 32 | 264.86 | 0.87185 | 0.0032918 |
| kvaerno5 | adaptive | default | 32768 | 637.09 | 4.6925 | 0.0073655 |
| kvaerno5 | adaptive | default | 512 | 233.32 | 1.2096 | 0.0051842 |
| kvaerno5 | adaptive | default | 524288 | 8187.5 | 53.288 | 0.0065085 |
| kvaerno5 | adaptive | default | 8 | 97.786 | 0.87112 | 0.0089085 |
| kvaerno5 | adaptive | default | 8192 | 272.18 | 2.154 | 0.0079141 |
| kvaerno5 | adaptive | default | 8388608 | 1.2886e+05 | 874.83 | 0.006789 |
| kvaerno5 | adaptive | default | 128 | 232.04 | 0.87618 | 0.0037759 |
| kvaerno5 | adaptive | default | 131072 | 2146.8 | 13.524 | 0.0062995 |
| kvaerno5 | adaptive | default | 2048 | 245.92 | 1.4818 | 0.0060257 |
| kvaerno5 | adaptive | default | 2097152 | 32338 | 196.09 | 0.0060638 |
| kvaerno5 | adaptive | default | 32 | 264.52 | 0.82537 | 0.0031202 |
| kvaerno5 | adaptive | default | 32768 | 636.62 | 4.4015 | 0.0069138 |
| kvaerno5 | adaptive | default | 512 | 232.98 | 1.1591 | 0.0049752 |
| kvaerno5 | adaptive | default | 524288 | 8186 | 49.978 | 0.0061054 |
| kvaerno5 | adaptive | default | 8 | 97.474 | 0.82456 | 0.0084592 |
| kvaerno5 | adaptive | default | 8192 | 271.8 | 2.0634 | 0.0075916 |
| kvaerno5 | adaptive | default | 8388608 | 1.2882e+05 | 780.15 | 0.0060562 |
| kvaerno5 | adaptive | pi | 128 | 85.394 | 0.91901 | 0.010762 |
| kvaerno5 | adaptive | pi | 131072 | 766.89 | 14.443 | 0.018832 |
| kvaerno5 | adaptive | pi | 2048 | 92.183 | 1.5384 | 0.016689 |
| kvaerno5 | adaptive | pi | 2097152 | 11471 | 208.73 | 0.018197 |
| kvaerno5 | adaptive | pi | 32 | 101.85 | 0.87185 | 0.0085603 |
| kvaerno5 | adaptive | pi | 32768 | 232.49 | 4.6925 | 0.020184 |
| kvaerno5 | adaptive | pi | 512 | 90.082 | 1.2096 | 0.013428 |
| kvaerno5 | adaptive | pi | 524288 | 2910.9 | 53.288 | 0.018307 |
| kvaerno5 | adaptive | pi | 8 | 38.38 | 0.87112 | 0.022697 |
| kvaerno5 | adaptive | pi | 8192 | 106 | 2.154 | 0.02032 |
| kvaerno5 | adaptive | pi | 8388608 | 45663 | 874.83 | 0.019158 |
| kvaerno5 | adaptive | pi | 128 | 85.072 | 0.87618 | 0.010299 |
| kvaerno5 | adaptive | pi | 131072 | 766.2 | 13.524 | 0.017651 |
| kvaerno5 | adaptive | pi | 2048 | 91.856 | 1.4818 | 0.016132 |
| kvaerno5 | adaptive | pi | 2097152 | 11463 | 196.09 | 0.017107 |
| kvaerno5 | adaptive | pi | 32 | 101.53 | 0.82537 | 0.008129 |
| kvaerno5 | adaptive | pi | 32768 | 232.06 | 4.4015 | 0.018967 |
| kvaerno5 | adaptive | pi | 512 | 89.766 | 1.1591 | 0.012913 |
| kvaerno5 | adaptive | pi | 524288 | 2908 | 49.978 | 0.017187 |
| kvaerno5 | adaptive | pi | 8 | 38.069 | 0.82456 | 0.021659 |
| kvaerno5 | adaptive | pi | 8192 | 105.66 | 2.0634 | 0.019528 |
| kvaerno5 | adaptive | pi | 8388608 | 45622 | 780.15 | 0.0171 |
| kvaerno5 | fixed | fixed | 128 | 7.1237 | 6.1108 | 0.85782 |
| kvaerno5 | fixed | fixed | 131072 | 108.22 | 59.669 | 0.55137 |
| kvaerno5 | fixed | fixed | 2048 | 7.0907 | 6.61 | 0.93219 |
| kvaerno5 | fixed | fixed | 2097152 | 1692.5 | 834.57 | 0.49311 |
| kvaerno5 | fixed | fixed | 32 | 7.034 | 5.9567 | 0.84684 |
| kvaerno5 | fixed | fixed | 32768 | 29.891 | 19.951 | 0.66745 |
| kvaerno5 | fixed | fixed | 512 | 7.0962 | 6.4205 | 0.90478 |
| kvaerno5 | fixed | fixed | 524288 | 425.88 | 211.59 | 0.49684 |
| kvaerno5 | fixed | fixed | 8 | 6.9385 | 5.9734 | 0.86091 |
| kvaerno5 | fixed | fixed | 8192 | 8.3931 | 8.2177 | 0.9791 |
| kvaerno5 | fixed | fixed | 8388608 | 6759.7 | 3281.6 | 0.48547 |
| kvaerno5 | fixed | fixed | 128 | 6.8128 | 6.0675 | 0.8906 |
| kvaerno5 | fixed | fixed | 131072 | 107.44 | 58.689 | 0.54625 |
| kvaerno5 | fixed | fixed | 2048 | 6.7773 | 6.5515 | 0.96669 |
| kvaerno5 | fixed | fixed | 2097152 | 1682.8 | 811.02 | 0.48195 |
| kvaerno5 | fixed | fixed | 32 | 6.7284 | 5.9072 | 0.87795 |
| kvaerno5 | fixed | fixed | 32768 | 29.448 | 19.695 | 0.66883 |
| kvaerno5 | fixed | fixed | 512 | 6.7819 | 6.373 | 0.9397 |
| kvaerno5 | fixed | fixed | 524288 | 423.31 | 208.04 | 0.49145 |
| kvaerno5 | fixed | fixed | 8 | 6.6241 | 5.926 | 0.89462 |
| kvaerno5 | fixed | fixed | 8192 | 8.0402 | 8.1186 | 1.0098 |
| kvaerno5 | fixed | fixed | 8388608 | 6721.7 | 3187.5 | 0.47421 |
| rosenbrock23_sciml | adaptive | default | 128 | 4.0521 | 3.2017 | 0.79012 |
| rosenbrock23_sciml | adaptive | default | 131072 | 30.439 | 27.543 | 0.90487 |
| rosenbrock23_sciml | adaptive | default | 2048 | 3.8482 | 3.2049 | 0.83284 |
| rosenbrock23_sciml | adaptive | default | 2097152 | 445.98 | 415.19 | 0.93095 |
| rosenbrock23_sciml | adaptive | default | 32 | 5.4017 | 3.1637 | 0.58568 |
| rosenbrock23_sciml | adaptive | default | 32768 | 9.8741 | 8.4779 | 0.85859 |
| rosenbrock23_sciml | adaptive | default | 512 | 3.8759 | 3.2201 | 0.83082 |
| rosenbrock23_sciml | adaptive | default | 524288 | 112.18 | 102.45 | 0.91327 |
| rosenbrock23_sciml | adaptive | default | 8 | 5.096 | 3.1364 | 0.61547 |
| rosenbrock23_sciml | adaptive | default | 8192 | 4.4208 | 3.8977 | 0.88167 |
| rosenbrock23_sciml | adaptive | default | 8388608 | 1764.3 | 1637.7 | 0.92828 |
| rosenbrock23_sciml | adaptive | default | 128 | 3.7413 | 3.1489 | 0.84165 |
| rosenbrock23_sciml | adaptive | default | 131072 | 29.642 | 26.662 | 0.89947 |
| rosenbrock23_sciml | adaptive | default | 2048 | 3.5179 | 3.1407 | 0.89276 |
| rosenbrock23_sciml | adaptive | default | 2097152 | 432.57 | 391.63 | 0.90535 |
| rosenbrock23_sciml | adaptive | default | 32 | 5.0894 | 3.1122 | 0.61151 |
| rosenbrock23_sciml | adaptive | default | 32768 | 9.442 | 8.1885 | 0.86725 |
| rosenbrock23_sciml | adaptive | default | 512 | 3.5453 | 3.1644 | 0.89257 |
| rosenbrock23_sciml | adaptive | default | 524288 | 109.65 | 99.108 | 0.90387 |
| rosenbrock23_sciml | adaptive | default | 8 | 4.7901 | 3.0844 | 0.64391 |
| rosenbrock23_sciml | adaptive | default | 8192 | 4.0665 | 3.7793 | 0.92938 |
| rosenbrock23_sciml | adaptive | default | 8388608 | 1726.2 | 1543.9 | 0.89438 |
| rosenbrock23_sciml | adaptive | pi | 128 | 4.902 | 3.2017 | 0.65314 |
| rosenbrock23_sciml | adaptive | pi | 131072 | 38.074 | 27.543 | 0.72341 |
| rosenbrock23_sciml | adaptive | pi | 2048 | 4.6793 | 3.2049 | 0.68492 |
| rosenbrock23_sciml | adaptive | pi | 2097152 | 556.82 | 415.19 | 0.74565 |
| rosenbrock23_sciml | adaptive | pi | 32 | 6.5305 | 3.1637 | 0.48444 |
| rosenbrock23_sciml | adaptive | pi | 32768 | 12.332 | 8.4779 | 0.68746 |
| rosenbrock23_sciml | adaptive | pi | 512 | 4.6956 | 3.2201 | 0.68578 |
| rosenbrock23_sciml | adaptive | pi | 524288 | 141.24 | 102.45 | 0.72535 |
| rosenbrock23_sciml | adaptive | pi | 8 | 6.0508 | 3.1364 | 0.51834 |
| rosenbrock23_sciml | adaptive | pi | 8192 | 5.3567 | 3.8977 | 0.72764 |
| rosenbrock23_sciml | adaptive | pi | 8388608 | 2221.3 | 1637.7 | 0.73729 |
| rosenbrock23_sciml | adaptive | pi | 128 | 4.5896 | 3.1489 | 0.68609 |
| rosenbrock23_sciml | adaptive | pi | 131072 | 37.282 | 26.662 | 0.71516 |
| rosenbrock23_sciml | adaptive | pi | 2048 | 4.3461 | 3.1407 | 0.72264 |
| rosenbrock23_sciml | adaptive | pi | 2097152 | 547.41 | 391.63 | 0.71542 |
| rosenbrock23_sciml | adaptive | pi | 32 | 6.2181 | 3.1122 | 0.50051 |
| rosenbrock23_sciml | adaptive | pi | 32768 | 11.892 | 8.1885 | 0.68856 |
| rosenbrock23_sciml | adaptive | pi | 512 | 4.3665 | 3.1644 | 0.7247 |
| rosenbrock23_sciml | adaptive | pi | 524288 | 138.68 | 99.108 | 0.71467 |
| rosenbrock23_sciml | adaptive | pi | 8 | 5.7467 | 3.0844 | 0.53673 |
| rosenbrock23_sciml | adaptive | pi | 8192 | 5.0017 | 3.7793 | 0.75561 |
| rosenbrock23_sciml | adaptive | pi | 8388608 | 2183.3 | 1543.9 | 0.70713 |
| rosenbrock23_sciml | fixed | fixed | 128 | 1.8481 | 1.1325 | 0.61279 |
| rosenbrock23_sciml | fixed | fixed | 131072 | 17.713 | 7.3559 | 0.41528 |
| rosenbrock23_sciml | fixed | fixed | 2048 | 1.8552 | 1.1844 | 0.63841 |
| rosenbrock23_sciml | fixed | fixed | 2097152 | 260.52 | 122.06 | 0.46855 |
| rosenbrock23_sciml | fixed | fixed | 32 | 1.8068 | 1.1258 | 0.62307 |
| rosenbrock23_sciml | fixed | fixed | 32768 | 5.3919 | 2.3244 | 0.43109 |
| rosenbrock23_sciml | fixed | fixed | 512 | 1.8746 | 1.1733 | 0.62592 |
| rosenbrock23_sciml | fixed | fixed | 524288 | 66.631 | 29.071 | 0.43631 |
| rosenbrock23_sciml | fixed | fixed | 8 | 1.7473 | 1.1256 | 0.64422 |
| rosenbrock23_sciml | fixed | fixed | 8192 | 2.2463 | 1.2944 | 0.57625 |
| rosenbrock23_sciml | fixed | fixed | 8388608 | 1037.9 | 485.05 | 0.46735 |
| rosenbrock23_sciml | fixed | fixed | 128 | 1.5311 | 1.0787 | 0.7045 |
| rosenbrock23_sciml | fixed | fixed | 131072 | 16.921 | 6.4778 | 0.38281 |
| rosenbrock23_sciml | fixed | fixed | 2048 | 1.5253 | 1.1232 | 0.73638 |
| rosenbrock23_sciml | fixed | fixed | 2097152 | 250.91 | 98.612 | 0.39302 |
| rosenbrock23_sciml | fixed | fixed | 32 | 1.4941 | 1.0741 | 0.71887 |
| rosenbrock23_sciml | fixed | fixed | 32768 | 4.9596 | 2.0377 | 0.41087 |
| rosenbrock23_sciml | fixed | fixed | 512 | 1.5357 | 1.1189 | 0.72858 |
| rosenbrock23_sciml | fixed | fixed | 524288 | 64.071 | 25.813 | 0.40287 |
| rosenbrock23_sciml | fixed | fixed | 8 | 1.4465 | 1.0729 | 0.74175 |
| rosenbrock23_sciml | fixed | fixed | 8192 | 1.8921 | 1.189 | 0.62837 |
| rosenbrock23_sciml | fixed | fixed | 8388608 | 1000.1 | 391.18 | 0.39112 |
| tsit5 | adaptive | default | 128 | 0.75332 | 0.34943 | 0.46386 |
| tsit5 | adaptive | default | 131072 | 2.3415 | 4.3705 | 1.8666 |
| tsit5 | adaptive | default | 2048 | 0.77095 | 0.48963 | 0.6351 |
| tsit5 | adaptive | default | 2097152 | 28.328 | 72.978 | 2.5762 |
| tsit5 | adaptive | default | 32 | 0.78863 | 0.33119 | 0.41995 |
| tsit5 | adaptive | default | 32768 | 1.1288 | 1.4402 | 1.2759 |
| tsit5 | adaptive | default | 512 | 0.76532 | 0.38195 | 0.49907 |
| tsit5 | adaptive | default | 524288 | 7.5163 | 18.651 | 2.4813 |
| tsit5 | adaptive | default | 8 | 0.77846 | 0.32425 | 0.41653 |
| tsit5 | adaptive | default | 8192 | 0.82214 | 0.67193 | 0.8173 |
| tsit5 | adaptive | default | 8388608 | 111.88 | 289.98 | 2.592 |
| tsit5 | adaptive | default | 128 | 0.44505 | 0.31136 | 0.69961 |
| tsit5 | adaptive | default | 131072 | 1.5529 | 3.4867 | 2.2452 |
| tsit5 | adaptive | default | 2048 | 0.44894 | 0.42871 | 0.95492 |
| tsit5 | adaptive | default | 2097152 | 18.543 | 49.471 | 2.6679 |
| tsit5 | adaptive | default | 32 | 0.47997 | 0.29392 | 0.61237 |
| tsit5 | adaptive | default | 32768 | 0.70963 | 1.1652 | 1.642 |
| tsit5 | adaptive | default | 512 | 0.45149 | 0.34145 | 0.75628 |
| tsit5 | adaptive | default | 524288 | 4.942 | 12.66 | 2.5617 |
| tsit5 | adaptive | default | 8 | 0.47081 | 0.28537 | 0.60613 |
| tsit5 | adaptive | default | 8192 | 0.48068 | 0.53208 | 1.1069 |
| tsit5 | adaptive | default | 8388608 | 73.017 | 196.59 | 2.6924 |
| tsit5 | adaptive | pi | 128 | 0.72429 | 0.34943 | 0.48244 |
| tsit5 | adaptive | pi | 131072 | 2.0919 | 4.3705 | 2.0892 |
| tsit5 | adaptive | pi | 2048 | 0.72672 | 0.48963 | 0.67375 |
| tsit5 | adaptive | pi | 2097152 | 24.705 | 72.978 | 2.954 |
| tsit5 | adaptive | pi | 32 | 0.74212 | 0.33119 | 0.44627 |
| tsit5 | adaptive | pi | 32768 | 1.0471 | 1.4402 | 1.3755 |
| tsit5 | adaptive | pi | 512 | 0.71503 | 0.38195 | 0.53418 |
| tsit5 | adaptive | pi | 524288 | 6.6091 | 18.651 | 2.8219 |
| tsit5 | adaptive | pi | 8 | 0.74066 | 0.32425 | 0.43779 |
| tsit5 | adaptive | pi | 8192 | 0.78104 | 0.67193 | 0.8603 |
| tsit5 | adaptive | pi | 8388608 | 97.266 | 289.98 | 2.9813 |
| tsit5 | adaptive | pi | 128 | 0.41421 | 0.31136 | 0.75168 |
| tsit5 | adaptive | pi | 131072 | 1.3073 | 3.4867 | 2.6671 |
| tsit5 | adaptive | pi | 2048 | 0.41221 | 0.42871 | 1.04 |
| tsit5 | adaptive | pi | 2097152 | 14.945 | 49.471 | 3.3103 |
| tsit5 | adaptive | pi | 32 | 0.43694 | 0.29392 | 0.67268 |
| tsit5 | adaptive | pi | 32768 | 0.62529 | 1.1652 | 1.8635 |
| tsit5 | adaptive | pi | 512 | 0.40563 | 0.34145 | 0.84178 |
| tsit5 | adaptive | pi | 524288 | 4.0227 | 12.66 | 3.1472 |
| tsit5 | adaptive | pi | 8 | 0.43421 | 0.28537 | 0.65722 |
| tsit5 | adaptive | pi | 8192 | 0.44095 | 0.53208 | 1.2067 |
| tsit5 | adaptive | pi | 8388608 | 58.192 | 196.59 | 3.3783 |
| tsit5 | fixed | fixed | 128 | 0.92666 | 0.44272 | 0.47777 |
| tsit5 | fixed | fixed | 131072 | 7.461 | 10.554 | 1.4145 |
| tsit5 | fixed | fixed | 2048 | 0.93676 | 0.4897 | 0.52275 |
| tsit5 | fixed | fixed | 2097152 | 109.25 | 157.82 | 1.4445 |
| tsit5 | fixed | fixed | 32 | 0.92169 | 0.4939 | 0.53586 |
| tsit5 | fixed | fixed | 32768 | 2.4256 | 2.6224 | 1.0811 |
| tsit5 | fixed | fixed | 512 | 0.92693 | 0.44284 | 0.47774 |
| tsit5 | fixed | fixed | 524288 | 27.785 | 41.201 | 1.4829 |
| tsit5 | fixed | fixed | 8 | 0.92668 | 0.4891 | 0.5278 |
| tsit5 | fixed | fixed | 8192 | 1.1192 | 0.74695 | 0.66741 |
| tsit5 | fixed | fixed | 8388608 | 434.75 | 625.6 | 1.439 |
| tsit5 | fixed | fixed | 128 | 0.6136 | 0.40398 | 0.65837 |
| tsit5 | fixed | fixed | 131072 | 6.6715 | 8.933 | 1.339 |
| tsit5 | fixed | fixed | 2048 | 0.61548 | 0.44425 | 0.7218 |
| tsit5 | fixed | fixed | 2097152 | 99.365 | 133.99 | 1.3484 |
| tsit5 | fixed | fixed | 32 | 0.612 | 0.43002 | 0.70264 |
| tsit5 | fixed | fixed | 32768 | 2.0052 | 2.3319 | 1.1629 |
| tsit5 | fixed | fixed | 512 | 0.61407 | 0.40268 | 0.65575 |
| tsit5 | fixed | fixed | 524288 | 25.186 | 35.242 | 1.3992 |
| tsit5 | fixed | fixed | 8 | 0.6145 | 0.44115 | 0.7179 |
| tsit5 | fixed | fixed | 8192 | 0.77659 | 0.65744 | 0.84657 |
| tsit5 | fixed | fixed | 8388608 | 395.94 | 532.06 | 1.3438 |
| vern7 | adaptive | default | 128 | 0.82413 | 0.6325 | 0.76747 |
| vern7 | adaptive | default | 131072 | 3.7035 | 8.9727 | 2.4228 |
| vern7 | adaptive | default | 2048 | 0.83154 | 0.72611 | 0.87321 |
| vern7 | adaptive | default | 2097152 | 49.661 | 134.45 | 2.7073 |
| vern7 | adaptive | default | 32 | 0.85971 | 0.60843 | 0.70772 |
| vern7 | adaptive | default | 32768 | 1.5202 | 2.6426 | 1.7383 |
| vern7 | adaptive | default | 512 | 0.82528 | 0.6724 | 0.81475 |
| vern7 | adaptive | default | 524288 | 12.865 | 34.683 | 2.6959 |
| vern7 | adaptive | default | 8 | 0.85047 | 0.60895 | 0.71602 |
| vern7 | adaptive | default | 8192 | 0.9262 | 0.98486 | 1.0633 |
| vern7 | adaptive | default | 8388608 | 195.57 | 574.15 | 2.9358 |
| vern7 | adaptive | default | 128 | 0.51309 | 0.58546 | 1.141 |
| vern7 | adaptive | default | 131072 | 2.9315 | 8.1085 | 2.766 |
| vern7 | adaptive | default | 2048 | 0.51112 | 0.6681 | 1.3071 |
| vern7 | adaptive | default | 2097152 | 40.105 | 121.81 | 3.0373 |
| vern7 | adaptive | default | 32 | 0.54698 | 0.56593 | 1.0347 |
| vern7 | adaptive | default | 32768 | 1.1028 | 2.3714 | 2.1503 |
| vern7 | adaptive | default | 512 | 0.51416 | 0.6241 | 1.2138 |
| vern7 | adaptive | default | 524288 | 10.356 | 30.755 | 2.97 |
| vern7 | adaptive | default | 8 | 0.54315 | 0.56044 | 1.0318 |
| vern7 | adaptive | default | 8192 | 0.58525 | 0.89398 | 1.5275 |
| vern7 | adaptive | default | 8388608 | 157.77 | 481 | 3.0488 |
| vern7 | adaptive | pi | 128 | 0.76878 | 0.6325 | 0.82273 |
| vern7 | adaptive | pi | 131072 | 3.1582 | 8.9727 | 2.8411 |
| vern7 | adaptive | pi | 2048 | 0.78417 | 0.72611 | 0.92596 |
| vern7 | adaptive | pi | 2097152 | 41.339 | 134.45 | 3.2523 |
| vern7 | adaptive | pi | 32 | 0.79819 | 0.60843 | 0.76227 |
| vern7 | adaptive | pi | 32768 | 1.3554 | 2.6426 | 1.9496 |
| vern7 | adaptive | pi | 512 | 0.7698 | 0.6724 | 0.87347 |
| vern7 | adaptive | pi | 524288 | 10.773 | 34.683 | 3.2194 |
| vern7 | adaptive | pi | 8 | 0.7873 | 0.60895 | 0.77347 |
| vern7 | adaptive | pi | 8192 | 0.85925 | 0.98486 | 1.1462 |
| vern7 | adaptive | pi | 8388608 | 162.58 | 574.15 | 3.5315 |
| vern7 | adaptive | pi | 128 | 0.45745 | 0.58546 | 1.2798 |
| vern7 | adaptive | pi | 131072 | 2.3802 | 8.1085 | 3.4066 |
| vern7 | adaptive | pi | 2048 | 0.46235 | 0.6681 | 1.445 |
| vern7 | adaptive | pi | 2097152 | 31.764 | 121.81 | 3.8349 |
| vern7 | adaptive | pi | 32 | 0.4871 | 0.56593 | 1.1618 |
| vern7 | adaptive | pi | 32768 | 0.93421 | 2.3714 | 2.5384 |
| vern7 | adaptive | pi | 512 | 0.45616 | 0.6241 | 1.3682 |
| vern7 | adaptive | pi | 524288 | 8.254 | 30.755 | 3.7261 |
| vern7 | adaptive | pi | 8 | 0.48401 | 0.56044 | 1.1579 |
| vern7 | adaptive | pi | 8192 | 0.51633 | 0.89398 | 1.7314 |
| vern7 | adaptive | pi | 8388608 | 124.83 | 481 | 3.8532 |
| vern7 | fixed | fixed | 128 | 1.0863 | 0.71126 | 0.65478 |
| vern7 | fixed | fixed | 131072 | 12.283 | 22.441 | 1.8269 |
| vern7 | fixed | fixed | 2048 | 1.0947 | 0.75056 | 0.68562 |
| vern7 | fixed | fixed | 2097152 | 183.64 | 338.03 | 1.8407 |
| vern7 | fixed | fixed | 32 | 1.0834 | 0.70556 | 0.65122 |
| vern7 | fixed | fixed | 32768 | 3.7289 | 7.1968 | 1.93 |
| vern7 | fixed | fixed | 512 | 1.0823 | 0.73699 | 0.68092 |
| vern7 | fixed | fixed | 524288 | 46.574 | 85.309 | 1.8317 |
| vern7 | fixed | fixed | 8 | 1.075 | 0.69778 | 0.64907 |
| vern7 | fixed | fixed | 8192 | 1.4786 | 1.2476 | 0.84377 |
| vern7 | fixed | fixed | 8388608 | 732.56 | 1348.3 | 1.8406 |
| vern7 | fixed | fixed | 128 | 0.77166 | 0.66215 | 0.85809 |
| vern7 | fixed | fixed | 131072 | 11.504 | 21.56 | 1.8742 |
| vern7 | fixed | fixed | 2048 | 0.7745 | 0.69289 | 0.89462 |
| vern7 | fixed | fixed | 2097152 | 174.02 | 314.07 | 1.8048 |
| vern7 | fixed | fixed | 32 | 0.7745 | 0.65957 | 0.85161 |
| vern7 | fixed | fixed | 32768 | 3.3158 | 6.9087 | 2.0836 |
| vern7 | fixed | fixed | 512 | 0.77022 | 0.68609 | 0.89077 |
| vern7 | fixed | fixed | 524288 | 44.05 | 79.593 | 1.8069 |
| vern7 | fixed | fixed | 8 | 0.7707 | 0.65143 | 0.84525 |
| vern7 | fixed | fixed | 8192 | 1.1352 | 1.1564 | 1.0187 |
| vern7 | fixed | fixed | 8388608 | 694.72 | 1252.7 | 1.8032 |

## Numerical equivalence

Per-trajectory Float32 finals are retained beneath `finals/`. Mutual metrics are elementwise at t=1.

| algorithm | mode | cubie_tier | setting | mutual_rmse | mutual_p99_abs | mutual_max_abs | failed_pairs |
|---|---|---|---|---|---|---|---|
| tsit5 | fixed | fixed | 0.125 | 4.2182e-05 | 0.00020026 | 0.00055885 | 0 |
| tsit5 | fixed | fixed | 0.0625 | 1.3053e-05 | 5.272e-05 | 0.0001106 | 0 |
| tsit5 | fixed | fixed | 0.03125 | 5.6772e-06 | 2.1596e-05 | 4.576e-05 | 0 |
| tsit5 | fixed | fixed | 0.015625 | 3.0163e-06 | 1.1419e-05 | 2.7669e-05 | 0 |
| tsit5 | fixed | fixed | 0.0078125 | 2.5527e-06 | 9.541e-06 | 2.0034e-05 | 0 |
| tsit5 | fixed | fixed | 0.00390625 | 2.3941e-06 | 8.8921e-06 | 2.0017e-05 | 0 |
| tsit5 | fixed | fixed | 0.001953125 | 2.5502e-06 | 1.0445e-05 | 1.9316e-05 | 0 |
| tsit5 | fixed | fixed | 0.0009765625 | 2.4147e-06 | 9.5085e-06 | 2.1935e-05 | 0 |
| tsit5 | fixed | fixed | 0.00048828125 | 2.353e-06 | 8.6769e-06 | 1.883e-05 | 0 |
| tsit5 | fixed | fixed | 0.000244140625 | 2.4234e-06 | 9.3522e-06 | 1.8832e-05 | 0 |
| tsit5 | fixed | fixed | 0.0001220703125 | 2.3523e-06 | 9.5338e-06 | 1.5739e-05 | 0 |
| tsit5 | adaptive | default | 0.01 | 0.045814 | 0.14006 | 0.14332 | 0 |
| tsit5 | adaptive | default | 0.001 | 0.0039447 | 0.012882 | 0.013998 | 0 |
| tsit5 | adaptive | default | 0.0001 | 0.00038198 | 0.0010432 | 0.0011521 | 0 |
| tsit5 | adaptive | default | 1e-05 | 2.2766e-05 | 6.4861e-05 | 8.9621e-05 | 0 |
| tsit5 | adaptive | default | 1e-06 | 3.6842e-06 | 1.4124e-05 | 2.9587e-05 | 0 |
| tsit5 | adaptive | pi | 0.01 | 0.044212 | 0.18251 | 0.19211 | 0 |
| tsit5 | adaptive | pi | 0.001 | 0.0034107 | 0.0089219 | 0.010269 | 0 |
| tsit5 | adaptive | pi | 0.0001 | 9.0628e-05 | 0.00026757 | 0.00031257 | 0 |
| tsit5 | adaptive | pi | 1e-05 | 8.1388e-06 | 3.0181e-05 | 5.4402e-05 | 0 |
| tsit5 | adaptive | pi | 1e-06 | 3.4437e-06 | 1.2158e-05 | 2.5713e-05 | 0 |
| vern7 | fixed | fixed | 0.125 | 2.2631e-05 | 9.727e-05 | 0.00016642 | 0 |
| vern7 | fixed | fixed | 0.0625 | 5.8556e-06 | 2.2896e-05 | 4.1933e-05 | 0 |
| vern7 | fixed | fixed | 0.03125 | 2.6368e-06 | 1.141e-05 | 2.4791e-05 | 0 |
| vern7 | fixed | fixed | 0.015625 | 1.6349e-06 | 5.8868e-06 | 1.1484e-05 | 0 |
| vern7 | fixed | fixed | 0.0078125 | 1.8177e-06 | 7.5956e-06 | 1.6236e-05 | 0 |
| vern7 | fixed | fixed | 0.00390625 | 1.8706e-06 | 7.5931e-06 | 1.5214e-05 | 0 |
| vern7 | fixed | fixed | 0.001953125 | 1.8294e-06 | 7.5838e-06 | 1.5258e-05 | 0 |
| vern7 | fixed | fixed | 0.0009765625 | 1.9243e-06 | 7.9378e-06 | 1.9076e-05 | 0 |
| vern7 | fixed | fixed | 0.00048828125 | 1.716e-06 | 6.6318e-06 | 1.9547e-05 | 0 |
| vern7 | fixed | fixed | 0.000244140625 | 1.8723e-06 | 7.4291e-06 | 1.8839e-05 | 0 |
| vern7 | fixed | fixed | 0.0001220703125 | 1.9347e-06 | 7.9365e-06 | 1.4348e-05 | 0 |
| vern7 | adaptive | default | 0.01 | 0.0038351 | 0.01713 | 0.034893 | 0 |
| vern7 | adaptive | default | 0.001 | 0.00012276 | 0.00042483 | 0.00081155 | 0 |
| vern7 | adaptive | default | 0.0001 | 1.4977e-05 | 6.3177e-05 | 0.00019841 | 0 |
| vern7 | adaptive | default | 1e-05 | 3.6784e-06 | 1.3354e-05 | 3.2404e-05 | 0 |
| vern7 | adaptive | default | 1e-06 | 2.5682e-06 | 9.4881e-06 | 1.6223e-05 | 0 |
| vern7 | adaptive | pi | 0.01 | 0.0053826 | 0.023208 | 0.029528 | 0 |
| vern7 | adaptive | pi | 0.001 | 6.8807e-05 | 0.00027507 | 0.00060702 | 0 |
| vern7 | adaptive | pi | 0.0001 | 1.3208e-05 | 5.8934e-05 | 0.00016884 | 0 |
| vern7 | adaptive | pi | 1e-05 | 3.7118e-06 | 1.2439e-05 | 3.4369e-05 | 0 |
| vern7 | adaptive | pi | 1e-06 | 2.5287e-06 | 9.1959e-06 | 1.8089e-05 | 0 |
| rosenbrock23_sciml | fixed | fixed | 0.125 | 2.2336e-06 | 8.536e-06 | 1.719e-05 | 0 |
| rosenbrock23_sciml | fixed | fixed | 0.0625 | 1.4314e-06 | 4.7993e-06 | 1.2418e-05 | 0 |
| rosenbrock23_sciml | fixed | fixed | 0.03125 | 1.5489e-06 | 5.076e-06 | 9.4884e-06 | 0 |
| rosenbrock23_sciml | fixed | fixed | 0.015625 | 2.1021e-06 | 6.6837e-06 | 1.7173e-05 | 0 |
| rosenbrock23_sciml | fixed | fixed | 0.0078125 | 3.0034e-06 | 1.0227e-05 | 2.3797e-05 | 0 |
| rosenbrock23_sciml | fixed | fixed | 0.00390625 | 4.0908e-06 | 1.4462e-05 | 2.763e-05 | 0 |
| rosenbrock23_sciml | fixed | fixed | 0.001953125 | 5.8789e-06 | 2.1582e-05 | 5.0542e-05 | 0 |
| rosenbrock23_sciml | fixed | fixed | 0.0009765625 | 8.8719e-06 | 3.0522e-05 | 6.4879e-05 | 0 |
| rosenbrock23_sciml | fixed | fixed | 0.00048828125 | 1.6084e-05 | 5.2454e-05 | 9.2497e-05 | 0 |
| rosenbrock23_sciml | fixed | fixed | 0.000244140625 | 5.244e-05 | 0.00017955 | 0.00023551 | 0 |
| rosenbrock23_sciml | fixed | fixed | 0.0001220703125 | 0.00023576 | 0.00087592 | 0.00099659 | 0 |
| rosenbrock23_sciml | adaptive | default | 0.01 | 0.072737 | 0.18655 | 0.21437 | 0 |
| rosenbrock23_sciml | adaptive | default | 0.001 | 0.005731 | 0.012535 | 0.014807 | 0 |
| rosenbrock23_sciml | adaptive | default | 0.0001 | 0.00065577 | 0.001557 | 0.0018029 | 0 |
| rosenbrock23_sciml | adaptive | default | 1e-05 | 0.00017242 | 0.00035665 | 0.00042153 | 0 |
| rosenbrock23_sciml | adaptive | default | 1e-06 | 4.0775e-05 | 9.642e-05 | 0.00012402 | 0 |
| rosenbrock23_sciml | adaptive | pi | 0.01 | 0.0094901 | 0.024455 | 0.025971 | 0 |
| rosenbrock23_sciml | adaptive | pi | 0.001 | 0.0016345 | 0.003768 | 0.0043355 | 0 |
| rosenbrock23_sciml | adaptive | pi | 0.0001 | 0.00094103 | 0.002545 | 0.002811 | 0 |
| rosenbrock23_sciml | adaptive | pi | 1e-05 | 0.00020874 | 0.00049645 | 0.00052934 | 0 |
| rosenbrock23_sciml | adaptive | pi | 1e-06 | 4.9435e-05 | 0.00011185 | 0.00013638 | 0 |
| kvaerno3 | fixed | fixed | 0.015625 | 4.0944e-05 | 0.00013094 | 0.00014782 | 0 |
| kvaerno3 | fixed | fixed | 0.0078125 | 1.3194e-05 | 4.8629e-05 | 6.1966e-05 | 0 |
| kvaerno3 | fixed | fixed | 0.00390625 | 5.1914e-06 | 1.725e-05 | 3.3414e-05 | 0 |
| kvaerno3 | fixed | fixed | 0.001953125 | 7.403e-06 | 2.5752e-05 | 6.7703e-05 | 0 |
| kvaerno3 | fixed | fixed | 0.0009765625 | 1.0638e-05 | 3.6521e-05 | 7.8239e-05 | 0 |
| kvaerno3 | fixed | fixed | 0.00048828125 | 1.4639e-05 | 5.0542e-05 | 0.00010108 | 0 |
| kvaerno3 | fixed | fixed | 0.000244140625 | 1.9795e-05 | 6.772e-05 | 0.00015828 | 0 |
| kvaerno3 | fixed | fixed | 0.0001220703125 | 2.7265e-05 | 9.3115e-05 | 0.00018501 | 0 |
| kvaerno3 | adaptive | default | 0.01 | 0.017358 | 0.045667 | 0.063603 | 0 |
| kvaerno3 | adaptive | default | 0.001 | 0.0020574 | 0.0052252 | 0.0053739 | 0 |
| kvaerno3 | adaptive | default | 0.0001 | 0.00026924 | 0.00069906 | 0.00075145 | 0 |
| kvaerno3 | adaptive | default | 1e-05 | 2.9926e-05 | 8.2938e-05 | 0.0001364 | 0 |
| kvaerno3 | adaptive | default | 1e-06 | 1.0894e-05 | 4.0193e-05 | 7.439e-05 | 0 |
| kvaerno3 | adaptive | pi | 0.01 | 0.019687 | 0.064499 | 0.069169 | 0 |
| kvaerno3 | adaptive | pi | 0.001 | 0.0010754 | 0.0035132 | 0.0037365 | 0 |
| kvaerno3 | adaptive | pi | 0.0001 | 4.7399e-05 | 0.0001284 | 0.00016689 | 0 |
| kvaerno3 | adaptive | pi | 1e-05 | 9.3133e-06 | 3.2089e-05 | 7.2526e-05 | 0 |
| kvaerno3 | adaptive | pi | 1e-06 | 1.0858e-05 | 3.8061e-05 | 7.2454e-05 | 0 |
| kvaerno5 | fixed | fixed | 0.015625 | 3.9247e-06 | 1.3487e-05 | 3.3408e-05 | 0 |
| kvaerno5 | fixed | fixed | 0.0078125 | 4.5787e-06 | 1.5226e-05 | 3.3336e-05 | 0 |
| kvaerno5 | fixed | fixed | 0.00390625 | 6.3763e-06 | 2.2535e-05 | 4.006e-05 | 0 |
| kvaerno5 | fixed | fixed | 0.001953125 | 8.733e-06 | 3.0822e-05 | 6.6767e-05 | 0 |
| kvaerno5 | fixed | fixed | 0.0009765625 | 1.1901e-05 | 4.1127e-05 | 6.7233e-05 | 0 |
| kvaerno5 | fixed | fixed | 0.00048828125 | 1.7109e-05 | 6.1515e-05 | 0.00011158 | 0 |
| kvaerno5 | fixed | fixed | 0.000244140625 | 2.3101e-05 | 7.879e-05 | 0.00020598 | 0 |
| kvaerno5 | fixed | fixed | 0.0001220703125 | 3.6503e-05 | 0.00012936 | 0.00031377 | 0 |
| kvaerno5 | adaptive | default | 0.01 | 0.43374 | 1.1583 | 1.3617 | 0 |
| kvaerno5 | adaptive | default | 0.001 | 0.026273 | 0.092844 | 0.10173 | 0 |
| kvaerno5 | adaptive | default | 0.0001 | 0.0025425 | 0.0077308 | 0.0090933 | 0 |
| kvaerno5 | adaptive | default | 1e-05 | 0.00026951 | 0.0007123 | 0.00076679 | 0 |
| kvaerno5 | adaptive | default | 1e-06 | 3.8704e-05 | 0.00010126 | 0.00016786 | 0 |
| kvaerno5 | adaptive | pi | 0.01 | 0.011025 | 0.028507 | 0.036075 | 0 |
| kvaerno5 | adaptive | pi | 0.001 | 0.0064177 | 0.020344 | 0.029638 | 0 |
| kvaerno5 | adaptive | pi | 0.0001 | 0.0016266 | 0.0046933 | 0.005619 | 0 |
| kvaerno5 | adaptive | pi | 1e-05 | 0.00016175 | 0.00046469 | 0.00069329 | 0 |
| kvaerno5 | adaptive | pi | 1e-06 | 1.125e-05 | 3.6759e-05 | 7.151e-05 | 0 |

## Observed fixed-step convergence order

| framework | algorithm | tier | observed_order | usable_intervals |
|---|---|---|---|---|
| cubie | kvaerno3 | fixed | 2.9432 | 4 |
| cubie | kvaerno5 | fixed | 2.507 | 1 |
| cubie | rosenbrock23_sciml | fixed | 1.9427 | 8 |
| cubie | tsit5 | fixed | 6.8424 | 3 |
| cubie | vern7 | fixed | 2.177 | 4 |
| julia | kvaerno3 | fixed | 2.7862 | 9 |
| julia | kvaerno5 | fixed | 3.3923 | 6 |
| julia | rosenbrock23_sciml | fixed | 1.9427 | 8 |
| julia | tsit5 | fixed | 5.3935 | 4 |
| julia | vern7 | fixed | 2.1609 | 4 |

## Work-precision

`work_precision.csv` joins every work-point timing distribution (min/p05/median/p95/max) to golden RMSE. `plots/work_precision.png` plots median runtime on the x-axis against golden RMSE on the y-axis; it is an error-work plot, not another setting sweep.

## Failures and non-finite results

26 point failures were recorded. See the framework failure CSVs for full messages. Non-finite trajectory counts are retained per successful point in the metric CSVs.

| framework | algorithm | phase | mode | tier | setting_kind | setting | error_type | message |
|---|---|---|---|---|---|---|---|---|
| cubie | tsit5 | numerical | fixed | fixed | dt | 0.5 | FloatingPointError | non-finite result: 82/1024 trajectories valid |
| cubie | tsit5 | numerical | fixed | fixed | dt | 0.25 | FloatingPointError | non-finite result: 298/1024 trajectories valid |
| cubie | vern7 | numerical | fixed | fixed | dt | 0.5 | FloatingPointError | non-finite result: 27/1024 trajectories valid |
| cubie | vern7 | numerical | fixed | fixed | dt | 0.25 | FloatingPointError | non-finite result: 722/1024 trajectories valid |
| cubie | rosenbrock23_sciml | numerical | fixed | fixed | dt | 0.5 | FloatingPointError | non-finite result: 472/1024 trajectories valid |
| cubie | rosenbrock23_sciml | numerical | fixed | fixed | dt | 0.25 | FloatingPointError | non-finite result: 708/1024 trajectories valid |
| cubie | kvaerno3 | numerical | fixed | fixed | dt | 0.5 | FloatingPointError | non-finite result: 1/1024 trajectories valid |
| cubie | kvaerno3 | numerical | fixed | fixed | dt | 0.25 | FloatingPointError | non-finite result: 229/1024 trajectories valid |
| cubie | kvaerno3 | numerical | fixed | fixed | dt | 0.125 | FloatingPointError | non-finite result: 444/1024 trajectories valid |
| cubie | kvaerno3 | numerical | fixed | fixed | dt | 0.0625 | FloatingPointError | non-finite result: 817/1024 trajectories valid |
| cubie | kvaerno3 | numerical | fixed | fixed | dt | 0.03125 | FloatingPointError | non-finite result: 1018/1024 trajectories valid |
| cubie | kvaerno3 | work_precision | fixed | fixed | dt | 0.0625 | FloatingPointError | non-finite result: 25705/32768 trajectories valid |
| cubie | kvaerno3 | work_precision | fixed | fixed | dt | 0.03125 | FloatingPointError | non-finite result: 32593/32768 trajectories valid |
| cubie | kvaerno3 | work_precision | fixed | fixed | dt | 0.015625 | FloatingPointError | non-finite result: 32763/32768 trajectories valid |
| cubie | kvaerno5 | numerical | fixed | fixed | dt | 0.5 | FloatingPointError | non-finite result: 151/1024 trajectories valid |
| cubie | kvaerno5 | numerical | fixed | fixed | dt | 0.25 | FloatingPointError | non-finite result: 258/1024 trajectories valid |
| cubie | kvaerno5 | numerical | fixed | fixed | dt | 0.125 | FloatingPointError | non-finite result: 448/1024 trajectories valid |
| cubie | kvaerno5 | numerical | fixed | fixed | dt | 0.0625 | FloatingPointError | non-finite result: 976/1024 trajectories valid |
| cubie | kvaerno5 | numerical | fixed | fixed | dt | 0.03125 | FloatingPointError | non-finite result: 1005/1024 trajectories valid |
| cubie | kvaerno5 | work_precision | fixed | fixed | dt | 0.0625 | FloatingPointError | non-finite result: 31379/32768 trajectories valid |
| cubie | kvaerno5 | work_precision | fixed | fixed | dt | 0.03125 | FloatingPointError | non-finite result: 32055/32768 trajectories valid |
| cubie | kvaerno5 | work_precision | fixed | fixed | dt | 0.015625 | FloatingPointError | non-finite result: 32741/32768 trajectories valid |
| julia | tsit5 | numerical | fixed | fixed | dt | 0.5 | ErrorException | non-finite result: 35/1024 trajectories valid |
| julia | tsit5 | numerical | fixed | fixed | dt | 0.25 | ErrorException | non-finite result: 289/1024 trajectories valid |
| julia | vern7 | numerical | fixed | fixed | dt | 0.5 | ErrorException | non-finite result: 9/1024 trajectories valid |
| julia | vern7 | numerical | fixed | fixed | dt | 0.25 | ErrorException | non-finite result: 708/1024 trajectories valid |

## Artifacts

Plots: `plots/performance_scaling.png`, `plots/numerical_equivalence.png`, and `plots/work_precision.png`. Raw and derived CSVs in this directory are algorithm-, mode-, tier-, N-, and setting-keyed.
