# GPU_ODE_JuliaKernels

Precompiled benchmark systems for the DiffEqGPU kernel path. `SYSTEMS_CODEGEN` (`runner_scripts/julia_systems.jl`) evals the ModelingToolkit-generated functions into this module, and the `@compile_workload` runs one zero-step fixed and adaptive solve per (problem × algorithm) leg through `runner_scripts/julia_prob.jl`, for every problem julia_gpu implements and every algorithm with a `julia_gpu` constructor in `runner_scripts/julia_algorithms.csv`; kernels specialize on types, so those solves warm the same kernels the full-size sweeps use in bounded time. `GPU_ODE_Julia/bench_ode_gpu.jl` merges `GPU_ODE_JuliaKernels.ENTRIES` over its lazy entry table and takes the precompiled entry whenever a trial's construction parameters match it; a cold leg, or a lorenz96 size the catalogue does not carry, builds its entry at run time instead.

## What persists across processes

Everything the workload executes lands in this package's pkgimage. Measured on the RTX 4070 SUPER (lorenz96_20 rosenbrock23 adaptive, first solve in a fresh process; cold cost 70.5s): entries and host-side inference always persist; GPU-side kernel inference persists on julia 1.13.0-rc3 (residual 10.9s) but loads back invalidated on julia 1.12.6 (residual 39.7s). Compiled PTX never persists: CUDA.jl 6.3.0 does not implement GPUCompiler 2.x's `can_persist_results`/relocatable-IR hooks, so the LLVM+ptxas residual recurs per process on every julia version.

## When it recompiles

Triggers: package sources or any included `runner_scripts/*.jl` file; `problems.csv`/`julia_algorithms.csv` (`include_dependency`); any `Manifest.toml` dependency version, transitively; the julia version or depot; the package's `precompile_workload` preference. Non-triggers: bench-script edits, trial files, sweep sizes, tolerances, watchdog settings, and environment variables. `julia_driver.py` runs `Pkg.instantiate(); Pkg.precompile()` once before its legs, which builds the package; on a GPU-less machine skip the workload with `Preferences.set_preferences!(GPU_ODE_JuliaKernels, "precompile_workload" => false)`.
