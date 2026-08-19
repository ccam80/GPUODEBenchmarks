# GPU_ODE_JuliaKernels

Precompiled benchmark systems for the DiffEqGPU kernel path. `SYSTEMS_CODEGEN` (`runner_scripts/julia_systems.jl`) evals the ModelingToolkit-generated functions into this module, and the `@compile_workload` runs one N=4 fixed and adaptive solve per (problem × algorithm) leg through the shared constructors in `runner_scripts/julia_prob.jl`; N, dt, and tolerances are runtime values, so those solves warm the same kernels the full-size sweeps use. `bench_ode_gpu.jl` merges `GPU_ODE_JuliaKernels.ENTRIES` over its lazy entry table. States-sweep grid sizes run under `julia_driver.py run_states` instead.

## GPUCompiler disk cache

`LocalPreferences.toml` sets GPUCompiler's `disk_cache = "true"`; the preference binds only through the root project's `[extras]` GPUCompiler entry. GPUCompiler disk-caches a kernel only when its inferred CodeInstance carries a pkgimage build id (`GPUCompiler/src/execution.jl:171`). On julia 1.12.6 + GPUCompiler 1.23.0 + CUDA 6.2.1 the GPU-owned CodeInstances load back unvalidated (`max_world = 0`, foreign-method CIs dropped), so the cache stays empty and GPU kernels recompile per process; the pkgimage still serves the entries and the host-side solve-path inference. A CUDA stack on GPUCompiler 2.x (CompilerCaching.jl) activates the disk cache unchanged.

## When it recompiles

Triggers: package sources or any included `runner_scripts/*.jl` file; `problems.csv`/`algorithms.csv` (`include_dependency`); any `Manifest.toml` dependency version, transitively; the julia version or depot; the package's `precompile_workload` preference. Non-triggers: bench-script edits, sweep sizes, tolerances, watchdog settings, environment variables, and GPUCompiler's runtime-read `disk_cache` preference. The bench's `Pkg.instantiate(); Pkg.precompile()` preamble and the suite's `-a warm` stage build the package; on a GPU-less machine skip the workload with `Preferences.set_preferences!(GPU_ODE_JuliaKernels, "precompile_workload" => false)`.
