# P7 system module: one MTK definition per problem, Float32 and Float64

The golden is the Float64 solve of the same MTK system the kernels and the Float32 CPU sweeps run. `reference_systems.jl`, `generate_golden.jl` and `golden/retcode_sidecar.jl` are deleted; `julia_systems.jl` is the only system module.

Measured basis (julia 1.13.0-rc3, OrdinaryDiffEqCore 4.14.3, ModelingToolkit 11.39.1, i7-12700 20 threads):

| problem | Float64 MTK form | golden solver, tol | result | per-step heap |
|---|---|---|---|---|
| pollu | 20 unknowns | Rodas5P 1e-13 | 6.43e-15 vs published; rhs matches the hand-written Float64 form to 1.9e-16 | 5 B |
| ring_modulator | 15 unknowns, identity mass | Rodas5P 1e-10 | 6.2e-10 vs published (Table II.3.2, Cs 2e-12); 1e-12 gives 7.5e-11 | 1 B |
| ring_modulator_index2 | 15 unknowns, rows 10–13 algebraic (index 2, no reduction) | RadauIIA5 1e-10 | 2.7e-9 / 3.1e-9 vs the committed RadauIIA9 golden at rows 65536 / 131072 | 1 B |
| nand_gate | 22 unknowns (14 + 8 dummy derivatives) | Rodas5P 1e-10, no tstops | 2.2e-9 vs the committed DFBDF golden, which is 1.2e-9 vs published | — |
| lorenz, lorenz96, lorenz96_20, pleiades | same equations | Vern9 1e-13 | pleiades hand-written form 2.1e-12 vs published | 0–6 B |

Threaded golden loop, 128 trajectories over the grid, 20 threads: ring 26.8 s (7.6 h for 131072), pleiades 3.2 s (54 min), index2 RadauIIA9 0.87 s (RadauIIA5 takes 2.4× the steps, so about 35 min), nand 0.29 s (5 min), pollu 0.04 s, lorenz family under 10 s.

## 1. `runner_scripts/julia_systems.jl`

### 1.1 Element type

Every builder takes the element type:

```julia
_lorenz_entry(::Type{T}) where {T}
_lorenz96_entry(::Type{T}, n) where {T}
_pleiades_entry(::Type{T}) where {T}
_pollu_entry(::Type{T}) where {T}
_ring_modulator_entry(::Type{T}) where {T}
_ring_modulator_index2_entry(::Type{T}) where {T}
_nand_gate_entry(::Type{T}) where {T}
_build_entry(raw, ::Type{T}; u0map, golden_vars, consistent_u0 = false) where {T}
```

Every numeric literal in a builder or helper becomes `T(x)` (`10.0f0` → `T(10)`, `8.0f0 / 3.0f0` → `T(8) / T(3)`, `Float32[...]` → `T[...]`, `Float32(i)` → `T(i)`, `0.0f0` in `ifelse` → `zero(T)`). Helpers that carry literals (`_rm_q`, `_nand_pulse`, `_nand_cb`, `_nand_ibs`, `_nand_gdsp`, `_nand_gdsm`, `_nand_ids1`, `_nand_ids2`) take `T` as their first argument. `@parameters` defaults are `T(x)`. `_ordered_values` returns `T`. `_build_entry` produces `SMatrix{n, n, T}` mass matrix, `SVector{n, T}` u0, and `_consistent_u0` returns `SVector{n, T}`; its internal Newton step stays Float64.

The Jacobian is required: `generate_jacobian` failing is an error, not `nothing`; `tgrad` keeps its `nothing` fallback.

Both element types go through the same `mtkcompile(raw; split = false)` and `generate_*` calls. `SYSTEMS_CODEGEN` is unchanged.

### 1.2 Registry

```julia
const _ENTRY_BUILDERS = Dict{String, Function}(
    "lorenz" => T -> _lorenz_entry(T),
    "lorenz96" => T -> _lorenz96_entry(T, 32),
    "lorenz96_20" => T -> _lorenz96_entry(T, 20),
    ...)
const _ENTRIES = Dict{Tuple{String, DataType}, Any}()
julia_system(problem, ::Type{T} = Float32) where {T}
```

`julia_system(problem)` keeps its one-argument form for every existing caller (`julia_prob.jl`, `bench_ode_gpu.jl`, `julia_worker.jl`, `ne_diffeq.jl`); `julia_system(problem, Float64)` is the golden path. `golden_finals` is unchanged.

### 1.3 Shared CPU problem constructor

```julia
"In-place ODEProblem of one trajectory for the CPU solvers; u0 carries the consistent algebraic states."
function cpu_problem(system, problem, p)
    T = eltype(system.u0)
    f = system.mass_matrix === nothing ?
        ODEFunction{true}(system.rhs!; jac = system.jac!) :
        ODEFunction{true}(system.rhs!; jac = system.jac!,
            mass_matrix = Matrix{T}(system.mass_matrix))
    return ODEProblem{true}(f, Vector{T}(system.u0_for(T(p))), (zero(T), T(problem["duration"])), T[p])
end
```

`ne_diffeq.jl` (P7's `bench_ode_cpu.jl`) builds its Float32 base problem and its `remake` targets with it; the golden trials build the Float64 problem with it. Solver kwargs on the golden: `abstol = reltol = golden_tol`, `save_everystep = false`, `save_start = false`, `dense = false`, `maxiters = 10^8`, `verbose = DEVerbosity(SciMLLogging.None())`, nothing else. No tstops on any problem. The golden row's finals are `sol.u[end][system.golden_index]`; `converged = sol.retcode == ReturnCode.Success`.

### 1.4 Solver table

`reference_solver` moves into `julia_systems.jl` as `golden_solver(name)` with exactly: `Vern9()`, `Rodas5P()`, `RadauIIA5()`. No `linsolve` argument anywhere. Any other `golden_algorithm` is an error.

## 2. `runner_scripts/problems.csv`

| problem | golden_algorithm | golden_tol |
|---|---|---|
| lorenz, lorenz96, lorenz96_20, pleiades | Vern9 | 1e-13 |
| pollu | Rodas5P | 1e-13 |
| ring_modulator | Rodas5P | 1e-10 |
| ring_modulator_index2 | RadauIIA5 | 1e-10 |
| nand_gate | Rodas5P | 1e-10 |

`nand_gate` sweeps `c9` over [2.5e-5, 1e-4] linear (PR #111). The `c9` builder keeps VDD = 5 as a constant.

## 3. `GPU_ODE_JuliaKernels`

`_ENTRIES[(row["problem"], Float32)] = Base.structdiff(_ENTRY_BUILDERS[row["problem"]](Float32), NamedTuple{(:sys,)})`. `ENTRIES` stays the exported constant; `merge!(_ENTRIES, GPU_ODE_JuliaKernels.ENTRIES)` in the consumers is unchanged. The Float64 entries are never precompiled; the golden process pays one `mtkcompile` per problem (nand 25 s, the rest seconds). The package re-precompiles once after this change (about 30 minutes on the 4070).

## 4. `runner_scripts/golden/verify_references.jl`

Keeps its published-value checks and runs them on `julia_system(name, Float64)` through `cpu_problem` and `golden_solver`:

| check | reference | pass threshold |
|---|---|---|
| pollu, k1 = 0.35, Rodas5P 1e-13 | `POLLU_REF` | 1e-13 |
| pleiades, m1 = 1.0, Vern9 1e-13 | `PLEI_REF` | 1e-10 |
| nand_gate, c9 = 5e-5, Rodas5P 1e-10 | `NAND_REF` | 1e-8 |
| ring_modulator, Cs = 2e-12, Rodas5P 1e-10 | Test Set report II-3 Table II.3.2 (values below) | 1e-8 |
| lorenz96, F = 8, Vern9 vs RadauIIA5 at 1e-13 | each other | 1e-7 |
| ring_modulator_index2, amplitude 0.5, RadauIIA5 vs RadauIIA9 at 1e-10 | each other | 1e-8 |
| every problem: Float32 entry rhs vs Float64 entry rhs at 20 random states, mapped by `golden_index` | each other | 1e-5 relative |
| every problem, both element types: `@allocated rhs!(du, u, p, t)` after one warm call | 0 | 0 bytes |

Published ring modulator final state at t = 1e-3 (Cs = 2e-12), y1..y15:

```
-0.2339057358486745e-1, -0.7367485485540825e-2, 0.2582956709291169, -0.4064465721283450,
-0.4039455665149794, 0.2607966765422943, 0.1106761861269975, 0.2939904342435596e-6,
-0.2840029933642329e-7, 0.7267198267264553e-3, 0.7929487196960840e-3, -0.7255283495698965e-3,
-0.7941401968526521e-3, 0.7088495416976114e-4, 0.2390059075236570e-4
```

The Float32-vs-Float64 check replaces the current hand-written-vs-MTK check; the two entries share the equations, so a failure means a literal escaped the `T(...)` conversion.

## 5. Deletions

`runner_scripts/reference_systems.jl`, `runner_scripts/golden/generate_golden.jl`, `runner_scripts/golden/retcode_sidecar.jl`, `data/numerical/golden_*_retcodes.csv`. Every `include` of `reference_systems.jl` goes (`generate_golden.jl`, `verify_references.jl`). `problems.jl`'s `golden_path` goes with the P2 store conversion.

## 6. Tests

`runner_scripts/tests/test_julia_systems.jl`: for every problem row and both element types, the entry builds, `eltype(u0) == T`, `mass_matrix` is `nothing` or `SMatrix{n,n,T}`, `rhs!` allocates 0 bytes, `u0_for(sweep_min)` leaves every algebraic row of `rhs!` below `1e-6 * eps(T)^0` scaled as `1e-12` for Float64 and `1e-6` for Float32, and `golden_index` is a permutation into `1:n` of length `problem["states"]`. `verify_references.jl` runs in the same test invocation.

## 7. Regeneration

Every golden is regenerated through the P7 `golden` set once the module lands, in this order and with these expected threaded wall times on the 4070 host: lorenz family (seconds), pollu (1 min), nand_gate (5 min), ring_modulator_index2 (about 35 min), pleiades (55 min), ring_modulator (7.6 h). Expected movement against the committed files: lorenz family ≤ 5e-13, pollu ≤ 1e-15, pleiades ≤ 4e-11, nand 2.2e-9, index2 3e-9, ring 4e-9 at the stiffest Cs.

## 8. Solver facts to carry as protocol outcomes

Measured on the Float64 22-unknown nand form and the 15-unknown ring forms; these are recorded results, not settings to tune:

- FBDF returns `Success` after 2–6 steps with finals 3.8 off on nand_gate at every tolerance 1e-2..1e-8.
- Kvaerno5 is `Unstable` or `MaxIters` on nand_gate at every tolerance, and `ConvergenceFailure` at every fixed dt down to 80·2⁻¹⁴.
- Rodas5P is `Unstable` on ring_modulator_index2 at 1e-10.
- In-place RadauIIA9 is 2.3e-6 off the published ring value at 1e-13 for every linear solver (OrdinaryDiffEqFIRK 2.7.0); RadauIIA9 is not a golden solver.
- Newton-based methods in Float32 fail on the nand form at every tolerance and every fixed dt tried; Rosenbrock23 in Float32 floors near 3e-4 absolute from tol 1e-5.
