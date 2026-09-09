# P7 system module: one MTK definition per problem, Float32 and Float64

`julia_systems.jl` is the only system module; the golden is the Float64 solve of the same compiled system the kernels and the Float32 sweeps run.

## 1. `runner_scripts/julia_systems.jl`

Builders take the element type: `_lorenz_entry(::Type{T})`, `_lorenz96_entry(::Type{T}, n)`, `_pleiades_entry(::Type{T})`, `_pollu_entry(::Type{T})`, `_ring_modulator_entry(::Type{T})`, `_ring_modulator_index2_entry(::Type{T})`, `_nand_gate_entry(::Type{T})`, `_build_entry(raw, ::Type{T}; u0map, golden_vars, consistent_u0 = false)`.

Every literal is `T(x)`; `ifelse` thresholds are `zero(T)`; `@parameters` defaults are `T(x)`. Helpers with literals (`_rm_q`, `_nand_pulse`, `_nand_cb`, `_nand_ibs`, `_nand_gdsp`, `_nand_gdsm`, `_nand_ids1`, `_nand_ids2`) take `T` first. `_build_entry` returns `SMatrix{n,n,T}` mass matrix, `SVector{n,T}` u0 and `u0_for`; the Newton step inside `_consistent_u0` stays Float64. `generate_jacobian` failing is an error; `tgrad` keeps its `nothing` fallback. `mtkcompile(raw; split = false)` and `SYSTEMS_CODEGEN` are unchanged.

Registry: `_ENTRY_BUILDERS` values are `T -> _x_entry(T)` (`lorenz96` binds `n`); `_ENTRIES::Dict{Tuple{String, DataType}, Any}`; `julia_system(problem, ::Type{T} = Float32)`. Existing one-argument callers are unchanged.

```julia
"In-place ODEProblem of one trajectory for the CPU solvers; u0 carries the consistent algebraic states."
function cpu_problem(system, problem, p)
    T = eltype(system.u0)
    f = system.mass_matrix === nothing ?
        ODEFunction{true}(system.rhs!; jac = system.jac!) :
        ODEFunction{true}(system.rhs!; jac = system.jac!, mass_matrix = Matrix{T}(system.mass_matrix))
    return ODEProblem{true}(f, Vector{T}(system.u0_for(T(p))), (zero(T), T(problem["duration"])), T[p])
end
```

The Float32 sweep and the Float64 golden both build through `cpu_problem`. Golden solve kwargs: `abstol = reltol = golden_tol`, `save_everystep = false`, `save_start = false`, `dense = false`, `maxiters = 10^8`, `verbose = DEVerbosity(SciMLLogging.None())`; no tstops. Finals are `sol.u[end][system.golden_index]`; `converged = sol.retcode == ReturnCode.Success`.

`golden_solver(name)`: `Vern9()`, `Rodas5P()`, `RadauIIA5()`; anything else errors.

## 2. `runner_scripts/problems.csv`

| problem | golden_algorithm | golden_tol |
|---|---|---|
| lorenz, lorenz96, lorenz96_20, pleiades | Vern9 | 1e-13 |
| pollu | Rodas5P | 1e-13 |
| ring_modulator | Rodas5P | 1e-10 |
| ring_modulator_index2 | RadauIIA5 | 1e-10 |
| nand_gate (sweep `c9`, VDD = 5 constant) | Rodas5P | 1e-10 |

## 3. `GPU_ODE_JuliaKernels`

`_ENTRIES[(name, Float32)] = Base.structdiff(_ENTRY_BUILDERS[name](Float32), NamedTuple{(:sys,)})`; `ENTRIES` and the consumers' `merge!` are unchanged. Float64 entries compile at golden time (nand 25 s, others seconds). Re-precompile once (about 30 min on the 4070).

## 4. `runner_scripts/golden/verify_references.jl`

Runs on `julia_system(name, Float64)` through `cpu_problem` and `golden_solver`:

| check | reference | pass |
|---|---|---|
| pollu, k1 = 0.35, Rodas5P 1e-13 | `POLLU_REF` | 1e-13 |
| pleiades, m1 = 1.0, Vern9 1e-13 | `PLEI_REF` | 1e-10 |
| nand_gate, c9 = 5e-5, Rodas5P 1e-10 | `NAND_REF` | 1e-8 |
| ring_modulator, Cs = 2e-12, Rodas5P 1e-10 | `RING_REF` below | 1e-8 |
| lorenz96, F = 8, Vern9 vs RadauIIA5 at 1e-13 | each other | 1e-7 |
| ring_modulator_index2, amplitude 0.5, RadauIIA5 vs RadauIIA9 at 1e-10 | each other | 1e-8 |
| every problem, Float32 rhs vs Float64 rhs at 20 random states via `golden_index` | each other | 1e-5 relative |
| every problem, both types, `@allocated rhs!` after one warm call | 0 | 0 B |

`RING_REF` (Test Set report II-3, Table II.3.2, t = 1e-3):

```
-0.2339057358486745e-1, -0.7367485485540825e-2, 0.2582956709291169, -0.4064465721283450,
-0.4039455665149794, 0.2607966765422943, 0.1106761861269975, 0.2939904342435596e-6,
-0.2840029933642329e-7, 0.7267198267264553e-3, 0.7929487196960840e-3, -0.7255283495698965e-3,
-0.7941401968526521e-3, 0.7088495416976114e-4, 0.2390059075236570e-4
```

## 5. Deletions

`runner_scripts/reference_systems.jl`, `runner_scripts/golden/generate_golden.jl`, `runner_scripts/golden/retcode_sidecar.jl`, `data/numerical/golden_*_retcodes.csv`, every `include` of `reference_systems.jl`.

## 6. Tests

`runner_scripts/tests/test_julia_systems.jl`, every problem row, both element types: entry builds; `eltype(u0) == T`; `mass_matrix` is `nothing` or `SMatrix{n,n,T}`; `rhs!` allocates 0 B; algebraic rows of `rhs!` at `u0_for(sweep_min)` are below 1e-12 (Float64) and 1e-6 (Float32); `golden_index` is `problem["states"]` distinct indices in `1:n`. `verify_references.jl` runs in the same invocation.

## 7. Regeneration

Through the `golden` set, in order, threaded wall on the 4070 host: lorenz family (seconds), pollu (1 min), nand_gate (5 min), ring_modulator_index2 (35 min), pleiades (55 min), ring_modulator (7.6 h). Expected movement against the committed files: lorenz family ≤ 5e-13, pollu ≤ 1e-15, pleiades ≤ 4e-11, nand 2.2e-9, index2 3e-9, ring 4e-9.

## 8. Recorded solver outcomes

- FBDF: `Success` after 2–6 steps with finals 3.8 off on nand_gate, tol 1e-2..1e-8.
- Kvaerno5: `Unstable`/`MaxIters` on nand_gate at every tolerance; `ConvergenceFailure` at every fixed dt down to 80·2⁻¹⁴.
- Rodas5P: `Unstable` on ring_modulator_index2 at 1e-10.
- In-place RadauIIA9: 2.3e-6 from the published ring value at 1e-13 with any linear solver (OrdinaryDiffEqFIRK 2.7.0).
- Float32 Newton methods on nand_gate: fail at every tolerance and fixed dt tried; Rosenbrock23 floors near 3e-4 from tol 1e-5.
