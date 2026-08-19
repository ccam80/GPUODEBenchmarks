module GPU_ODE_JuliaKernels

# Precompiled benchmark systems for the DiffEqGPU kernel path; see README.md.

using LinearAlgebra
using StaticArrays
using ModelingToolkit
using SciMLBase
using DiffEqGPU
using CUDA
using PrecompileTools: @setup_workload, @compile_workload

const REPO_ROOT = dirname(dirname(@__DIR__))
include(joinpath(REPO_ROOT, "runner_scripts", "problems.jl"))
include(joinpath(REPO_ROOT, "runner_scripts", "algorithms.jl"))
include(joinpath(REPO_ROOT, "runner_scripts", "julia_systems.jl"))
include(joinpath(REPO_ROOT, "runner_scripts", "julia_prob.jl"))
# Changing a registry CSV re-precompiles this package.
Base.include_dependency(PROBLEMS_CSV)
Base.include_dependency(ALGORITHMS_CSV)

SYSTEMS_CODEGEN = (expression = Val(false), eval_expression = true,
    eval_module = @__MODULE__)

# Entries built at precompile time; `sys` is unused and stays out of the image.
for row in resolve_problems("all", "julia")
    entry = _ENTRY_BUILDERS[row["problem"]]()
    _ENTRIES[row["problem"]] = Base.structdiff(entry, NamedTuple{(:sys,)})
end
const ENTRIES = _ENTRIES

# Mirrors TIMING_TOL in runner_scripts/wp_common.py and bench_ode_gpu.jl.
const TIMING_TOL = 1.0f-8
# Trajectory count is runtime data, not a kernel specialization axis.
const WORKLOAD_N = 4

"One solve per supported mode, with the bench writer's exact call shape."
function _warm_leg(row, algorithm)
    solver = gpu_solver(algorithm)
    system, prob, duration = build_prob_parts(ENTRIES[row["problem"]], row)
    dt0 = Float32(problem_timing_dt(row))
    # Kernels specialize on types, so a zero-step tspan warms them in bounded time.
    prob = remake(prob, tspan = (0.0f0, 0.0f0))
    probs_host, probs = build_ensemble(system, prob, row, WORKLOAD_N)
    if algorithm in supported_algorithms("julia", "fixed")
        sol = CUDA.@sync DiffEqGPU.vectorized_solve(probs, prob, solver;
            saveat = 0.0f0, save_everystep = false, dt = dt0)
        Array(sol[1])
        Array(sol[2])
    end
    if algorithm in supported_algorithms("julia", "adaptive")
        sol = CUDA.@sync DiffEqGPU.vectorized_asolve(probs, prob, solver;
            saveat = 0.0f0, save_everystep = false,
            reltol = TIMING_TOL, abstol = TIMING_TOL, dt = dt0)
        Array(sol[1])
        Array(sol[2])
    end
    return nothing
end

# States-sweep grid sizes stay with julia_driver.py's cancellation control.
@setup_workload begin
    if CUDA.functional()
        @compile_workload begin
            for row in resolve_problems("all", "julia")
                for algorithm in supported_algorithms("julia")
                    problem_supports(row, "julia") || continue
                    elapsed = @elapsed try
                        _warm_leg(row, algorithm)
                    catch err
                        @warn "kernel warm workload leg failed" problem=row["problem"] algorithm err
                    end
                    @info "kernel warm workload leg" problem=row["problem"] algorithm elapsed
                end
            end
        end
    else
        @warn "CUDA not functional; the GPU kernel warm workload was skipped"
    end
end

end # module
