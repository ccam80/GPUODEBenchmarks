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
# Changing a catalogue CSV re-precompiles this package.
Base.include_dependency(PROBLEMS_CSV)
Base.include_dependency(JULIA_ALGORITHMS_CSV)

SYSTEMS_CODEGEN = (expression = Val(false), eval_expression = true,
    eval_module = @__MODULE__)

# Entries built at precompile time; `sys` is unused and stays out of the image.
for row in resolve_problems("all", "julia_gpu")
    entry = _ENTRY_BUILDERS[row["problem"]]()
    _ENTRIES[row["problem"]] = Base.structdiff(entry, NamedTuple{(:sys,)})
end
const ENTRIES = _ENTRIES

# Trajectory count is runtime data, not a kernel specialization axis.
const WORKLOAD_N = 4
const WORKLOAD_TOL = 1.0f-5

"One fixed and one adaptive solve through the shared kernel-path functions."
function _warm_leg(row, algorithm)
    solver = gpu_solver(algorithm)
    system = ENTRIES[row["problem"]]
    # Kernels specialize on types, so a zero-step tspan warms them in bounded time.
    prob = remake(build_prob(system, row["duration"]), tspan = (0.0f0, 0.0f0))
    probs_host, probs = build_ensemble(system, prob,
        range(row["sweep_min"], row["sweep_max"], length = WORKLOAD_N))
    dt = Float32(row["duration"]) * 2.0f0^-10
    for (controller, dt0, tol) in (("fixed", dt, NaN), ("default", NaN, WORKLOAD_TOL))
        sol = CUDA.@sync gpu_solve(probs, prob, solver, controller, dt0, tol, tol;
            saveat = 0.0f0)
        Array(sol[1])
        Array(sol[2])
    end
    return nothing
end

@setup_workload begin
    if CUDA.functional()
        @compile_workload begin
            for row in resolve_problems("all", "julia_gpu")
                for algorithm in package_algorithms("julia_gpu")
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
