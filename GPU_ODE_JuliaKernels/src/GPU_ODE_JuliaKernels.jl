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

# Trajectory count is runtime data, not a kernel specialization axis.
const WORKLOAD_N = 4

"Every algorithm the timed sweeps or the overlap suite run on the kernel path."
workload_algorithms() = unique(vcat(supported_algorithms("julia"),
    [row["algorithm"] for row in overlap_algorithms()]))

"One fixed and one adaptive solve through the shared kernel-path functions."
function _warm_leg(row, algorithm)
    solver = gpu_solver(algorithm)
    system, prob, duration = build_prob_parts(ENTRIES[row["problem"]], row)
    # Kernels specialize on types, so a zero-step tspan warms them in bounded time.
    prob = remake(prob, tspan = (0.0f0, 0.0f0))
    probs_host, probs = build_ensemble(system, prob, row, WORKLOAD_N)
    for (mode, setting) in (("fixed", problem_timing_dt(row)),
                            ("adaptive", TIMING_TOL))
        sol = CUDA.@sync gpu_solve(probs, prob, solver, mode, setting, row;
            saveat = 0.0f0)
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
                for algorithm in workload_algorithms()
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
