# The ne ensemble grid and golden for the DifferentialEquations.jl and DiffEqGPU ne sweeps; include problems.jl first.

using DelimitedFiles

"The ne ensemble grid: golden_ne's own column for an ne_legacy_grid problem, else the first N_NE points of the N_WP sweep."
function ne_sweep(problem)
    name = problem["problem"]
    if name in NE_LEGACY_GRID
        path = golden_ne_path(problem)
        isfile(path) || error("$(path) not found; it defines the legacy ne grid of $(name)")
        return Float32.(readdlm(path, ',')[:, 1])
    end
    return Float32.(problem_sweep(problem, N_WP))[1:N_NE]
end

"Path of the standalone ne golden of an ne_legacy_grid problem, as ne_common.golden_ne_path."
golden_ne_path(problem) = joinpath(dirname(@__DIR__), "data", "numerical",
    "golden_ne_$(problem["problem"])_$(N_NE).csv")

"The Float64 golden states of the ne ensemble, (N_NE, states)."
function ne_golden_states(problem)
    if problem["problem"] in NE_LEGACY_GRID
        return readdlm(golden_ne_path(problem), ',')[:, 2:end]
    end
    path = golden_path(problem)
    isfile(path) || error("$(path) not found - generate it first with `julia -t auto --project=. runner_scripts/golden/generate_golden.jl --problem $(problem["problem"])`")
    return readdlm(path, ',')[1:N_NE, :]
end
