# The ne ensemble grid and golden for the DifferentialEquations.jl and DiffEqGPU ne sweeps; include problems.jl first.

using DelimitedFiles

"The ne ensemble grid: the first N_NE points of the N_WP sweep."
ne_sweep(problem) = Float32.(problem_sweep(problem, N_WP))[1:N_NE]

"The Float64 golden states of the ne ensemble, (N_NE, states): the first N_NE rows of the wp golden."
function ne_golden_states(problem)
    path = golden_path(problem)
    isfile(path) || error("$(path) not found - generate it first with `julia -t auto --project=. runner_scripts/golden/generate_golden.jl --problem $(problem["problem"])`")
    return readdlm(path, ',')[1:N_NE, :]
end
