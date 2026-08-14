# Golden references for the numerical-equivalence sweeps, one file per problem.
#
# Same protocol as the wp golden at N=1024, and the Float32-rounded parameter
# grid is stored in column 1 so every consumer integrates bit-identical
# parameters instead of rebuilding its own linspace.
#
# Output: data/numerical/golden_ne_<problem>_1024.csv, columns
# <swept parameter> then the final state, no header. Machine independent.
#
# Run from the repo root:
#   julia -t auto --project=. runner_scripts/numerical_equivalence/generate_golden_ne.jl [--problem <name|all>] [--force]
#
# An existing file is kept unless --force is given.

using OrdinaryDiffEq
# The slim OrdinaryDiffEq v7 umbrella doesn't re-export the ensemble API.
using SciMLBase: ODEProblem, ODEFunction, remake
using DelimitedFiles

include(joinpath(dirname(@__DIR__), "problems.jl"))
include(joinpath(dirname(@__DIR__), "reference_systems.jl"))

const N = 1024

requested = "all"
force = false
let i = 1
    while i <= length(ARGS)
        if ARGS[i] == "--force"
            global force = true
            i += 1
        elseif ARGS[i] == "--problem" || ARGS[i] == "-s"
            i < length(ARGS) || error("--problem requires a value")
            global requested = ARGS[i + 1]
            i += 2
        elseif startswith(ARGS[i], "--problem=")
            global requested = split(ARGS[i], "=", limit = 2)[2]
            i += 1
        else
            error("unexpected argument $(ARGS[i])")
        end
    end
end

function generate(problem)
    outdir = joinpath(dirname(dirname(@__DIR__)), "data", "numerical")
    outfile = joinpath(outdir, "golden_ne_$(problem["problem"])_$(N).csv")
    if isfile(outfile) && !force
        @info "Keeping $(outfile)"
        return
    end
    system = reference_system(problem)
    nstates = problem["states"]
    grid = Float64.(Float32.(problem_sweep(problem, N)))
    f = system.mass_matrix === nothing ? ODEFunction(system.rhs) :
        ODEFunction(system.rhs; mass_matrix = system.mass_matrix)
    prob = ODEProblem(f, system.u0, (0.0, problem["duration"]), [grid[1]])
    solver = reference_solver(problem["golden_algorithm"])

    @info "Solving $(N)-trajectory Float64 $(problem["problem"]) reference " *
          "($(problem["golden_algorithm"]), tol 1e-13) on " *
          "$(Threads.nthreads()) threads..."
    out = Matrix{Float64}(undef, N, nstates + 1)
    out[:, 1] .= grid
    @time Threads.@threads for i in 1:N
        sol = solve(remake(prob, p = [grid[i]]), solver;
            abstol = 1e-13, reltol = 1e-13, save_everystep = false,
            save_start = false, dense = false)
        out[i, 2:end] .= sol.u[end]
    end

    mkpath(outdir)
    open(outfile, "w") do io
        writedlm(io, out, ',')
    end
    @info "Wrote $(outfile)"
end

for problem in resolve_problems(requested)
    generate(problem)
end
