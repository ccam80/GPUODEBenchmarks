# Float64 ne golden references at N=1024, column 1 the Float32-rounded swept parameter, kept unless --force.
#
# Usage: julia -t auto --project=. runner_scripts/numerical_equivalence/generate_golden_ne.jl [--problem <name|all>] [--force]

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
    tol = problem["golden_tol"]

    @info "Solving $(N)-trajectory Float64 $(problem["problem"]) reference " *
          "($(problem["golden_algorithm"]), tol $(tol)) on " *
          "$(Threads.nthreads()) threads..."
    out = Matrix{Float64}(undef, N, nstates + 1)
    out[:, 1] .= grid
    @time Threads.@threads for i in 1:N
        # Stiff references need far more steps than the default cap allows.
        sol = solve(remake(prob, p = [grid[i]]), solver;
            abstol = tol, reltol = tol, save_everystep = false,
            save_start = false, dense = false, maxiters = 10^8)
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
