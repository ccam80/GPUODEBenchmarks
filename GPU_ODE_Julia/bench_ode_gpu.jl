using Pkg

Pkg.instantiate()
Pkg.precompile()

using CUDA
using DiffEqGPU, OrdinaryDiffEq, StaticArrays
using CSV, DataFrames, DelimitedFiles

# CLI: <N>|wp [algorithm|all] [--problem <name|all>]; wp always runs at N_WP.
@show ARGS
#settings
CUDA.allowscalar(false)

# Dataset key "<os>_<gpu>" keys output files per machine.
include(joinpath(dirname(@__DIR__), "runner_scripts", "bench_key.jl"))
include(joinpath(dirname(@__DIR__), "runner_scripts", "problems.jl"))
include(joinpath(dirname(@__DIR__), "runner_scripts", "algorithms.jl"))
include(joinpath(dirname(@__DIR__), "runner_scripts", "julia_systems.jl"))
include(joinpath(dirname(@__DIR__), "runner_scripts", "watchdog.jl"))
const DATASET_KEY = dataset_key()
const REPO_ROOT = dirname(@__DIR__)

requested_algorithm = "all"
requested_problem = "all"
let i = 2
    while i <= length(ARGS)
        tok = ARGS[i]
        if tok == "--problem" || tok == "-s"
            i += 1
            i <= length(ARGS) || error("--problem requires a value")
            global requested_problem = ARGS[i]
        elseif startswith(tok, "--problem=")
            global requested_problem = split(tok, "=", limit = 2)[2]
        else
            global requested_algorithm = tok
        end
        i += 1
    end
end
const ALGORITHMS = resolve_algorithms(requested_algorithm, "julia")
const FIXED_ALGORITHMS = supported_algorithms("julia", "fixed")
const ADAPTIVE_ALGORITHMS = supported_algorithms("julia", "adaptive")
if isempty(ALGORITHMS)
    println("Julia (DiffEqGPU kernel path) runs none of the requested ",
        "algorithms; skipping.")
    exit(0)
end
const PROBLEMS = resolve_problems(requested_problem, "julia")
if isempty(PROBLEMS)
    println("Julia runs none of the requested problems; skipping.")
    exit(0)
end

# Fixed sample count to match the other frameworks.
const REPEATS = 20
const WP_MODE = !isempty(ARGS) && ARGS[1] == "wp"
# Mirrors TIMING_TOL and N_WP in runner_scripts/wp_common.py.
const TIMING_TOL = 1.0f-8
const N_WP = 131072
numberOfParameters = isinteractive() ? 8192 :
                     (WP_MODE ? N_WP : parse(Int64, ARGS[1]))

"DiffEqGPU kernel solver for an algorithm name; autodiff off matches the overlap suite."
function gpu_solver(algorithm)
    algorithm == "tsit5" && return GPUTsit5()
    algorithm == "rosenbrock23_sciml" && return GPURosenbrock23(autodiff = Val(false))
    algorithm == "kvaerno3" && return GPUKvaerno3(autodiff = Val(false))
    error("no DiffEqGPU kernel solver for '$(algorithm)'")
end

"Ensemble l2-at-final error against the Float64 golden reference."
function ensemble_error(system, us, golden)
    final = Array(us[end, :])
    m = Matrix{Float64}(undef, length(final), length(system.golden_index))
    for i in eachindex(final)
        m[i, :] .= Float64.(final[i][system.golden_index])
    end
    return sqrt(sum(abs2, m .- golden) / length(m))
end

function build_probs(problem)
    system = julia_system(problem)
    duration = Float32(problem["duration"])
    f = system.mass_matrix === nothing ?
        ODEFunction{false}(system.rhs; jac = system.jac,
            tgrad = system.tgrad) :
        ODEFunction{false}(system.rhs; jac = system.jac,
            tgrad = system.tgrad, mass_matrix = system.mass_matrix)
    prob = ODEProblem{false}(f, system.u0, (0.0f0, duration),
        @SArray [Float32(problem["sweep_max"])])
    grid = Float32.(collect(problem_sweep(problem, numberOfParameters)))
    probs_host = map(1:numberOfParameters) do i
        DiffEqGPU.make_prob_compatible(remake(prob,
            u0 = system.u0_for(grid[i]), p = @SVector [grid[i]]))
    end
    return system, prob, probs_host, cu(probs_host), duration
end

"An algorithm that cannot run this system is a NaN row, not an aborted run."
function failed(what, err)
    println("FAILED $(what): $(err)")
    return NaN
end

# One wp sweep; a watchdog breach fills the remaining settings with NaN rows.
function wp_sweep(solve_once, system, path, settings, golden, label)
    open(path, "w") do io
        breached = false
        for (index, setting) in enumerate(settings)
            if breached
                println(io, setting, " NaN NaN")
                continue
            end
            on_breach = () -> begin
                for s in settings[index:end]
                    println(io, s, " NaN NaN")
                end
                flush(io)
                println("WATCHDOG $(label) setting=$(setting): run never returned")
            end
            t_ms, err = try
                warm = @elapsed sol = run_watchdogged(
                    () -> solve_once(setting), on_breach)
                if warm > WATCHDOG_SECONDS
                    (NaN, NaN)
                else
                    e = ensemble_error(system, sol[2], golden)
                    t = watchdogged_min_ms(() -> solve_once(setting),
                        on_breach, REPEATS)
                    (t, isnan(t) ? NaN : e)
                end
            catch err
                (failed("wp $(label) setting=$(setting)", err), NaN)
            end
            if isnan(t_ms)
                println("WATCHDOG wp $(label) setting=$(setting): run exceeded the cap")
                breached = true
            end
            println(io, setting, " ", t_ms, " ", err)
            flush(io)
            println("wp $(label) setting=$(setting): $(t_ms) ms, err=$(err)")
        end
    end
end

# Sweeps fixed dt and adaptive tolerance at N=N_WP; grids mirror runner_scripts/wp_common.py.
function run_wp(problem)
    golden = readdlm(
        joinpath(REPO_ROOT, "data", "numerical",
            "golden_$(problem["problem"])_$(N_WP).csv"), ',', Float64)
    system, prob, probs_host, probs, duration = build_probs(problem)
    outdir = data_dir(REPO_ROOT, "Julia", DATASET_KEY, problem)
    dt0 = Float32(problem_timing_dt(problem))

    for algorithm in ALGORITHMS
        solver = gpu_solver(algorithm)
        label = "$(problem["problem"]) $(algorithm)"

        if algorithm in FIXED_ALGORITHMS
            wp_sweep(system,
                joinpath(outdir, "Julia_wp_fixed_$(algorithm).txt"),
                collect(problem_dts(problem)), golden,
                "$(label) fixed") do dt
                CUDA.@sync sol = DiffEqGPU.vectorized_solve(probs, prob,
                    solver; saveat = duration, save_everystep = false,
                    dt = Float32(dt))
                ts = Array(sol[1])
                us = Array(sol[2])
                sol
            end
        end

        if algorithm in ADAPTIVE_ALGORITHMS
            wp_sweep(system,
                joinpath(outdir, "Julia_wp_adaptive_$(algorithm).txt"),
                [10.0^-k for k in 2:8], golden,
                "$(label) adaptive") do tol
                CUDA.@sync sol = DiffEqGPU.vectorized_asolve(probs, prob,
                    solver; saveat = duration, save_everystep = false,
                    reltol = Float32(tol), abstol = Float32(tol), dt = dt0)
                ts = Array(sol[1])
                us = Array(sol[2])
                sol
            end
        end
    end
end

function run_times(problem)
    system, prob, probs_host, probs, duration = build_probs(problem)
    dt0 = Float32(problem_timing_dt(problem))
    outdir = data_dir(REPO_ROOT, "Julia", DATASET_KEY, problem)

    for algorithm in ALGORITHMS
        solver = gpu_solver(algorithm)

        for mode in ("fixed", "adaptive")
            mode == "fixed" && !(algorithm in FIXED_ALGORITHMS) && continue
            mode == "adaptive" && !(algorithm in ADAPTIVE_ALGORITHMS) && continue
            @info "Solving $(problem["problem"]) on GPU ($(mode) dt, $(algorithm))"

            device_solve = () -> begin
                # Device-only: probs already resident, results left there.
                if mode == "fixed"
                    CUDA.@sync DiffEqGPU.vectorized_solve(probs, prob, solver,
                        saveat = duration, save_everystep = false, dt = dt0)
                else
                    CUDA.@sync DiffEqGPU.vectorized_asolve(probs, prob, solver,
                        saveat = duration, save_everystep = false,
                        reltol = TIMING_TOL, abstol = TIMING_TOL, dt = dt0)
                end
            end
            full_solve = () -> begin
                # Array(ts), Array(us) mirror what the higher-level wrapper transfers back.
                probs_d = cu(probs_host)
                sol = if mode == "fixed"
                    CUDA.@sync DiffEqGPU.vectorized_solve(probs_d, prob, solver,
                        saveat = duration, save_everystep = false, dt = dt0)
                else
                    CUDA.@sync DiffEqGPU.vectorized_asolve(probs_d, prob, solver,
                        saveat = duration, save_everystep = false,
                        reltol = TIMING_TOL, abstol = TIMING_TOL, dt = dt0)
                end
                ts = Array(sol[1])
                us = Array(sol[2])
                sol
            end
            outfile = joinpath(outdir, "Julia_times_$(mode)_$(algorithm).txt")
            on_breach = () -> begin
                isinteractive() || open(outfile, "a+") do io
                    println(io, numberOfParameters, " NaN NaN")
                end
                println("WATCHDOG $(problem["problem"]) $(mode) $(algorithm): run never returned")
            end

            t_ms, t_dev_ms, ran = try
                t_dev = watchdogged_min_ms(device_solve, on_breach, REPEATS)
                t = isnan(t_dev) ? NaN :
                    watchdogged_min_ms(full_solve, on_breach, REPEATS)
                isnan(t) && println("WATCHDOG $(problem["problem"]) " *
                                    "$(mode) $(algorithm): run exceeded the cap")
                (t, t_dev, !isnan(t))
            catch err
                (failed("$(problem["problem"]) $(mode) $(algorithm)", err), NaN, false)
            end

            if !isinteractive()
                open(outfile, "a+") do io
                    println(io, numberOfParameters, " ", t_ms, " ", t_dev_ms)
                end
            end

            # Save numerical output for 32768-trajectory run
            if ran && !isinteractive() && numberOfParameters == 32768 &&
               algorithm == "tsit5"
                sol = device_solve()
                write_finals(system, problem, sol,
                    mode == "fixed" ? "julia_fixed.csv" : "julia_adaptive.csv")
            end

            println("Parameter number: " * string(numberOfParameters))
            println("Minimum time: " * string(t_ms) * " ms")
        end
    end
end

"Write the per-trajectory final states for the pairwise numerical check."
function write_finals(system, problem, sol, name)
    final_states = Array(sol[2][end, :])  # convert to CPU Array
    df = DataFrame([Tuple(s[system.golden_index]) for s in final_states],
        :auto)
    CSV.write(joinpath(data_dir(REPO_ROOT, "numerical", DATASET_KEY, problem),
            name), df, header = false)
end

for problem in PROBLEMS
    if WP_MODE
        run_wp(problem)
    else
        run_times(problem)
    end
end
