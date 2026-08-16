using Pkg

Pkg.instantiate()
Pkg.precompile()

using CUDA
using BenchmarkTools, DiffEqGPU, OrdinaryDiffEq, StaticArrays
using CSV, DataFrames, DelimitedFiles

# CLI: <N> [wp] [algorithm|all] [--problem <name|all>]
@show ARGS
#settings
CUDA.allowscalar(false)
numberOfParameters = isinteractive() ? 8192 : parse(Int64, ARGS[1])

# Dataset key "<os>_<gpu>" keys output files per machine.
include(joinpath(dirname(@__DIR__), "runner_scripts", "bench_key.jl"))
include(joinpath(dirname(@__DIR__), "runner_scripts", "problems.jl"))
include(joinpath(dirname(@__DIR__), "runner_scripts", "algorithms.jl"))
include(joinpath(dirname(@__DIR__), "runner_scripts", "protocol.jl"))
include(joinpath(dirname(@__DIR__), "runner_scripts", "julia_systems.jl"))
const DATASET_KEY = dataset_key()
const REPO_ROOT = dirname(@__DIR__)

requested_algorithm = "all"
requested_problem = "all"
let i = 2
    while i <= length(ARGS)
        tok = ARGS[i]
        if tok == "wp"
        elseif tok == "--problem" || tok == "-s"
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
const WP_MODE = "wp" in ARGS
# The kernel solvers fix their own controller; only the tolerance is shared.
const TIMING_TOL = Float32(load_protocol()["timing_tol"])

"DiffEqGPU kernel solver for an algorithm name; autodiff off matches the overlap suite."
function gpu_solver(algorithm)
    algorithm == "tsit5" && return GPUTsit5()
    algorithm == "rosenbrock23_sciml" && return GPURosenbrock23(autodiff = Val(false))
    algorithm == "kvaerno3" && return GPUKvaerno3(autodiff = Val(false))
    error("no DiffEqGPU kernel solver for '$(algorithm)'")
end

"Ensemble l2-at-final error against the Float64 golden reference."
function ensemble_error(us, golden, nstates)
    final = Array(us[end, :])
    m = Matrix{Float64}(undef, length(final), nstates)
    for i in eachindex(final)
        m[i, :] .= Float64.(final[i])
    end
    return sqrt(sum(abs2, m .- golden) / length(m))
end

function build_probs(problem)
    system = julia_system(problem)
    duration = Float32(problem["duration"])
    f = system.mass_matrix === nothing ? ODEFunction{false}(system.rhs) :
        ODEFunction{false}(system.rhs; mass_matrix = system.mass_matrix)
    prob = ODEProblem{false}(f, system.u0, (0.0f0, duration),
        @SArray [Float32(problem["sweep_max"])])
    grid = Float32.(collect(problem_sweep(problem, numberOfParameters)))
    probs_host = map(1:numberOfParameters) do i
        DiffEqGPU.make_prob_compatible(remake(prob, p = @SVector [grid[i]]))
    end
    return prob, probs_host, cu(probs_host), duration
end

"An algorithm that cannot run this system is a NaN row, not an aborted run."
function failed(what, err)
    println("FAILED $(what): $(err)")
    return NaN
end

# Sweeps fixed dt and adaptive tolerance at N=32768; grids mirror runner_scripts/wp_common.py.
function run_wp(problem)
    numberOfParameters == 32768 || error("wp mode must be run with N = 32768")
    golden = readdlm(
        joinpath(REPO_ROOT, "data", "numerical",
            "golden_$(problem["problem"])_32768.csv"), ',', Float64)
    nstates = problem["states"]
    prob, _, probs, duration = build_probs(problem)
    outdir = data_dir(REPO_ROOT, "Julia", DATASET_KEY, problem)

    for algorithm in ALGORITHMS
        solver = gpu_solver(algorithm)

        if algorithm in FIXED_ALGORITHMS
            open(joinpath(outdir, "Julia_wp_fixed_$(algorithm).txt"), "w") do io
                for dt in problem_dts(problem)
                    dt32 = Float32(dt)
                    t_ms, err = try
                        CUDA.@sync sol = DiffEqGPU.vectorized_solve(probs, prob,
                            solver; saveat = duration, save_everystep = false,
                            dt = dt32)
                        e = ensemble_error(sol[2], golden, nstates)
                        data = @benchmark begin
                            CUDA.@sync sol = DiffEqGPU.vectorized_solve($probs,
                                $prob, $solver; saveat = $duration,
                                save_everystep = false, dt = $dt32)
                            ts = Array(sol[1])
                            us = Array(sol[2])
                        end
                        (minimum(data.times) / 1e6, e)
                    catch err
                        (failed("wp $(problem["problem"]) fixed $(algorithm) dt=$(dt)", err), NaN)
                    end
                    println(io, dt, " ", t_ms, " ", err)
                    println("wp $(problem["problem"]) fixed $(algorithm) dt=$(dt): $(t_ms) ms, err=$(err)")
                end
            end
        end

        if algorithm in ADAPTIVE_ALGORITHMS
            open(joinpath(outdir, "Julia_wp_adaptive_$(algorithm).txt"), "w") do io
                for tol in [10.0^-k for k in 2:8]
                    tol32 = Float32(tol)
                    dt0 = Float32(problem_timing_dt(problem))
                    t_ms, err = try
                        CUDA.@sync sol = DiffEqGPU.vectorized_asolve(probs, prob,
                            solver; saveat = duration, save_everystep = false,
                            reltol = tol32, abstol = tol32, dt = dt0)
                        e = ensemble_error(sol[2], golden, nstates)
                        data = @benchmark begin
                            CUDA.@sync sol = DiffEqGPU.vectorized_asolve($probs,
                                $prob, $solver; saveat = $duration,
                                save_everystep = false, reltol = $tol32,
                                abstol = $tol32, dt = $dt0)
                            ts = Array(sol[1])
                            us = Array(sol[2])
                        end
                        (minimum(data.times) / 1e6, e)
                    catch err
                        (failed("wp $(problem["problem"]) adaptive $(algorithm) tol=$(tol)", err), NaN)
                    end
                    println(io, tol, " ", t_ms, " ", err)
                    println("wp $(problem["problem"]) adaptive $(algorithm) tol=$(tol): $(t_ms) ms, err=$(err)")
                end
            end
        end
    end
end

function run_times(problem)
    prob, probs_host, probs, duration = build_probs(problem)
    dt0 = Float32(problem_timing_dt(problem))
    outdir = data_dir(REPO_ROOT, "Julia", DATASET_KEY, problem)

    for algorithm in ALGORITHMS
        solver = gpu_solver(algorithm)

        if algorithm in FIXED_ALGORITHMS
            @info "Solving $(problem["problem"]) on GPU (fixed dt, $(algorithm))"
            t_ms, t_dev_ms, ran = try
                # Device-only: probs already resident, results left there.
                data_dev = @benchmark begin
                    CUDA.@sync DiffEqGPU.vectorized_solve($probs, $prob,
                        $solver, saveat = $duration, save_everystep = false,
                        dt = $dt0)
                end samples=REPEATS evals=1 seconds=1e9
                data = @benchmark begin
                    # Array(ts), Array(us) mirror what the higher-level wrapper transfers back.
                    probs_d = cu($probs_host)
                    CUDA.@sync sol = DiffEqGPU.vectorized_solve(probs_d, $prob,
                        $solver, saveat = $duration, save_everystep = false,
                        dt = $dt0)
                    ts = Array(sol[1])
                    us = Array(sol[2])
                end samples=REPEATS evals=1 seconds=1e9
                (minimum(data.times) / 1e6, minimum(data_dev.times) / 1e6, true)
            catch err
                (failed("$(problem["problem"]) fixed $(algorithm)", err), NaN, false)
            end

            if !isinteractive()
                open(joinpath(outdir, "Julia_times_fixed_$(algorithm).txt"), "a+") do io
                    println(io, numberOfParameters, " ", t_ms, " ", t_dev_ms)
                end
            end

            # Save numerical output for 32768-trajectory run
            if ran && !isinteractive() && numberOfParameters == 32768 &&
               algorithm == "tsit5"
                CUDA.@sync sol = DiffEqGPU.vectorized_solve(probs, prob, solver,
                    saveat = duration, save_everystep = false, dt = dt0)
                write_finals(problem, sol, "julia_fixed.csv")
            end

            println("Parameter number: " * string(numberOfParameters))
            println("Minimum time: " * string(t_ms) * " ms")
        end

        if algorithm in ADAPTIVE_ALGORITHMS
            @info "Solving $(problem["problem"]) on GPU (adaptive dt, $(algorithm))"
            t_ms, t_dev_ms, ran = try
                # Device-only: probs already resident, results left there.
                data_dev = @benchmark begin
                    CUDA.@sync DiffEqGPU.vectorized_asolve($probs, $prob,
                        $solver, saveat = $duration, save_everystep = false,
                        reltol = TIMING_TOL, abstol = TIMING_TOL, dt = $dt0)
                end samples=REPEATS evals=1 seconds=1e9
                data = @benchmark begin
                    probs_d = cu($probs_host)
                    CUDA.@sync sol = DiffEqGPU.vectorized_asolve(probs_d, $prob,
                        $solver, saveat = $duration, save_everystep = false,
                        reltol = TIMING_TOL, abstol = TIMING_TOL, dt = $dt0)
                    ts = Array(sol[1])
                    us = Array(sol[2])
                end samples=REPEATS evals=1 seconds=1e9
                (minimum(data.times) / 1e6, minimum(data_dev.times) / 1e6, true)
            catch err
                (failed("$(problem["problem"]) adaptive $(algorithm)", err), NaN, false)
            end

            if !isinteractive()
                open(joinpath(outdir, "Julia_times_adaptive_$(algorithm).txt"), "a+") do io
                    println(io, numberOfParameters, " ", t_ms, " ", t_dev_ms)
                end
            end

            println("Parameter number: " * string(numberOfParameters))
            println("Minimum time: " * string(t_ms) * " ms")

            if ran && !isinteractive() && numberOfParameters == 32768 &&
               algorithm == "tsit5"
                CUDA.@sync sol = DiffEqGPU.vectorized_asolve(probs, prob, solver,
                    saveat = duration, save_everystep = false,
                    reltol = TIMING_TOL, abstol = TIMING_TOL, dt = dt0)
                write_finals(problem, sol, "julia_adaptive.csv")
            end
        end
    end
end

"Write the per-trajectory final states for the pairwise numerical check."
function write_finals(problem, sol, name)
    final_states = Array(sol[2][end, :])  # convert to CPU Array
    df = DataFrame([Tuple(s) for s in final_states], :auto)
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
