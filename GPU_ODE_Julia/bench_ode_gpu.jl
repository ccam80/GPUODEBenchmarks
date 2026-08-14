using Pkg

Pkg.instantiate()
Pkg.precompile()

using CUDA
using BenchmarkTools, DiffEqGPU, OrdinaryDiffEq, StaticArrays
using CSV, DataFrames, DelimitedFiles

# CLI: <N> [wp] [algorithm|all] [--problem <name|all>]; only Tsit5 runs here.
@show ARGS
#settings
CUDA.allowscalar(false)
numberOfParameters = isinteractive() ? 8192 : parse(Int64, ARGS[1])

# Dataset key "<os>_<gpu>" keys output files per machine.
include(joinpath(dirname(@__DIR__), "runner_scripts", "bench_key.jl"))
include(joinpath(dirname(@__DIR__), "runner_scripts", "problems.jl"))
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
if !(requested_algorithm in ("all", "tsit5"))
    println("Julia (DiffEqGPU kernel path) does not support algorithm '",
        requested_algorithm, "'; skipping.")
    exit(0)
end
const ALGORITHM = "tsit5"
const PROBLEMS = resolve_problems(requested_problem, "julia")
if isempty(PROBLEMS)
    println("Julia runs none of the requested problems; skipping.")
    exit(0)
end

# Fixed sample count to match the other frameworks.
const REPEATS = 20
const WP_MODE = "wp" in ARGS

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
    prob = ODEProblem{false}(system.rhs, system.u0, (0.0f0, duration),
        @SArray [Float32(problem["sweep_max"])])
    grid = Float32.(collect(problem_sweep(problem, numberOfParameters)))
    probs_host = map(1:numberOfParameters) do i
        DiffEqGPU.make_prob_compatible(remake(prob, p = @SVector [grid[i]]))
    end
    return prob, probs_host, cu(probs_host), duration
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

    open(joinpath(outdir, "Julia_wp_fixed_$(ALGORITHM).txt"), "w") do io
        for dt in problem_dts(problem)
            dt32 = Float32(dt)
            CUDA.@sync sol = DiffEqGPU.vectorized_solve(probs, prob, GPUTsit5();
                saveat = duration, save_everystep = false, dt = dt32)
            err = ensemble_error(sol[2], golden, nstates)
            data = @benchmark begin
                CUDA.@sync sol = DiffEqGPU.vectorized_solve($probs, $prob,
                    GPUTsit5(); saveat = $duration, save_everystep = false,
                    dt = $dt32)
                ts = Array(sol[1])
                us = Array(sol[2])
            end
            t_ms = minimum(data.times) / 1e6
            println(io, dt, " ", t_ms, " ", err)
            println("wp $(problem["problem"]) fixed dt=$(dt): $(t_ms) ms, err=$(err)")
        end
    end

    open(joinpath(outdir, "Julia_wp_adaptive_$(ALGORITHM).txt"), "w") do io
        for tol in [10.0^-k for k in 2:8]
            tol32 = Float32(tol)
            dt0 = Float32(problem_timing_dt(problem))
            CUDA.@sync sol = DiffEqGPU.vectorized_asolve(probs, prob,
                GPUTsit5(); saveat = duration, save_everystep = false,
                reltol = tol32, abstol = tol32, dt = dt0)
            err = ensemble_error(sol[2], golden, nstates)
            data = @benchmark begin
                CUDA.@sync sol = DiffEqGPU.vectorized_asolve($probs, $prob,
                    GPUTsit5(); saveat = $duration, save_everystep = false,
                    reltol = $tol32, abstol = $tol32, dt = $dt0)
                ts = Array(sol[1])
                us = Array(sol[2])
            end
            t_ms = minimum(data.times) / 1e6
            println(io, tol, " ", t_ms, " ", err)
            println("wp $(problem["problem"]) adaptive tol=$(tol): $(t_ms) ms, err=$(err)")
        end
    end
end

function run_times(problem)
    prob, probs_host, probs, duration = build_probs(problem)
    dt0 = Float32(problem_timing_dt(problem))
    outdir = data_dir(REPO_ROOT, "Julia", DATASET_KEY, problem)

    @info "Solving $(problem["problem"]) on GPU (fixed dt)"
    # Device-only: probs already resident, results left there.
    data_dev = @benchmark begin
        CUDA.@sync DiffEqGPU.vectorized_solve($probs, $prob, GPUTsit5(),
            saveat = $duration, save_everystep = false, dt = $dt0)
    end samples=REPEATS evals=1 seconds=1e9
    data = @benchmark begin
        # Array(ts), Array(us) mirror what the higher-level wrapper transfers back.
        probs_d = cu($probs_host)
        CUDA.@sync sol = DiffEqGPU.vectorized_solve(probs_d, $prob, GPUTsit5(),
            saveat = $duration, save_everystep = false, dt = $dt0)
        ts = Array(sol[1])
        us = Array(sol[2])
    end samples=REPEATS evals=1 seconds=1e9

    if !isinteractive()
        open(joinpath(outdir, "Julia_times_fixed_$(ALGORITHM).txt"), "a+") do io
            println(io, numberOfParameters, " ", minimum(data.times) / 1e6,
                " ", minimum(data_dev.times) / 1e6)
        end
    end

    # Save numerical output for 32768-trajectory run
    if !isinteractive() && numberOfParameters == 32768
        CUDA.@sync sol = DiffEqGPU.vectorized_solve(probs, prob, GPUTsit5(),
            saveat = duration, save_everystep = false, dt = dt0)
        write_finals(problem, sol, "julia_fixed.csv")
    end

    println("Parameter number: " * string(numberOfParameters))
    println("Minimum time: " * string(minimum(data.times) / 1e6) * " ms")
    println("Allocs: " * string(data.allocs))

    @info "Solving $(problem["problem"]) on GPU (adaptive dt)"
    # Device-only: probs already resident, results left there.
    data_dev = @benchmark begin
        CUDA.@sync DiffEqGPU.vectorized_asolve($probs, $prob, GPUTsit5(),
            saveat = $duration, save_everystep = false,
            reltol = 1.0f-8, abstol = 1.0f-8, dt = $dt0)
    end samples=REPEATS evals=1 seconds=1e9
    data = @benchmark begin
        probs_d = cu($probs_host)
        CUDA.@sync sol = DiffEqGPU.vectorized_asolve(probs_d, $prob, GPUTsit5(),
            saveat = $duration, save_everystep = false,
            reltol = 1.0f-8, abstol = 1.0f-8, dt = $dt0)
        ts = Array(sol[1])
        us = Array(sol[2])
    end samples=REPEATS evals=1 seconds=1e9

    if !isinteractive()
        open(joinpath(outdir, "Julia_times_adaptive_$(ALGORITHM).txt"), "a+") do io
            println(io, numberOfParameters, " ", minimum(data.times) / 1f6,
                " ", minimum(data_dev.times) / 1f6)
        end
    end

    println("Parameter number: " * string(numberOfParameters))
    println("Minimum time: " * string(minimum(data.times) / 1f6) * " ms")
    println("Allocs: " * string(data.allocs))

    if !isinteractive() && numberOfParameters == 32768
        CUDA.@sync sol = DiffEqGPU.vectorized_asolve(probs, prob, GPUTsit5(),
            saveat = duration, save_everystep = false,
            reltol = 1.0f-8, abstol = 1.0f-8, dt = dt0)
        write_finals(problem, sol, "julia_adaptive.csv")
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
