using Pkg

Pkg.instantiate()
Pkg.precompile()

using CUDA
using DiffEqGPU, OrdinaryDiffEq, StaticArrays
using CSV, DataFrames, DelimitedFiles
using FileWatching.Pidfile: mkpidlock
using GPU_ODE_JuliaKernels

# CLI: <N|N,N,...>|wp|states:<nstates>:<N> [algorithm|all] [--problem <name|all>] [--mode <fixed|adaptive|all>].
@show ARGS
#settings
CUDA.allowscalar(false)

# Dataset key "<os>_<gpu>" keys output files per machine.
include(joinpath(dirname(@__DIR__), "runner_scripts", "bench_key.jl"))
include(joinpath(dirname(@__DIR__), "runner_scripts", "problems.jl"))
include(joinpath(dirname(@__DIR__), "runner_scripts", "algorithms.jl"))
include(joinpath(dirname(@__DIR__), "runner_scripts", "julia_systems.jl"))
include(joinpath(dirname(@__DIR__), "runner_scripts", "julia_prob.jl"))
include(joinpath(dirname(@__DIR__), "runner_scripts", "watchdog.jl"))
include(joinpath(dirname(@__DIR__), "runner_scripts", "samples.jl"))
include(joinpath(dirname(@__DIR__), "runner_scripts", "resume.jl"))
# Precompiled entries take precedence over runtime-built ones.
merge!(_ENTRIES, GPU_ODE_JuliaKernels.ENTRIES)
const DATASET_KEY = dataset_key()
const REPO_ROOT = dirname(@__DIR__)

requested_algorithm = "all"
requested_problem = "all"
requested_mode = "all"
let i = 2
    while i <= length(ARGS)
        tok = ARGS[i]
        if tok == "--problem" || tok == "-s"
            i += 1
            i <= length(ARGS) || error("--problem requires a value")
            global requested_problem = ARGS[i]
        elseif startswith(tok, "--problem=")
            global requested_problem = split(tok, "=", limit = 2)[2]
        elseif tok == "--mode"
            i += 1
            i <= length(ARGS) || error("--mode requires a value")
            global requested_mode = ARGS[i]
        elseif startswith(tok, "--mode=")
            global requested_mode = split(tok, "=", limit = 2)[2]
        else
            global requested_algorithm = tok
        end
        i += 1
    end
end
requested_mode in ("fixed", "adaptive", "all") ||
    error("--mode takes fixed, adaptive or all, got '$(requested_mode)'")
const REQUESTED_MODE = requested_mode
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

# Repeat ceiling; the count per leg follows its first timed run's duration.
const REPEATS = 20
const WP_MODE = !isempty(ARGS) && ARGS[1] == "wp"
const STATES_MODE = !isempty(ARGS) && startswith(ARGS[1], "states:")
# states:<nstates>:<ensemble>, one system size per process.
const STATES_ARGS = STATES_MODE ? parse.(Int, split(ARGS[1], ':')[2:3]) : Int[]
# Mirrors TIMING_TOL and N_WP in runner_scripts/wp_common.py.
const TIMING_TOL = 1.0f-5
const N_WP = 131072
# The N sweep runs ascending inside one process so each kernel compiles once.
const NS = isinteractive() ? [8192] :
           (WP_MODE ? [N_WP] :
            (STATES_MODE ? [STATES_ARGS[2]] :
             sort(parse.(Int64, split(ARGS[1], ',')))))

"Modes the algorithm runs under this framework and --mode, fixed first."
algorithm_modes(algorithm) = [mode
                              for (mode, list) in
                                  (("fixed", FIXED_ALGORITHMS),
                                   ("adaptive", ADAPTIVE_ALGORITHMS))
                              if algorithm in list &&
                                 REQUESTED_MODE in ("all", mode)]

"Ensemble l2-at-final error against the Float64 golden reference."
function ensemble_error(system, us, golden)
    final = Array(us[end, :])
    m = Matrix{Float64}(undef, length(final), length(system.golden_index))
    for i in eachindex(final)
        m[i, :] .= Float64.(final[i][system.golden_index])
    end
    return sqrt(sum(abs2, m .- golden) / length(m))
end

"An algorithm that cannot run this system is a NaN row, not an aborted run."
function failed(what, err)
    println("FAILED $(what): $(err)")
    return NaN
end

"Append `n NaN NaN` timing rows for the given sweep sizes; --floor merges them so recorded rows survive."
function nan_rows(outfile, ns)
    isinteractive() && return
    if floor_enabled()
        for n in ns
            merge_min_row(outfile, n, (NaN, NaN))
        end
        return
    end
    open(outfile, "a+") do io
        for n in ns
            println(io, n, " NaN NaN")
        end
        flush(io)
    end
end

# One wp sweep; a watchdog breach fills the remaining settings with NaN rows.
function wp_sweep(solve_once, system, problem, algorithm, mode, path, settings,
        golden, label)
    if skip_wp_leg(problem["problem"], algorithm, mode, path, length(settings))
        println("-- resume: skipping wp $(label) (already covered)")
        return
    end
    samples_file = samples_outfile(REPO_ROOT, "Julia", DATASET_KEY, "Julia",
        "wp", mode, algorithm, problem)
    setting_kind = mode == "fixed" ? "dt" : "tol"
    # --floor merges the new times in; the log gains a fresh series instead.
    floor_enabled() || reset_samples(samples_file)
    open(path, floor_enabled() ? "a" : "w") do io
        breached = false
        compiled = false
        for (index, setting) in enumerate(settings)
            if breached
                write_wp_row(io, path, setting, NaN, NaN)
                continue
            end
            point = sample_point("wp", problem["problem"], algorithm, mode,
                N_WP, problem["states"]; setting_kind = setting_kind,
                setting = setting)
            on_breach = () -> begin
                for s in settings[index:end]
                    write_wp_row(io, path, s, NaN, NaN)
                end
                flush(io)
                println("WATCHDOG $(label) setting=$(setting): run never returned")
            end
            t_ms, err = try
                if !compiled
                    # The first solve carries the kernel compile, off the GPU lock.
                    run_watchdogged(() -> solve_once(setting), on_breach)
                    compiled = true
                end
                with_gpu_lock() do
                    warm = @elapsed sol = run_watchdogged(
                        () -> solve_once(setting), on_breach)
                    if warm > WATCHDOG_SECONDS
                        (NaN, NaN)
                    else
                        e = ensemble_error(system, sol[2], golden)
                        t, samples = watchdogged_min_ms(
                            () -> solve_once(setting), on_breach, REPEATS)
                        # The ensemble is resident, so only the d2h is timed.
                        append_samples(samples_file, point, "d2h", samples)
                        (t, isnan(t) ? NaN : e)
                    end
                end
            catch err
                (failed("wp $(label) setting=$(setting)", err), NaN)
            end
            if isnan(t_ms)
                println("WATCHDOG wp $(label) setting=$(setting): run exceeded the cap")
                breached = true
            end
            write_wp_row(io, path, setting, t_ms, err)
            println("wp $(label) setting=$(setting): $(t_ms) ms, err=$(err)")
        end
    end
end

# Sweeps fixed dt and adaptive tolerance at N=N_WP; grids mirror runner_scripts/wp_common.py.
function run_wp(problem)
    golden = readdlm(
        joinpath(REPO_ROOT, "data", "numerical",
            "golden_$(problem["problem"])_$(N_WP).csv"), ',', Float64)
    system, prob, duration = build_prob(problem)
    probs_host, probs = build_ensemble(system, prob, problem, N_WP)
    outdir = data_dir(REPO_ROOT, "Julia", DATASET_KEY, problem)
    dt0 = Float32(problem_timing_dt(problem))

    for algorithm in ALGORITHMS
        problem_supports(problem, "julia") || continue
        solver = gpu_solver(algorithm)
        label = "$(problem["problem"]) $(algorithm)"

        if "fixed" in algorithm_modes(algorithm)
            wp_sweep(system, problem, algorithm, "fixed",
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

        if "adaptive" in algorithm_modes(algorithm)
            wp_sweep(system, problem, algorithm, "adaptive",
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

# One (algorithm, mode) leg: every sweep size ascending on one compiled kernel.
function run_leg(problem, system, prob, duration, algorithm, mode, later_legs)
    solver = gpu_solver(algorithm)
    dt0 = Float32(problem_timing_dt(problem))
    outdir = data_dir(REPO_ROOT, "Julia", DATASET_KEY, problem)
    outfile = joinpath(outdir, "Julia_times_$(mode)_$(algorithm).txt")
    samples_file = samples_outfile(REPO_ROOT, "Julia", DATASET_KEY, "Julia",
        "times", mode, algorithm, problem)
    compiled = false

    run_ns = [n for n in NS
              if !skip_point(problem["problem"], algorithm, mode, n, outfile)]
    if isempty(run_ns)
        println("-- resume: skipping $(problem["problem"]) $(mode) " *
                "$(algorithm) (already covered)")
        return
    end
    if length(run_ns) < length(NS)
        println("-- resume: $(problem["problem"]) $(mode) $(algorithm) " *
                "runs N=" * join(run_ns, ","))
    end
    # Drop stale rows for the points about to rerun.
    prune_reruns(outfile, run_ns)

    for (index, n) in enumerate(run_ns)
        @info "Solving $(problem["problem"]) on GPU ($(mode) dt, $(algorithm), N=$(n))"
        probs_host, probs = build_ensemble(system, prob, problem, n)

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
        # NaN rows for every uncovered point this process will no longer reach.
        on_breach = () -> begin
            nan_rows(outfile, run_ns[index:end])
            for (later_algorithm, later_mode) in later_legs()
                later_out = joinpath(outdir,
                    "Julia_times_$(later_mode)_$(later_algorithm).txt")
                nan_rows(later_out,
                    [m for m in NS
                     if !skip_point(problem["problem"], later_algorithm,
                         later_mode, m, later_out)])
            end
            println("WATCHDOG $(problem["problem"]) $(mode) $(algorithm) " *
                    "N=$(n): run never returned")
        end

        t_ms, t_dev_ms, breached = try
            if !compiled
                # The first solve carries the kernel compile, off the GPU lock.
                run_watchdogged(device_solve, on_breach)
                compiled = true
            end
            point = sample_point("times", problem["problem"], algorithm, mode,
                n, problem["states"])
            t_dev, t = with_gpu_lock() do
                td, samples = watchdogged_min_ms(device_solve, on_breach,
                    REPEATS)
                append_samples(samples_file, point, "none", samples)
                isnan(td) && return (td, NaN)
                tt, samples = watchdogged_min_ms(full_solve, on_breach,
                    REPEATS)
                append_samples(samples_file, point, "both", samples)
                (td, tt)
            end
            (t, t_dev, isnan(t))
        catch err
            (failed("$(problem["problem"]) $(mode) $(algorithm) N=$(n)", err),
                NaN, false)
        end
        ran = !isnan(t_ms)

        isinteractive() || record_row(outfile, n, (t_ms, t_dev_ms))

        # Save numerical output for 32768-trajectory run
        if ran && !isinteractive() && n == 32768 && algorithm == "tsit5"
            sol = device_solve()
            write_finals(system, problem, sol,
                mode == "fixed" ? "julia_fixed.csv" : "julia_adaptive.csv",
                duration)
        end

        # Ensembles are per-size; only the compiled kernels carry over.
        probs_host = nothing
        probs = nothing
        GC.gc()
        CUDA.reclaim()

        println("Parameter number: " * string(n))
        println("Minimum time: " * string(t_ms) * " ms")

        if breached
            # Larger sweep sizes are slower, so the leg ends here.
            println("WATCHDOG $(problem["problem"]) $(mode) $(algorithm) " *
                    "N=$(n): run exceeded the cap")
            nan_rows(outfile, run_ns[(index + 1):end])
            return
        end
    end
end

# Serialize timed GPU sections on a pidfile; stale_age breaks dead owners' locks.
function with_gpu_lock(f)
    path = get(ENV, "BENCH_GPU_LOCK", "")
    isempty(path) && return f()
    gpu_lock = mkpidlock(path; wait = true, stale_age = 120)
    try
        return f()
    finally
        close(gpu_lock)
    end
end

# One system size per process; the driver enforces the compile budget and
# backfills rows for processes that never wrote them.
function run_states(nstates, n)
    # Runtime entry at every size, so build_s is a cold compile.
    entry = _lorenz96_entry(nstates)
    row = copy(get_problem("lorenz96"))
    row["states"] = nstates
    system, prob, duration = build_prob_parts(entry, row)
    dt0 = Float32(problem_timing_dt(row))
    outdir = data_dir(REPO_ROOT, "Julia", DATASET_KEY, row)

    for algorithm in ALGORITHMS
        solver = gpu_solver(algorithm)
        for mode in algorithm_modes(algorithm)
            outfile = joinpath(outdir, "Julia_states_$(mode)_$(algorithm).txt")
            if skip_point(row["problem"], algorithm, mode, nstates, outfile)
                println("-- resume: skipping states=$(nstates) $(mode) " *
                        "$(algorithm) (already covered)")
                continue
            end
            samples_file = samples_outfile(REPO_ROOT, "Julia", DATASET_KEY,
                "Julia", "states", mode, algorithm, row)
            @info "Solving lorenz96 states=$(nstates) on GPU ($(mode) dt, $(algorithm), N=$(n))"
            t_ms, t_dev_ms, build_s = try
                probs_host, probs = build_ensemble(system, prob, row, n)
                device_solve = () -> begin
                    if mode == "fixed"
                        CUDA.@sync DiffEqGPU.vectorized_solve(probs, prob,
                            solver, saveat = duration,
                            save_everystep = false, dt = dt0)
                    else
                        CUDA.@sync DiffEqGPU.vectorized_asolve(probs, prob,
                            solver, saveat = duration,
                            save_everystep = false,
                            reltol = TIMING_TOL, abstol = TIMING_TOL,
                            dt = dt0)
                    end
                end
                full_solve = () -> begin
                    probs_d = cu(probs_host)
                    sol = if mode == "fixed"
                        CUDA.@sync DiffEqGPU.vectorized_solve(probs_d, prob,
                            solver, saveat = duration,
                            save_everystep = false, dt = dt0)
                    else
                        CUDA.@sync DiffEqGPU.vectorized_asolve(probs_d, prob,
                            solver, saveat = duration,
                            save_everystep = false,
                            reltol = TIMING_TOL, abstol = TIMING_TOL,
                            dt = dt0)
                    end
                    ts = Array(sol[1])
                    us = Array(sol[2])
                    sol
                end
                # Uncapped: the first solve carries the kernel compile.
                build = @elapsed device_solve()
                marker = get(ENV, "BENCH_STATES_MARKER", "")
                isempty(marker) || touch(marker)
                on_breach = () -> println("WATCHDOG lorenz96 " *
                    "states=$(nstates) $(mode) $(algorithm) N=$(n): " *
                    "run never returned")
                point = sample_point("states", row["problem"], algorithm,
                    mode, n, nstates)
                t_dev, t = with_gpu_lock() do
                    td, samples = watchdogged_min_ms(device_solve, on_breach,
                        REPEATS)
                    append_samples(samples_file, point, "none", samples)
                    isnan(td) && return (td, NaN)
                    tt, samples = watchdogged_min_ms(full_solve, on_breach,
                        REPEATS)
                    append_samples(samples_file, point, "both", samples)
                    (td, tt)
                end
                isnan(t) &&
                    println("WATCHDOG lorenz96 states=$(nstates) $(mode) " *
                            "$(algorithm) N=$(n): run exceeded the cap")
                (t, t_dev, build)
            catch err
                (failed("lorenz96 states=$(nstates) $(mode) $(algorithm) " *
                        "N=$(n)", err), NaN, NaN)
            end
            isinteractive() ||
                record_row(outfile, nstates, (t_ms, t_dev_ms, build_s))
            GC.gc()
            CUDA.reclaim()
            println("states=$(nstates) $(mode) $(algorithm): $(t_ms) ms")
        end
    end
end

function run_times(problem)
    legs = [(algorithm, mode) for algorithm in ALGORITHMS
            if problem_supports(problem, "julia")
            for mode in algorithm_modes(algorithm)]
    if isempty(legs)
        println("Julia runs none of the requested algorithms on ",
            problem["problem"], "; skipping.")
        return
    end
    system, prob, duration = build_prob(problem)
    for (index, (algorithm, mode)) in enumerate(legs)
        run_leg(problem, system, prob, duration, algorithm, mode,
            () -> legs[(index + 1):end])
    end
end

"Write the per-trajectory final states for the pairwise numerical check; a trajectory that never reached `duration` is a NaN row."
function write_finals(system, problem, sol, name, duration)
    # Do not count solves that never wrote a final time
    final_times = Array(sol[1][end, :])
    final_states = Array(sol[2][end, :])
    # One row per trajectory, one column per golden state.
    m = Matrix{Float64}(undef, length(final_states),
        length(system.golden_index))
    arrived = 0
    for i in eachindex(final_states)
        if isapprox(Float64(final_times[i]), Float64(duration); rtol = 1.0f-4)
            m[i, :] .= Float64.(final_states[i][system.golden_index])
            arrived += 1
        else
            m[i, :] .= NaN
        end
    end
    if arrived < length(final_states)
        @warn "$(name): $(length(final_states) - arrived) of " *
              "$(length(final_states)) trajectories stopped before " *
              "t=$(duration); written as NaN rows"
    end
    df = DataFrame(m, :auto)
    CSV.write(joinpath(data_dir(REPO_ROOT, "numerical", DATASET_KEY, problem),
            name), df, header = false)
end

if STATES_MODE
    run_states(STATES_ARGS[1], STATES_ARGS[2])
else
    for problem in PROBLEMS
        if WP_MODE
            run_wp(problem)
        else
            run_times(problem)
        end
    end
end
