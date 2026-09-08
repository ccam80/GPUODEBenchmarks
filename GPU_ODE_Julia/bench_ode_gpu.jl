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
include(joinpath(dirname(@__DIR__), "runner_scripts", "errored.jl"))
include(joinpath(dirname(@__DIR__), "runner_scripts", "results.jl"))
include(joinpath(dirname(@__DIR__), "runner_scripts", "resume.jl"))
# Precompiled entries take precedence over runtime-built ones.
merge!(_ENTRIES, GPU_ODE_JuliaKernels.ENTRIES)
const DATASET_KEY = dataset_key()
const REPO_ROOT = dirname(@__DIR__)
const STORE = result_store(REPO_ROOT, "julia", DATASET_KEY)

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

const REPEATS = REPEAT_CAP
const WP_MODE = !isempty(ARGS) && ARGS[1] == "wp"
const STATES_MODE = !isempty(ARGS) && startswith(ARGS[1], "states:")
# states:<nstates>:<ensemble>, one system size per process.
const STATES_ARGS = STATES_MODE ? parse.(Int, split(ARGS[1], ':')[2:3]) : Int[]
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
    m = Float64.(final_states(system, Array(us[end, :])))
    return sqrt(sum(abs2, m .- golden) / length(m))
end

"An algorithm that cannot run this system is a NaN row, not an aborted run."
function failed(what, err)
    println("FAILED $(what): $(err)")
    return NaN
end

"Record NaN rows for the given sweep sizes of a leg."
function nan_rows(problem, algorithm, mode, ns)
    isinteractive() && return
    for n in ns
        result_record_times(STORE, "julia", DATASET_KEY, "times", problem,
            algorithm, mode, n, problem["states"], NaN, NaN, 100.0)
    end
end

"Record NaN wp rows for the given settings of a leg."
function nan_wp_rows(problem, algorithm, mode, settings)
    for setting in settings
        result_record_wp(STORE, "julia", DATASET_KEY, problem, algorithm, mode,
            setting, NaN, NaN, 100.0; transfers = "d2h")
    end
end

"True when one N or states point of a julia leg is covered."
function point_covered(analysis, problem, algorithm, mode, n, states)
    kind, setting = timing_setting(problem, mode)
    key = analysis == "states" ? states : n
    return skip_point(STORE, analysis, problem["problem"], algorithm, mode, key,
        n, states, kind, setting)
end

# One wp sweep; a watchdog breach fills the remaining settings with NaN rows.
function wp_sweep(solve_once, system, problem, algorithm, mode, settings,
        golden, label)
    if skip_wp_leg(STORE, problem["problem"], algorithm, mode, settings,
            problem["states"])
        println("-- resume: skipping wp $(label) (already covered)")
        return
    end
    compiled = false
    for (index, setting) in enumerate(settings)
        on_breach = () -> begin
            nan_wp_rows(problem, algorithm, mode, settings[index:end])
            println("WATCHDOG $(label) setting=$(setting): run never returned")
        end
        samples = nothing
        t_ms, err, pct = try
            if !compiled
                # The first solve carries the kernel compile, off the GPU lock.
                run_watchdogged(() -> solve_once(setting), on_breach)
                compiled = true
            end
            with_gpu_lock() do
                warm = @elapsed sol = run_watchdogged(
                    () -> solve_once(setting), on_breach)
                if warm > WATCHDOG_SECONDS
                    (NaN, NaN, 100.0)
                else
                    e = ensemble_error(system, sol[2], golden)
                    p = errored_pct(@view sol[2][end, :])
                    # The ensemble is resident, so only the d2h is timed.
                    t, samples, _ = watchdogged_min_ms(
                        () -> solve_once(setting), on_breach, REPEATS)
                    (t, isnan(t) ? NaN : e, p)
                end
            end
        catch err
            (failed("wp $(label) setting=$(setting)", err), NaN, 100.0)
        end
        result_record_wp(STORE, "julia", DATASET_KEY, problem, algorithm, mode,
            setting, t_ms, err, pct; transfers = "d2h", samples = samples)
        println("wp $(label) setting=$(setting): $(t_ms) ms, err=$(err), " *
                "errored=$(round(pct, digits = 1))%")
        if isnan(t_ms)
            println("WATCHDOG wp $(label) setting=$(setting): run exceeded the cap")
            nan_wp_rows(problem, algorithm, mode, settings[(index + 1):end])
            break
        end
    end
end

# Sweeps fixed dt and adaptive tolerance at N=N_WP; grids mirror runner_scripts/wp_common.py.
function run_wp(problem)
    golden = readdlm(golden_path(problem), ',', Float64)
    system, prob, duration = build_prob(problem)
    probs_host, probs = build_ensemble(system, prob, problem, N_WP)

    for algorithm in ALGORITHMS
        problem_supports(problem, "julia") || continue
        solver = gpu_solver(algorithm)
        label = "$(problem["problem"]) $(algorithm)"
        settings = Dict("fixed" => collect(problem_dts(problem, algorithm)),
            "adaptive" => TOLS)
        for mode in algorithm_modes(algorithm)
            # The ensemble is resident; each solve is timed with its d2h.
            wp_sweep(system, problem, algorithm, mode, settings[mode], golden,
                "$(label) $(mode)") do setting
                gpu_solve_d2h(probs, prob, solver, mode, setting, problem)[1]
            end
        end
    end
end

# One (algorithm, mode) leg: every sweep size ascending on one compiled kernel.
function run_leg(problem, system, prob, duration, algorithm, mode, later_legs)
    solver = gpu_solver(algorithm)
    _, setting = timing_setting(problem, mode)
    compiled = false

    run_ns = [n for n in NS
              if !point_covered("times", problem, algorithm, mode, n,
                  problem["states"])]
    if isempty(run_ns)
        println("-- resume: skipping $(problem["problem"]) $(mode) " *
                "$(algorithm) (already covered)")
        return
    end
    if length(run_ns) < length(NS)
        println("-- resume: $(problem["problem"]) $(mode) $(algorithm) " *
                "runs N=" * join(run_ns, ","))
    end

    for (index, n) in enumerate(run_ns)
        @info "Solving $(problem["problem"]) on GPU ($(mode) dt, $(algorithm), N=$(n))"
        probs_host, probs = build_ensemble(system, prob, problem, n)

        device_solve = () -> gpu_solve_device(probs, prob, solver, mode,
            setting, problem)
        full_solve = () -> gpu_solve_host(probs_host, prob, solver, mode,
            setting, problem)[1]
        # NaN rows for every uncovered point this process will no longer reach.
        on_breach = () -> begin
            nan_rows(problem, algorithm, mode, run_ns[index:end])
            for (later_algorithm, later_mode) in later_legs()
                nan_rows(problem, later_algorithm, later_mode,
                    [m for m in NS
                     if !point_covered("times", problem, later_algorithm,
                         later_mode, m, problem["states"])])
            end
            println("WATCHDOG $(problem["problem"]) $(mode) $(algorithm) " *
                    "N=$(n): run never returned")
        end

        samples_none = samples_both = nothing
        t_ms, t_dev_ms, pct, breached = try
            if !compiled
                # The first solve carries the kernel compile, off the GPU lock.
                run_watchdogged(device_solve, on_breach)
                compiled = true
            end
            t_dev, t, pct = with_gpu_lock() do
                td, samples_none, dev_sol = watchdogged_min_ms(device_solve,
                    on_breach, REPEATS)
                isnan(td) && return (td, NaN, 100.0)
                p = errored_pct(@view dev_sol[2][end, :])
                tt, samples_both, _ = watchdogged_min_ms(full_solve, on_breach,
                    REPEATS)
                (td, tt, p)
            end
            (t, t_dev, pct, isnan(t))
        catch err
            (failed("$(problem["problem"]) $(mode) $(algorithm) N=$(n)", err),
                NaN, 100.0, false)
        end
        ran = !isnan(t_ms)

        isinteractive() || result_record_times(STORE, "julia", DATASET_KEY,
            "times", problem, algorithm, mode, n, problem["states"], t_ms,
            t_dev_ms, pct; samples_both = samples_both,
            samples_none = samples_none)

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
            nan_rows(problem, algorithm, mode, run_ns[(index + 1):end])
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

    for algorithm in ALGORITHMS
        solver = gpu_solver(algorithm)
        for mode in algorithm_modes(algorithm)
            _, setting = timing_setting(row, mode)
            if point_covered("states", row, algorithm, mode, n, nstates)
                println("-- resume: skipping states=$(nstates) $(mode) " *
                        "$(algorithm) (already covered)")
                continue
            end
            @info "Solving lorenz96 states=$(nstates) on GPU ($(mode) dt, $(algorithm), N=$(n))"
            samples_none = samples_both = nothing
            t_ms, t_dev_ms, build_s, pct = try
                probs_host, probs = build_ensemble(system, prob, row, n)
                device_solve = () -> gpu_solve_device(probs, prob, solver,
                    mode, setting, row)
                full_solve = () -> gpu_solve_host(probs_host, prob, solver,
                    mode, setting, row)[1]
                # Uncapped: the first solve carries the kernel compile.
                build = @elapsed device_solve()
                marker = get(ENV, "BENCH_STATES_MARKER", "")
                isempty(marker) || touch(marker)
                on_breach = () -> println("WATCHDOG lorenz96 " *
                    "states=$(nstates) $(mode) $(algorithm) N=$(n): " *
                    "run never returned")
                t_dev, t, pct = with_gpu_lock() do
                    td, samples_none, dev_sol = watchdogged_min_ms(device_solve,
                        on_breach, REPEATS)
                    isnan(td) && return (td, NaN, 100.0)
                    p = errored_pct(@view dev_sol[2][end, :])
                    tt, samples_both, _ = watchdogged_min_ms(full_solve,
                        on_breach, REPEATS)
                    (td, tt, p)
                end
                isnan(t) &&
                    println("WATCHDOG lorenz96 states=$(nstates) $(mode) " *
                            "$(algorithm) N=$(n): run exceeded the cap")
                (t, t_dev, build, pct)
            catch err
                (failed("lorenz96 states=$(nstates) $(mode) $(algorithm) " *
                        "N=$(n)", err), NaN, NaN, 100.0)
            end
            isinteractive() || result_record_times(STORE, "julia", DATASET_KEY,
                "states", row, algorithm, mode, n, nstates, t_ms, t_dev_ms, pct;
                samples_both = samples_both, samples_none = samples_none,
                build_s = build_s)
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
    m = Float64.(final_states(system, Array(sol[2][end, :])))
    arrived = 0
    for i in eachindex(final_times)
        if isapprox(Float64(final_times[i]), Float64(duration); rtol = 1.0f-4)
            arrived += 1
        else
            m[i, :] .= NaN
        end
    end
    if arrived < length(final_times)
        @warn "$(name): $(length(final_times) - arrived) of " *
              "$(length(final_times)) trajectories stopped before " *
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
