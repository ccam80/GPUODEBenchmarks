#!/usr/bin/env julia
# DiffEqGPU worker for the direct Cubie overlap suite.  Deliberately does not
# instantiate or precompile: environment setup is an explicit prerequisite.

using CUDA
using DiffEqGPU
using SciMLBase: ODEFunction, ODEProblem, remake
using StaticArrays
using CSV
using DelimitedFiles
using Printf
using Statistics
using GPU_ODE_JuliaKernels

CUDA.allowscalar(false)

const HERE = @__DIR__
const REPO_ROOT = dirname(dirname(HERE))
include(joinpath(REPO_ROOT, "runner_scripts", "problems.jl"))
include(joinpath(REPO_ROOT, "runner_scripts", "algorithms.jl"))
include(joinpath(REPO_ROOT, "runner_scripts", "julia_systems.jl"))
include(joinpath(REPO_ROOT, "runner_scripts", "julia_prob.jl"))
include(joinpath(REPO_ROOT, "runner_scripts", "watchdog.jl"))
# Precompiled entries take precedence over runtime-built ones.
merge!(_ENTRIES, GPU_ODE_JuliaKernels.ENTRIES)
# The fixed step is a fraction of the duration.
const FIXED_DT = 2.0^-TIMING_DT_K
const ADAPTIVE_TOL = OVERLAP_TOL
const PERFORMANCE_REPEATS = REPEAT_CAP
const WORK_REPEATS = REPEAT_CAP

function cli_args(args)
    out = Dict{String, String}()
    i = 1
    while i <= length(args)
        startswith(args[i], "--") || error("unexpected argument $(args[i])")
        key = args[i][3:end]
        i < length(args) || error("--$(key) requires a value")
        out[key] = args[i + 1]
        i += 2
    end
    return out
end

const OPT = cli_args(ARGS)
const OUT = abspath(haskey(OPT, "output") ? OPT["output"] : error("--output is required"))
const ANALYSIS = get(OPT, "analysis", "all")
const NMAX = get(OPT, "nmax", string(NMAX_DEFAULT))
const FROM_N = parse(Int, get(OPT, "from-n", "0"))
const ALGORITHM = get(OPT, "algorithm", "all")
const PROBLEM = get_problem(get(OPT, "problem", "lorenz"))
const NSTATES = PROBLEM["states"]
const DURATION = Float32(PROBLEM["duration"])

mkpath(OUT)

function protocol()
    ns = parse_ns(NMAX, FROM_N)
    return (performance_ns = ns, performance_repeats = PERFORMANCE_REPEATS,
        ne_n = N_NE, ne_dts = fixed_dts(1.0, NE_K), ne_tols = TOLS,
        wp_n = N_WP, wp_dts = fixed_dts(1.0, WP_K), wp_tols = TOLS,
        work_repeats = WORK_REPEATS)
end
const PROTOCOL = protocol()

const TIMING_HEADER = "framework,algorithm,phase,mode,tier,transfers,n,setting_kind,setting,samples,min_ms,p05_ms,median_ms,p95_ms,max_ms"
const METRIC_HEADER = "framework,algorithm,phase,mode,tier,n,setting_kind,setting,golden_rmse,finite_trajectories,failed_trajectories,finals_path"
const FAILURE_HEADER = "framework,algorithm,phase,mode,tier,n,setting_kind,setting,error_type,message"
const TIMING_FILE = joinpath(OUT, "julia_timings.csv")
const METRIC_FILE = joinpath(OUT, "julia_metrics.csv")
const FAILURE_FILE = joinpath(OUT, "julia_failures.csv")

# Append-only: the launcher clears the rows a run replaces.
function init_csv(path, header)
    if !isfile(path)
        open(path, "w") do io
            println(io, header)
        end
    end
end
init_csv(TIMING_FILE, TIMING_HEADER)
init_csv(METRIC_FILE, METRIC_HEADER)
init_csv(FAILURE_FILE, FAILURE_HEADER)

"""Reduce one point's repeats to the persisted timing statistics.

`quantile` defaults to linear interpolation, matching `np.percentile`, so
both workers' columns are computed identically.
"""
function timing_stats(values)
    v = collect(Float64, values)
    return (length(v), minimum(v), quantile(v, 0.05), median(v),
        quantile(v, 0.95), maximum(v))
end

clean(value) = replace(replace(string(value), ',' => ';'), '\n' => ' ')
function append_row(path, values...)
    open(path, "a") do io
        println(io, join(clean.(values), ','))
        flush(io)
    end
end

const SYSTEM, PROB, _ = build_prob(PROBLEM)

const golden_ne_all = readdlm(joinpath(REPO_ROOT, "data", "numerical",
    "golden_ne_$(PROBLEM["problem"])_1024.csv"), ',', Float64)
const golden_wp_all = readdlm(golden_path(PROBLEM), ',', Float64)

function sweep_grid(kind, n)
    if kind == "numerical"
        return Float32.(golden_ne_all[1:n, 1])
    elseif kind == "work_precision"
        return Float32.(problem_sweep(PROBLEM, N_WP))[1:n]
    end
    return Float32.(problem_sweep(PROBLEM, n))
end

"Host and device ensembles over one phase's grid, and the shared problem."
function build_problems(kind, n)
    probs_host, probs = build_ensemble(SYSTEM, PROB, sweep_grid(kind, n))
    return probs_host, probs, PROB
end

# The armed point's identity, for the watchdog's failure row.
const WATCHDOG_POINT = Ref(("", "", "", "", 0, "", NaN))

function watchdog_breach()
    alias, phase, mode, tier, n, setting_kind, setting = WATCHDOG_POINT[]
    append_row(FAILURE_FILE, "julia", alias, phase, mode, tier, n,
        setting_kind, setting, "Watchdog",
        "run never returned within $(WATCHDOG_SECONDS) s")
    println("WATCHDOG julia $(alias) $(phase) $(mode) " *
            "$(setting_kind)=$(setting): run never returned")
end

"Time one solve including the h2d and d2h transfers; the reshape is untimed."
function solve_end_to_end(probs_host, prob, alg, mode, setting)
    CUDA.synchronize()
    start = time_ns()
    host_us = run_watchdogged(watchdog_breach) do
        _, _, us = gpu_solve_host(probs_host, prob, alg, mode, setting, PROBLEM)
        us
    end
    elapsed_ms = (time_ns() - start) / 1.0e6
    elapsed_ms > WATCHDOG_SECONDS * 1000.0 &&
        error("watchdog: run exceeded $(WATCHDOG_SECONDS) s")
    finals = final_states(SYSTEM, host_us[end, :])
    size(finals) == (length(probs_host), NSTATES) || error(
        "unexpected final-state size $(size(finals)); expected " *
        "($(length(probs_host)), $(NSTATES))")
    return finals, elapsed_ms
end

"Time one solve with neither transfer: probs already resident, results left there."
function solve_device_only(probs, prob, alg, mode, setting)
    CUDA.synchronize()
    start = time_ns()
    run_watchdogged(watchdog_breach) do
        gpu_solve_device(probs, prob, alg, mode, setting, PROBLEM)
    end
    elapsed_ms = (time_ns() - start) / 1.0e6
    elapsed_ms > WATCHDOG_SECONDS * 1000.0 &&
        error("watchdog: run exceeded $(WATCHDOG_SECONDS) s")
    return elapsed_ms
end

function finite_counts(finals)
    good = [all(isfinite, @view finals[i, :]) for i in axes(finals, 1)]
    return count(identity, good), count(!, good)
end

function golden_rmse(finals, golden)
    good = [all(isfinite, @view finals[i, :]) for i in axes(finals, 1)]
    any(good) || return NaN
    delta = Float64.(finals[good, :]) .- golden[good, :]
    return sqrt(sum(abs2, delta) / length(delta))
end

slug(value) = replace(replace(replace(@sprintf("%.10g", value), "-" => "m"),
    "+" => "p"), "." => "p")
function write_finals(alias, mode, tier, setting_kind, setting, finals)
    relative = joinpath("finals", "julia", alias,
        "$(mode)_$(tier)_$(setting_kind)_$(slug(setting)).csv")
    path = joinpath(OUT, relative)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "traj," * join(["s$(s)" for s in 1:NSTATES], ","))
        for i in axes(finals, 1)
            fields = join([@sprintf("%.9g", finals[i, s]) for s in 1:NSTATES], ",")
            println(io, "$(i - 1),$(fields)")
        end
    end
    return replace(relative, '\\' => '/')
end

function record_failure(alias, phase, mode, tier, n, setting_kind, setting, err)
    POINT_FAILURE_COUNT[] += 1
    append_row(FAILURE_FILE, "julia", alias, phase, mode, tier, n,
        setting_kind, setting, nameof(typeof(err)), sprint(showerror, err)[1:min(end, 2000)])
    println("FAILED julia $(alias) $(phase) $(mode) $(setting_kind)=$(setting): $(sprint(showerror, err))")
end

const POINT_FAILURE_COUNT = Ref(0)

table = overlap_algorithms(ALGORITHM)
isempty(table) && error("'$(ALGORITHM)' is not in the overlap set; see runner_scripts/algorithms.csv")
phases = ANALYSIS == "all" ? ("performance", "numerical", "work_precision") :
    (replace(ANALYSIS, "-" => "_"),)

for row in table
    alias = row["algorithm"]
    alg = try
        gpu_solver(alias)
    catch err
        for phase in phases
            record_failure(alias, phase, "all", "julia", 0, "constructor", NaN, err)
        end
        continue
    end
    for phase in phases
        points = Tuple{String, String, Float64, Int}[]
        repeats = 1
        # Scale dt fractions by the duration; the recorded setting is the absolute dt.
        if phase == "performance"
            for n in PROTOCOL.performance_ns
                push!(points, ("fixed", "dt", DURATION * FIXED_DT, n))
                push!(points, ("adaptive", "tol", ADAPTIVE_TOL, n))
            end
            repeats = PROTOCOL.performance_repeats
        elseif phase == "numerical"
            # erk-family rows run no fixed numerical sweep.
            if runs_fixed_ne(row)
                append!(points, [("fixed", "dt", DURATION * dt, PROTOCOL.ne_n) for dt in PROTOCOL.ne_dts])
            end
            append!(points, [("adaptive", "tol", tol, PROTOCOL.ne_n) for tol in PROTOCOL.ne_tols])
        else
            append!(points, [("fixed", "dt", DURATION * dt, PROTOCOL.wp_n) for dt in PROTOCOL.wp_dts])
            append!(points, [("adaptive", "tol", tol, PROTOCOL.wp_n) for tol in PROTOCOL.wp_tols])
            repeats = PROTOCOL.work_repeats
        end

        # Memory need is linear in N, so once a mode OOMs, larger N cannot run.
        oom_ceiling = Dict{String, Int}()

        for (mode, setting_kind, setting, n) in points
            tier = mode == "fixed" ? "fixed" : "julia"
            if haskey(oom_ceiling, mode) && n >= oom_ceiling[mode]
                println("SKIPPED julia $(alias) $(phase) $(mode) $(setting_kind)=$(setting) " *
                        "N=$(n): at or above the N=$(oom_ceiling[mode]) out-of-memory ceiling")
                continue
            end
            WATCHDOG_POINT[] = (alias, phase, mode, tier, n, setting_kind,
                Float64(setting))
            try
                probs_host, probs, prob = build_problems(phase, n)
                if phase == "numerical"
                    # Accuracy only: one untimed solve.
                    finals, _ = solve_end_to_end(probs_host, prob, alg, mode, setting)
                    finite, failed = finite_counts(finals)
                    finals_path = write_finals(alias, mode, tier, setting_kind,
                        setting, finals)
                    append_row(METRIC_FILE, "julia", alias, phase, mode, tier, n,
                        setting_kind, setting,
                        golden_rmse(finals, golden_ne_all[1:n, 2:(1 + NSTATES)]),
                        finite, failed, finals_path)
                    println("OK julia $(alias) $(phase) $(mode) $(setting_kind)=$(setting) N=$(n)")
                    continue
                end
                # One warmup covers both transfer paths.
                solve_end_to_end(probs_host, prob, alg, mode, setting)
                finals = Matrix{Float32}(undef, n, NSTATES)
                # Unbroken block per transfer variant; repeats follow the first timed run's duration.
                end_to_end = Float64[]
                lo = hi = 0
                while true
                    finals, elapsed = solve_end_to_end(probs_host, prob, alg, mode, setting)
                    finite, failed = finite_counts(finals)
                    if failed > 0 || finite != n
                        append_row(METRIC_FILE, "julia", alias, phase, mode, tier,
                            n, setting_kind, setting, "", finite, failed, "")
                        error("non-finite result: $(finite)/$(n) trajectories valid")
                    end
                    push!(end_to_end, elapsed)
                    length(end_to_end) == 1 &&
                        ((lo, hi) = repeat_bounds(end_to_end[1] / 1000.0, repeats))
                    repeats_done(end_to_end, lo, hi) && break
                end
                device_only = Float64[]
                lo = hi = 0
                while true
                    push!(device_only, solve_device_only(probs, prob, alg, mode, setting))
                    length(device_only) == 1 &&
                        ((lo, hi) = repeat_bounds(device_only[1] / 1000.0, repeats))
                    repeats_done(device_only, lo, hi) && break
                end
                for (transfers, samples) in (("both", end_to_end), ("none", device_only))
                    append_row(TIMING_FILE, "julia", alias, phase, mode, tier,
                        transfers, n, setting_kind, setting, timing_stats(samples)...)
                end
                finite, failed = finite_counts(finals)
                if phase == "performance"
                    append_row(METRIC_FILE, "julia", alias, phase, mode, tier,
                        n, setting_kind, setting, "", finite, failed, "")
                else
                    append_row(METRIC_FILE, "julia", alias, phase, mode, tier, n,
                        setting_kind, setting,
                        golden_rmse(finals, golden_wp_all[1:n, :]),
                        finite, failed, "")
                end
                println("OK julia $(alias) $(phase) $(mode) $(setting_kind)=$(setting) N=$(n)")
            catch err
                record_failure(alias, phase, mode, tier, n, setting_kind, setting, err)
                if phase == "performance" && err isa CUDA.OutOfGPUMemoryError
                    oom_ceiling[mode] = n
                end
            end
        end
    end
end

open(joinpath(OUT, "julia_metadata.json"), "w") do io
    println(io, "{")
    println(io, "  \"framework\": \"DiffEqGPU\",")
    println(io, "  \"julia_version\": \"$(VERSION)\",")
    println(io, "  \"diffeqgpu_version\": \"$(pkgversion(DiffEqGPU))\",")
    println(io, "  \"cuda_runtime\": \"$(CUDA.runtime_version())\"")
    println(io, "}")
end

exit(POINT_FAILURE_COUNT[] == 0 ? 0 : 1)
