# The julia_gpu runner: `bench_ode_gpu.jl --trials <path> [--floor] [--gpu-lock <pidfile>] [--store-python <exe>]` runs one leg's trials on the DiffEqGPU kernel path, recording each through results.jl; exit 0 when done, the watchdog code when a run never returned.

using CUDA
using DiffEqGPU
using SciMLBase
using StaticArrays
using Dates
using JSON
using TOML
using FileWatching.Pidfile: mkpidlock
using GPU_ODE_JuliaKernels

const REPO_ROOT = dirname(@__DIR__)
const RUNNER_SCRIPTS = joinpath(REPO_ROOT, "runner_scripts")
include(joinpath(RUNNER_SCRIPTS, "bench_key.jl"))
include(joinpath(RUNNER_SCRIPTS, "problems.jl"))
include(joinpath(RUNNER_SCRIPTS, "algorithms.jl"))
include(joinpath(RUNNER_SCRIPTS, "julia_systems.jl"))
include(joinpath(RUNNER_SCRIPTS, "julia_prob.jl"))
include(joinpath(RUNNER_SCRIPTS, "grid.jl"))
include(joinpath(RUNNER_SCRIPTS, "watchdog.jl"))
include(joinpath(RUNNER_SCRIPTS, "errored.jl"))
include(joinpath(RUNNER_SCRIPTS, "results.jl"))
# Precompiled entries take precedence over runtime-built ones.
merge!(_ENTRIES, GPU_ODE_JuliaKernels.ENTRIES)

CUDA.allowscalar(false)

const WARM_N = 8
const TRIAL_FLOAT_FIELDS = ("duration", "grid_min", "grid_max", "dt", "dt_min", "dt_max",
    "atol", "rtol", "newton_atol", "newton_rtol")
const CONTROLLERS = ("fixed", "default")

# ------------------------------------------------------------------ inputs

function parse_cli(args)
    trials = ""
    floor = false
    lock = ""
    i = 1
    while i <= length(args)
        tok = args[i]
        if tok == "--trials"
            i += 1
            i <= length(args) || error("--trials requires a path")
            trials = args[i]
        elseif tok == "--floor"
            floor = true
        elseif tok == "--gpu-lock"
            i += 1
            i <= length(args) || error("--gpu-lock requires a path")
            lock = args[i]
        elseif tok == "--store-python"
            i += 1
            i <= length(args) || error("--store-python requires a path")
            STORE_PYTHON[] = args[i]
        else
            error("unknown argument '$(tok)'")
        end
        i += 1
    end
    isempty(trials) && error("--trials <path> is required")
    return (trials = trials, floor = floor, lock = lock)
end

"The trial records of a JSONL file in file order; null floats come back as NaN."
function read_trials(path)
    trials = Dict{String, Any}[]
    for line in eachline(path)
        isempty(strip(line)) && continue
        record = Dict{String, Any}(JSON.parse(line))
        for field in TRIAL_FLOAT_FIELDS
            record[field] = record[field] === nothing ? NaN : Float64(record[field])
        end
        push!(trials, record)
    end
    return trials
end

"Trials grouped by leg, legs in first appearance."
function by_leg(trials)
    legs = Dict{String, Vector{Dict{String, Any}}}()
    order = String[]
    for trial in trials
        haskey(legs, trial["leg"]) || push!(order, trial["leg"])
        push!(get!(() -> Dict{String, Any}[], legs, trial["leg"]), trial)
    end
    return [(leg, legs[leg]) for leg in order]
end

function write_progress(path, trial)
    stamp = Dates.format(Dates.now(Dates.UTC), "yyyy-mm-ddTHH:MM:SS") * "Z"
    open(path, "w") do io
        JSON.print(io, Dict("trial_id" => trial["trial_id"], "kind" => trial["kind"],
            "started_utc" => stamp))
    end
end

"The DiffEqGPU version pinned by Manifest.toml."
function diffeqgpu_version()
    manifest = TOML.parsefile(joinpath(REPO_ROOT, "Manifest.toml"))
    return String(manifest["deps"]["DiffEqGPU"][1]["version"])
end

"The reason a trial cannot run on the kernel path, or nothing: a controller other than fixed or default, a precision other than float32, a dt floor, a dt cap or explicit gains."
function reject_reason(trial)
    controller = trial["controller"]
    controller in CONTROLLERS || return "error: unknown controller $(controller)"
    trial["precision"] == "float32" ||
        return "error: unsupported precision $(trial["precision"])"
    isnan(trial["dt_min"]) || return "error: unsupported dt_min: the kernel path has no dt floor to set"
    isnan(trial["dt_max"]) || return "error: unsupported dt_max: the kernel path has no dt cap to set"
    trial["gains"] in ("", "{}") || return "error: unsupported gains $(trial["gains"])"
    return nothing
end

# ----------------------------------------------------------------- systems

"The state count of a trial's system from its construction parameters, else the catalogue's."
function trial_states(trial)
    params = JSON.parse(trial["system_params"])
    haskey(params, "states") && return Int(params["states"])
    return Int(get_problem(trial["problem"])["states"])
end

"The Float32 system of a trial: the compiled entry when it matches the construction parameters, else built now; a cold leg always builds now."
function trial_system(trial, cold)
    problem = trial["problem"]
    params = JSON.parse(trial["system_params"])
    states = haskey(params, "states") ? Int(params["states"]) : 0
    haskey(_ENTRY_BUILDERS, problem) ||
        error("no ModelingToolkit definition for problem '$(problem)'")
    if !cold && haskey(_ENTRIES, (problem, Float32)) &&
       (states == 0 || _ENTRIES[(problem, Float32)].n == states)
        return _ENTRIES[(problem, Float32)]
    end
    if states > 0 && startswith(problem, "lorenz96")
        return _lorenz96_entry(Float32, states)
    end
    system = cold ? _ENTRY_BUILDERS[problem](Float32) : julia_system(problem, Float32)
    (states == 0 || system.n == states) ||
        error("$(problem) has $(system.n) states, not $(states)")
    return system
end

"(system, prob, solver) of a leg from one of its trials."
function leg_parts(trial, cold)
    system = trial_system(trial, cold)
    return (system = system, prob = build_prob(system, trial["duration"]),
        solver = gpu_solver(trial["algorithm"]))
end

# ------------------------------------------------------------------ solves

struct Outcome
    kind::String          # ok, timeout, oom, error, abandoned
    min_ms::Float64
    samples::Vector{Float64}
    reason::String
    result::Any           # the last solve's (ts, us), or nothing
end

failed(kind, reason) = Outcome(kind, NaN, Float64[], reason, nothing)

"oom or error, with the reason text: '<kind>: <Type>: <message[:200]>'."
function classify(err)
    text = sprint(showerror, err)
    oom = err isa CUDA.OutOfGPUMemoryError || occursin("OUT_OF_MEMORY", text) ||
          occursin("out of memory", lowercase(text))
    kind = oom ? "oom" : "error"
    message = replace(first(text, 200), r"\s+" => " ")
    return kind, "$(kind): $(nameof(typeof(err))): $(message)"
end

function with_gpu_lock(f, path)
    isempty(path) && return f()
    gpu_lock = mkpidlock(path; wait = true, stale_age = 120)
    try
        return f()
    finally
        close(gpu_lock)
    end
end

"One untimed warm-up and the repeats schedule of protocol.toml over f, under the watchdog; a run past the soft cap is a timeout, an exception an oom or error."
function timed(f, label)
    on_breach = () -> begin
        println("WATCHDOG $(label): run never returned")
        flush(stdout)
    end
    try
        ms, samples, result = watchdogged_min_ms(f, on_breach, REPEAT_CAP)
        isnan(ms) && return Outcome("timeout", NaN, samples,
            "timeout: run exceeded $(WATCHDOG_SECONDS)s", result)
        return Outcome("ok", ms, samples, "", result)
    catch err
        return failed(classify(err)...)
    end
end

"Per leg: the transfers already abandoned (with the reason), the cold build time and the build failure's reason when it did not build."
mutable struct LegState
    abandoned::Dict{String, String}
    build_s::Float64
    failure::String
end

"The abandon rule: after a timeout or oom at ordinal k, every higher ordinal of the leg with the same transfers is recorded abandoned instead of run."
function abandon!(state, transfers, outcome, ordinal)
    outcome.kind in ("timeout", "oom") || return
    haskey(state.abandoned, transfers) && return
    state.abandoned[transfers] = "abandoned: $(outcome.kind) at ordinal $(ordinal)"
    return
end

"One solve at n = 8 in the leg's process, off the GPU lock; the kernel compile lands here."
function warm_leg(parts, trial, label)
    values = grid_values(trial["grid_scale"], trial["grid_min"], trial["grid_max"], WARM_N)
    probs_host, probs = build_ensemble(parts.system, parts.prob, values)
    on_breach = () -> begin
        println("WATCHDOG $(label): warm run never returned")
        flush(stdout)
    end
    run_watchdogged(on_breach) do
        gpu_solve_device(probs, parts.prob, parts.solver, trial["controller"], trial["dt"],
            trial["atol"], trial["rtol"])
    end
    return nothing
end

"Time each listed transfers leg of a solve trial, then record every transfers row (and the finals when kept) in one batch."
function run_solve(trial, parts, state, cli, version, rev, key)
    n = Int(trial["n"])
    duration = trial["duration"]
    label = "$(trial["leg"]) ordinal $(trial["ordinal"]) n=$(n)"
    rejection = reject_reason(trial)
    outcomes = Dict{String, Outcome}()
    ensemble = nothing
    ensemble_failure = nothing
    result = nothing
    for transfers in trial["transfers"]
        outcome = if haskey(state.abandoned, transfers)
            failed("abandoned", state.abandoned[transfers])
        elseif rejection !== nothing
            failed("error", rejection)
        elseif parts === nothing
            failed("error", isempty(state.failure) ?
                "error: the leg's system, problem or solver could not be built" : state.failure)
        else
            if ensemble === nothing && ensemble_failure === nothing
                try
                    values = grid_values(trial["grid_scale"], trial["grid_min"],
                        trial["grid_max"], n)
                    ensemble = build_ensemble(parts.system, parts.prob, values)
                catch err
                    ensemble_failure = failed(classify(err)...)
                end
            end
            if ensemble_failure !== nothing
                ensemble_failure
            else
                probs_host, probs = ensemble
                solve = transfers == "both" ?
                    () -> gpu_solve_host(probs_host, parts.prob, parts.solver,
                        trial["controller"], trial["dt"], trial["atol"], trial["rtol"]) :
                    () -> gpu_solve_device(probs, parts.prob, parts.solver,
                        trial["controller"], trial["dt"], trial["atol"], trial["rtol"])
                with_gpu_lock(cli.lock) do
                    timed(solve, "$(label) $(transfers)")
                end
            end
        end
        abandon!(state, transfers, outcome, Int(trial["ordinal"]))
        outcome.result === nothing || (result = outcome.result)
        outcomes[transfers] = outcome
        println("$(label) $(transfers): $(outcome.min_ms) ms $(outcome.reason)")
        flush(stdout)
    end
    pct = NaN
    finals = ""
    if result !== nothing
        # The kernel path reports no retcode; a failed trajectory shows in its state or final time.
        m = final_states(parts.system, Array(result[2][end, :]))
        t_final = Float64.(Array(result[1][end, :]))
        retcode = fill("", size(m, 1))
        pct = errored_pct(m, t_final, retcode, duration)
        if trial["finals"]
            finals = store_finals(merge(trial, Dict("key" => key)), m, t_final; retcode)
        end
    end
    states = parts === nothing ? trial_states(trial) : parts.system.n
    rows = [store_row(merge(trial, Dict("transfers" => transfers, "key" => key));
                states = states, min_ms = outcomes[transfers].min_ms,
                samples_ms = outcomes[transfers].samples, errored_pct = pct,
                build_s = state.build_s, reason = outcomes[transfers].reason, finals = finals,
                package_version = version, suite_rev = rev)
            for transfers in trial["transfers"]]
    store_record(rows; floor = cli.floor)
    # Ensembles are per-trial; only the compiled kernels carry over.
    ensemble = nothing
    result = nothing
    outcomes = nothing
    GC.gc()
    CUDA.reclaim()
    return nothing
end

"A leg: its warm line (cold builds the system now and times the build), then every solve in ordinal order; optimize lines do not apply to the kernel path."
function run_leg(leg, trials, cli, version, rev, key, progress_path)
    warm = findfirst(t -> t["kind"] == "warm", trials)
    cold = warm !== nothing && trials[warm]["cold"] == true
    state = LegState(Dict{String, String}(), NaN, "")
    parts = nothing
    built = false
    for trial in trials
        write_progress(progress_path, trial)
        kind = trial["kind"]
        if kind == "optimize"
            println("$(leg): optimize lines do not apply to the DiffEqGPU kernel path; skipped")
            continue
        end
        if !built
            built = true
            elapsed = @elapsed begin
                rejection = reject_reason(trial)
                if rejection !== nothing
                    println("$(leg): $(rejection)")
                else
                    try
                        parts = leg_parts(trial, cold)
                        kind == "warm" && warm_leg(parts, trial, leg)
                    catch err
                        state.failure = classify(err)[2]
                        println("$(leg): build failed: $(state.failure)")
                    end
                end
            end
            cold && (state.build_s = elapsed)
            kind == "warm" && println("$(leg): warm at n=$(WARM_N) in $(elapsed)s")
        end
        kind == "solve" || continue
        run_solve(trial, parts, state, cli, version, rev, key)
    end
    return nothing
end

function main(args)
    cli = parse_cli(args)
    trials = read_trials(cli.trials)
    version = diffeqgpu_version()
    rev = store_suite_rev(REPO_ROOT)
    key = dataset_key()
    progress_path = cli.trials * ".progress"
    for (leg, leg_trials) in by_leg(trials)
        run_leg(leg, leg_trials, cli, version, rev, key, progress_path)
    end
    return 0
end

exit(main(ARGS))
