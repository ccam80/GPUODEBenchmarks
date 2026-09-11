# The julia_gpu runner: `bench_ode_gpu.jl --trials <path> [--floor] [--store-python <exe>]` runs one build's trials on the DiffEqGPU kernel path, recording each through results.jl; exit 0 when done, the watchdog code when a run never returned.

using CUDA
using DiffEqGPU
using SciMLBase
using StaticArrays
using Dates
using JSON
using TOML
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
    "atol", "rtol", "newton_atol", "newton_rtol", "watchdog_s")
const CONTROLLERS = ("fixed", "default")

# ------------------------------------------------------------------ inputs

function parse_cli(args)
    trials = ""
    floor = false
    i = 1
    while i <= length(args)
        tok = args[i]
        if tok == "--trials"
            i += 1
            i <= length(args) || error("--trials requires a path")
            trials = args[i]
        elseif tok == "--floor"
            floor = true
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
    return (trials = trials, floor = floor)
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

function write_progress(path, trial, stage)
    stamp = Dates.format(Dates.now(Dates.UTC), "yyyy-mm-ddTHH:MM:SS") * "Z"
    open(path, "w") do io
        JSON.print(io, Dict("trial_id" => trial["trial_id"], "stage" => stage,
            "started_utc" => stamp))
    end
end

"(n, states, -dt, -tol) of a trial: each entry grows with the cost of the run."
function difficulty(trial)
    params = JSON.parse(trial["system_params"])
    states = haskey(params, "states") ? Int(params["states"]) : 0
    finite(x) = isnan(x) ? 0.0 : x
    return (Int(trial["n"]), states, -finite(trial["dt"]), -finite(trial["atol"]))
end

"True when a is at least as hard as b in every difficulty entry and harder in one."
function harder(a, b)
    da, db = difficulty(a), difficulty(b)
    return all(x >= y for (x, y) in zip(da, db)) && da != db
end

"The fields the abandon rule compares within."
family(trial) = (trial["problem"], trial["precision"], trial["algorithm"], trial["controller"], trial["gains"])

"The abandon rule: 'abandoned: <outcome> at <trial_id>' when a timeout or oom of the trial's family on these transfers is no harder than it, else nothing; `failures` maps transfers to [(trial, outcome)]."
function abandon_reason(trial, transfers, failures)
    for (failed, outcome) in get(failures, transfers, Tuple{Dict{String, Any}, String}[])
        family(failed) == family(trial) && harder(trial, failed) &&
            return "abandoned: $(outcome) at $(failed["trial_id"])"
    end
    return nothing
end

function note_failure!(failures, trial, transfers, outcome)
    outcome in ("timeout", "oom") || return
    push!(get!(() -> Tuple{Dict{String, Any}, String}[], failures, transfers), (trial, outcome))
    return
end

"'<problem> <algorithm> <controller> n=<n> dt=<dt>|tol=<tol> [<system_params>]'."
function trial_label(trial)
    text = "$(trial["problem"]) $(trial["algorithm"]) $(trial["controller"]) n=$(Int(trial["n"]))"
    text *= trial["controller"] == "fixed" ? " dt=$(trial["dt"])" : " tol=$(trial["atol"])"
    trial["system_params"] in ("", "{}") || (text *= " " * trial["system_params"])
    return text
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

"The Float32 system of a trial: the compiled entry when it matches the construction parameters, else built now; a cold line always builds now."
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

"(system, prob, solver) of a build from one of its trials."
function build_parts(trial, cold)
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

"One untimed warm-up and the repeats schedule of protocol.toml over f, under the watchdog with the trial's cap; a run past the cap is a timeout, an exception an oom or error; an untimed line runs once with no warm-up."
function timed(f, label, cap_s, is_timed = true)
    on_breach = () -> begin
        println("WATCHDOG $(label): run never returned")
        flush(stdout)
    end
    try
        if !is_timed
            elapsed = @elapsed result = run_watchdogged(f, on_breach; budget_s = cap_s + 30.0)
            ms = elapsed * 1000.0
            elapsed > cap_s || return Outcome("ok", ms, [ms], "", result)
            return Outcome("timeout", NaN, [ms], "timeout: run exceeded $(cap_s)s", result)
        end
        ms, samples, result = watchdogged_min_ms(f, on_breach, REPEAT_CAP; cap_s)
        isnan(ms) && return Outcome("timeout", NaN, samples,
            "timeout: run exceeded $(cap_s)s", result)
        return Outcome("ok", ms, samples, "", result)
    catch err
        return failed(classify(err)...)
    end
end

"One solve at n = 8 under the watchdog's hard exit; the kernel compile lands here."
function warm_build(parts, trial, label)
    values = grid_values(trial["grid_scale"], trial["grid_min"], trial["grid_max"], WARM_N)
    probs_host, probs = build_ensemble(parts.system, parts.prob, values)
    on_breach = () -> begin
        println("WATCHDOG $(label): warm run never returned")
        flush(stdout)
    end
    run_watchdogged(on_breach; budget_s = trial["watchdog_s"] + 30.0) do
        gpu_solve_device(probs, parts.prob, parts.solver, trial["controller"], trial["dt"],
            trial["atol"], trial["rtol"])
    end
    return nothing
end

"Time each listed transfers of a trial, then record every transfers row (and the finals when kept) in one batch; `failure` is the build's failure reason when it did not build."
function run_solve(trial, parts, failure, failures, build_s, cli, version, rev, key)
    n = Int(trial["n"])
    duration = trial["duration"]
    label = trial_label(trial)
    rejection = reject_reason(trial)
    outcomes = Dict{String, Outcome}()
    ensemble = nothing
    ensemble_failure = nothing
    result = nothing
    for transfers in trial["transfers"]
        abandoned = abandon_reason(trial, transfers, failures)
        outcome = if abandoned !== nothing
            failed("abandoned", abandoned)
        elseif rejection !== nothing
            failed("error", rejection)
        elseif parts === nothing
            failed("error", isempty(failure) ?
                "error: the system, problem or solver could not be built" : failure)
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
                timed(solve, "$(label) $(transfers)", trial["watchdog_s"], get(trial, "timed", true))
            end
        end
        note_failure!(failures, trial, transfers, outcome.kind)
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
                build_s = build_s, reason = outcomes[transfers].reason, finals = finals,
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

"Every trial of the file in order: the first line builds (a cold line rebuilds and times the build), every line with transfers solves; optimize does not apply to the kernel path."
function run_trials(trials, cli, version, rev, key, progress_path)
    parts = nothing
    built = false
    failure = ""
    failures = Dict{String, Vector{Tuple{Dict{String, Any}, String}}}()
    for trial in trials
        build_s = NaN
        if !built || trial["cold"] == true
            write_progress(progress_path, trial, "build")
            cold = trial["cold"] == true
            elapsed = @elapsed begin
                rejection = reject_reason(trial)
                if rejection !== nothing
                    println("$(trial_label(trial)): $(rejection)")
                else
                    try
                        parts = build_parts(trial, cold)
                        cold && warm_build(parts, trial, trial_label(trial))
                        failure = ""
                    catch err
                        parts = nothing
                        failure = classify(err)[2]
                        println("$(trial_label(trial)): build failed: $(failure)")
                    end
                end
            end
            built = true
            cold && (build_s = elapsed)
            cold && println("$(trial_label(trial)): built cold in $(elapsed)s")
        end
        trial["optimize"] === nothing ||
            println("$(trial_label(trial)): optimize does not apply to the DiffEqGPU kernel path; skipped")
        isempty(trial["transfers"]) && continue
        write_progress(progress_path, trial, "solve")
        run_solve(trial, parts, failure, failures, build_s, cli, version, rev, key)
    end
    return nothing
end

function main(args)
    cli = parse_cli(args)
    trials = read_trials(cli.trials)
    version = diffeqgpu_version()
    rev = store_suite_rev(REPO_ROOT)
    key = dataset_key()
    run_trials(trials, cli, version, rev, key, cli.trials * ".progress")
    return 0
end

exit(main(ARGS))
