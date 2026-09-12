# The julia_cpu runner: DifferentialEquations.jl on EnsembleThreads over a trial file, one system and solver per build (consecutive lines of one system, algorithm, controller and precision) in the trial's element type; every transfers entry times the same call, finals land when asked, and each adaptive build's resolved controller goes to controllers/<problem>.csv under the run key.
#   julia -t auto --project=. GPU_ODE_Julia/bench_ode_cpu.jl --trials <path> [--floor]

using OrdinaryDiffEq
using OrdinaryDiffEqLowOrderRK, OrdinaryDiffEqHighOrderRK, OrdinaryDiffEqExplicitRK
using OrdinaryDiffEqVerner, OrdinaryDiffEqSDIRK, OrdinaryDiffEqFIRK
using OrdinaryDiffEqRosenbrock, OrdinaryDiffEqBDF
import OrdinaryDiffEqCore
# The slim OrdinaryDiffEq v7 umbrella doesn't re-export the ensemble API.
using SciMLBase: EnsembleProblem, EnsembleThreads, ReturnCode, remake, init
using Dates
using JSON
using Printf
using TOML

const REPO_ROOT = dirname(@__DIR__)
const SCRIPTS = joinpath(REPO_ROOT, "runner_scripts")
include(joinpath(SCRIPTS, "bench_key.jl"))
include(joinpath(SCRIPTS, "problems.jl"))
include(joinpath(SCRIPTS, "algorithms.jl"))
include(joinpath(SCRIPTS, "julia_tableaus.jl"))
include(joinpath(SCRIPTS, "grid.jl"))
include(joinpath(SCRIPTS, "julia_systems.jl"))
include(joinpath(SCRIPTS, "protocol.jl"))
include(joinpath(SCRIPTS, "watchdog.jl"))
include(joinpath(SCRIPTS, "errored.jl"))
include(joinpath(SCRIPTS, "results.jl"))

const PACKAGE = "julia_cpu"
const WARM_N = 8
const CONTROLLERS = ("fixed", "default")
const CONTROLLER_COLUMNS = ("algorithm", "controller", "beta1", "beta2", "qmin", "qmax",
    "gamma", "order")
const FLOAT_FIELDS = ("duration", "grid_min", "grid_max", "dt", "dt_min", "dt_max", "atol",
    "rtol", "newton_atol", "newton_rtol", "watchdog_s")
# The solvers' own warnings (dt below eps, instability) are per trajectory; the row's retcodes carry them.
const QUIET = OrdinaryDiffEqCore.DEVerbosity(OrdinaryDiffEqCore.SciMLLogging.None())

# ------------------------------------------------------------------ inputs

function parse_cli(args)
    path = ""
    floor = false
    i = 1
    while i <= length(args)
        if args[i] == "--trials"
            i < length(args) || error("--trials requires a path")
            path = args[i + 1]
            i += 2
        elseif args[i] == "--floor"
            floor = true
            i += 1
        else
            error("unexpected argument $(args[i])")
        end
    end
    isempty(path) && error("usage: bench_ode_cpu.jl --trials <path> [--floor]")
    return path, floor
end

"The trial records of a JSONL file; null floats come back as NaN."
function read_trials(path)
    trials = Dict{String, Any}[]
    for line in eachline(path)
        isempty(strip(line)) && continue
        record = JSON.parse(line)
        for field in FLOAT_FIELDS
            value = get(record, field, nothing)
            record[field] = value === nothing ? NaN : Float64(value)
        end
        push!(trials, record)
    end
    return trials
end

build_key(trial) = (trial["problem"], trial["system_params"], trial["precision"], trial["algorithm"],
    trial["controller"], trial["gains"])

"Builds in file order: each run of consecutive lines with one build key."
function builds_of(trials)
    builds = Pair{Any, Vector{Dict{String, Any}}}[]
    for trial in trials
        key = build_key(trial)
        if isempty(builds) || builds[end].first != key
            push!(builds, key => Dict{String, Any}[])
        end
        push!(builds[end].second, trial)
    end
    return builds
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


element_type(precision) = precision == "float64" ? Float64 :
                          precision == "float32" ? Float32 :
                          error("precision '$(precision)' is not float32 or float64")

"The states entry of a trial's system_params, or nothing."
function requested_states(trial)
    params = JSON.parse(trial["system_params"])
    return haskey(params, "states") ? Int(params["states"]) : nothing
end

"OrdinaryDiffEq's version pinned by Manifest.toml."
function package_version()
    manifest = TOML.parsefile(joinpath(REPO_ROOT, "Manifest.toml"))
    return String(manifest["deps"]["OrdinaryDiffEq"][1]["version"])
end

# --------------------------------------------------------------- the solve

const _SIZED_ENTRIES = Dict{Tuple{String, DataType, Int}, Any}()

"The compiled system of a trial in T; a lorenz96 size other than the catalogue's builds its own entry."
function build_system(trial, ::Type{T}) where {T}
    system = julia_system(trial["problem"], T)
    states = requested_states(trial)
    (states === nothing || states == length(system.golden_index)) && return system
    startswith(trial["problem"], "lorenz96") ||
        error("system_params states=$(states) does not fit $(trial["problem"])")
    return get!(() -> _lorenz96_entry(T, states), _SIZED_ENTRIES, (trial["problem"], T, states))
end

"The reason a trial cannot run on this package, or empty."
function rejection(trial, alg)
    controller = trial["controller"]
    controller in CONTROLLERS || return "error: unknown controller $(controller)"
    if controller == "fixed"
        isnan(trial["dt"]) && return "error: a fixed step needs dt"
    else
        OrdinaryDiffEqCore.isadaptive(alg) ||
            return "error: $(trial["algorithm"]) has no adaptive controller in OrdinaryDiffEq"
        # OrdinaryDiffEq terminates its Newton iterations against the step tolerances.
        for (newton, tol) in (("newton_atol", "atol"), ("newton_rtol", "rtol"))
            isnan(trial[newton]) || trial[newton] == trial[tol] ||
                return "error: OrdinaryDiffEq takes the Newton tolerance from $(tol); $(newton) must equal it"
        end
    end
    return ""
end

"Solve keywords of a trial in T: the fixed step or the tolerances with dt0, each pin only when set, the end state only, no verbosity."
function solve_kwargs(trial, ::Type{T}) where {T}
    kw = Dict{Symbol, Any}(:save_everystep => false, :save_start => false, :dense => false,
        :maxiters => 10^8, :verbose => QUIET)
    if trial["controller"] == "fixed"
        kw[:adaptive] = false
        kw[:dt] = T(trial["dt"])
        # With adaptive = false the tolerances only scale the Newton termination of the implicit solvers.
        isnan(trial["newton_atol"]) || (kw[:abstol] = T(trial["newton_atol"]))
        isnan(trial["newton_rtol"]) || (kw[:reltol] = T(trial["newton_rtol"]))
    else
        kw[:abstol] = T(trial["atol"])
        kw[:reltol] = T(trial["rtol"])
        isnan(trial["dt"]) || (kw[:dt] = T(trial["dt"]))
    end
    isnan(trial["dt_min"]) || (kw[:dtmin] = T(trial["dt_min"]))
    isnan(trial["dt_max"]) || (kw[:dtmax] = T(trial["dt_max"]))
    return kw
end

"One ensemble solve over `points`; returns (finals n x k in T, t_final, retcode text per trajectory, empty on Success)."
function ensemble_solve(system, prob, alg, points, kwargs)
    T = eltype(prob.u0)
    index = system.golden_index
    nan_state = fill(T(NaN), length(index))
    # SciMLBase's ensemble API passes an EnsembleContext (with .sim_id) as the second argument.
    eprob = EnsembleProblem(prob;
        prob_func = (pr, ctx) -> remake(pr,
            u0 = Vector{T}(system.u0_for(points[ctx.sim_id])),
            p = T[points[ctx.sim_id]]),
        output_func = (sol, ctx) -> ((isempty(sol.u) ? nan_state : sol.u[end][index],
                                      isempty(sol.t) ? NaN : Float64(sol.t[end]),
                                      sol.retcode), false),
        safetycopy = false)
    sim = solve(eprob, alg, EnsembleThreads(); trajectories = length(points), kwargs...)
    n = length(points)
    finals = Matrix{T}(undef, n, length(index))
    t_final = Vector{Float64}(undef, n)
    retcode = Vector{String}(undef, n)
    for i in 1:n
        u, t_end, code = sim.u[i]
        eltype(u) === T || error("element type discipline violated: trajectory $(i) returned $(eltype(u))")
        finals[i, :] .= u
        t_final[i] = t_end
        retcode[i] = code == ReturnCode.Success ? "" : string(code)
    end
    return finals, t_final, retcode
end

# ---------------------------------------------------------------- outcomes

function root_error(err)
    err isa TaskFailedException && return root_error(err.task.exception)
    err isa CompositeException && !isempty(err.exceptions) && return root_error(first(err.exceptions))
    return err
end

is_oom(err) = root_error(err) isa OutOfMemoryError

error_reason(err) = (e = root_error(err);
    "error: $(typeof(e).name.name): " * first(sprint(showerror, e), 200))

"(min_ms, samples, result, outcome, reason) of timing f under the watchdog with the trial's cap: ok, timeout (the cap passed but the run returned), oom or error; an untimed line runs once with no warm-up."
function time_trial(f, label, cap_s, is_timed = true)
    on_breach = () -> println("WATCHDOG $(label): run never returned")
    try
        if !is_timed
            elapsed = @elapsed result = run_watchdogged(f, on_breach; budget_s = cap_s + 30.0)
            ms = elapsed * 1000.0
            elapsed > cap_s || return ms, [ms], result, "ok", ""
            return NaN, [ms], result, "timeout",
                @sprintf("timeout: %.1f s over the %g s cap", elapsed, cap_s)
        end
        ms, samples, result = watchdogged_min_ms(f, on_breach, REPEAT_CAP; cap_s)
        isnan(ms) || return ms, samples, result, "ok", ""
        reason = @sprintf("timeout: %.1f s over the %g s cap", samples[end] / 1000, cap_s)
        return ms, samples, result, "timeout", reason
    catch err
        is_oom(err) && return NaN, Float64[], nothing, "oom", "oom: " * error_reason(err)[8:end]
        return NaN, Float64[], nothing, "error", error_reason(err)
    end
end

# -------------------------------------------------------------------- store

"One store row of a trial under a transfers value; the run key and version stamps come from ctx."
function trial_row(ctx, trial, transfers; states, min_ms = NaN, samples_ms = Float64[],
        errored_pct = NaN, build_s = NaN, reason = "", finals = "")
    spec = merge(trial, Dict{String, Any}("transfers" => transfers, "key" => ctx.key))
    return store_row(spec; states, min_ms, samples_ms, errored_pct, build_s, reason, finals,
        package_version = ctx.version, suite_rev = ctx.suite_rev)
end

function write_progress(ctx, trial, stage = "solve")
    stamp = Dates.format(Dates.now(Dates.UTC), "yyyy-mm-ddTHH:MM:SS.sss") * "Z"
    write(ctx.progress, JSON.json(Dict("trial_id" => trial["trial_id"], "stage" => stage,
        "started_utc" => stamp)))
end

controllers_path(ctx, problem) = joinpath(ctx.root, "key=" * ctx.key, "package=" * PACKAGE,
    "controllers", problem * ".csv")

"Resolve the step controller OrdinaryDiffEq picks for an adaptive build and merge its row into controllers/<problem>.csv under the key."
function export_controller(ctx, trial, prob, alg, ::Type{T}) where {T}
    kw = solve_kwargs(trial, T)
    integ = init(prob, alg; abstol = kw[:abstol], reltol = kw[:reltol],
        (haskey(kw, :dt) ? (dt = kw[:dt],) : (;))..., save_everystep = false, verbose = QUIET)
    ctrl = integ.controller_cache.controller
    basic = hasproperty(ctrl, :basic) ? ctrl.basic : ctrl
    field(obj, name) = hasproperty(obj, name) ? string(getproperty(obj, name)) : ""
    row = Dict("algorithm" => trial["algorithm"], "controller" => string(typeof(ctrl).name.name),
        "beta1" => field(ctrl, :beta1), "beta2" => field(ctrl, :beta2),
        "qmin" => field(basic, :qmin), "qmax" => field(basic, :qmax),
        "gamma" => field(basic, :gamma), "order" => string(OrdinaryDiffEqCore.alg_order(alg)))
    path = controllers_path(ctx, trial["problem"])
    # Other algorithms' rows stay; an algorithm or cubie_alias column names them.
    kept = Dict{String, String}[]
    if isfile(path)
        lines = filter(!isempty, strip.(readlines(path)))
        header = String.(split(lines[1], ','))
        for line in lines[2:end]
            fields = String.(split(line, ','))
            while length(fields) < length(header)
                push!(fields, "")
            end
            existing = Dict{String, String}(zip(header, fields))
            existing["algorithm"] = get(existing, "algorithm", get(existing, "cubie_alias", ""))
            existing["algorithm"] == row["algorithm"] || push!(kept, existing)
        end
    end
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join(CONTROLLER_COLUMNS, ","))
        for entry in vcat(kept, [row])
            println(io, join([get(entry, c, "") for c in CONTROLLER_COLUMNS], ","))
        end
    end
    println("  controller: $(row["controller"]) beta1=$(row["beta1"]) beta2=$(row["beta2"]) " *
            "qmin=$(row["qmin"]) qmax=$(row["qmax"]) gamma=$(row["gamma"]) -> $(path)")
end

# --------------------------------------------------------------------- builds

"Record every transfers row of a trial with one reason."
function record_failure(ctx, trial, reason)
    write_progress(ctx, trial)
    states = something(requested_states(trial), 0)
    rows = [trial_row(ctx, trial, transfers; states, reason) for transfers in trial["transfers"]]
    isempty(rows) || store_record(rows; root = ctx.root, floor = ctx.floor)
    println("  $(trial_label(trial)): $(reason)")
end

"One build's lines: the first builds the system and solver and compiles them with one solve at n = 8 (timed as build_s on a cold line), then every line with transfers solves."
function run_build(ctx, lines, failures)
    lead = first(lines)
    println("=== $(trial_label(lead)): $(count(l -> !isempty(l["transfers"]), lines)) trials ===")
    T = element_type(lead["precision"])
    write_progress(ctx, lead, "build")
    built = nothing
    build_s = NaN
    elapsed = @elapsed begin
        built = try
            system = build_system(lead, T)
            alg = julia_solver(lead["algorithm"], PACKAGE, T)
            (system, alg, cpu_problem(system, lead, lead["grid_min"]))
        catch err
            reason = error_reason(err)
            for trial in lines
                record_failure(ctx, trial, reason)
            end
            nothing
        end
        if built !== nothing && isempty(rejection(lead, built[2])) &&
           any(get(l, "timed", true) for l in lines)
            points = grid(merge(lead, Dict("n" => WARM_N)))
            try
                ensemble_solve(built[1], built[3], built[2], points, solve_kwargs(lead, T))
            catch err
                println("  warm failed: $(error_reason(err))")
            end
        end
    end
    built === nothing && return
    system, alg, prob = built
    lead["cold"] == true && (build_s = elapsed)
    println(@sprintf("  built%s: %.2f s", lead["cold"] == true ? " cold" : "", elapsed))
    states = length(system.golden_index)

    first_default = findfirst(l -> l["controller"] == "default" && !isempty(l["transfers"]) &&
        isempty(rejection(l, alg)), lines)
    if first_default !== nothing
        try
            export_controller(ctx, lines[first_default], prob, alg, T)
        catch err
            println("  controller export failed: $(error_reason(err))")
        end
    end

    for trial in lines
        isempty(trial["transfers"]) && continue
        write_progress(ctx, trial)
        reason = rejection(trial, alg)
        kwargs = solve_kwargs(trial, T)
        points = grid(trial)
        duration = trial["duration"]
        label = trial_label(trial)
        line_build_s = trial === lead ? build_s : NaN
        finals_path = ""
        rows = Dict{String, Any}[]
        for transfers in trial["transfers"]
            abandoned = abandon_reason(trial, transfers, failures)
            if abandoned !== nothing
                push!(rows, trial_row(ctx, trial, transfers; states, reason = abandoned))
                continue
            end
            if !isempty(reason)
                push!(rows, trial_row(ctx, trial, transfers; states, reason))
                continue
            end
            ms, samples, result, outcome, why = time_trial(
                () -> ensemble_solve(system, prob, alg, points, kwargs), "$(label) $(transfers)",
                trial["watchdog_s"], get(trial, "timed", true))
            pct = NaN
            if result !== nothing
                finals, t_final, retcode = result
                pct = errored_pct(finals, t_final, retcode, duration)
                if trial["finals"] && isempty(finals_path)
                    spec = merge(trial, Dict{String, Any}("key" => ctx.key))
                    finals_path = store_finals(spec, finals, t_final; retcode, root = ctx.root)
                end
            end
            push!(rows, trial_row(ctx, trial, transfers; states, min_ms = ms, samples_ms = samples,
                errored_pct = pct, build_s = line_build_s, reason = why, finals = finals_path))
            println(@sprintf("  %s %s: %s ms, errored=%s%%%s", label, transfers,
                isnan(ms) ? "nan" : @sprintf("%.3f", ms), isnan(pct) ? "nan" : @sprintf("%.1f", pct),
                isempty(why) ? "" : "  [" * why * "]"))
            note_failure!(failures, trial, transfers, outcome)
        end
        isempty(rows) || store_record(rows; root = ctx.root, floor = ctx.floor)
    end
end

function main(args)
    path, floor = parse_cli(args)
    trials = read_trials(path)
    ctx = (key = dataset_key(), root = joinpath(REPO_ROOT, "data"), floor = floor,
        suite_rev = store_suite_rev(REPO_ROOT), version = package_version(),
        progress = path * ".progress")
    println("julia_cpu: $(length(trials)) trial lines, key $(ctx.key), OrdinaryDiffEq $(ctx.version), " *
            "$(Threads.nthreads()) threads")
    failures = Dict{String, Vector{Tuple{Dict{String, Any}, String}}}()
    for (_, lines) in builds_of(trials)
        run_build(ctx, lines, failures)
    end
    return 0
end

exit(main(ARGS))
