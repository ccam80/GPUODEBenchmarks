# Continuation of partial runs against the result store, mirroring resume.py; include problems.jl, algorithms.jl and results.jl first.

const RESUME_MODES = ("fixed", "adaptive")

resume_enabled() = !(get(ENV, "BENCH_RESUME", "") in ("", "0"))

no_overwrite_enabled() = !(get(ENV, "BENCH_NO_OVERWRITE", "") in ("", "0"))

"True when BENCH_FLOOR asks to merge re-runs by keeping the lower time."
floor_enabled() = result_floor_enabled()

"BENCH_RESUME_FROM spec -> cursor tuple; omitted parts are `nothing`."
function parse_resume_cursor(spec)
    parts = String.(split(spec, ':'))
    (isempty(parts) || isempty(parts[1])) &&
        error("BENCH_RESUME_FROM requires a problem name, got '$(spec)'")
    get_problem(parts[1])
    problem = findfirst(==(parts[1]), problem_names())
    algorithm = nothing
    mode = nothing
    n = nothing
    for tok in parts[2:end]
        if occursin(r"^\d+$", tok)
            n === nothing ||
                error("BENCH_RESUME_FROM '$(spec)': more than one N")
            n = parse(Int, tok)
        elseif tok in RESUME_MODES
            (algorithm === nothing || mode !== nothing || n !== nothing) &&
                error("BENCH_RESUME_FROM '$(spec)': the mode goes after " *
                      "the algorithm and before N")
            mode = findfirst(==(tok), collect(RESUME_MODES))
        else
            (algorithm !== nothing || n !== nothing) &&
                error("BENCH_RESUME_FROM '$(spec)': expected " *
                      "problem[:algorithm][:fixed|adaptive][:N]")
            get_algorithm(tok)
            algorithm = findfirst(==(tok), algorithm_names())
        end
    end
    (algorithm !== nothing && mode === nothing) && (mode = 1)
    return (problem = problem, algorithm = algorithm, mode = mode, n = n)
end

const _RESUME_CURSOR = Ref{Any}(:unparsed)

"The parsed BENCH_RESUME_FROM cursor, or `nothing`; parsed once."
function resume_cursor()
    if _RESUME_CURSOR[] === :unparsed
        spec = get(ENV, "BENCH_RESUME_FROM", "")
        _RESUME_CURSOR[] = isempty(spec) ? nothing : parse_resume_cursor(spec)
    end
    return _RESUME_CURSOR[]
end

resume_active() = resume_enabled() || no_overwrite_enabled() ||
                  resume_cursor() !== nothing

"True when (problem, algorithm, mode[, n]) is before the cursor."
function cursor_skips(problem, algorithm, mode, n = nothing)
    cur = resume_cursor()
    cur === nothing && return false
    pi = findfirst(==(problem), problem_names())
    pi == cur.problem || return pi < cur.problem
    if cur.algorithm === nothing
        return cur.n !== nothing && n !== nothing && n < cur.n
    end
    ai = findfirst(==(algorithm), algorithm_names())
    mi = findfirst(==(mode), collect(RESUME_MODES))
    (ai, mi) == (cur.algorithm, cur.mode) ||
        return (ai, mi) < (cur.algorithm, cur.mode)
    return cur.n !== nothing && n !== nothing && n < cur.n
end

"Whether a recorded status is covered under the active flags."
function _status_skips(status)
    resume_enabled() && status != "absent" && return true
    return no_overwrite_enabled() && status == "finite"
end

"True when one N or states point is covered; `key` is N, or the state count in the states sweep."
function skip_point(store, analysis, problem, algorithm, mode, key, n, states,
        setting_kind, setting)
    cursor_skips(problem, algorithm, mode, key) && return true
    status = result_status(store; analysis = analysis, problem = problem,
        algorithm = algorithm, mode = mode, setting_kind = setting_kind,
        setting = _result_fmt(Float64(setting)), n = string(n),
        states = string(states))
    return _status_skips(status)
end

"True when every setting of a work-precision leg is covered."
function skip_wp_leg(store, problem, algorithm, mode, settings, states)
    cursor_skips(problem, algorithm, mode) && return true
    (resume_enabled() || no_overwrite_enabled()) || return false
    kind = mode == "fixed" ? "dt" : "tol"
    return all(settings) do setting
        _status_skips(result_status(store; analysis = "wp", problem = problem,
            algorithm = algorithm, mode = mode, setting_kind = kind,
            setting = _result_fmt(Float64(setting)), n = string(N_WP),
            states = string(states)))
    end
end
