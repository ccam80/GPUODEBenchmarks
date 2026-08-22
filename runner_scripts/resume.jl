# Continuation of partial runs, mirroring resume.py; include problems.jl and algorithms.jl first.

const RESUME_MODES = ("fixed", "adaptive")

resume_enabled() = !(get(ENV, "BENCH_RESUME", "") in ("", "0"))

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

resume_active() = resume_enabled() || resume_cursor() !== nothing

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

"First-column integers of the rows already in an output file."
function recorded_values(path)
    values = Set{Int}()
    isfile(path) || return values
    for line in eachline(path)
        fields = split(line)
        length(fields) < 2 && continue
        value = tryparse(Float64, fields[1])
        value === nothing || push!(values, Int(round(value)))
    end
    return values
end

"True when one (problem, algorithm, mode, n) sweep point is covered."
function skip_point(problem, algorithm, mode, n, outfile)
    cursor_skips(problem, algorithm, mode, n) && return true
    return resume_enabled() && n in recorded_values(outfile)
end

"True when a wp leg's file already holds `expected` rows or the cursor skips it."
function skip_wp_leg(problem, algorithm, mode, outfile, expected)
    cursor_skips(problem, algorithm, mode) && return true
    resume_enabled() || return false
    isfile(outfile) || return false
    rows = count(line -> length(split(line)) >= 2, eachline(outfile))
    return rows >= expected
end
