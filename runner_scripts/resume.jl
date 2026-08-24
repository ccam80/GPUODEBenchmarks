# Continuation of partial runs, mirroring resume.py; include problems.jl and algorithms.jl first.

const RESUME_MODES = ("fixed", "adaptive")

resume_enabled() = !(get(ENV, "BENCH_RESUME", "") in ("", "0"))

no_overwrite_enabled() = !(get(ENV, "BENCH_NO_OVERWRITE", "") in ("", "0"))

"True when BENCH_FLOOR asks to merge re-runs by keeping the lower time."
floor_enabled() = !(get(ENV, "BENCH_FLOOR", "") in ("", "0"))

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

"The token as a finite float, or `nothing`."
function _finite(token)
    value = tryparse(Float64, token)
    return (value !== nothing && isfinite(value)) ? value : nothing
end

"First-column integers of the rows whose time field is finite."
function numeric_values(path)
    values = Set{Int}()
    isfile(path) || return values
    for line in eachline(path)
        fields = split(line)
        (length(fields) < 2 || _finite(fields[2]) === nothing) && continue
        value = tryparse(Float64, fields[1])
        value === nothing || push!(values, Int(round(value)))
    end
    return values
end

"Drop the rows for the points about to rerun."
function prune_reruns(outfile, ns)
    # --floor merges at write time; the recorded rows must survive.
    floor_enabled() && return
    (resume_active() && !isempty(ns) && isfile(outfile)) || return
    rerun = Set(Int.(ns))
    stale = line -> begin
        fields = split(line)
        length(fields) < 2 && return false
        value = tryparse(Float64, fields[1])
        return value !== nothing && Int(round(value)) in rerun
    end
    lines = readlines(outfile)
    kept = [line for line in lines if !stale(line)]
    if length(kept) < length(lines)
        open(outfile, "w") do io
            for line in kept
                println(io, line)
            end
        end
    end
end

"True when one (problem, algorithm, mode, n) sweep point is covered."
function skip_point(problem, algorithm, mode, n, outfile)
    cursor_skips(problem, algorithm, mode, n) && return true
    resume_enabled() && n in recorded_values(outfile) && return true
    return no_overwrite_enabled() && n in numeric_values(outfile)
end

"The lower of two times; NaN loses to any finite value."
_lower_time(recorded, new) = isnan(recorded) ? new :
                             (isnan(new) ? recorded : min(recorded, new))

"--floor: merge `key values...` into path, keeping the lower value per column; recorded fields beyond the merged columns survive."
function merge_min_row(path, key, values)
    out = String[]
    merged = false
    for line in (isfile(path) ? readlines(path) : String[])
        fields = split(line)
        parsed = isempty(fields) ? nothing : tryparse(Float64, fields[1])
        if merged || parsed === nothing || Int(round(parsed)) != key
            push!(out, line)
            continue
        end
        recorded = [begin
                        raw = i + 1 <= length(fields) ?
                              tryparse(Float64, fields[i + 1]) : nothing
                        raw === nothing ? NaN : raw
                    end
                    for i in eachindex(values)]
        combined = [_lower_time(r, Float64(v))
                    for (r, v) in zip(recorded, values)]
        tail = String.(fields[(length(values) + 2):end])
        push!(out, join(vcat([string(key)], string.(combined), tail), " "))
        merged = true
    end
    merged ||
        push!(out,
            join(vcat([string(key)], [string(Float64(v)) for v in values]),
                " "))
    open(path, "w") do io
        for line in out
            println(io, line)
        end
    end
end

"--floor: merge one wp row, keeping the (time, error) pair with the lower time."
function merge_wp_row(path, setting, t_ms, err)
    new_row = string(setting, " ", t_ms, " ", err)
    out = String[]
    merged = false
    for line in (isfile(path) ? readlines(path) : String[])
        fields = split(line)
        parsed = isempty(fields) ? nothing : tryparse(Float64, fields[1])
        if merged || parsed === nothing ||
           !isapprox(parsed, Float64(setting); rtol = 1e-8)
            push!(out, line)
            continue
        end
        merged = true
        recorded_t = length(fields) >= 2 ? tryparse(Float64, fields[2]) :
                     nothing
        recorded_t === nothing && (recorded_t = NaN)
        if isnan(recorded_t) || (!isnan(t_ms) && t_ms < recorded_t)
            push!(out, new_row)
        else
            push!(out, line)
        end
    end
    merged || push!(out, new_row)
    open(path, "w") do io
        for line in out
            println(io, line)
        end
    end
end

"Append one timing row, or --floor-merge it keeping the lower time per column."
function record_row(outfile, key, values)
    if floor_enabled()
        merge_min_row(outfile, key, values)
        return
    end
    open(outfile, "a+") do io
        println(io, key, " ", join(values, " "))
        flush(io)
    end
end

"Write one wp row through the open handle, or --floor-merge it into path."
function write_wp_row(io, path, setting, t_ms, err)
    if floor_enabled()
        merge_wp_row(path, setting, t_ms, err)
        return
    end
    println(io, setting, " ", t_ms, " ", err)
    flush(io)
end

"True when a wp leg's file already holds `expected` rows or the cursor skips it."
function skip_wp_leg(problem, algorithm, mode, outfile, expected)
    cursor_skips(problem, algorithm, mode) && return true
    isfile(outfile) || return false
    if resume_enabled()
        rows = count(line -> length(split(line)) >= 2, eachline(outfile))
        rows >= expected && return true
    end
    no_overwrite_enabled() || return false
    finite_rows = count(eachline(outfile)) do line
        fields = split(line)
        length(fields) >= 2 && _finite(fields[2]) !== nothing
    end
    return finite_rows >= expected
end
