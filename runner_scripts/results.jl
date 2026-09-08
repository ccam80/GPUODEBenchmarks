# The result store for the Julia writers, mirroring results.py: one results.csv per package and machine key.

using Dates

include(joinpath(@__DIR__, "protocol.jl"))

const RESULT_IDENTITY = ("package", "key", "analysis", "problem", "algorithm",
    "mode", "setting_kind", "setting", "n", "states", "tier", "transfers")
# samples_ms: every attempt in ms, warm-up first, ';'-joined; min_ms is the minimum after the warm-up.
const RESULT_VALUES = ("min_ms", "samples_ms", "errored_pct", "error", "build_s",
    "recorded_utc")
const RESULT_FIELDS = (RESULT_IDENTITY..., RESULT_VALUES...)
const RESULT_PACKAGE_DIRS = Dict("cubie" => "CUBIE", "cubie_mlir" => "CUBIE_MLIR",
    "julia" => "Julia", "cpp" => "CPP", "jax" => "JAX", "pytorch" => "PYTORCH",
    "myokit_cuda" => "MYOKIT_CUDA")

"data/<PACKAGE_DIR>/<key>/results.csv; the directory is created."
function result_store(repo_root, package, key)
    dir = joinpath(repo_root, "data", RESULT_PACKAGE_DIRS[package], key)
    mkpath(dir)
    return joinpath(dir, "results.csv")
end

result_floor_enabled() = !(get(ENV, "BENCH_FLOOR", "") in ("", "0"))

_result_float(text) = something(tryparse(Float64, String(text)), NaN)

_result_fmt(value::AbstractFloat) = isnan(value) ? "nan" : string(round(value, sigdigits = 10))
_result_fmt(value) = string(value)

"Two settings name the same point within a relative 1e-8."
function result_setting_matches(a, b)
    x, y = _result_float(a), _result_float(b)
    (isnan(x) && isnan(y)) && return true
    return isapprox(x, y; rtol = 1e-8, atol = 0.0)
end

"True when a row carries the identity columns of ident."
function result_same_point(row, ident)
    for field in RESULT_IDENTITY
        haskey(ident, field) || continue
        if field == "setting"
            result_setting_matches(row[field], ident[field]) || return false
        elseif string(row[field]) != string(ident[field])
            return false
        end
    end
    return true
end

"Every row of a store as Dict{String,String}; a missing file is empty."
function result_load(path)
    rows = Dict{String, String}[]
    isfile(path) || return rows
    lines = filter(!isempty, readlines(path))
    isempty(lines) && return rows
    header = String.(split(lines[1], ','))
    for line in lines[2:end]
        fields = String.(split(line, ','; keepempty = true))
        length(fields) < length(header) && continue
        push!(rows, Dict{String, String}(zip(header, fields)))
    end
    return rows
end

function _result_save(path, rows)
    scratch = path * ".partial"
    open(scratch, "w") do io
        println(io, join(RESULT_FIELDS, ","))
        for row in rows
            println(io, join([get(row, f, "") for f in RESULT_FIELDS], ","))
        end
    end
    mv(scratch, path; force = true)
end

"Run f under the mkdir lock beside the store; a stale lock is taken over."
function result_locked(f, path)
    lock = path * ".lock"
    deadline = time() + 120.0
    while true
        try
            mkdir(lock)
            break
        catch
            age = try time() - mtime(lock) catch; 0.0 end
            if age > 300.0
                try rm(lock; force = true, recursive = true) catch end
                continue
            end
            time() > deadline && error("result store locked: $(lock)")
            sleep(0.05)
        end
    end
    try
        return f()
    finally
        try rm(lock; force = true, recursive = true) catch end
    end
end

"The attempts of a row in ms, warm-up first; empty when none were recorded."
result_samples(row) = [parse(Float64, v) for v in split(get(row, "samples_ms", ""), ';') if !isempty(v)]

"One store row; samples is every attempt in ms, warm-up first."
function result_row(package, key, analysis, problem, algorithm, mode,
        setting_kind, setting, n, states; tier = "default", transfers = "both",
        min_ms = NaN, samples = nothing, errored_pct = NaN, error = NaN,
        build_s = NaN)
    attempts = samples === nothing ? Float64[] : Float64.(samples)
    return Dict{String, String}(
        "package" => package, "key" => key, "analysis" => analysis,
        "problem" => problem, "algorithm" => algorithm, "mode" => mode,
        "setting_kind" => setting_kind, "setting" => _result_fmt(Float64(setting)),
        "n" => string(Int(n)), "states" => string(Int(states)), "tier" => tier,
        "transfers" => transfers, "min_ms" => _result_fmt(Float64(min_ms)),
        "samples_ms" => join(_result_fmt.(attempts), ";"),
        "errored_pct" => _result_fmt(Float64(errored_pct)),
        "error" => _result_fmt(Float64(error)),
        "build_s" => _result_fmt(Float64(build_s)),
        "recorded_utc" => Dates.format(Dates.now(Dates.UTC), "yyyy-mm-ddTHH:MM:SSZ"))
end

"True when the new row's time beats the recorded one; NaN loses."
function _result_lower_wins(recorded, new)
    old, fresh = _result_float(recorded["min_ms"]), _result_float(new["min_ms"])
    isnan(old) && return true
    isnan(fresh) && return false
    return fresh < old
end

"Replace the row with this identity, or under --floor keep whichever has the lower time."
function result_record(path, row; floor = result_floor_enabled())
    result_locked(path) do
        rows = result_load(path)
        replaced = false
        for (index, existing) in enumerate(rows)
            result_same_point(existing, row) || continue
            if !floor || _result_lower_wins(existing, row)
                rows[index] = row
            end
            replaced = true
            break
        end
        replaced || push!(rows, row)
        _result_save(path, rows)
    end
end

"Rows matching the given identity columns."
result_rows(path; ident...) =
    [row for row in result_load(path)
     if result_same_point(row, Dict(string(k) => v for (k, v) in ident))]

"\"absent\", \"nan\" or \"finite\" for the rows matching ident."
function result_status(path; ident...)
    matched = result_rows(path; ident...)
    isempty(matched) && return "absent"
    any(row -> isfinite(_result_float(row["min_ms"])), matched) && return "finite"
    return "nan"
end

"Every store row under data/, each with its `os` and `gpu` split from the key."
function result_rows_under(root)
    rows = Dict{String, String}[]
    for (package, dir) in RESULT_PACKAGE_DIRS
        package_root = joinpath(root, dir)
        isdir(package_root) || continue
        for key in sort(readdir(package_root))
            path = joinpath(package_root, key, "results.csv")
            isfile(path) || continue
            parts = split(key, '_')
            length(parts) == 2 || continue
            for row in result_load(path)
                row["os"] = String(parts[1])
                row["gpu"] = String(parts[2])
                push!(rows, row)
            end
        end
    end
    return rows
end

"Display name of a package, as the figures label it."
const RESULT_DISPLAY = Dict("julia" => "Julia", "cpp" => "MPGOS", "jax" => "JAX",
    "pytorch" => "PYTORCH", "cubie" => "CUBIE", "cubie_mlir" => "CUBIE_MLIR",
    "myokit_cuda" => "MYOKIT CUDA")

"(setting_kind, setting) of the N and states sweeps for a problem row."
timing_setting(problem, mode) =
    mode == "fixed" ? ("dt", problem_timing_dt(problem)) : ("tol", TIMING_TOL)

"Record the two transfer legs of one N or states point."
function result_record_times(store, package, key, analysis, problem, algorithm,
        mode, n, states, t_both, t_none, errored_pct; samples_both = nothing,
        samples_none = nothing, build_s = NaN)
    kind, setting = timing_setting(problem, mode)
    name = problem["problem"]
    for (transfers, t_ms, samples) in (("both", t_both, samples_both),
                                       ("none", t_none, samples_none))
        result_record(store, result_row(package, key, analysis, name, algorithm,
            mode, kind, setting, n, states; transfers = transfers,
            min_ms = t_ms, samples = samples, errored_pct = errored_pct,
            build_s = build_s))
    end
end

"Record one work-precision point; every package times the resident solve alone."
function result_record_wp(store, package, key, problem, algorithm, mode, setting,
        t_ms, error, errored_pct; samples = nothing)
    kind = mode == "fixed" ? "dt" : "tol"
    result_record(store, result_row(package, key, "wp", problem["problem"],
        algorithm, mode, kind, setting, N_WP, problem["states"];
        transfers = "none", min_ms = t_ms, samples = samples,
        errored_pct = errored_pct, error = error))
end
