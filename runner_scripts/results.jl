# The result store for the Julia writers: the store_* shim serialises rows with JSON.jl and calls store.py; the result_* CSV functions serve bench_ode_gpu.jl, resume.jl and the plot scripts.

using Dates
using JSON

include(joinpath(@__DIR__, "protocol.jl"))

const STORE_CLI = joinpath(@__DIR__, "store.py")
const STORE_REPO_ROOT = dirname(@__DIR__)
const STORE_PACKAGES = ("cubie", "cubie_mlir", "jax", "pytorch", "myokit_cuda", "cpp",
    "julia_gpu", "julia_cpu")

# An explicit interpreter wins over the resolved one; the test script sets it from its argument.
const STORE_PYTHON = Ref("")

"The suite interpreter (GPU_ODE_CUBIE/venv, as launch.suite_python resolves it), else python on PATH."
function store_python(repo_root = STORE_REPO_ROOT)
    isempty(STORE_PYTHON[]) || return STORE_PYTHON[]
    venv = joinpath(repo_root, "GPU_ODE_CUBIE", "venv")
    for candidate in (joinpath(venv, "Scripts", "python.exe"),
                      joinpath(venv, "bin", "python3"), joinpath(venv, "bin", "python"))
        isfile(candidate) && return candidate
    end
    names = Sys.iswindows() ? ("python", "python3") : ("python3", "python")
    for name in names
        Sys.which(name) === nothing || return name
    end
    return names[1]
end

function _store_cmd(args; root = nothing)
    argv = [store_python(), STORE_CLI]
    root === nothing || append!(argv, ["--root", String(root)])
    return Cmd(vcat(argv, String.(args)))
end

"`git rev-parse --short HEAD`, suffixed -dirty when tracked files have changed; unknown outside git."
function store_suite_rev(repo_root = STORE_REPO_ROOT)
    rev = try
        strip(read(setenv(`git rev-parse --short HEAD`; dir = repo_root), String))
    catch
        return "unknown"
    end
    dirty = try
        !isempty(strip(read(setenv(`git status --porcelain --untracked-files=no`;
            dir = repo_root), String)))
    catch
        false
    end
    return rev * (dirty ? "-dirty" : "")
end

"One complete store row (section 1.2): the identity as given, every value column defaulted."
function store_row(package, key, problem, algorithm, mode, setting_kind, setting, n,
        states; tier = "default", transfers = "both", min_ms = NaN, samples_ms = Float64[],
        errored_pct = NaN, error = NaN, build_s = NaN, reason = "", finals = "",
        package_version = "", suite_rev = "", recorded_utc = nothing)
    package in STORE_PACKAGES || throw(ArgumentError("package '$(package)' is not a store package"))
    stamp = recorded_utc === nothing ? Dates.now(Dates.UTC) : recorded_utc
    return Dict{String, Any}(
        "package" => String(package), "key" => String(key), "problem" => String(problem),
        "algorithm" => String(algorithm), "mode" => String(mode),
        "setting_kind" => String(setting_kind), "setting" => Float64(setting),
        "n" => Int(n), "states" => Int(states), "tier" => String(tier),
        "transfers" => String(transfers), "min_ms" => Float64(min_ms),
        "samples_ms" => Float64[samples_ms...], "errored_pct" => Float64(errored_pct),
        "error" => Float64(error), "build_s" => Float64(build_s),
        "reason" => String(reason), "finals" => String(finals),
        "package_version" => String(package_version), "suite_rev" => String(suite_rev),
        "recorded_utc" => Dates.format(stamp, "yyyy-mm-ddTHH:MM:SS.sss") * "Z")
end

"Record rows (a Dict or a vector of them) through the store CLI; floor keeps the lower finite time."
function store_record(rows; root = nothing, floor = false)
    batch = rows isa AbstractDict ? [rows] : collect(rows)
    args = ["record", "-"]
    floor && push!(args, "--floor")
    payload = JSON.json(batch; allownan = true)
    run(pipeline(_store_cmd(args; root = root); stdin = IOBuffer(payload)))
    return nothing
end

"Write the finals file of a trial (finals is rows x states, converged one flag per row); returns the path relative to the package dir."
function store_finals(identity, finals::AbstractMatrix, converged; root = nothing)
    size(finals, 1) == length(converged) ||
        throw(ArgumentError("converged has one flag per finals row"))
    ident_path, csv_path = tempname() * ".json", tempname() * ".csv"
    write(ident_path, JSON.json(identity; allownan = true))
    open(csv_path, "w") do io
        println(io, join(vcat(["traj"], ["s$(k)" for k in 1:size(finals, 2)], ["converged"]), ","))
        for (index, row) in enumerate(eachrow(finals))
            println(io, join(vcat([string(index - 1)],
                [repr(Float64(v)) for v in row], [converged[index] ? "1" : "0"]), ","))
        end
    end
    try
        return String(strip(read(_store_cmd(["finals", ident_path, csv_path]; root = root), String)))
    finally
        rm(ident_path; force = true)
        rm(csv_path; force = true)
    end
end

"\"absent\", \"nan\" or \"finite\" for the rows carrying the identity columns given."
function store_status(identity; root = nothing)
    ident_path = tempname() * ".json"
    write(ident_path, JSON.json(identity; allownan = true))
    try
        return String(strip(read(_store_cmd(["status", ident_path]; root = root), String)))
    finally
        rm(ident_path; force = true)
    end
end

"The CSV text of a SQL statement over the `results` view."
store_query(sql; root = nothing) = read(_store_cmd(["query", sql]; root = root), String)

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
