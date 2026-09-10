# The result store for the Julia writers: rows serialise with JSON.jl and land through store.py, which validates, hashes and writes.

using Dates
using JSON

const STORE_CLI = joinpath(@__DIR__, "store.py")
const STORE_REPO_ROOT = dirname(@__DIR__)
const STORE_PACKAGES = ("cubie", "cubie_mlir", "jax", "pytorch", "myokit_cuda", "cpp",
    "julia_gpu", "julia_cpu")

# The run spec in table order; trial_id hashes every field but transfers and key, group_id the system and stepping fields only.
const STORE_SPEC_FIELDS = ("problem", "system_params", "duration", "precision",
    "parameter", "grid_scale", "grid_min", "grid_max", "n", "grid_dtype",
    "algorithm", "controller", "dt", "dt_min", "dt_max", "atol", "rtol", "gains",
    "newton_atol", "newton_rtol", "transfers", "package", "key")
const STORE_TRIAL_FIELDS = Tuple(f for f in STORE_SPEC_FIELDS if !(f in ("transfers", "key")))
const STORE_FINALS_FIELDS = (STORE_TRIAL_FIELDS..., "key")

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

"`git rev-parse --short HEAD`; unknown outside git."
function store_suite_rev(repo_root = STORE_REPO_ROOT)
    try
        strip(read(setenv(`git rev-parse --short HEAD`; dir = repo_root), String))
    catch
        "unknown"
    end
end

"The named spec fields of a trial or row Dict (extra keys ignored); every one must be present and the package known."
function store_spec(fields, names = STORE_SPEC_FIELDS)
    missing = [f for f in names if !haskey(fields, f)]
    isempty(missing) || throw(ArgumentError("spec incomplete: " * join(missing, ", ")))
    if "package" in names
        fields["package"] in STORE_PACKAGES ||
            throw(ArgumentError("package '$(fields["package"])' is not a store package"))
    end
    return Dict{String, Any}(f => fields[f] for f in names)
end

"One complete store row: the spec fields of a trial or spec Dict with the value columns; store.py hashes run_id, trial_id and group_id."
function store_row(spec; states, min_ms = NaN, samples_ms = Float64[], errored_pct = NaN,
        build_s = NaN, reason = "", finals = "", package_version = "", suite_rev = "",
        recorded_utc = nothing)
    row = store_spec(spec)
    stamp = recorded_utc === nothing ? Dates.now(Dates.UTC) : recorded_utc
    row["states"] = Int(states)
    row["min_ms"] = Float64(min_ms)
    row["samples_ms"] = Float64[samples_ms...]
    row["errored_pct"] = Float64(errored_pct)
    row["build_s"] = Float64(build_s)
    row["reason"] = String(reason)
    row["finals"] = String(finals)
    row["package_version"] = String(package_version)
    row["suite_rev"] = String(suite_rev)
    row["recorded_utc"] = Dates.format(stamp, "yyyy-mm-ddTHH:MM:SS.sss") * "Z"
    return row
end

"Record rows (a Dict or a vector of them) through the store CLI in one batch; floor keeps the lower finite time."
function store_record(rows; root = nothing, floor = false)
    batch = rows isa AbstractDict ? [rows] : collect(rows)
    args = ["record", "-"]
    floor && push!(args, "--floor")
    payload = JSON.json(batch; allownan = true)
    run(pipeline(_store_cmd(args; root = root); stdin = IOBuffer(payload)))
    return nothing
end

function _with_spec_file(f, spec, names)
    path = tempname() * ".json"
    write(path, JSON.json(store_spec(spec, names); allownan = true))
    try
        return f(path)
    finally
        rm(path; force = true)
    end
end

"Write the finals file of a trial (finals is n x states in grid order, t_final each trajectory's final time, retcode the package's failure code text, empty on success); returns finals/<trial_id>.parquet relative to the package dir."
function store_finals(spec, finals::AbstractMatrix, t_final; retcode = nothing, root = nothing)
    size(finals, 1) == length(t_final) ||
        throw(ArgumentError("t_final has one time per finals row"))
    codes = retcode === nothing ? fill("", size(finals, 1)) : string.(retcode)
    length(codes) == size(finals, 1) ||
        throw(ArgumentError("retcode has one code per finals row"))
    csv_path = tempname() * ".csv"
    open(csv_path, "w") do io
        println(io, join(vcat(["traj"], ["s$(k)" for k in 1:size(finals, 2)], ["t_final", "retcode"]), ","))
        for (index, row) in enumerate(eachrow(finals))
            println(io, join(vcat([string(index - 1)], [repr(Float64(v)) for v in row],
                [repr(Float64(t_final[index])), codes[index]]), ","))
        end
    end
    try
        return _with_spec_file(spec, STORE_FINALS_FIELDS) do spec_path
            String(strip(read(_store_cmd(["finals", spec_path, csv_path]; root = root), String)))
        end
    finally
        rm(csv_path; force = true)
    end
end

"(trial_id, run_id, group_id) of a spec, hashed by store.py."
function store_hash(spec; root = nothing)
    ids = _with_spec_file(spec, STORE_SPEC_FIELDS) do spec_path
        JSON.parse(read(_store_cmd(["hash", spec_path]; root = root), String))
    end
    return (String(ids["trial_id"]), String(ids["run_id"]), String(ids["group_id"]))
end

"\"absent\", \"nan\" or \"finite\" for the row of a run_id (or of a spec's run_id)."
function store_status(run; root = nothing)
    id = run isa AbstractDict ? store_hash(run; root = root)[2] : String(run)
    return String(strip(read(_store_cmd(["status", id]; root = root), String)))
end

"The CSV text of a SQL statement over the `results` view."
store_query(sql; root = nothing) = read(_store_cmd(["query", sql]; root = root), String)
