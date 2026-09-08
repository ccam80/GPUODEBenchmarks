# The result store for the Julia writers: rows serialise with JSON.jl and land through store.py.

using Dates
using JSON

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
