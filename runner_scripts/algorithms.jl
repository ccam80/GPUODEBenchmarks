# The algorithm axis: one row per integration algorithm in algorithms.csv, mirrored by algorithms.py.

const ALGORITHMS_CSV = joinpath(@__DIR__, "algorithms.csv")

const _MODES = ("fixed", "adaptive")
const _BOOLS = ("ne", "ne_adaptive")

"Split one CSV line on the commas outside double quotes."
function _csv_fields(line)
    fields = String[]
    buffer = IOBuffer()
    quoted = false
    for c in line
        if c == '"'
            quoted = !quoted
        elseif c == ',' && !quoted
            push!(fields, String(take!(buffer)))
        else
            write(buffer, c)
        end
    end
    push!(fields, String(take!(buffer)))
    return fields
end

"Every algorithm in declaration order, as a vector of Dict{String,Any}."
function load_algorithms()
    lines = filter(!isempty, strip.(readlines(ALGORITHMS_CSV)))
    header = String.(split(lines[1], ','))
    algorithms = Dict{String, Any}[]
    for line in lines[2:end]
        fields = _csv_fields(line)
        while length(fields) < length(header)
            push!(fields, "")
        end
        row = Dict{String, Any}(zip(header, fields))
        for mode in _MODES
            row[mode] = String.(filter(!isempty, split(row[mode], '|')))
        end
        for flag in _BOOLS
            row[flag] = lowercase(strip(row[flag])) == "true"
        end
        row["order"] = parse(Int, row["order"])
        push!(algorithms, row)
    end
    return algorithms
end

algorithm_names() = [row["algorithm"] for row in load_algorithms()]

"One algorithm by name; errors on an unknown name."
function get_algorithm(name)
    for row in load_algorithms()
        row["algorithm"] == name && return row
    end
    error("unknown algorithm '$(name)' (expected one of: all, " *
          join(algorithm_names(), ", ") * ")")
end

"True when the framework times this algorithm, in the mode if given."
function algorithm_supports(row, framework, mode = nothing)
    modes = mode === nothing ? _MODES : (mode,)
    return any(framework in row[m] for m in modes)
end

"Algorithm names a framework times, in declaration order."
function supported_algorithms(framework, mode = nothing)
    return [row["algorithm"] for row in load_algorithms()
            if algorithm_supports(row, framework, mode)]
end

"Resolve \"all\" or a comma list to the algorithms a framework times."
function resolve_algorithms(request, framework)
    supported = supported_algorithms(framework)
    (request === nothing || request == "" || request == "all") && return supported
    names = [String(name) for name in split(request, ',') if !isempty(name)]
    for name in names
        get_algorithm(name)
    end
    return [name for name in names if name in supported]
end

"Rows named by \"all\" or a comma list; an unknown name errors."
function _select(rows, request)
    (request === nothing || request == "" || request == "all") && return rows
    names = [String(name) for name in split(request, ',') if !isempty(name)]
    for name in names
        get_algorithm(name)
    end
    return [row for row in rows if row["algorithm"] in names]
end

"The numerical-equivalence rows, narrowed by name."
ne_algorithms(request = "all") =
    _select([row for row in load_algorithms() if row["ne"]], request)

"The cubie-DiffEqGPU overlap rows, narrowed by name."
overlap_algorithms(request = "all") =
    _select([row for row in load_algorithms() if !isempty(row["julia_gpu"])],
            request)

"The fixed-step ne sweep excludes the erk family."
runs_fixed_ne(row) = row["family"] != "erk"
