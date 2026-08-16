# The algorithm axis: one row per integration algorithm in algorithms.csv, mirrored by algorithms.py.

const ALGORITHMS_CSV = joinpath(@__DIR__, "algorithms.csv")

const _MODES = ("fixed", "adaptive")

"Every algorithm in declaration order, as a vector of Dict{String,Any}."
function load_algorithms()
    lines = filter(!isempty, strip.(readlines(ALGORITHMS_CSV)))
    header = String.(split(lines[1], ','))
    algorithms = Dict{String, Any}[]
    for line in lines[2:end]
        fields = String.(split(line, ','))
        # A trailing empty field is dropped by split, so pad the row out.
        while length(fields) < length(header)
            push!(fields, "")
        end
        row = Dict{String, Any}(zip(header, fields))
        row["order"] = parse(Int, row["order"])
        for mode in _MODES
            row[mode] = String.(filter(!isempty, split(row[mode], '|')))
        end
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

"True when the framework runs this algorithm, in the mode if given."
function algorithm_supports(row, framework, mode = nothing)
    modes = mode === nothing ? _MODES : (mode,)
    return any(framework in row[m] for m in modes)
end

"Algorithm names a framework runs, in declaration order."
function supported_algorithms(framework, mode = nothing)
    return [row["algorithm"] for row in load_algorithms()
            if algorithm_supports(row, framework, mode)]
end

"Resolve \"all\" or a comma list to the algorithms a framework runs."
function resolve_algorithms(request, framework)
    supported = supported_algorithms(framework)
    (request === nothing || request == "" || request == "all") && return supported
    names = [String(name) for name in split(request, ',') if !isempty(name)]
    for name in names
        get_algorithm(name)
    end
    return [name for name in names if name in supported]
end
