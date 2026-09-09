# The Julia constructor table: one row per algorithm in julia_algorithms.csv with the OrdinaryDiffEq (julia_cpu) and DiffEqGPU (julia_gpu) constructor expressions.

const JULIA_ALGORITHMS_CSV = joinpath(@__DIR__, "julia_algorithms.csv")

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

"Every row in declaration order, as a vector of Dict{String,String}."
function load_algorithms()
    lines = filter(!isempty, strip.(readlines(JULIA_ALGORITHMS_CSV)))
    header = String.(split(lines[1], ','))
    algorithms = Dict{String, String}[]
    for line in lines[2:end]
        fields = _csv_fields(line)
        while length(fields) < length(header)
            push!(fields, "")
        end
        push!(algorithms, Dict{String, String}(zip(header, fields)))
    end
    return algorithms
end

algorithm_names() = [row["algorithm"] for row in load_algorithms()]

"One algorithm by name; errors on an unknown name."
function get_algorithm(name)
    for row in load_algorithms()
        row["algorithm"] == name && return row
    end
    error("unknown algorithm '$(name)' (expected one of: " *
          join(algorithm_names(), ", ") * ")")
end

"The constructor expression of an algorithm for a package (julia_cpu or julia_gpu); errors when the package has none."
function julia_constructor(name, package)
    expr = get_algorithm(name)[package]
    isempty(expr) && error("no $(package) constructor for '$(name)'")
    return expr
end

"The algorithm names with a constructor for a package, in declaration order."
package_algorithms(package) = [row["algorithm"] for row in load_algorithms() if !isempty(row[package])]
