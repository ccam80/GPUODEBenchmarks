# The problem catalogue: one row per benchmark ODE/DAE in problems.csv, mirrored by problems.py.

const PROBLEMS_CSV = joinpath(@__DIR__, "problems.csv")
const DEFAULT_PROBLEM = "lorenz"

const _INT_FIELDS = ("states",)
const _FLOAT_FIELDS = ("duration", "sweep_min", "sweep_max", "golden_tol")

"Every problem in declaration order, as a vector of Dict{String,Any}."
function load_problems()
    lines = filter(!isempty, strip.(readlines(PROBLEMS_CSV)))
    header = String.(split(lines[1], ','))
    problems = Dict{String, Any}[]
    for line in lines[2:end]
        fields = String.(split(line, ','))
        # A trailing empty field is dropped by split, so pad the row out.
        while length(fields) < length(header)
            push!(fields, "")
        end
        row = Dict{String, Any}(zip(header, fields))
        for field in _INT_FIELDS
            row[field] = parse(Int, row[field])
        end
        for field in _FLOAT_FIELDS
            row[field] = parse(Float64, row[field])
        end
        row["frameworks"] = String.(split(row["frameworks"], '|'))
        push!(problems, row)
    end
    return problems
end

problem_names() = [row["problem"] for row in load_problems()]

"One problem by name; errors on an unknown name."
function get_problem(name)
    for row in load_problems()
        row["problem"] == name && return row
    end
    error("unknown problem '$(name)' (expected one of: " *
          join(problem_names(), ", ") * ")")
end

"Resolve \"all\" or a comma list to the problems a package implements."
function resolve_problems(request, package = nothing)
    selected = if request === nothing || request == "" || request == "all"
        load_problems()
    else
        [get_problem(String(name)) for name in split(request, ',') if !isempty(name)]
    end
    package === nothing && return selected
    return [row for row in selected if package in row["frameworks"]]
end

"True when the package implements this problem."
problem_supports(row, package) = package in row["frameworks"]
