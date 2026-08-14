# The problem axis: one row per benchmark ODE/DAE in problems.csv, mirrored by problems.py.

const PROBLEMS_CSV = joinpath(@__DIR__, "problems.csv")
const DEFAULT_PROBLEM = "lorenz"

const _INT_FIELDS = ("states", "dae_index", "wp_k_min", "wp_k_max",
    "euler_k_min", "euler_k_max", "ne_k_min", "ne_k_max")
const _FLOAT_FIELDS = ("duration", "sweep_min", "sweep_max")

"Every problem in declaration order, as a vector of Dict{String,Any}."
function load_problems()
    lines = filter(!isempty, strip.(readlines(PROBLEMS_CSV)))
    header = String.(split(lines[1], ','))
    problems = Dict{String, Any}[]
    for line in lines[2:end]
        fields = String.(split(line, ','))
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

"Resolve \"all\" or a comma list to the problems a framework runs."
function resolve_problems(request, framework = nothing)
    selected = if request === nothing || request == "" || request == "all"
        load_problems()
    else
        [get_problem(String(name)) for name in split(request, ',') if !isempty(name)]
    end
    framework === nothing && return selected
    return [row for row in selected if framework in row["frameworks"]]
end

"Fixed step used by the N-sweep: 1000 steps for every problem."
problem_timing_dt(problem) = problem["duration"] / 1000.0

"Fixed-step dt grid for the work-precision sweep."
function problem_dts(problem, algorithm = nothing)
    lo, hi = algorithm == "euler" ?
             (problem["euler_k_min"], problem["euler_k_max"]) :
             (problem["wp_k_min"], problem["wp_k_max"])
    return [problem["duration"] * 2.0^-k for k in lo:hi]
end

"Fixed-step dt grid for the numerical-equivalence sweep."
problem_ne_dts(problem) = [problem["duration"] * 2.0^-k
                           for k in problem["ne_k_min"]:problem["ne_k_max"]]

"The ensemble parameter grid: n values over the sweep range."
problem_sweep(problem, n) = range(problem["sweep_min"],
    stop = problem["sweep_max"], length = n)
