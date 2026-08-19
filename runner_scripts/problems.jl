# The problem axis: one row per benchmark ODE/DAE in problems.csv, mirrored by problems.py.

const PROBLEMS_CSV = joinpath(@__DIR__, "problems.csv")
const DEFAULT_PROBLEM = "lorenz"

const _INT_FIELDS = ("states",)
const _FLOAT_FIELDS = ("duration", "sweep_min", "sweep_max", "golden_tol")

# Dyadic dt-grid exponents as duration fractions; mirrored in problems.py.
const WP_K = (4, 13)
const EULER_K = (8, 17)
const NE_K = (1, 13)
const TIMING_DT_K = 10

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

"True when the framework runs this problem."
problem_supports(row, framework) = framework in row["frameworks"]

"Fixed step used by the N-sweep: duration * 2^-10."
problem_timing_dt(problem) = problem["duration"] * 2.0^-TIMING_DT_K

"Fixed-step dt grid for the work-precision sweep."
function problem_dts(problem, algorithm = nothing)
    lo, hi = algorithm == "euler" ? EULER_K : WP_K
    return [problem["duration"] * 2.0^-k for k in lo:hi]
end

"Fixed-step dt grid for the numerical-equivalence sweep."
problem_ne_dts(problem) = [problem["duration"] * 2.0^-k
                           for k in NE_K[1]:NE_K[2]]

"The ensemble parameter grid: n values over the sweep range."
function problem_sweep(problem, n)
    lo, hi = problem["sweep_min"], problem["sweep_max"]
    if problem["sweep_scale"] == "log"
        lo > 0 || error("problem '$(problem["problem"])': a log sweep needs sweep_min > 0")
        return 10 .^ range(log10(lo), stop = log10(hi), length = n)
    end
    return range(lo, stop = hi, length = n)
end
