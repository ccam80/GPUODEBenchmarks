# Float64 CPU definitions used for the golden references, one entry per problem.

"Lorenz with the exact Float64 constants; the swept rho is p[1]."
function lorenz_reference(u, p, t)
    du1 = 10.0 * (u[2] - u[1])
    du2 = u[1] * (p[1] - u[3]) - u[2]
    du3 = u[1] * u[2] - (8.0 / 3.0) * u[3]
    return [du1, du2, du3]
end

const REFERENCE_SYSTEMS = Dict{String, Any}(
    "lorenz" => (rhs = lorenz_reference, u0 = [1.0, 0.0, 0.0],
        mass_matrix = nothing),
)

"Right-hand side, initial state and mass matrix for a problem row or name."
function reference_system(problem)
    name = problem isa AbstractDict ? problem["problem"] : problem
    haskey(REFERENCE_SYSTEMS, name) ||
        error("no Float64 reference definition for problem '$(name)'")
    return REFERENCE_SYSTEMS[name]
end

"Solver instance named by a problem's golden_algorithm column."
function reference_solver(name)
    name == "Vern9" && return Vern9()
    name == "Rodas5P" && return Rodas5P()
    name == "RadauIIA5" && return RadauIIA5()
    error("unknown golden algorithm '$(name)'")
end
