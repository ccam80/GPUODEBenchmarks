# DiffEqGPU system definitions: an out-of-place SVector RHS and Float32 initial state per problem, with the swept scalar in p[1].

using StaticArrays

function lorenz_rhs(u, p, t)
    du1 = 10.0f0 * (u[2] - u[1])
    du2 = p[1] * u[1] - u[2] - u[1] * u[3]
    du3 = u[1] * u[2] - (8.0f0 / 3.0f0) * u[3]
    return @SVector [du1, du2, du3]
end

# In-place Float32 twin used by the CPU numerical-equivalence sweeps.
function lorenz_rhs!(du, u, p, t)
    du[1] = 10.0f0 * (u[2] - u[1])
    du[2] = u[1] * (p[1] - u[3]) - u[2]
    du[3] = u[1] * u[2] - (8.0f0 / 3.0f0) * u[3]
    return nothing
end

function lorenz_jac(u, p, t)
    return @SMatrix [-10.0f0 10.0f0 0.0f0;
        p[1]-u[3] -1.0f0 -u[1];
        u[2] u[1] -(8.0f0/3.0f0)]
end

lorenz_tgrad(u, p, t) = @SVector [0.0f0, 0.0f0, 0.0f0]

const JULIA_SYSTEMS = Dict{String, Any}(
    "lorenz" => (rhs = lorenz_rhs, rhs! = lorenz_rhs!,
        jac = lorenz_jac, tgrad = lorenz_tgrad,
        u0 = @SVector([1.0f0, 0.0f0, 0.0f0])),
)

"Right-hand sides and initial state for a problem row or name."
function julia_system(problem)
    name = problem isa AbstractDict ? problem["problem"] : problem
    haskey(JULIA_SYSTEMS, name) ||
        error("no DiffEqGPU definition for problem '$(name)'")
    return JULIA_SYSTEMS[name]
end
