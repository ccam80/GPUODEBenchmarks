# Float64 CPU definitions used for the golden references, one entry per problem.

using LinearAlgebra
using StaticArrays
# The Radau family is not in the slim OrdinaryDiffEq umbrella's default set.
using OrdinaryDiffEqFIRK

"Lorenz with the exact Float64 constants; the swept rho is p[1]."
function lorenz_reference(u, p, t)
    du1 = 10.0 * (u[2] - u[1])
    du2 = u[1] * (p[1] - u[3]) - u[2]
    du3 = u[1] * u[2] - (8.0 / 3.0) * u[3]
    return [du1, du2, du3]
end

# Ring modulator constants (Test Set for IVP Solvers, problem II-3).
const RM_C = 1.6e-8
const RM_CP = 1.0e-8
const RM_LH = 4.45
const RM_LS1 = 0.002
const RM_LS2 = 5.0e-4
const RM_LS3 = 5.0e-4
const RM_GAMMA = 40.67286402e-9
const RM_R = 25000.0
const RM_RP = 50.0
const RM_RG1 = 36.3
const RM_RG2 = 17.3
const RM_RG3 = 17.3
const RM_RI = 50.0
const RM_RC = 600.0
const RM_DELTA = 17.7493332
const RM_U0 = @SVector zeros(15)

rm_q(u) = RM_GAMMA * (exp(RM_DELTA * u) - 1.0)

"Ring modulator right-hand side; `cs` is the capacitance and `amp` the Uin1 amplitude."
function ring_modulator_rhs(u, cs, amp, t)
    uin1 = amp * sin(2000.0 * pi * t)
    uin2 = 2.0 * sin(20000.0 * pi * t)
    ud1 = u[3] - u[5] - u[7] - uin2
    ud2 = -u[4] + u[6] - u[7] - uin2
    ud3 = u[4] + u[5] + u[7] + uin2
    ud4 = -u[3] - u[6] + u[7] + uin2
    q1, q2, q3, q4 = rm_q(ud1), rm_q(ud2), rm_q(ud3), rm_q(ud4)
    # Rows 3 to 6 are divided by cs by the caller, or left as residuals when cs = 0.
    return @SVector [
        (u[8] - 0.5 * u[10] + 0.5 * u[11] + u[14] - u[1] / RM_R) / RM_C,
        (u[9] - 0.5 * u[12] + 0.5 * u[13] + u[15] - u[2] / RM_R) / RM_C,
        u[10] - q1 + q4,
        -u[11] + q2 - q3,
        u[12] + q1 - q3,
        -u[13] - q2 + q4,
        (-u[7] / RM_RP + q1 + q2 - q3 - q4) / RM_CP,
        -u[1] / RM_LH,
        -u[2] / RM_LH,
        (0.5 * u[1] - u[3] - RM_RG2 * u[10]) / RM_LS2,
        (-0.5 * u[1] + u[4] - RM_RG3 * u[11]) / RM_LS3,
        (0.5 * u[2] - u[5] - RM_RG2 * u[12]) / RM_LS2,
        (-0.5 * u[2] + u[6] - RM_RG3 * u[13]) / RM_LS3,
        (-u[1] + uin1 - (RM_RI + RM_RG1) * u[14]) / RM_LS1,
        (-u[2] - (RM_RC + RM_RG1) * u[15]) / RM_LS1,
    ]
end

"Stiff ODE form: the swept Cs is p[1]."
function ring_modulator_reference(u, p, t)
    du = ring_modulator_rhs(u, p[1], 0.5, t)
    scale = @SVector [1.0, 1.0, 1.0 / p[1], 1.0 / p[1], 1.0 / p[1], 1.0 / p[1],
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    return du .* scale
end

"Index-2 form: Cs = 0 leaves rows 3 to 6 as residuals; the swept Uin1 amplitude is p[1]."
function ring_modulator_index2_reference!(du, u, p, t)
    du .= ring_modulator_rhs(u, 0.0, p[1], t)
    return nothing
end

# Rows 3 to 6 carry no derivative in the index-2 form.
const RM_INDEX2_MASS = Diagonal([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

const REFERENCE_SYSTEMS = Dict{String, Any}(
    "lorenz" => (rhs = lorenz_reference, u0 = [1.0, 0.0, 0.0],
        mass_matrix = nothing),
    "ring_modulator" => (rhs = ring_modulator_reference, u0 = RM_U0,
        mass_matrix = nothing),
    # A mass matrix needs the mutating form, so this one keeps a plain vector.
    "ring_modulator_index2" => (rhs = ring_modulator_index2_reference!,
        u0 = zeros(15), mass_matrix = RM_INDEX2_MASS),
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
    name == "RadauIIA9" && return RadauIIA9()
    error("unknown golden algorithm '$(name)'")
end
