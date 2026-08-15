# DiffEqGPU system definitions: an out-of-place SVector RHS and Float32 initial state per problem, with the swept scalar in p[1].

using LinearAlgebra
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

# Ring modulator constants (Test Set for IVP Solvers, problem II-3).
const RM_C_F = 1.6f-8
const RM_CP_F = 1.0f-8
const RM_LH_F = 4.45f0
const RM_LS1_F = 0.002f0
const RM_LS2_F = 5.0f-4
const RM_LS3_F = 5.0f-4
const RM_GAMMA_F = 40.67286402f-9
const RM_R_F = 25000.0f0
const RM_RP_F = 50.0f0
const RM_RG1_F = 36.3f0
const RM_RG2_F = 17.3f0
const RM_RG3_F = 17.3f0
const RM_RI_F = 50.0f0
const RM_RC_F = 600.0f0
const RM_DELTA_F = 17.7493332f0
const RM_W1_F = 6283.1855f0
const RM_W2_F = 62831.855f0
const RM_U0_F = @SVector zeros(Float32, 15)

rm_q_f(u) = RM_GAMMA_F * (exp(RM_DELTA_F * u) - 1.0f0)

"Ring modulator right-hand side; `cs` is the capacitance and `amp` the Uin1 amplitude."
function ring_modulator_rhs_f(u, cs, amp, t)
    uin1 = amp * sin(RM_W1_F * t)
    uin2 = 2.0f0 * sin(RM_W2_F * t)
    ud1 = u[3] - u[5] - u[7] - uin2
    ud2 = -u[4] + u[6] - u[7] - uin2
    ud3 = u[4] + u[5] + u[7] + uin2
    ud4 = -u[3] - u[6] + u[7] + uin2
    q1, q2, q3, q4 = rm_q_f(ud1), rm_q_f(ud2), rm_q_f(ud3), rm_q_f(ud4)
    return @SVector [
        (u[8] - 0.5f0 * u[10] + 0.5f0 * u[11] + u[14] - u[1] / RM_R_F) / RM_C_F,
        (u[9] - 0.5f0 * u[12] + 0.5f0 * u[13] + u[15] - u[2] / RM_R_F) / RM_C_F,
        (u[10] - q1 + q4) / cs,
        (-u[11] + q2 - q3) / cs,
        (u[12] + q1 - q3) / cs,
        (-u[13] - q2 + q4) / cs,
        (-u[7] / RM_RP_F + q1 + q2 - q3 - q4) / RM_CP_F,
        -u[1] / RM_LH_F,
        -u[2] / RM_LH_F,
        (0.5f0 * u[1] - u[3] - RM_RG2_F * u[10]) / RM_LS2_F,
        (-0.5f0 * u[1] + u[4] - RM_RG3_F * u[11]) / RM_LS3_F,
        (0.5f0 * u[2] - u[5] - RM_RG2_F * u[12]) / RM_LS2_F,
        (-0.5f0 * u[2] + u[6] - RM_RG3_F * u[13]) / RM_LS3_F,
        (-u[1] + uin1 - (RM_RI_F + RM_RG1_F) * u[14]) / RM_LS1_F,
        (-u[2] - (RM_RC_F + RM_RG1_F) * u[15]) / RM_LS1_F,
    ]
end

"Stiff ODE form: the swept Cs is p[1]."
ring_modulator_rhs(u, p, t) = ring_modulator_rhs_f(u, p[1], 0.5f0, t)

function ring_modulator_rhs!(du, u, p, t)
    du .= ring_modulator_rhs_f(u, p[1], 0.5f0, t)
    return nothing
end

# Rows 3 to 6 carry no derivative when Cs = 0; the swept Uin1 amplitude is p[1].
ring_modulator_index2_rhs(u, p, t) = ring_modulator_rhs_f(u, 1.0f0, p[1], t)

function ring_modulator_index2_rhs!(du, u, p, t)
    du .= ring_modulator_rhs_f(u, 1.0f0, p[1], t)
    return nothing
end

# Static so a problem carrying it stays isbits for cu(probs).
const RM_INDEX2_MASS_F = Diagonal(SVector{15, Float32}(1, 1, 0, 0, 0, 0, 1, 1,
    1, 1, 1, 1, 1, 1, 1))

const JULIA_SYSTEMS = Dict{String, Any}(
    "lorenz" => (rhs = lorenz_rhs, rhs! = lorenz_rhs!,
        jac = lorenz_jac, tgrad = lorenz_tgrad,
        u0 = @SVector([1.0f0, 0.0f0, 0.0f0]), mass_matrix = nothing),
    "ring_modulator" => (rhs = ring_modulator_rhs, rhs! = ring_modulator_rhs!,
        jac = nothing, tgrad = nothing, u0 = RM_U0_F, mass_matrix = nothing),
    "ring_modulator_index2" => (rhs = ring_modulator_index2_rhs,
        rhs! = ring_modulator_index2_rhs!, jac = nothing, tgrad = nothing,
        u0 = RM_U0_F, mass_matrix = RM_INDEX2_MASS_F),
)

"Right-hand sides and initial state for a problem row or name."
function julia_system(problem)
    name = problem isa AbstractDict ? problem["problem"] : problem
    haskey(JULIA_SYSTEMS, name) ||
        error("no DiffEqGPU definition for problem '$(name)'")
    return JULIA_SYSTEMS[name]
end
