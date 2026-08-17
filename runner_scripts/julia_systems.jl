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

"Lorenz 96 with cyclic coupling; the swept forcing F is p[1]."
function lorenz96_rhs(u, p, t)
    return SVector(ntuple(
        i -> (u[mod1(i + 1, 40)] - u[mod1(i - 2, 40)]) * u[mod1(i - 1, 40)] -
             u[i] + p[1],
        Val(40)))
end

function lorenz96_rhs!(du, u, p, t)
    du .= lorenz96_rhs(u, p, t)
    return nothing
end

# Uniform state 8 with x1 perturbed to 9.
const LORENZ96_U0_F = SVector{40, Float32}(i == 1 ? 9.0f0 : 8.0f0
                                           for i in 1:40)

# Pleiades masses 2..7; the swept m1 is p[1].
const PLEI_MASS_F = (2.0f0, 3.0f0, 4.0f0, 5.0f0, 6.0f0, 7.0f0)

"Seven-body planar gravitation, u = (x, y, x', y'); the swept m1 is p[1]."
function pleiades_rhs(u, p, t)
    accel = ntuple(Val(7)) do i
        sumx = 0.0f0
        sumy = 0.0f0
        for j in 1:7
            j == i && continue
            mj = j == 1 ? p[1] : PLEI_MASS_F[j - 1]
            rij = (u[i] - u[j])^2 + (u[i + 7] - u[j + 7])^2
            rij32 = rij * sqrt(rij)
            sumx += mj * (u[j] - u[i]) / rij32
            sumy += mj * (u[j + 7] - u[i + 7]) / rij32
        end
        (sumx, sumy)
    end
    return SVector(ntuple(Val(28)) do i
        i <= 14 && return u[i + 14]
        i <= 21 && return accel[i - 14][1]
        return accel[i - 21][2]
    end)
end

function pleiades_rhs!(du, u, p, t)
    du .= pleiades_rhs(u, p, t)
    return nothing
end

const PLEI_U0_F = SVector{28, Float32}(3.0f0, 3.0f0, -1.0f0, -3.0f0, 2.0f0,
    -2.0f0, 2.0f0, 3.0f0, -3.0f0, 2.0f0, 0.0f0, 0.0f0, -4.0f0, 4.0f0,
    0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 1.75f0, -1.5f0,
    0.0f0, 0.0f0, 0.0f0, -1.25f0, 1.0f0, 0.0f0, 0.0f0)

# Pollution problem rate constants k2..k25 (k1 is swept).
const POLLU_K_F = (26.6f0, 1.23f4, 8.6f-4, 8.2f-4, 1.5f4, 1.3f-4, 2.4f4,
    1.65f4, 9.0f3, 2.2f-2, 1.2f4, 1.88f0, 1.63f4, 4.8f6, 3.5f-4, 1.75f-2,
    1.0f8, 4.44f11, 1.24f3, 2.1f0, 5.78f0, 4.74f-2, 1.78f3, 3.12f0)

"Verwer's air pollution mechanism; the swept photolysis rate k1 is p[1]."
function pollu_rhs(u, p, t)
    k = POLLU_K_F
    r1 = p[1] * u[1]
    r2 = k[1] * u[2] * u[4]
    r3 = k[2] * u[5] * u[2]
    r4 = k[3] * u[7]
    r5 = k[4] * u[7]
    r6 = k[5] * u[7] * u[6]
    r7 = k[6] * u[9]
    r8 = k[7] * u[9] * u[6]
    r9 = k[8] * u[11] * u[2]
    r10 = k[9] * u[11] * u[1]
    r11 = k[10] * u[13]
    r12 = k[11] * u[10] * u[2]
    r13 = k[12] * u[14]
    r14 = k[13] * u[1] * u[6]
    r15 = k[14] * u[3]
    r16 = k[15] * u[4]
    r17 = k[16] * u[4]
    r18 = k[17] * u[16]
    r19 = k[18] * u[16]
    r20 = k[19] * u[17] * u[6]
    r21 = k[20] * u[19]
    r22 = k[21] * u[19]
    r23 = k[22] * u[1] * u[4]
    r24 = k[23] * u[19] * u[1]
    r25 = k[24] * u[20]
    return @SVector [
        -r1 - r10 - r14 - r23 - r24 + r2 + r3 + r9 + r11 + r12 + r22 + r25,
        -r2 - r3 - r9 - r12 + r1 + r21,
        -r15 + r1 + r17 + r19 + r22,
        -r2 - r16 - r17 - r23 + r15,
        -r3 + 2.0f0 * r4 + r6 + r7 + r13 + r20,
        -r6 - r8 - r14 - r20 + r3 + 2.0f0 * r18,
        -r4 - r5 - r6 + r13,
        r4 + r5 + r6 + r7,
        -r7 - r8,
        -r12 + r7 + r9,
        -r9 - r10 + r8 + r11,
        r9,
        -r11 + r10,
        -r13 + r12,
        r14,
        -r18 - r19 + r16,
        -r20,
        r20,
        -r21 - r22 - r24 + r23 + r25,
        -r25 + r24,
    ]
end

function pollu_rhs!(du, u, p, t)
    du .= pollu_rhs(u, p, t)
    return nothing
end

const POLLU_U0_F = SVector{20, Float32}(0.0f0, 0.2f0, 0.0f0, 0.04f0, 0.0f0,
    0.0f0, 0.1f0, 0.3f0, 0.01f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0,
    0.0f0, 0.007f0, 0.0f0, 0.0f0, 0.0f0)

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
    "lorenz96" => (rhs = lorenz96_rhs, rhs! = lorenz96_rhs!,
        jac = nothing, tgrad = nothing, u0 = LORENZ96_U0_F,
        mass_matrix = nothing),
    "pleiades" => (rhs = pleiades_rhs, rhs! = pleiades_rhs!,
        jac = nothing, tgrad = nothing, u0 = PLEI_U0_F,
        mass_matrix = nothing),
    "pollu" => (rhs = pollu_rhs, rhs! = pollu_rhs!,
        jac = nothing, tgrad = nothing, u0 = POLLU_U0_F,
        mass_matrix = nothing),
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
