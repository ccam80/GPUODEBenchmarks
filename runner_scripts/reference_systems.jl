# Float64 CPU definitions used for the golden references, one entry per problem.

using LinearAlgebra
using StaticArrays
# The Radau family is not in the slim OrdinaryDiffEq umbrella's default set.
using OrdinaryDiffEqFIRK
# DFBDF integrates the fully implicit NAND gate.
using OrdinaryDiffEqBDF
using SciMLBase: ODEProblem, ODEFunction, DAEProblem

"Lorenz with the exact Float64 constants; the swept rho is p[1]."
function lorenz_reference(u, p, t)
    du1 = 10.0 * (u[2] - u[1])
    du2 = u[1] * (p[1] - u[3]) - u[2]
    du3 = u[1] * u[2] - (8.0 / 3.0) * u[3]
    return [du1, du2, du3]
end

"Lorenz 96 with cyclic coupling, sized by u; the swept forcing F is p[1]."
function lorenz96_reference(u, p, t)
    n = length(u)
    return [(u[mod1(i + 1, n)] - u[mod1(i - 2, n)]) * u[mod1(i - 1, n)] -
            u[i] + p[1] for i in 1:n]
end

# Uniform state 8 with x1 perturbed to 9.
lorenz96_u0(n) = [i == 1 ? 9.0 : 8.0 for i in 1:n]

# Pleiades (Test Set for IVP Solvers, celestial mechanics): u = (x, y, x', y').
"Seven-body planar gravitation with masses (m1, 2, ..., 7); the swept m1 is p[1]."
function pleiades_reference(u, p, t)
    du = zeros(28)
    for i in 1:14
        du[i] = u[i + 14]
    end
    for i in 1:7
        sumx = 0.0
        sumy = 0.0
        for j in 1:7
            j == i && continue
            mj = j == 1 ? p[1] : Float64(j)
            rij = (u[i] - u[j])^2 + (u[i + 7] - u[j + 7])^2
            rij32 = rij^1.5
            sumx += mj * (u[j] - u[i]) / rij32
            sumy += mj * (u[j + 7] - u[i + 7]) / rij32
        end
        du[i + 14] = sumx
        du[i + 21] = sumy
    end
    return du
end

const PLEIADES_U0 = [3.0, 3.0, -1.0, -3.0, 2.0, -2.0, 2.0,
    3.0, -3.0, 2.0, 0.0, 0.0, -4.0, 4.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 1.75, -1.5,
    0.0, 0.0, 0.0, -1.25, 1.0, 0.0, 0.0]

# Pollution problem (Test Set for IVP Solvers): 25 reactions over 20 species.
const POLLU_K = (0.35, 26.6, 1.23e4, 8.6e-4, 8.2e-4, 1.5e4, 1.3e-4, 2.4e4,
    1.65e4, 9.0e3, 2.2e-2, 1.2e4, 1.88, 1.63e4, 4.8e6, 3.5e-4, 1.75e-2,
    1.0e8, 4.44e11, 1.24e3, 2.1, 5.78, 4.74e-2, 1.78e3, 3.12)

"Verwer's air pollution mechanism; the swept photolysis rate k1 is p[1]."
function pollu_reference(u, p, t)
    k = POLLU_K
    r1 = p[1] * u[1]
    r2 = k[2] * u[2] * u[4]
    r3 = k[3] * u[5] * u[2]
    r4 = k[4] * u[7]
    r5 = k[5] * u[7]
    r6 = k[6] * u[7] * u[6]
    r7 = k[7] * u[9]
    r8 = k[8] * u[9] * u[6]
    r9 = k[9] * u[11] * u[2]
    r10 = k[10] * u[11] * u[1]
    r11 = k[11] * u[13]
    r12 = k[12] * u[10] * u[2]
    r13 = k[13] * u[14]
    r14 = k[14] * u[1] * u[6]
    r15 = k[15] * u[3]
    r16 = k[16] * u[4]
    r17 = k[17] * u[4]
    r18 = k[18] * u[16]
    r19 = k[19] * u[16]
    r20 = k[20] * u[17] * u[6]
    r21 = k[21] * u[19]
    r22 = k[22] * u[19]
    r23 = k[23] * u[1] * u[4]
    r24 = k[24] * u[19] * u[1]
    r25 = k[25] * u[20]
    return [
        -r1 - r10 - r14 - r23 - r24 + r2 + r3 + r9 + r11 + r12 + r22 + r25,
        -r2 - r3 - r9 - r12 + r1 + r21,
        -r15 + r1 + r17 + r19 + r22,
        -r2 - r16 - r17 - r23 + r15,
        -r3 + 2.0 * r4 + r6 + r7 + r13 + r20,
        -r6 - r8 - r14 - r20 + r3 + 2.0 * r18,
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

const POLLU_U0 = [0.0, 0.2, 0.0, 0.04, 0.0, 0.0, 0.1, 0.3, 0.01, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.007, 0.0, 0.0, 0.0]

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

# NAND gate (Test Set for IVP Solvers): C(y) y' = f(y, t), index-0 IDE.
const NAND_RGS = 4.0
const NAND_RGD = 4.0
const NAND_RBS = 10.0
const NAND_RBD = 10.0
const NAND_CGS = 0.6e-4
const NAND_CGD = 0.6e-4
const NAND_CBD = 2.4e-5
const NAND_C9 = 0.5e-4
const NAND_DELTA = 0.2e-1
const NAND_CURIS = 1.0e-14
const NAND_VTH = 25.85
const NAND_VBB = -2.5
const NAND_PHIB = 0.87

"Trapezoidal test-set input signal; returns the voltage and its derivative."
function nand_pulse(t, delay, t1, t2, t3, period)
    low, high = 0.0, 5.0
    time = mod(t, period)
    if time > delay + t1 + t2 + t3
        return low, 0.0
    elseif time > delay + t1 + t2
        return ((high - low) / t3) * (delay + t1 + t2 + t3 - time) + low,
        -(high - low) / t3
    elseif time > delay + t1
        return high, 0.0
    elseif time > delay
        return ((high - low) / t1) * (time - delay) + low, (high - low) / t1
    else
        return low, 0.0
    end
end

"Bulk junction capacitance CBD(V) = CBS(V)."
nand_cbdbs(v) = v <= 0.0 ? NAND_CBD / sqrt(1.0 - v / NAND_PHIB) :
                NAND_CBD * (1.0 + v / (2.0 * NAND_PHIB))

"Bulk junction current IBD(V) = IBS(V)."
nand_ibdbs(v) = v <= 0.0 ? -NAND_CURIS * (exp(v / NAND_VTH) - 1.0) : zero(v)

"Shichman-Hodges drain current for VDS > 0; ned 1 is depletion, 2 enhancement."
function nand_gdsp(ned, vds, vgs, vbs)
    vt0, cgamma, phi, beta = ned == 1 ? (-2.43, 0.2, 1.28, 5.35e-4) :
                             (0.2, 0.035, 1.01, 1.748e-3)
    # Outside the sqrt domain: return NaN so the solver rejects the step.
    phi - vbs < 0.0 && return oftype(vbs, NaN)
    vte = vt0 + cgamma * (sqrt(phi - vbs) - sqrt(phi))
    vgs - vte <= 0.0 && return zero(vte)
    vgs - vte <= vds && return -beta * (vgs - vte)^2 * (1.0 + NAND_DELTA * vds)
    return -beta * vds * (2.0 * (vgs - vte) - vds) * (1.0 + NAND_DELTA * vds)
end

"Shichman-Hodges drain current for VDS < 0; nand.f uses beta 1.748e-4 here."
function nand_gdsm(ned, vds, vgd, vbd)
    vt0, cgamma, phi, beta = ned == 1 ? (-2.43, 0.2, 1.28, 5.35e-4) :
                             (0.2, 0.035, 1.01, 1.748e-4)
    # Outside the sqrt domain: return NaN so the solver rejects the step.
    phi - vbd < 0.0 && return oftype(vbd, NaN)
    vte = vt0 + cgamma * (sqrt(phi - vbd) - sqrt(phi))
    vgd - vte <= 0.0 && return zero(vte)
    vgd - vte <= -vds && return beta * (vgd - vte)^2 * (1.0 - NAND_DELTA * vds)
    return -beta * vds * (2.0 * (vgd - vte) + vds) * (1.0 - NAND_DELTA * vds)
end

function nand_ids(ned, vds, vgs, vbs, vgd, vbd)
    vds > 0.0 && return nand_gdsp(ned, vds, vgs, vbs)
    vds < 0.0 && return nand_gdsm(ned, vds, vgd, vbd)
    return zero(vds)
end

"Right-hand side f(y, t) of the NAND network equation; the swept VDD is p[1]."
function nand_rhs(u, p, t)
    vdd = p[1]
    v1, v1d = nand_pulse(t, 5.0, 5.0, 5.0, 5.0, 20.0)
    v2, v2d = nand_pulse(t, 15.0, 5.0, 15.0, 5.0, 40.0)
    ids1 = nand_ids(1, u[2] - u[1], u[5] - u[1], u[3] - u[5], u[5] - u[2],
        u[4] - vdd)
    ids2 = nand_ids(2, u[7] - u[6], v1 - u[6], u[8] - u[10], v1 - u[7],
        u[9] - u[5])
    ids3 = nand_ids(2, u[12] - u[11], v2 - u[11], u[13], v2 - u[12],
        u[14] - u[10])
    return [
        -(u[1] - u[5]) / NAND_RGS - ids1,
        -(u[2] - vdd) / NAND_RGD + ids1,
        -(u[3] - NAND_VBB) / NAND_RBS + nand_ibdbs(u[3] - u[5]),
        -(u[4] - NAND_VBB) / NAND_RBD + nand_ibdbs(u[4] - vdd),
        -(u[5] - u[1]) / NAND_RGS - nand_ibdbs(u[3] - u[5]) -
        (u[5] - u[7]) / NAND_RGD - nand_ibdbs(u[9] - u[5]),
        NAND_CGS * v1d - (u[6] - u[10]) / NAND_RGS - ids2,
        NAND_CGD * v1d - (u[7] - u[5]) / NAND_RGD + ids2,
        -(u[8] - NAND_VBB) / NAND_RBS + nand_ibdbs(u[8] - u[10]),
        -(u[9] - NAND_VBB) / NAND_RBD + nand_ibdbs(u[9] - u[5]),
        -(u[10] - u[6]) / NAND_RGS - nand_ibdbs(u[8] - u[10]) -
        (u[10] - u[12]) / NAND_RGD - nand_ibdbs(u[14] - u[10]),
        NAND_CGS * v2d - u[11] / NAND_RGS - ids3,
        NAND_CGD * v2d - (u[12] - u[10]) / NAND_RGD + ids3,
        -(u[13] - NAND_VBB) / NAND_RBS + nand_ibdbs(u[13]),
        -(u[14] - NAND_VBB) / NAND_RBD + nand_ibdbs(u[14] - u[10]),
    ]
end

"Voltage-dependent capacitance matrix C(y); the swept VDD is p[1]."
function nand_capacitance(u, p)
    vdd = p[1]
    c = zeros(eltype(u), 14, 14)
    cb35 = nand_cbdbs(u[3] - u[5])
    cb95 = nand_cbdbs(u[9] - u[5])
    cb810 = nand_cbdbs(u[8] - u[10])
    cb1410 = nand_cbdbs(u[14] - u[10])
    c[1, 1] = NAND_CGS
    c[1, 5] = -NAND_CGS
    c[2, 2] = NAND_CGD
    c[2, 5] = -NAND_CGD
    c[3, 3] = cb35
    c[3, 5] = -cb35
    c[4, 4] = nand_cbdbs(u[4] - vdd)
    c[5, 1] = -NAND_CGS
    c[5, 2] = -NAND_CGD
    c[5, 3] = -cb35
    c[5, 5] = NAND_CGS + NAND_CGD + cb35 + cb95 + NAND_C9
    c[5, 9] = -cb95
    c[6, 6] = NAND_CGS
    c[7, 7] = NAND_CGD
    c[8, 8] = cb810
    c[8, 10] = -cb810
    c[9, 5] = -cb95
    c[9, 9] = cb95
    c[10, 8] = -cb810
    c[10, 10] = cb810 + cb1410 + NAND_C9
    c[10, 14] = -cb1410
    c[11, 11] = NAND_CGS
    c[12, 12] = NAND_CGD
    c[13, 13] = nand_cbdbs(u[13])
    c[14, 10] = -cb1410
    c[14, 14] = cb1410
    return c
end

"Fully implicit residual C(y) y' - f(y, t) for DFBDF."
function nand_residual!(res, du, u, p, t)
    res .= nand_capacitance(u, p) * du .- nand_rhs(u, p, t)
    return nothing
end

"Consistent y'(0) for a swept VDD: C(y0) y0' = f(y0, 0)."
nand_du0(p) = nand_capacitance(NAND_U0, p) \ nand_rhs(NAND_U0, p, 0.0)

const NAND_U0 = [5.0, 5.0, NAND_VBB, NAND_VBB, 5.0, 3.62385, 5.0, NAND_VBB,
    NAND_VBB, 3.62385, 0.0, 3.62385, NAND_VBB, NAND_VBB]

const REFERENCE_SYSTEMS = Dict{String, Any}(
    "lorenz" => (rhs = lorenz_reference, u0 = [1.0, 0.0, 0.0],
        mass_matrix = nothing),
    "lorenz96" => (rhs = lorenz96_reference, u0 = lorenz96_u0(40),
        mass_matrix = nothing),
    "lorenz96_20" => (rhs = lorenz96_reference, u0 = lorenz96_u0(20),
        mass_matrix = nothing),
    "pleiades" => (rhs = pleiades_reference, u0 = PLEIADES_U0,
        mass_matrix = nothing),
    "pollu" => (rhs = pollu_reference, u0 = POLLU_U0, mass_matrix = nothing),
    "ring_modulator" => (rhs = ring_modulator_reference, u0 = RM_U0,
        mass_matrix = nothing),
    # A mass matrix needs the mutating form, so this one keeps a plain vector.
    "ring_modulator_index2" => (rhs = ring_modulator_index2_reference!,
        u0 = zeros(15), mass_matrix = RM_INDEX2_MASS),
    # Fully implicit; only the golden generator integrates it.
    "nand_gate" => (residual = nand_residual!, u0 = NAND_U0, du0 = nand_du0,
        tstops = collect(0.0:5.0:80.0), mass_matrix = nothing),
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
    name == "DFBDF" && return DFBDF()
    error("unknown golden algorithm '$(name)'")
end

"Reference problem instance for one swept value; ODE or fully implicit DAE."
function reference_problem(system, problem, p)
    tspan = (0.0, problem["duration"])
    if haskey(system, :residual)
        return DAEProblem(system.residual, system.du0([p]), system.u0, tspan,
            [p]; differential_vars = trues(length(system.u0)))
    end
    f = system.mass_matrix === nothing ? ODEFunction(system.rhs) :
        ODEFunction(system.rhs; mass_matrix = system.mass_matrix)
    return ODEProblem(f, system.u0, tspan, [p])
end

"Extra solve kwargs a reference needs, e.g. pulse-corner tstops."
reference_solve_kwargs(system) = haskey(system, :tstops) ?
                                 (; tstops = system.tstops) : (;)
