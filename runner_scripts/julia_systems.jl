# ModelingToolkit system definitions, one per problem, compiled and code-generated in a chosen element type; include problems.jl first (the lorenz96 builders read their default size from it).

using LinearAlgebra
using StaticArrays
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using SciMLBase: ODEFunction, ODEProblem

"Unknown-order values for a symbolic->value map in T; absent unknowns get zero."
function _ordered_values(sys, valmap, ::Type{T}) where {T}
    entries = collect(valmap)
    return [begin
        i = findfirst(pr -> isequal(pr.first, v), entries)
        i === nothing ? zero(T) : T(entries[i].second)
    end for v in unknowns(sys)]
end

"Position of each golden variable in the compiled system's unknown order."
function _golden_index(sys, golden_vars)
    us = unknowns(sys)
    return [findfirst(u -> isequal(v, u), us) for v in golden_vars]
end

# Codegen options for generate_*; GPU_ODE_JuliaKernels overrides them.
SYSTEMS_CODEGEN = (expression = Val(false),)

"Compile a raw system and generate the numeric artifacts every suite uses, in element type T."
function _build_entry(raw, ::Type{T}; u0map, golden_vars, consistent_u0 = false) where {T}
    sys = mtkcompile(raw; split = false)
    n = length(unknowns(sys))
    rhs, rhs! = ModelingToolkit.generate_rhs(sys; SYSTEMS_CODEGEN...)
    jac, jac! = ModelingToolkit.generate_jacobian(sys; SYSTEMS_CODEGEN...)
    # A piecewise input pulse can defeat the symbolic time derivative; nothing means finite differences.
    tgrad = try
        ModelingToolkit.generate_tgrad(sys; SYSTEMS_CODEGEN...)[1]
    catch
        nothing
    end
    mm = calculate_massmatrix(sys)
    mass_matrix = mm === I ? nothing : SMatrix{n, n, T}(T.(mm))
    u0 = SVector{n, T}(_ordered_values(sys, u0map, T))
    index = _golden_index(sys, golden_vars)
    any(isnothing, index) && error("golden variable missing from unknowns")
    u0_for = consistent_u0 ?
             _consistent_u0(rhs, jac, u0, findall(iszero, diag(mm)), n, T) :
             (p -> u0)
    return (sys = sys, n = n, rhs = rhs, rhs! = rhs!, jac = jac, jac! = jac!,
        tgrad = tgrad, mass_matrix = mass_matrix, u0 = u0, u0_for = u0_for,
        golden_index = SVector{length(index), Int}(index))
end

"Per-parameter u0 with the algebraic unknowns solved by one Newton step in Float64; the algebraic rows are linear in them."
function _consistent_u0(rhs, jac, u0, zidx, n, ::Type{T}) where {T}
    function u0_for(pval)
        p = SVector{1, Float64}(Float64(pval))
        u = SVector{n, Float64}(u0)
        A = Matrix(jac(u, p, 0.0))[zidx, zidx]
        g = Vector(rhs(u, p, 0.0))[zidx]
        base = Vector{Float64}(u)
        base[zidx] .-= A \ g
        return SVector{n, T}(T.(base))
    end
    return u0_for
end

# --- lorenz ----------------------------------------------------------------
function _lorenz_entry(::Type{T}) where {T}
    @parameters rho = T(21)
    @variables x(t) y(t) z(t)
    eqs = [D(x) ~ T(10) * (y - x),
        D(y) ~ x * (rho - z) - y,
        D(z) ~ x * y - (T(8) / T(3)) * z]
    @named lorenz = System(eqs, t)
    return _build_entry(lorenz, T;
        u0map = [x => T(1), y => zero(T), z => zero(T)],
        golden_vars = [x, y, z])
end

# --- lorenz 96 -------------------------------------------------------------
# n is the state count, from the problem row or the states-sweep grid.
function _lorenz96_entry(::Type{T}, n) where {T}
    @parameters F = T(8)
    @variables (x(t))[1:n]
    xs = collect(x)
    eqs = [D(xs[i]) ~ (xs[mod1(i + 1, n)] - xs[mod1(i - 2, n)]) *
                      xs[mod1(i - 1, n)] - xs[i] + F for i in 1:n]
    @named lorenz96 = System(eqs, t)
    return _build_entry(lorenz96, T;
        u0map = [xs[i] => (i == 1 ? T(9) : T(8)) for i in 1:n],
        golden_vars = xs)
end

# --- pleiades --------------------------------------------------------------
function _pleiades_entry(::Type{T}) where {T}
    @parameters m1 = T(1)
    @variables (x(t))[1:7] (y(t))[1:7] (u(t))[1:7] (v(t))[1:7]
    xs, ys, us, vs = collect(x), collect(y), collect(u), collect(v)
    masses = [i == 1 ? m1 : T(i) for i in 1:7]
    eqs = Equation[]
    for i in 1:7
        push!(eqs, D(xs[i]) ~ us[i])
        push!(eqs, D(ys[i]) ~ vs[i])
    end
    for i in 1:7
        sumx = zero(T)
        sumy = zero(T)
        for j in 1:7
            j == i && continue
            rij = (xs[i] - xs[j])^2 + (ys[i] - ys[j])^2
            rij32 = rij * sqrt(rij)
            sumx += masses[j] * (xs[j] - xs[i]) / rij32
            sumy += masses[j] * (ys[j] - ys[i]) / rij32
        end
        push!(eqs, D(us[i]) ~ sumx)
        push!(eqs, D(vs[i]) ~ sumy)
    end
    @named pleiades = System(eqs, t)
    x0 = T[3, 3, -1, -3, 2, -2, 2]
    y0 = T[3, -3, 2, 0, 0, -4, 4]
    u0 = T[0, 0, 0, 0, 0, 1.75, -1.5]
    v0 = T[0, 0, 0, -1.25, 1, 0, 0]
    u0map = vcat([xs[i] => x0[i] for i in 1:7], [ys[i] => y0[i] for i in 1:7],
        [us[i] => u0[i] for i in 1:7], [vs[i] => v0[i] for i in 1:7])
    return _build_entry(pleiades, T; u0map = u0map,
        golden_vars = vcat(xs, ys, us, vs))
end

# --- pollution problem -----------------------------------------------------
function _pollu_entry(::Type{T}) where {T}
    @parameters k1 = T(0.35)
    @variables (y(t))[1:20]
    ys = collect(y)
    k = T[0, 26.6, 1.23e4, 8.6e-4, 8.2e-4, 1.5e4, 1.3e-4, 2.4e4,
        1.65e4, 9.0e3, 2.2e-2, 1.2e4, 1.88, 1.63e4, 4.8e6, 3.5e-4, 1.75e-2,
        1.0e8, 4.44e11, 1.24e3, 2.1, 5.78, 4.74e-2, 1.78e3, 3.12]
    r1 = k1 * ys[1]
    r2 = k[2] * ys[2] * ys[4]
    r3 = k[3] * ys[5] * ys[2]
    r4 = k[4] * ys[7]
    r5 = k[5] * ys[7]
    r6 = k[6] * ys[7] * ys[6]
    r7 = k[7] * ys[9]
    r8 = k[8] * ys[9] * ys[6]
    r9 = k[9] * ys[11] * ys[2]
    r10 = k[10] * ys[11] * ys[1]
    r11 = k[11] * ys[13]
    r12 = k[12] * ys[10] * ys[2]
    r13 = k[13] * ys[14]
    r14 = k[14] * ys[1] * ys[6]
    r15 = k[15] * ys[3]
    r16 = k[16] * ys[4]
    r17 = k[17] * ys[4]
    r18 = k[18] * ys[16]
    r19 = k[19] * ys[16]
    r20 = k[20] * ys[17] * ys[6]
    r21 = k[21] * ys[19]
    r22 = k[22] * ys[19]
    r23 = k[23] * ys[1] * ys[4]
    r24 = k[24] * ys[19] * ys[1]
    r25 = k[25] * ys[20]
    two = T(2)
    eqs = [
        D(ys[1]) ~ -r1 - r10 - r14 - r23 - r24 + r2 + r3 + r9 + r11 + r12 +
                   r22 + r25,
        D(ys[2]) ~ -r2 - r3 - r9 - r12 + r1 + r21,
        D(ys[3]) ~ -r15 + r1 + r17 + r19 + r22,
        D(ys[4]) ~ -r2 - r16 - r17 - r23 + r15,
        D(ys[5]) ~ -r3 + two * r4 + r6 + r7 + r13 + r20,
        D(ys[6]) ~ -r6 - r8 - r14 - r20 + r3 + two * r18,
        D(ys[7]) ~ -r4 - r5 - r6 + r13,
        D(ys[8]) ~ r4 + r5 + r6 + r7,
        D(ys[9]) ~ -r7 - r8,
        D(ys[10]) ~ -r12 + r7 + r9,
        D(ys[11]) ~ -r9 - r10 + r8 + r11,
        D(ys[12]) ~ r9,
        D(ys[13]) ~ -r11 + r10,
        D(ys[14]) ~ -r13 + r12,
        D(ys[15]) ~ r14,
        D(ys[16]) ~ -r18 - r19 + r16,
        D(ys[17]) ~ -r20,
        D(ys[18]) ~ r20,
        D(ys[19]) ~ -r21 - r22 - r24 + r23 + r25,
        D(ys[20]) ~ -r25 + r24,
    ]
    @named pollu = System(eqs, t)
    u0vals = zeros(T, 20)
    u0vals[2] = T(0.2)
    u0vals[4] = T(0.04)
    u0vals[7] = T(0.1)
    u0vals[8] = T(0.3)
    u0vals[9] = T(0.01)
    u0vals[17] = T(0.007)
    return _build_entry(pollu, T;
        u0map = [ys[i] => u0vals[i] for i in 1:20], golden_vars = ys)
end

# --- ring modulator (Test Set for IVP Solvers, problem II-3) ---------------
_rm_q(::Type{T}, v) where {T} = T(40.67286402e-9) * (exp(T(17.7493332) * v) - one(T))

"Equations and variables in T; `cs` and `amp` are symbolic or literal."
function _ring_modulator_equations(::Type{T}, cs, amp) where {T}
    @variables U1(t) U2(t) U3(t) U4(t) U5(t) U6(t) U7(t)
    @variables I1(t) I2(t) I3(t) I4(t) I5(t) I6(t) I7(t) I8(t)
    C = T(1.6e-8)
    Cp = T(1.0e-8)
    Lh = T(4.45)
    Ls1 = T(0.002)
    Ls2 = T(5.0e-4)
    Ls3 = T(5.0e-4)
    R = T(25000)
    Rp = T(50)
    Rg1 = T(36.3)
    Rg2 = T(17.3)
    Rg3 = T(17.3)
    Ri = T(50)
    Rc = T(600)
    w1 = T(2000 * pi)
    w2 = T(20000 * pi)
    half = T(0.5)
    Uin1 = amp * sin(w1 * t)
    Uin2 = T(2) * sin(w2 * t)
    UD1 = U3 - U5 - U7 - Uin2
    UD2 = -U4 + U6 - U7 - Uin2
    UD3 = U4 + U5 + U7 + Uin2
    UD4 = -U3 - U6 + U7 + Uin2
    q1, q2, q3, q4 = _rm_q(T, UD1), _rm_q(T, UD2), _rm_q(T, UD3), _rm_q(T, UD4)
    eqs = [
        C * D(U1) ~ I1 - half * I3 + half * I4 + I7 - U1 / R,
        C * D(U2) ~ I2 - half * I5 + half * I6 + I8 - U2 / R,
        cs * D(U3) ~ I3 - q1 + q4,
        cs * D(U4) ~ -I4 + q2 - q3,
        cs * D(U5) ~ I5 + q1 - q3,
        cs * D(U6) ~ -I6 - q2 + q4,
        Cp * D(U7) ~ -U7 / Rp + q1 + q2 - q3 - q4,
        Lh * D(I1) ~ -U1,
        Lh * D(I2) ~ -U2,
        Ls2 * D(I3) ~ half * U1 - U3 - Rg2 * I3,
        Ls3 * D(I4) ~ -half * U1 + U4 - Rg3 * I4,
        Ls2 * D(I5) ~ half * U2 - U5 - Rg2 * I5,
        Ls3 * D(I6) ~ -half * U2 + U6 - Rg3 * I6,
        Ls1 * D(I7) ~ -U1 + Uin1 - (Ri + Rg1) * I7,
        Ls1 * D(I8) ~ -U2 - (Rc + Rg1) * I8,
    ]
    return eqs, [U1, U2, U3, U4, U5, U6, U7, I1, I2, I3, I4, I5, I6, I7, I8]
end

"Stiff form: the swept capacitance Cs stays a parameter."
function _ring_modulator_entry(::Type{T}) where {T}
    @parameters Cs = T(2.0e-12)
    eqs, vars = _ring_modulator_equations(T, Cs, T(0.5))
    @named ring_modulator = System(eqs, t)
    return _build_entry(ring_modulator, T;
        u0map = [v => zero(T) for v in vars], golden_vars = vars)
end

"Index-2 form: Cs = 0 substituted at definition, the Uin1 amplitude swept."
function _ring_modulator_index2_entry(::Type{T}) where {T}
    @parameters Uin1_amplitude = T(0.5)
    eqs, vars = _ring_modulator_equations(T, zero(T), Uin1_amplitude)
    @named ring_modulator_index2 = System(eqs, t)
    return _build_entry(ring_modulator_index2, T;
        u0map = [v => zero(T) for v in vars], golden_vars = vars)
end

# --- NAND gate (Test Set for IVP Solvers) ----------------------------------
"Trapezoidal test-set input pulse and its time derivative."
function _nand_pulse(::Type{T}, delay, t1, t2, t3, period) where {T}
    tp = t - period * floor(t / period)
    hi = T(5)
    z = zero(T)
    vin = ifelse(tp > delay + t1 + t2 + t3, z,
        ifelse(tp > delay + t1 + t2, (hi / t3) * (delay + t1 + t2 + t3 - tp),
            ifelse(tp > delay + t1, hi,
                ifelse(tp > delay, (hi / t1) * (tp - delay), z))))
    vind = ifelse(tp > delay + t1 + t2 + t3, z,
        ifelse(tp > delay + t1 + t2, -hi / t3,
            ifelse(tp > delay + t1, z,
                ifelse(tp > delay, hi / t1, z))))
    return vin, vind
end

_nand_cb(::Type{T}, v) where {T} = ifelse(v <= zero(T),
    T(2.4e-5) / sqrt(one(T) - v / T(0.87)),
    T(2.4e-5) * (one(T) + v / (T(2) * T(0.87))))
_nand_ibs(::Type{T}, v) where {T} = ifelse(v <= zero(T),
    -T(1.0e-14) * (exp(v / T(25.85)) - one(T)), zero(T))

function _nand_gdsp(::Type{T}, vds, vgs, vbs, vt0, cgamma, phi, beta) where {T}
    vte = vt0 + cgamma * (sqrt(phi - vbs) - sqrt(phi))
    return ifelse(vgs - vte <= zero(T), zero(T),
        ifelse(vgs - vte <= vds,
            -beta * (vgs - vte)^2 * (one(T) + T(0.02) * vds),
            -beta * vds * (T(2) * (vgs - vte) - vds) *
            (one(T) + T(0.02) * vds)))
end

function _nand_gdsm(::Type{T}, vds, vgd, vbd, vt0, cgamma, phi, beta) where {T}
    vte = vt0 + cgamma * (sqrt(phi - vbd) - sqrt(phi))
    return ifelse(vgd - vte <= zero(T), zero(T),
        ifelse(vgd - vte <= -vds,
            beta * (vgd - vte)^2 * (one(T) - T(0.02) * vds),
            -beta * vds * (T(2) * (vgd - vte) + vds) *
            (one(T) - T(0.02) * vds)))
end

# nand.f: depletion beta 5.35e-4 in both branches; enhancement 1.748e-3 / 1.748e-4.
_nand_ids1(::Type{T}, vds, vgs, vbs, vgd, vbd) where {T} = ifelse(vds > zero(T),
    _nand_gdsp(T, vds, vgs, vbs, T(-2.43), T(0.2), T(1.28), T(5.35e-4)),
    ifelse(vds < zero(T),
        _nand_gdsm(T, vds, vgd, vbd, T(-2.43), T(0.2), T(1.28), T(5.35e-4)), zero(T)))
_nand_ids2(::Type{T}, vds, vgs, vbs, vgd, vbd) where {T} = ifelse(vds > zero(T),
    _nand_gdsp(T, vds, vgs, vbs, T(0.2), T(0.035), T(1.01), T(1.748e-3)),
    ifelse(vds < zero(T),
        _nand_gdsm(T, vds, vgd, vbd, T(0.2), T(0.035), T(1.01), T(1.748e-4)), zero(T)))

function _nand_gate_entry(::Type{T}) where {T}
    @parameters c9 = T(0.5e-4)
    @variables (y(t))[1:14]
    ys = collect(y)
    VDD = T(5)
    rgs = T(4)
    rgd = T(4)
    rbs = T(10)
    rbd = T(10)
    cgs = T(0.6e-4)
    cgd = T(0.6e-4)
    vbb = T(-2.5)
    v1, v1d = _nand_pulse(T, T(5), T(5), T(5), T(5), T(20))
    v2, v2d = _nand_pulse(T, T(15), T(5), T(15), T(5), T(40))
    i1 = _nand_ids1(T, ys[2] - ys[1], ys[5] - ys[1], ys[3] - ys[5],
        ys[5] - ys[2], ys[4] - VDD)
    i2 = _nand_ids2(T, ys[7] - ys[6], v1 - ys[6], ys[8] - ys[10], v1 - ys[7],
        ys[9] - ys[5])
    i3 = _nand_ids2(T, ys[12] - ys[11], v2 - ys[11], ys[13], v2 - ys[12],
        ys[14] - ys[10])
    cb35 = _nand_cb(T, ys[3] - ys[5])
    cb4 = _nand_cb(T, ys[4] - VDD)
    cb95 = _nand_cb(T, ys[9] - ys[5])
    cb810 = _nand_cb(T, ys[8] - ys[10])
    cb13 = _nand_cb(T, ys[13])
    cb1410 = _nand_cb(T, ys[14] - ys[10])
    eqs = [
        cgs * D(ys[1]) - cgs * D(ys[5]) ~ -(ys[1] - ys[5]) / rgs - i1,
        cgd * D(ys[2]) - cgd * D(ys[5]) ~ -(ys[2] - VDD) / rgd + i1,
        cb35 * D(ys[3]) - cb35 * D(ys[5]) ~ -(ys[3] - vbb) / rbs +
                                            _nand_ibs(T, ys[3] - ys[5]),
        cb4 * D(ys[4]) ~ -(ys[4] - vbb) / rbd + _nand_ibs(T, ys[4] - VDD),
        -cgs * D(ys[1]) - cgd * D(ys[2]) - cb35 * D(ys[3]) +
        (cgs + cgd + cb35 + cb95 + c9) * D(ys[5]) - cb95 * D(ys[9]) ~
            -(ys[5] - ys[1]) / rgs - _nand_ibs(T, ys[3] - ys[5]) -
            (ys[5] - ys[7]) / rgd - _nand_ibs(T, ys[9] - ys[5]),
        cgs * D(ys[6]) ~ cgs * v1d - (ys[6] - ys[10]) / rgs - i2,
        cgd * D(ys[7]) ~ cgd * v1d - (ys[7] - ys[5]) / rgd + i2,
        cb810 * D(ys[8]) - cb810 * D(ys[10]) ~ -(ys[8] - vbb) / rbs +
                                               _nand_ibs(T, ys[8] - ys[10]),
        -cb95 * D(ys[5]) + cb95 * D(ys[9]) ~ -(ys[9] - vbb) / rbd +
                                             _nand_ibs(T, ys[9] - ys[5]),
        -cb810 * D(ys[8]) + (cb810 + cb1410 + c9) * D(ys[10]) -
        cb1410 * D(ys[14]) ~
            -(ys[10] - ys[6]) / rgs - _nand_ibs(T, ys[8] - ys[10]) -
            (ys[10] - ys[12]) / rgd - _nand_ibs(T, ys[14] - ys[10]),
        cgs * D(ys[11]) ~ cgs * v2d - ys[11] / rgs - i3,
        cgd * D(ys[12]) ~ cgd * v2d - (ys[12] - ys[10]) / rgd + i3,
        cb13 * D(ys[13]) ~ -(ys[13] - vbb) / rbs + _nand_ibs(T, ys[13]),
        -cb1410 * D(ys[10]) + cb1410 * D(ys[14]) ~ -(ys[14] - vbb) / rbd +
                                                   _nand_ibs(T, ys[14] - ys[10]),
    ]
    @named nand_gate = System(eqs, t)
    u0vals = T[5, 5, -2.5, -2.5, 5, 3.62385, 5, -2.5, -2.5, 3.62385,
        0, 3.62385, -2.5, -2.5]
    return _build_entry(nand_gate, T;
        u0map = [ys[i] => u0vals[i] for i in 1:14], golden_vars = ys,
        consistent_u0 = true)
end

# Builders by problem name, each taking the element type.
const _ENTRY_BUILDERS = Dict{String, Function}(
    "lorenz" => _lorenz_entry,
    "lorenz96" => T -> _lorenz96_entry(T, get_problem("lorenz96")["states"]),
    "lorenz96_20" => T -> _lorenz96_entry(T, get_problem("lorenz96_20")["states"]),
    "pleiades" => _pleiades_entry,
    "pollu" => _pollu_entry,
    "ring_modulator" => _ring_modulator_entry,
    "ring_modulator_index2" => _ring_modulator_index2_entry,
    "nand_gate" => _nand_gate_entry,
)

const _ENTRIES = Dict{Tuple{String, DataType}, Any}()

"Compiled system artifacts for a problem row or name in element type T; built on first use."
function julia_system(problem, ::Type{T} = Float32) where {T}
    name = problem isa AbstractDict ? problem["problem"] : problem
    haskey(_ENTRY_BUILDERS, name) ||
        error("no ModelingToolkit definition for problem '$(name)'")
    return get!(() -> _ENTRY_BUILDERS[name](T), _ENTRIES, (name, T))
end

"In-place ODEProblem of one trajectory for the DifferentialEquations.jl solvers, in the system's element type: `problem` carries the duration, p is the swept value and the parameter vector."
function cpu_problem(system, problem, p)
    T = eltype(system.u0)
    f = system.mass_matrix === nothing ?
        ODEFunction{true}(system.rhs!; jac = system.jac!) :
        ODEFunction{true}(system.rhs!; jac = system.jac!,
            mass_matrix = Matrix{T}(system.mass_matrix))
    return ODEProblem{true}(f, Vector{T}(system.u0_for(T(p))),
        (zero(T), T(problem["duration"])), T[p])
end
