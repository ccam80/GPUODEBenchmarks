# ModelingToolkit system definitions, one per problem: the raw equations
# compile through mtkcompile and every numeric artifact (rhs, jacobian,
# tgrad, mass matrix, orderings) is generated from the compiled system.

using LinearAlgebra
using StaticArrays
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D

"Unknown-order values for a symbolic->value map; absent unknowns get 0."
function _ordered_values(sys, valmap)
    entries = collect(valmap)
    return [begin
        i = findfirst(pr -> isequal(pr.first, v), entries)
        Float32(i === nothing ? 0.0f0 : entries[i].second)
    end for v in unknowns(sys)]
end

"Position of each golden variable in the compiled system's unknown order."
function _golden_index(sys, golden_vars)
    us = unknowns(sys)
    return [findfirst(u -> isequal(v, u), us) for v in golden_vars]
end

"Compile a raw system and generate the numeric artifacts every suite uses."
function _build_entry(raw; u0map, golden_vars, consistent_u0 = false)
    sys = mtkcompile(raw; split = false)
    n = length(unknowns(sys))
    rhs, rhs! = ModelingToolkit.generate_rhs(sys; expression = Val(false))
    # Piecewise expressions can defeat symbolic differentiation; the solvers
    # fall back to finite differences when a derivative is nothing.
    jac, jac! = try
        ModelingToolkit.generate_jacobian(sys; expression = Val(false))
    catch
        nothing, nothing
    end
    tgrad = try
        ModelingToolkit.generate_tgrad(sys; expression = Val(false))[1]
    catch
        nothing
    end
    mm = calculate_massmatrix(sys)
    mass_matrix = mm === I ? nothing : SMatrix{n, n, Float32}(Float32.(mm))
    u0 = SVector{n, Float32}(_ordered_values(sys, u0map))
    index = _golden_index(sys, golden_vars)
    any(isnothing, index) && error("golden variable missing from unknowns")
    u0_for = consistent_u0 ?
             _consistent_u0(rhs, u0, findall(iszero, diag(mm)), n) :
             (p -> u0)
    return (sys = sys, n = n, rhs = rhs, rhs! = rhs!, jac = jac, jac! = jac!,
        tgrad = tgrad, mass_matrix = mass_matrix, u0 = u0, u0_for = u0_for,
        golden_index = SVector{length(index), Int}(index))
end

"Per-parameter u0 with the algebraic unknowns solved from their (linear) rows."
function _consistent_u0(rhs, u0, zidx, n)
    m = length(zidx)
    function u0_for(pval)
        p = SVector{1, Float64}(Float64(pval))
        base = Vector{Float64}(u0)
        function residual(z)
            u = copy(base)
            u[zidx] .= z
            return Float64.(rhs(SVector{n, Float64}(u), p, 0.0))[zidx]
        end
        g0 = residual(zeros(m))
        A = reduce(hcat, [residual([j == k ? 1.0 : 0.0 for k in 1:m]) .- g0
                          for j in 1:m])
        base[zidx] .= A \ (-g0)
        return SVector{n, Float32}(Float32.(base))
    end
    return u0_for
end

# --- lorenz ----------------------------------------------------------------
function _lorenz_entry()
    @parameters rho = 21.0f0
    @variables x(t) y(t) z(t)
    eqs = [D(x) ~ 10.0f0 * (y - x),
        D(y) ~ x * (rho - z) - y,
        D(z) ~ x * y - (8.0f0 / 3.0f0) * z]
    @named lorenz = System(eqs, t)
    return _build_entry(lorenz;
        u0map = [x => 1.0f0, y => 0.0f0, z => 0.0f0],
        golden_vars = [x, y, z])
end

# --- lorenz 96 -------------------------------------------------------------
# The state count is a parameter: lorenz96 runs the canonical 40, lorenz96_20
# the largest size the DiffEqGPU kernel-path implicit solvers handle.
function _lorenz96_entry(n)
    @parameters F = 8.0f0
    @variables (x(t))[1:n]
    xs = collect(x)
    eqs = [D(xs[i]) ~ (xs[mod1(i + 1, n)] - xs[mod1(i - 2, n)]) *
                      xs[mod1(i - 1, n)] - xs[i] + F for i in 1:n]
    @named lorenz96 = System(eqs, t)
    return _build_entry(lorenz96;
        u0map = [xs[i] => (i == 1 ? 9.0f0 : 8.0f0) for i in 1:n],
        golden_vars = xs)
end

# --- pleiades --------------------------------------------------------------
function _pleiades_entry()
    @parameters m1 = 1.0f0
    @variables (x(t))[1:7] (y(t))[1:7] (u(t))[1:7] (v(t))[1:7]
    xs, ys, us, vs = collect(x), collect(y), collect(u), collect(v)
    masses = [i == 1 ? m1 : Float32(i) for i in 1:7]
    eqs = Equation[]
    for i in 1:7
        push!(eqs, D(xs[i]) ~ us[i])
        push!(eqs, D(ys[i]) ~ vs[i])
    end
    for i in 1:7
        sumx = 0
        sumy = 0
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
    x0 = Float32[3, 3, -1, -3, 2, -2, 2]
    y0 = Float32[3, -3, 2, 0, 0, -4, 4]
    u0 = Float32[0, 0, 0, 0, 0, 1.75, -1.5]
    v0 = Float32[0, 0, 0, -1.25, 1, 0, 0]
    u0map = vcat([xs[i] => x0[i] for i in 1:7], [ys[i] => y0[i] for i in 1:7],
        [us[i] => u0[i] for i in 1:7], [vs[i] => v0[i] for i in 1:7])
    return _build_entry(pleiades; u0map = u0map,
        golden_vars = vcat(xs, ys, us, vs))
end

# --- pollution problem -----------------------------------------------------
function _pollu_entry()
    @parameters k1 = 0.35f0
    @variables (y(t))[1:20]
    ys = collect(y)
    k = Float32[0, 26.6, 1.23e4, 8.6e-4, 8.2e-4, 1.5e4, 1.3e-4, 2.4e4,
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
    eqs = [
        D(ys[1]) ~ -r1 - r10 - r14 - r23 - r24 + r2 + r3 + r9 + r11 + r12 +
                   r22 + r25,
        D(ys[2]) ~ -r2 - r3 - r9 - r12 + r1 + r21,
        D(ys[3]) ~ -r15 + r1 + r17 + r19 + r22,
        D(ys[4]) ~ -r2 - r16 - r17 - r23 + r15,
        D(ys[5]) ~ -r3 + 2.0f0 * r4 + r6 + r7 + r13 + r20,
        D(ys[6]) ~ -r6 - r8 - r14 - r20 + r3 + 2.0f0 * r18,
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
    u0vals = zeros(Float32, 20)
    u0vals[2] = 0.2f0
    u0vals[4] = 0.04f0
    u0vals[7] = 0.1f0
    u0vals[8] = 0.3f0
    u0vals[9] = 0.01f0
    u0vals[17] = 0.007f0
    return _build_entry(pollu;
        u0map = [ys[i] => u0vals[i] for i in 1:20], golden_vars = ys)
end

# --- ring modulator (Test Set for IVP Solvers, problem II-3) ---------------
_rm_q(v) = 40.67286402f-9 * (exp(17.7493332f0 * v) - 1.0f0)

"Equations and variables; `cs` and `amp` are symbolic or literal."
function _ring_modulator_equations(cs, amp)
    @variables U1(t) U2(t) U3(t) U4(t) U5(t) U6(t) U7(t)
    @variables I1(t) I2(t) I3(t) I4(t) I5(t) I6(t) I7(t) I8(t)
    C = 1.6f-8
    Cp = 1.0f-8
    Lh = 4.45f0
    Ls1 = 0.002f0
    Ls2 = 5.0f-4
    Ls3 = 5.0f-4
    R = 25000.0f0
    Rp = 50.0f0
    Rg1 = 36.3f0
    Rg2 = 17.3f0
    Rg3 = 17.3f0
    Ri = 50.0f0
    Rc = 600.0f0
    w1 = 6283.1855f0
    w2 = 62831.855f0
    Uin1 = amp * sin(w1 * t)
    Uin2 = 2.0f0 * sin(w2 * t)
    UD1 = U3 - U5 - U7 - Uin2
    UD2 = -U4 + U6 - U7 - Uin2
    UD3 = U4 + U5 + U7 + Uin2
    UD4 = -U3 - U6 + U7 + Uin2
    q1, q2, q3, q4 = _rm_q(UD1), _rm_q(UD2), _rm_q(UD3), _rm_q(UD4)
    eqs = [
        C * D(U1) ~ I1 - 0.5f0 * I3 + 0.5f0 * I4 + I7 - U1 / R,
        C * D(U2) ~ I2 - 0.5f0 * I5 + 0.5f0 * I6 + I8 - U2 / R,
        cs * D(U3) ~ I3 - q1 + q4,
        cs * D(U4) ~ -I4 + q2 - q3,
        cs * D(U5) ~ I5 + q1 - q3,
        cs * D(U6) ~ -I6 - q2 + q4,
        Cp * D(U7) ~ -U7 / Rp + q1 + q2 - q3 - q4,
        Lh * D(I1) ~ -U1,
        Lh * D(I2) ~ -U2,
        Ls2 * D(I3) ~ 0.5f0 * U1 - U3 - Rg2 * I3,
        Ls3 * D(I4) ~ -0.5f0 * U1 + U4 - Rg3 * I4,
        Ls2 * D(I5) ~ 0.5f0 * U2 - U5 - Rg2 * I5,
        Ls3 * D(I6) ~ -0.5f0 * U2 + U6 - Rg3 * I6,
        Ls1 * D(I7) ~ -U1 + Uin1 - (Ri + Rg1) * I7,
        Ls1 * D(I8) ~ -U2 - (Rc + Rg1) * I8,
    ]
    return eqs, [U1, U2, U3, U4, U5, U6, U7, I1, I2, I3, I4, I5, I6, I7, I8]
end

"Stiff form: the swept capacitance Cs stays a parameter."
function _ring_modulator_entry()
    @parameters Cs = 2.0f-12
    eqs, vars = _ring_modulator_equations(Cs, 0.5f0)
    @named ring_modulator = System(eqs, t)
    return _build_entry(ring_modulator;
        u0map = [v => 0.0f0 for v in vars], golden_vars = vars)
end

"Index-2 form: Cs = 0 substituted at definition, the Uin1 amplitude swept."
function _ring_modulator_index2_entry()
    @parameters Uin1_amplitude = 0.5f0
    eqs, vars = _ring_modulator_equations(0.0f0, Uin1_amplitude)
    @named ring_modulator_index2 = System(eqs, t)
    return _build_entry(ring_modulator_index2;
        u0map = [v => 0.0f0 for v in vars], golden_vars = vars)
end

# --- NAND gate (Test Set for IVP Solvers) ----------------------------------
"Trapezoidal test-set input pulse and its time derivative."
function _nand_pulse(delay, t1, t2, t3, period)
    tp = t - period * floor(t / period)
    hi = 5.0f0
    vin = ifelse(tp > delay + t1 + t2 + t3, 0.0f0,
        ifelse(tp > delay + t1 + t2, (hi / t3) * (delay + t1 + t2 + t3 - tp),
            ifelse(tp > delay + t1, hi,
                ifelse(tp > delay, (hi / t1) * (tp - delay), 0.0f0))))
    vind = ifelse(tp > delay + t1 + t2 + t3, 0.0f0,
        ifelse(tp > delay + t1 + t2, -hi / t3,
            ifelse(tp > delay + t1, 0.0f0,
                ifelse(tp > delay, hi / t1, 0.0f0))))
    return vin, vind
end

_nand_cb(v) = ifelse(v <= 0.0f0, 2.4f-5 / sqrt(1.0f0 - v / 0.87f0),
    2.4f-5 * (1.0f0 + v / (2.0f0 * 0.87f0)))
_nand_ibs(v) = ifelse(v <= 0.0f0,
    -1.0f-14 * (exp(v / 25.85f0) - 1.0f0), 0.0f0)

function _nand_gdsp(vds, vgs, vbs, vt0, cgamma, phi, beta)
    vte = vt0 + cgamma * (sqrt(phi - vbs) - sqrt(phi))
    return ifelse(vgs - vte <= 0.0f0, 0.0f0,
        ifelse(vgs - vte <= vds,
            -beta * (vgs - vte)^2 * (1.0f0 + 0.02f0 * vds),
            -beta * vds * (2.0f0 * (vgs - vte) - vds) *
            (1.0f0 + 0.02f0 * vds)))
end

function _nand_gdsm(vds, vgd, vbd, vt0, cgamma, phi, beta)
    vte = vt0 + cgamma * (sqrt(phi - vbd) - sqrt(phi))
    return ifelse(vgd - vte <= 0.0f0, 0.0f0,
        ifelse(vgd - vte <= -vds,
            beta * (vgd - vte)^2 * (1.0f0 - 0.02f0 * vds),
            -beta * vds * (2.0f0 * (vgd - vte) + vds) *
            (1.0f0 - 0.02f0 * vds)))
end

# nand.f: depletion beta 5.35e-4 in both branches; enhancement 1.748e-3 / 1.748e-4.
_nand_ids1(vds, vgs, vbs, vgd, vbd) = ifelse(vds > 0.0f0,
    _nand_gdsp(vds, vgs, vbs, -2.43f0, 0.2f0, 1.28f0, 5.35f-4),
    ifelse(vds < 0.0f0,
        _nand_gdsm(vds, vgd, vbd, -2.43f0, 0.2f0, 1.28f0, 5.35f-4), 0.0f0))
_nand_ids2(vds, vgs, vbs, vgd, vbd) = ifelse(vds > 0.0f0,
    _nand_gdsp(vds, vgs, vbs, 0.2f0, 0.035f0, 1.01f0, 1.748f-3),
    ifelse(vds < 0.0f0,
        _nand_gdsm(vds, vgd, vbd, 0.2f0, 0.035f0, 1.01f0, 1.748f-4), 0.0f0))

function _nand_gate_entry()
    @parameters VDD = 5.0f0
    @variables (y(t))[1:14]
    ys = collect(y)
    rgs = 4.0f0
    rgd = 4.0f0
    rbs = 10.0f0
    rbd = 10.0f0
    cgs = 0.6f-4
    cgd = 0.6f-4
    c9 = 0.5f-4
    vbb = -2.5f0
    v1, v1d = _nand_pulse(5.0f0, 5.0f0, 5.0f0, 5.0f0, 20.0f0)
    v2, v2d = _nand_pulse(15.0f0, 5.0f0, 15.0f0, 5.0f0, 40.0f0)
    i1 = _nand_ids1(ys[2] - ys[1], ys[5] - ys[1], ys[3] - ys[5],
        ys[5] - ys[2], ys[4] - VDD)
    i2 = _nand_ids2(ys[7] - ys[6], v1 - ys[6], ys[8] - ys[10], v1 - ys[7],
        ys[9] - ys[5])
    i3 = _nand_ids2(ys[12] - ys[11], v2 - ys[11], ys[13], v2 - ys[12],
        ys[14] - ys[10])
    cb35 = _nand_cb(ys[3] - ys[5])
    cb4 = _nand_cb(ys[4] - VDD)
    cb95 = _nand_cb(ys[9] - ys[5])
    cb810 = _nand_cb(ys[8] - ys[10])
    cb13 = _nand_cb(ys[13])
    cb1410 = _nand_cb(ys[14] - ys[10])
    eqs = [
        cgs * D(ys[1]) - cgs * D(ys[5]) ~ -(ys[1] - ys[5]) / rgs - i1,
        cgd * D(ys[2]) - cgd * D(ys[5]) ~ -(ys[2] - VDD) / rgd + i1,
        cb35 * D(ys[3]) - cb35 * D(ys[5]) ~ -(ys[3] - vbb) / rbs +
                                            _nand_ibs(ys[3] - ys[5]),
        cb4 * D(ys[4]) ~ -(ys[4] - vbb) / rbd + _nand_ibs(ys[4] - VDD),
        -cgs * D(ys[1]) - cgd * D(ys[2]) - cb35 * D(ys[3]) +
        (cgs + cgd + cb35 + cb95 + c9) * D(ys[5]) - cb95 * D(ys[9]) ~
            -(ys[5] - ys[1]) / rgs - _nand_ibs(ys[3] - ys[5]) -
            (ys[5] - ys[7]) / rgd - _nand_ibs(ys[9] - ys[5]),
        cgs * D(ys[6]) ~ cgs * v1d - (ys[6] - ys[10]) / rgs - i2,
        cgd * D(ys[7]) ~ cgd * v1d - (ys[7] - ys[5]) / rgd + i2,
        cb810 * D(ys[8]) - cb810 * D(ys[10]) ~ -(ys[8] - vbb) / rbs +
                                               _nand_ibs(ys[8] - ys[10]),
        -cb95 * D(ys[5]) + cb95 * D(ys[9]) ~ -(ys[9] - vbb) / rbd +
                                             _nand_ibs(ys[9] - ys[5]),
        -cb810 * D(ys[8]) + (cb810 + cb1410 + c9) * D(ys[10]) -
        cb1410 * D(ys[14]) ~
            -(ys[10] - ys[6]) / rgs - _nand_ibs(ys[8] - ys[10]) -
            (ys[10] - ys[12]) / rgd - _nand_ibs(ys[14] - ys[10]),
        cgs * D(ys[11]) ~ cgs * v2d - ys[11] / rgs - i3,
        cgd * D(ys[12]) ~ cgd * v2d - (ys[12] - ys[10]) / rgd + i3,
        cb13 * D(ys[13]) ~ -(ys[13] - vbb) / rbs + _nand_ibs(ys[13]),
        -cb1410 * D(ys[10]) + cb1410 * D(ys[14]) ~ -(ys[14] - vbb) / rbd +
                                                   _nand_ibs(ys[14] - ys[10]),
    ]
    @named nand_gate = System(eqs, t)
    u0vals = Float32[5, 5, -2.5, -2.5, 5, 3.62385, 5, -2.5, -2.5, 3.62385,
        0, 3.62385, -2.5, -2.5]
    return _build_entry(nand_gate;
        u0map = [ys[i] => u0vals[i] for i in 1:14], golden_vars = ys,
        consistent_u0 = true)
end

const _ENTRY_BUILDERS = Dict{String, Function}(
    "lorenz" => _lorenz_entry,
    "lorenz96" => () -> _lorenz96_entry(40),
    "lorenz96_20" => () -> _lorenz96_entry(20),
    "pleiades" => _pleiades_entry,
    "pollu" => _pollu_entry,
    "ring_modulator" => _ring_modulator_entry,
    "ring_modulator_index2" => _ring_modulator_index2_entry,
    "nand_gate" => _nand_gate_entry,
)

const _ENTRIES = Dict{String, Any}()

"Compiled system artifacts for a problem row or name; built on first use."
function julia_system(problem)
    name = problem isa AbstractDict ? problem["problem"] : problem
    haskey(_ENTRY_BUILDERS, name) ||
        error("no ModelingToolkit definition for problem '$(name)'")
    return get!(() -> _ENTRY_BUILDERS[name](), _ENTRIES, name)
end

"Rows of an ensemble final-state collection in golden-reference order."
golden_finals(system, finals) = finals[:, system.golden_index]
