# The golden configurations against the published Test Set values and against
# each other, every Float32 system against its Float64 twin, and the
# allocation-free right-hand sides.
#
# Usage: julia --project=. runner_scripts/golden/verify_references.jl   (exit 1 when a check fails)
# tests/test_julia_systems.jl includes this file and asserts every check of verify_references().

using LinearAlgebra
using Printf
using Random
using StaticArrays
using OrdinaryDiffEq
using OrdinaryDiffEqBDF, OrdinaryDiffEqFIRK, OrdinaryDiffEqRosenbrock, OrdinaryDiffEqVerner

include(joinpath(dirname(@__DIR__), "problems.jl"))
include(joinpath(dirname(@__DIR__), "algorithms.jl"))
include(joinpath(dirname(@__DIR__), "julia_systems.jl"))

# Published reference values (solut subroutines of pollu.f, plei.f, nand.f and ringmod.f).
const POLLU_REF = [0.5646255480022769e-1, 0.1342484130422339, 0.4139734331099427e-8,
    0.5523140207484359e-2, 0.2018977262302196e-6, 0.1464541863493966e-6,
    0.7784249118997964e-1, 0.3245075353396018, 0.7494013383880406e-2,
    0.1622293157301561e-7, 0.1135863833257075e-7, 0.2230505975721359e-2,
    0.2087162882798630e-3, 0.1396921016840158e-4, 0.8964884856898295e-2,
    0.4352846369330103e-17, 0.6899219696263405e-2, 0.1007803037365946e-3,
    0.1772146513969984e-5, 0.5682943292316392e-4]

const PLEI_REF = [0.3706139143970502, 0.3237284092057233e1, -0.3222559032418324e1,
    0.6597091455775310, 0.3425581707156584, 0.1562172101400631e1,
    -0.7003092922212495, -0.3943437585517392e1, -0.3271380973972550e1,
    0.5225081843456543e1, -0.2590612434977470e1, 0.1198213693392275e1,
    -0.2429682344935824, 0.1091449240428980e1, 0.3417003806314313e1,
    0.1354584501625501e1, -0.2590065597810775e1, 0.2025053734714242e1,
    -0.1155815100160448e1, -0.8072988170223021, 0.5952396354208710,
    -0.3741244961234010e1, 0.3773459685750630, 0.9386858869551073,
    0.3667922227200571, -0.3474046353808490, 0.2344915448180937e1,
    -0.1947020434263292e1]

const NAND_REF = [0.4971088699385777e1, 0.4999752103929311e1, -0.2499998781491227e1,
    -0.2499999999999975e1, 0.4970837023296724e1, -0.2091214032073855,
    0.4970593243278363e1, -0.2500077409198803e1, -0.2499998781491227e1,
    -0.2090289583878100, -0.2399999999966269e-3, -0.2091214032073855,
    -0.2499999999999991e1, -0.2500077409198803e1]

# Test Set II-3 at t = 1e-3 with Cs = 2e-12 (RADAU at 1e-12).
const RING_REF = [-0.2339057358486745e-1, -0.7367485485540825e-2, 0.2582956709291169,
    -0.4064465721283450, -0.4039455665149794, 0.2607966765422943,
    0.1106761861269975, 0.2939904342435596e-6, -0.2840029933642329e-7,
    0.7267198267264553e-3, 0.7929487196960840e-3, -0.7255283495698965e-3,
    -0.7941401968526521e-3, 0.7088495416976114e-4, 0.2390059075236570e-4]

const RHS_TRIALS = 20
const RHS_SEED = 7

"Golden-ordered final state of one Float64 solve of a problem at the swept value p; the catalogue's golden algorithm and tolerance unless given."
function golden_final(name, p; algorithm = nothing, tol = nothing)
    row = get_problem(name)
    system = julia_system(name, Float64)
    prob = cpu_problem(system, row, p)
    alg = julia_solver(something(algorithm, row["golden_algorithm"]), "julia_cpu", Float64)
    tolerance = something(tol, row["golden_tol"])
    sol = solve(prob, alg; abstol = tolerance, reltol = tolerance, save_everystep = false,
        save_start = false, dense = false, maxiters = 10^8)
    return sol.u[end][system.golden_index]
end

published_deviation(name, p, reference) = maximum(abs.(golden_final(name, p) .- reference))

"Two independent integrators of Lorenz 96 at F = 8, both at 1e-13."
lorenz96_cross_check() = maximum(abs.(golden_final("lorenz96", 8.0) .-
                                      golden_final("lorenz96", 8.0; algorithm = "radau_iia_5", tol = 1e-13)))

"Unknown positions of the Float64 system in the Float32 system's order."
function _unknown_permutation(system64, system32)
    names32 = string.(unknowns(system32.sys))
    return [findfirst(==(name), names32) for name in string.(unknowns(system64.sys))]
end

"Worst relative deviation of the Float32 right-hand side from the Float64 one at RHS_TRIALS Float32-representable states around the consistent u0, 30% into the integration window, scaled by max(|du|, 1). Both sides see the same numbers, so the figure is the arithmetic and literal rounding alone."
function rhs_twin_deviation(name)
    row = get_problem(name)
    system64 = julia_system(name, Float64)
    system32 = julia_system(name, Float32)
    perm = _unknown_permutation(system64, system32)
    any(isnothing, perm) && error("$(name): the Float32 and Float64 systems disagree on their unknowns")
    p32 = Float32(0.5 * (row["sweep_min"] + row["sweep_max"]))
    t32 = Float32(0.3 * row["duration"])
    rng = MersenneTwister(RHS_SEED)
    base = Vector{Float64}(system64.u0_for(Float64(p32)))
    worst = 0.0
    for _ in 1:RHS_TRIALS
        u32 = zeros(Float32, system32.n)
        u32[perm] .= Float32.(base .+ 0.05 .* randn(rng, system64.n))
        u64 = Float64.(u32[perm])
        du64 = Vector(system64.rhs(SVector{system64.n, Float64}(u64), SVector{1, Float64}(Float64(p32)),
            Float64(t32)))
        du32 = Vector(system32.rhs(SVector{system32.n, Float32}(u32), SVector{1, Float32}(p32), t32))
        got = Float64.(du32[perm])
        worst = max(worst, maximum(abs.(got .- du64) ./ max.(abs.(du64), 1.0)))
    end
    return worst
end

_allocated_call(f!, du, u, p, t) = @allocated f!(du, u, p, t)

"Bytes the in-place right-hand side allocates on its second call in T."
function rhs_allocation(name, ::Type{T}) where {T}
    row = get_problem(name)
    system = julia_system(name, T)
    u = Vector{T}(system.u0_for(T(row["sweep_min"])))
    du = similar(u)
    p = T[row["sweep_min"]]
    _allocated_call(system.rhs!, du, u, p, zero(T))
    return _allocated_call(system.rhs!, du, u, p, zero(T))
end

"Every check as (name, measure, limit): a measured value at or below the limit passes."
function reference_checks()
    checks = Any[
        ("pollu k1=0.35 vs published", () -> published_deviation("pollu", 0.35, POLLU_REF), 1e-13),
        ("pleiades m1=1.0 vs published", () -> published_deviation("pleiades", 1.0, PLEI_REF), 1e-10),
        ("nand_gate c9=5e-5 vs published", () -> published_deviation("nand_gate", 0.5e-4, NAND_REF), 1e-8),
        ("ring_modulator Cs=2e-12 vs Test Set II-3 at t=1e-3",
            () -> published_deviation("ring_modulator", 2.0e-12, RING_REF), 1e-8),
        ("lorenz96 F=8 Vern9 vs RadauIIA5", lorenz96_cross_check, 1e-7),
    ]
    for row in load_problems()
        name = row["problem"]
        push!(checks, ("$(name) Float32 rhs vs Float64 rhs (relative)", () -> rhs_twin_deviation(name), 1e-5))
        for T in (Float32, Float64)
            push!(checks, ("$(name) $(T) rhs! bytes allocated", () -> rhs_allocation(name, T), 0))
        end
    end
    return checks
end

"Run every check; each result is (name, measured, limit, ok, seconds, note), note carrying an error text when the check could not run."
function verify_references()
    results = []
    for (name, measure, limit) in reference_checks()
        note = ""
        seconds = @elapsed measured = try
            measure()
        catch err
            note = first(sprint(showerror, err), 300)
            NaN
        end
        push!(results, (name = name, measured = measured, limit = limit,
            ok = isfinite(measured) && measured <= limit, seconds = seconds, note = note))
    end
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    results = verify_references()
    for r in results
        @printf("%-52s %-10.3e limit %-8.0e %s  (%.1f s)%s\n", r.name, r.measured, r.limit,
            r.ok ? "ok" : "FAIL", r.seconds, isempty(r.note) ? "" : "  " * r.note)
    end
    failed = count(r -> !r.ok, results)
    println("$(length(results) - failed) of $(length(results)) checks passed")
    exit(failed == 0 ? 0 : 1)
end
