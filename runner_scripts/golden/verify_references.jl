# Golden configurations vs the published Test Set values; Float32 rhs vs Float64 twins.
#
# Usage: julia --project=. runner_scripts/golden/verify_references.jl

using OrdinaryDiffEq
using StaticArrays

include(joinpath(dirname(@__DIR__), "problems.jl"))
include(joinpath(dirname(@__DIR__), "reference_systems.jl"))
include(joinpath(dirname(@__DIR__), "julia_systems.jl"))

# Published reference values (solut subroutines of pollu.f, plei.f, nand.f).
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

"Final state of one golden-configuration solve at the swept value p."
function golden_final(name, p)
    problem = get_problem(name)
    system = reference_system(problem)
    prob = reference_problem(system, problem, p)
    solver = reference_solver(problem["golden_algorithm"])
    tol = problem["golden_tol"]
    sol = solve(prob, solver; abstol = tol, reltol = tol,
        save_everystep = false, save_start = false, dense = false,
        maxiters = 10^8, reference_solve_kwargs(system)...)
    return sol.u[end]
end

for (name, p, ref) in (("pollu", 0.35, POLLU_REF),
    ("pleiades", 1.0, PLEI_REF),
    ("nand_gate", 5.0, NAND_REF))
    t = @elapsed final = golden_final(name, p)
    err = maximum(abs.(final .- ref))
    println("$(name): max |golden - published| = $(err)  ($(round(t; digits=2)) s/solve)")
end

# Lorenz 96 has no published value; compare two independent integrators.
let problem = get_problem("lorenz96")
    system = reference_system(problem)
    prob = reference_problem(system, problem, 8.0)
    a = solve(prob, reference_solver("Vern9"); abstol = 1e-13, reltol = 1e-13,
        save_everystep = false, save_start = false).u[end]
    b = solve(prob, RadauIIA9(); abstol = 1e-13, reltol = 1e-13,
        save_everystep = false, save_start = false).u[end]
    println("lorenz96: max |Vern9 - RadauIIA9| = $(maximum(abs.(a .- b)))")
end

# Float32 GPU right-hand sides against the Float64 references at random
# states, mapped through each compiled system's golden index.
using Random
rng = MersenneTwister(7)
for (name, nstates, p) in (("lorenz96", 32, 8.0), ("pleiades", 28, 1.0),
    ("pollu", 20, 0.35))
    system64 = reference_system(name)
    system32 = julia_system(name)
    worst = 0.0
    for trial in 1:20
        u = randn(rng, nstates) .* 2.0 .+ 0.5
        name == "pollu" && (u = abs.(u) .* 0.05)
        du64 = system64.rhs(u, [p], 0.3)
        umtk = zeros(Float32, system32.n)
        umtk[system32.golden_index] .= Float32.(u)
        du32 = system32.rhs(SVector{system32.n, Float32}(umtk),
            SVector{1, Float32}(Float32(p)), 0.3f0)
        got = Float64.(du32[system32.golden_index])
        scale = max.(abs.(du64), 1.0)
        worst = max(worst, maximum(abs.(got .- du64) ./ scale))
    end
    println("$(name): worst relative Float32-vs-Float64 rhs deviation = $(worst)")
end
