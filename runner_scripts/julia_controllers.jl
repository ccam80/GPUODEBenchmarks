# Writes runner_scripts/julia_controllers.csv: the step controller OrdinaryDiffEq resolves for every adaptive julia_cpu algorithm of julia_algorithms.csv at its defaults, in Float64, on a scalar test problem. The constants depend on the algorithm alone, so the table is checked in and sets.py reads it for the cubie packages.
#   julia --project=. runner_scripts/julia_controllers.jl

using OrdinaryDiffEq
using OrdinaryDiffEqLowOrderRK, OrdinaryDiffEqHighOrderRK, OrdinaryDiffEqExplicitRK
using OrdinaryDiffEqVerner, OrdinaryDiffEqSDIRK, OrdinaryDiffEqFIRK
using OrdinaryDiffEqRosenbrock, OrdinaryDiffEqBDF
import OrdinaryDiffEqCore
using SciMLBase: init

include(joinpath(@__DIR__, "algorithms.jl"))
include(joinpath(@__DIR__, "julia_tableaus.jl"))

const OUTPUT = joinpath(@__DIR__, "julia_controllers.csv")
const COLUMNS = ("algorithm", "controller", "beta1", "beta2", "qmin", "qmax", "gamma", "order")

"The controller row of an algorithm, or nothing when OrdinaryDiffEq does not step it adaptively."
function controller_row(name)
    alg = julia_solver(name, "julia_cpu", Float64)
    OrdinaryDiffEqCore.isadaptive(alg) || return nothing
    prob = ODEProblem((u, p, t) -> -u, 1.0, (0.0, 1.0))
    integ = init(prob, alg; abstol = 1e-6, reltol = 1e-6, dt = 0.1, save_everystep = false)
    ctrl = integ.controller_cache.controller
    basic = hasproperty(ctrl, :basic) ? ctrl.basic : ctrl
    field(obj, key) = hasproperty(obj, key) ? string(Float64(getproperty(obj, key))) : ""
    return Dict("algorithm" => name, "controller" => string(typeof(ctrl).name.name),
        "beta1" => field(ctrl, :beta1), "beta2" => field(ctrl, :beta2),
        "qmin" => field(basic, :qmin), "qmax" => field(basic, :qmax),
        "gamma" => field(basic, :gamma), "order" => string(OrdinaryDiffEqCore.alg_order(alg)))
end

rows = Dict{String, String}[]
for name in package_algorithms("julia_cpu")
    row = controller_row(name)
    row === nothing && continue
    push!(rows, row)
    println(join([row[c] for c in COLUMNS], ","))
end
open(OUTPUT, "w") do io
    println(io, join(COLUMNS, ","))
    for row in rows
        println(io, join([row[c] for c in COLUMNS], ","))
    end
end
println("wrote $(length(rows)) rows to $(OUTPUT)")
