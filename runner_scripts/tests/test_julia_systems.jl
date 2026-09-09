# The system module in both element types, the julia_cpu solver table and the golden configurations; run with `julia --project=. runner_scripts/tests/test_julia_systems.jl`.

using Test
using LinearAlgebra
using StaticArrays
# The solver table spans every OrdinaryDiffEq sub-library the runner loads.
using OrdinaryDiffEqLowOrderRK, OrdinaryDiffEqHighOrderRK, OrdinaryDiffEqExplicitRK
using OrdinaryDiffEqSDIRK, OrdinaryDiffEqFIRK, OrdinaryDiffEqRosenbrock, OrdinaryDiffEqVerner
using OrdinaryDiffEqBDF

include(joinpath(dirname(@__DIR__), "golden", "verify_references.jl"))
include(joinpath(dirname(@__DIR__), "julia_tableaus.jl"))

const RESIDUAL_LIMIT = Dict(Float64 => 1e-12, Float32 => 1e-6)

@testset "julia_systems.jl" begin
    for row in load_problems(), T in (Float32, Float64)
        name = row["problem"]
        @testset "$(name) $(T)" begin
            system = julia_system(name, T)
            @test julia_system(row, T) === system
            n = system.n
            @test system.u0 isa SVector{n, T}
            @test eltype(system.u0) == T
            @test system.mass_matrix === nothing || system.mass_matrix isa SMatrix{n, n, T}
            @test system.jac! !== nothing && system.jac !== nothing
            index = system.golden_index
            @test length(index) == row["states"]
            @test length(unique(index)) == length(index)
            @test all(1 .<= index .<= n)
            u = Vector{T}(system.u0_for(T(row["sweep_min"])))
            @test eltype(u) == T && length(u) == n
            du = similar(u)
            p = T[row["sweep_min"]]
            system.rhs!(du, u, p, zero(T))
            @test all(isfinite, du)
            @test rhs_allocation(name, T) == 0
            if system.mass_matrix !== nothing
                algebraic = findall(iszero, diag(system.mass_matrix))
                @test !isempty(algebraic)
                @test maximum(abs.(du[algebraic])) <= RESIDUAL_LIMIT[T]
            end
            problem = cpu_problem(system, row, row["sweep_min"])
            @test eltype(problem.u0) == T && problem.tspan == (zero(T), T(row["duration"]))
            @test problem.p == T[row["sweep_min"]]
            @test (problem.f.mass_matrix isa UniformScaling) == (system.mass_matrix === nothing)
        end
    end

    @testset "every julia_cpu constructor builds in both element types" begin
        for algorithm in load_algorithms(), T in (Float32, Float64)
            isempty(algorithm["julia_cpu"]) && continue
            @test julia_solver(algorithm["algorithm"], "julia_cpu", T) isa OrdinaryDiffEq.SciMLBase.AbstractODEAlgorithm ||
                  julia_solver(algorithm["algorithm"], "julia_cpu", T) isa OrdinaryDiffEq.SciMLBase.AbstractDEAlgorithm
        end
        @test eltype(julia_solver("cash-karp-54", "julia_cpu", Float64).tableau.A) == Float64
        @test eltype(julia_solver("fehlberg-45", "julia_cpu", Float32).tableau.A) == Float32
    end
end

@testset "verify_references.jl" begin
    for check in verify_references()
        @testset "$(check.name)" begin
            isempty(check.note) || println(check.name, ": ", check.note)
            @test check.measured <= check.limit
        end
    end
end
