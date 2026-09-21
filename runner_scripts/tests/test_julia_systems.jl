# The system module in both element types, the julia_cpu solver table and the golden configurations; run with `julia --project=. runner_scripts/tests/test_julia_systems.jl`.

using Test
using LinearAlgebra
using StaticArrays
# The solver table spans every OrdinaryDiffEq sub-library the runner loads.
using OrdinaryDiffEqLowOrderRK, OrdinaryDiffEqHighOrderRK, OrdinaryDiffEqExplicitRK
using OrdinaryDiffEqSDIRK, OrdinaryDiffEqFIRK, OrdinaryDiffEqRosenbrock, OrdinaryDiffEqVerner
using OrdinaryDiffEqBDF, OrdinaryDiffEqAdamsBashforthMoulton
using JSON

include(joinpath(dirname(@__DIR__), "golden", "verify_references.jl"))
include(joinpath(dirname(@__DIR__), "julia_tableaus.jl"))

const RESIDUAL_LIMIT = Dict(Float64 => 1e-12, Float32 => 1e-6)
const FABBRI_CHECK = joinpath(dirname(@__DIR__), "generated", "fabbri_linder_rhs_check.json")

@testset "fabbri.jl and the generated Fabbri-Linder right-hand side" begin
    # The head lattice carries the four corners; the fill reaches both range ends by bit reversal.
    @test fabbri_levels(0.0) == (0.0, 0.0)
    @test fabbri_levels(31.0) == (0.0, 1.0) && fabbri_levels(992.0) == (1.0, 0.0) && fabbri_levels(1023.0) == (1.0, 1.0)
    @test fabbri_inputs(1023.0, Float32) == (100.0f0, 1000.0f0)
    @test fabbri_inputs(1.0, Float64) == (0.0, 1000.0 / 31)
    @test fabbri_levels(Float64(fabbri_bit_reverse((255 << FABBRI_ISO_BITS) | 511))) == (1.0, 1.0)
    @test fabbri_lattice_index(131071.4) == 131071 && fabbri_lattice_index(-3.0) == 0 && fabbri_lattice_index(2.5) == 2
    head = [fabbri_levels(Float64(i)) for i in 0:1023]
    @test length(unique(head)) == 1024 && length(unique(first.(head))) == 32 && length(unique(last.(head))) == 32
    # The generated function reproduces the exporter's float64 check points.
    check = JSON.parsefile(FABBRI_CHECK)
    @test check["states"] == FABBRI_LINDER_STATES
    for point in eachindex(check["t"])
        u = Float64.(check["u"][point])
        du = similar(u)
        fabbri_linder_rhs!(du, u, check["ach"][point], check["iso"][point], check["t"][point])
        expected = Float64.(check["du"][point])
        @test maximum(abs.(du .- expected) ./ max.(abs.(expected), 1e-300)) <= 1e-9
    end
    # The Jacobian is finite on the ACh = 0 edge of the lattice, where 1/ACh terms take their limit.
    system = julia_system("fabbri_linder", Float64)
    J = zeros(Float64, system.n, system.n)
    system.jac!(J, Vector{Float64}(system.u0), [0.0], 0.0)
    @test all(isfinite, J)
    @test all(isfinite, system.jac(system.u0, SVector{1, Float64}(0.0), 0.0))
end

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
