# grid.jl against the numpy reference grids; run with `julia --project=. runner_scripts/tests/test_grid.jl`.

using Test

include(joinpath(dirname(@__DIR__), "grid.jl"))

const GRIDS_DIR = joinpath(@__DIR__, "grids")
const PROBLEMS_CSV = joinpath(dirname(@__DIR__), "problems.csv")
const REFERENCE_N = 131072

"The Float32 vector of a one-dimensional little-endian '<f4' .npy file."
function read_npy_f32(path)
    bytes = read(path)
    bytes[1:6] == UInt8[0x93, UInt8('N'), UInt8('U'), UInt8('M'), UInt8('P'), UInt8('Y')] ||
        error("$(path) is not an npy file")
    major = bytes[7]
    header_len, offset = major == 1 ? (Int(bytes[9]) | (Int(bytes[10]) << 8), 11) :
        (Int(bytes[9]) | (Int(bytes[10]) << 8) | (Int(bytes[11]) << 16) | (Int(bytes[12]) << 24), 13)
    header = String(bytes[offset:(offset + header_len - 1)])
    occursin("'<f4'", header) || error("$(path) is not '<f4': $(header)")
    occursin("'fortran_order': False", header) || error("$(path) is not C order")
    data = bytes[(offset + header_len):end]
    length(data) % 4 == 0 || error("$(path) has a partial float")
    return collect(reinterpret(Float32, data))
end

"(problem, scale, min, max) of every problems.csv row."
function problem_grids()
    lines = filter(!isempty, strip.(readlines(PROBLEMS_CSV)))
    header = String.(split(lines[1], ','))
    rows = []
    for line in lines[2:end]
        row = Dict(zip(header, String.(split(line, ','))))
        push!(rows, (row["problem"], row["sweep_scale"], parse(Float64, row["sweep_min"]),
            parse(Float64, row["sweep_max"])))
    end
    return rows
end

@testset "grid.jl" begin
    @testset "the formula, endpoints and precision" begin
        v = grid_values("linear", 0.0, 21.0, 8)
        @test v isa Vector{Float32}
        @test v[1] == 0.0f0 && v[end] == 21.0f0
        @test v == Float32.([0.0 + i * 3.0 for i in 0:7])
        w = grid_values("log", 1e-3, 1.0, 4)
        @test w[1] == 1.0f-3 && w[end] == 1.0f0
        @test w[2] == Float32(10.0^(-3.0 + 1.0)) && w[3] == Float32(10.0^(-3.0 + 2.0))
        spec = Dict("grid_scale" => "linear", "grid_min" => 0.0, "grid_max" => 21.0, "n" => 8,
            "precision" => "float64")
        wide = grid(spec)
        @test wide isa Vector{Float64} && wide == Float64.(v)
        @test grid(merge(spec, Dict("precision" => "float32"))) == v
        @test_throws ArgumentError grid_values("linear", 0.0, 1.0, 1)
        @test_throws ArgumentError grid_values("log", 0.0, 1.0, 8)
        @test_throws ArgumentError grid_values("cubic", 0.0, 1.0, 8)
        @test_throws ArgumentError grid(merge(spec, Dict("precision" => "float16")))
    end

    @testset "every problem grid equals the numpy file bit for bit" begin
        rows = problem_grids()
        @test length(rows) >= 8
        for (problem, scale, lo, hi) in rows
            reference = read_npy_f32(joinpath(GRIDS_DIR, "$(problem)_$(REFERENCE_N).npy"))
            @test length(reference) == REFERENCE_N
            ours = grid_values(scale, lo, hi, REFERENCE_N)
            mismatches = count(i -> reinterpret(UInt32, ours[i]) != reinterpret(UInt32, reference[i]),
                1:REFERENCE_N)
            @test mismatches == 0
            mismatches == 0 || println("$(problem): $(mismatches) mismatching points")
            # A 1024-point grid ending at the Float64 v[1023] reproduces the first 1024 reference points.
            point = grid_point(scale, lo, hi, REFERENCE_N, 1023)
            @test Float32(point) == reference[1024]
            prefix = grid_values(scale, lo, point, 1024)
            @test all(reinterpret(UInt32, prefix) .== reinterpret(UInt32, reference[1:1024]))
        end
        @test grid_point("linear", 0.0, 21.0, 8, 7) == 21.0
        @test_throws ArgumentError grid_point("linear", 0.0, 21.0, 8, 8)
    end
end
