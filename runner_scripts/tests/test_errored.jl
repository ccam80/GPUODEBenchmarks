# errored.jl against the rule of store.py; run with `julia --project=. runner_scripts/tests/test_errored.jl`.

using Test

include(joinpath(dirname(@__DIR__), "errored.jl"))

@testset "errored.jl" begin
    @testset "the mask flags a non-finite state, a short or NaN final time and a retcode" begin
        finals = Float32[1 2 3; NaN 2 3; 1 2 3; 1 2 3; 1 2 Inf; 1 2 3]
        t_final = [1.0, 1.0, 0.5, NaN, 1.0, 1.0 + 0.5e-4]
        none = fill("", 6)
        mask = errored_mask(finals, t_final, none, 1.0)
        @test mask == [false, true, true, true, true, false]
        @test errored_mask(finals, t_final, ["", "", "", "", "", "MaxIters"], 1.0)[6]
        @test_throws ArgumentError errored_mask(finals, t_final[1:2], none[1:2], 1.0)
        @test_throws ArgumentError errored_mask(finals, t_final, [""], 1.0)
    end

    @testset "the tolerance is relative to the duration" begin
        finals = Float32[1 2; 1 2]
        @test errored_mask(finals, [60.0 - 0.005, 60.0 - 0.007], ["", ""], 60.0) == [false, true]
    end

    @testset "the percent counts the flagged rows" begin
        finals = Float32[1 2; NaN 2; 1 2; 1 2]
        ts = Float32[1, 1, 0.5, 1]
        @test errored_pct(finals, ts, fill("", 4), 1.0) == 50.0
        @test errored_pct(finals, ts, ["", "", "", "Unstable"], 1.0) == 75.0
        @test isnan(errored_pct(zeros(Float32, 0, 2), Float32[], String[], 1.0))
        @test errored_pct(Float32[1 2; 3 4], [1.0, 1.0], ["", ""], 1.0) == 0.0
    end
end
