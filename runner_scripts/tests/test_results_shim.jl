# Round trip through the results.jl shim; run with `julia --project=. runner_scripts/tests/test_results_shim.jl [suite python]`.

using Test

include(joinpath(dirname(@__DIR__), "results.jl"))

isempty(ARGS) || (STORE_PYTHON[] = ARGS[1])

const KEY = "windows_RTX-4070-SUPER"

function query_rows(sql, root)
    lines = strip.(split(strip(store_query(sql; root = root)), '\n'))
    header = String.(split(lines[1], ','))
    return [Dict(zip(header, String.(split(line, ',')))) for line in lines[2:end]]
end

@testset "results.jl shim" begin
    root = mktempdir()
    ident = Dict("package" => "julia_gpu", "key" => KEY, "problem" => "lorenz",
        "algorithm" => "tsit5", "mode" => "adaptive", "setting_kind" => "tol",
        "setting" => 1e-5, "n" => 32768, "states" => 3, "tier" => "default")

    @testset "status of an empty store" begin
        @test store_status(merge(ident, Dict("transfers" => "both")); root = root) == "absent"
    end

    @testset "rows record with NaN, samples and the reason" begin
        rows = [
            store_row("julia_gpu", KEY, "lorenz", "tsit5", "adaptive", "tol", 1e-5, 32768, 3;
                transfers = "both", min_ms = 12.5, samples_ms = [20.0, 12.5, 13.0],
                errored_pct = 0.0, build_s = 3.25, package_version = "DiffEqGPU 3.4.1",
                suite_rev = store_suite_rev()),
            store_row("julia_gpu", KEY, "lorenz", "tsit5", "adaptive", "tol", 1e-5, 32768, 3;
                transfers = "none", reason = "abandoned: oom at ordinal 2"),
        ]
        store_record(rows; root = root)
        @test store_status(merge(ident, Dict("transfers" => "both")); root = root) == "finite"
        @test store_status(merge(ident, Dict("transfers" => "none")); root = root) == "nan"
        back = query_rows("SELECT transfers, min_ms, samples_ms, reason, build_s, package_version, " *
                          "suite_rev, states FROM results ORDER BY transfers", root)
        @test length(back) == 2
        both, none = back
        @test both["transfers"] == "both"
        @test parse(Float64, both["min_ms"]) == 12.5
        @test both["samples_ms"] == "20;12.5;13"
        @test parse(Float64, both["build_s"]) == 3.25
        @test both["package_version"] == "DiffEqGPU 3.4.1"
        @test both["suite_rev"] == store_suite_rev()
        @test both["states"] == "3"
        @test none["min_ms"] == "nan"
        @test none["samples_ms"] == ""
        @test none["reason"] == "abandoned: oom at ordinal 2"
    end

    @testset "floor keeps the lower finite time and NaN never wins" begin
        slower = store_row("julia_gpu", KEY, "lorenz", "tsit5", "adaptive", "tol", 1e-5, 32768, 3;
            transfers = "both", min_ms = 20.0)
        store_record(slower; root = root, floor = true)
        nan = store_row("julia_gpu", KEY, "lorenz", "tsit5", "adaptive", "tol", 1e-5, 32768, 3;
            transfers = "both")
        store_record(nan; root = root, floor = true)
        back = query_rows("SELECT min_ms FROM results WHERE transfers = 'both'", root)
        @test parse(Float64, back[1]["min_ms"]) == 12.5
        faster = store_row("julia_gpu", KEY, "lorenz", "tsit5", "adaptive", "tol", 1e-5, 32768, 3;
            transfers = "both", min_ms = 10.0)
        store_record(faster; root = root, floor = true)
        back = query_rows("SELECT min_ms FROM results WHERE transfers = 'both'", root)
        @test parse(Float64, back[1]["min_ms"]) == 10.0
    end

    @testset "finals land beside the leg and the row points at them" begin
        finals = Float32[1.5 2.5 3.5; 0.1 0.2 0.3]
        relative = store_finals(ident, finals, [true, false]; root = root)
        @test relative == "finals/lorenz__tsit5__adaptive__tol-1e-05__n32768__s3__default.parquet"
        @test isfile(joinpath(root, "key=" * KEY, "package=julia_gpu", "finals",
            "lorenz__tsit5__adaptive__tol-1e-05__n32768__s3__default.parquet"))
        store_record(store_row("julia_gpu", KEY, "lorenz", "tsit5", "adaptive", "tol", 1e-5, 32768, 3;
            transfers = "none", min_ms = 8.0, finals = relative); root = root)
        back = query_rows("SELECT finals FROM results WHERE transfers = 'none'", root)
        @test back[1]["finals"] == relative
        @test_throws ArgumentError store_finals(ident, finals, [true]; root = root)
    end

    @testset "a package outside the store vocabulary is refused" begin
        @test_throws ArgumentError store_row("julia", KEY, "lorenz", "tsit5", "fixed", "dt", 2.0^-10, 8, 3)
    end
end
