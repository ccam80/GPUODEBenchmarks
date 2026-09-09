# Round trip through the results.jl shim; run with `julia --project=. runner_scripts/tests/test_results_shim.jl [suite python]`.

using Test

include(joinpath(dirname(@__DIR__), "results.jl"))

isempty(ARGS) || (STORE_PYTHON[] = ARGS[1])

const KEY = "windows_RTX-4070-SUPER"

# The fixture of test_store.py: lorenz, adaptive tsit5 at tol 1e-5, n = 4, julia_gpu on the 4070.
function fixture(; overrides...)
    spec = Dict{String, Any}("problem" => "lorenz", "system_params" => Dict(), "duration" => 1.0,
        "precision" => "float32", "parameter" => "rho", "grid_scale" => "linear",
        "grid_min" => 0.0, "grid_max" => 21.0, "n" => 4, "grid_dtype" => "float32",
        "algorithm" => "tsit5", "controller" => "default", "dt" => 2.0^-10, "dt_min" => NaN,
        "dt_max" => NaN, "atol" => 1e-5, "rtol" => 1e-5, "gains" => Dict(),
        "newton_atol" => NaN, "newton_rtol" => NaN, "transfers" => "both",
        "package" => "julia_gpu", "key" => KEY)
    for (name, value) in overrides
        spec[String(name)] = value
    end
    return spec
end

function query_rows(sql, root)
    lines = strip.(split(strip(store_query(sql; root = root)), '\n'))
    header = String.(split(lines[1], ','))
    return [Dict(zip(header, String.(split(line, ',')))) for line in lines[2:end]]
end

@testset "results.jl shim" begin
    root = mktempdir()
    both = fixture()
    none = fixture(; transfers = "none")

    @testset "hashes come from store.py and status of an empty store" begin
        trial_both, run_both = store_hash(both; root = root)
        trial_none, run_none = store_hash(none; root = root)
        @test length(trial_both) == 16 && length(run_both) == 16
        @test trial_both == trial_none
        @test run_both != run_none
        @test store_status(both; root = root) == "absent"
        @test store_status(run_both; root = root) == "absent"
        # A trial Dict with extra fields hashes the same.
        @test store_hash(merge(both, Dict("kind" => "solve", "role" => "timed")); root = root) ==
              (trial_both, run_both)
    end

    @testset "rows record with NaN, samples and the reason" begin
        rows = [
            store_row(both; states = 3, min_ms = 12.5, samples_ms = [20.0, 12.5, 13.0],
                errored_pct = 0.0, build_s = 3.25, package_version = "DiffEqGPU 3.4.1",
                suite_rev = store_suite_rev()),
            store_row(merge(none, Dict("kind" => "solve")); states = 3,
                reason = "abandoned: oom at ordinal 2"),
        ]
        @test !haskey(rows[2], "kind")
        store_record(rows; root = root)
        @test store_status(both; root = root) == "finite"
        @test store_status(none; root = root) == "nan"
        back = query_rows("SELECT transfers, min_ms, samples_ms, reason, build_s, package_version, " *
                          "suite_rev, states, run_id, trial_id, system_params, gains, atol, dt_min " *
                          "FROM results ORDER BY transfers", root)
        @test length(back) == 2
        first, second = back
        @test first["transfers"] == "both"
        @test parse(Float64, first["min_ms"]) == 12.5
        @test first["samples_ms"] == "20;12.5;13"
        @test parse(Float64, first["build_s"]) == 3.25
        @test first["package_version"] == "DiffEqGPU 3.4.1"
        @test first["suite_rev"] == store_suite_rev()
        @test first["states"] == "3"
        @test first["system_params"] == "{}" && first["gains"] == "{}"
        @test parse(Float64, first["atol"]) == 1e-5
        @test first["dt_min"] == "nan"
        @test (first["trial_id"], first["run_id"]) == store_hash(both; root = root)
        @test second["min_ms"] == "nan"
        @test second["samples_ms"] == ""
        @test second["reason"] == "abandoned: oom at ordinal 2"
        @test second["trial_id"] == first["trial_id"]
        @test second["run_id"] == store_hash(none; root = root)[2]
    end

    @testset "floor keeps the lower finite time and NaN never wins" begin
        store_record(store_row(both; states = 3, min_ms = 20.0); root = root, floor = true)
        store_record(store_row(both; states = 3); root = root, floor = true)
        back = query_rows("SELECT min_ms FROM results WHERE transfers = 'both'", root)
        @test parse(Float64, back[1]["min_ms"]) == 12.5
        store_record(store_row(both; states = 3, min_ms = 10.0); root = root, floor = true)
        back = query_rows("SELECT min_ms FROM results WHERE transfers = 'both'", root)
        @test parse(Float64, back[1]["min_ms"]) == 10.0
    end

    @testset "finals land as finals/<trial_id>.parquet and the row points at them" begin
        finals = Float32[1.5 2.5 3.5; 0.1 0.2 0.3; 1.0 2.0 3.0; 4.0 5.0 6.0]
        relative = store_finals(none, finals, [true, false, true, true]; root = root)
        trial_id = store_hash(none; root = root)[1]
        @test relative == "finals/" * trial_id * ".parquet"
        @test isfile(joinpath(root, "key=" * KEY, "package=julia_gpu", "finals", trial_id * ".parquet"))
        store_record(store_row(none; states = 3, min_ms = 8.0, finals = relative); root = root)
        back = query_rows("SELECT finals FROM results WHERE transfers = 'none'", root)
        @test back[1]["finals"] == relative
        @test_throws ArgumentError store_finals(none, finals, [true]; root = root)
        # All n rows are required.
        @test_throws ProcessFailedException store_finals(none, finals[1:2, :], [true, false]; root = root)
    end

    @testset "specs outside the store vocabulary are refused" begin
        @test_throws ArgumentError store_row(fixture(; package = "julia"); states = 3)
        @test_throws ArgumentError store_row(delete!(fixture(), "gains"); states = 3)
        @test_throws ProcessFailedException store_record(store_row(fixture(; precision = "float16"); states = 3); root = root)
    end
end
