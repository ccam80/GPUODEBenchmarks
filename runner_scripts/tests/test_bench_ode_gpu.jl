# The GPU-free parts of GPU_ODE_Julia/bench_ode_gpu.jl: trial parsing, the rejections, the abandon rule and the outcome texts; run with `julia --project=. runner_scripts/tests/test_bench_ode_gpu.jl`.

using Test

const BENCH = joinpath(dirname(dirname(@__DIR__)), "GPU_ODE_Julia", "bench_ode_gpu.jl")

# Load the script's helpers without its package preamble or its exit line, with the GPU and store pieces stubbed.
function load_helpers()
    source = read(BENCH, String)
    body = split(source, "CUDA.allowscalar(false)")[2]
    body = replace(body, r"exit\(main\(ARGS\)\)\s*$" => "")
    stubs = """
    const REPEAT_CAP = 3
    const WATCHDOG_SECONDS = 120.0
    module CUDA
        struct OutOfGPUMemoryError <: Exception end
        Base.showerror(io::IO, ::OutOfGPUMemoryError) = print(io, "Out of GPU memory")
    end
    module Dates
        using Dates: now, UTC, format
    end
    watchdogged_min_ms(f, on_breach, repeats) = (1.5, [2.0, 1.5], f())
    run_watchdogged(f, on_breach) = f()
    mkpidlock(path; kwargs...) = nothing
    const STORE_PYTHON = Ref("")
    """
    Base.include_string(Main, stubs * body, "bench_ode_gpu_helpers.jl")
end

using JSON
load_helpers()

function trial(; overrides...)
    record = Dict{String, Any}("problem" => "lorenz", "system_params" => "{}", "duration" => 1.0,
        "precision" => "float32", "parameter" => "rho", "grid_scale" => "linear",
        "grid_min" => 0.0, "grid_max" => 21.0, "n" => 8, "grid_dtype" => "float32",
        "algorithm" => "tsit5", "controller" => "fixed", "dt" => 2.0^-10, "dt_min" => NaN,
        "dt_max" => NaN, "atol" => NaN, "rtol" => NaN, "gains" => "{}", "newton_atol" => NaN,
        "newton_rtol" => NaN, "package" => "julia_gpu", "trial_id" => "0123456789abcdef",
        "kind" => "solve", "finals" => false, "transfers" => ["both", "none"],
        "leg" => "lorenz/{}/tsit5/fixed/float32/n", "axis" => "n", "ordinal" => 0, "cold" => false)
    for (name, value) in overrides
        record[String(name)] = value
    end
    return record
end

@testset "bench_ode_gpu.jl helpers" begin
    @testset "the CLI" begin
        cli = parse_cli(["--trials", "x.jsonl", "--gpu-lock", "x.lock", "--floor"])
        @test cli.trials == "x.jsonl" && cli.lock == "x.lock" && cli.floor
        @test !parse_cli(["--trials", "x.jsonl"]).floor
        parse_cli(["--trials", "x.jsonl", "--store-python", "C:/venv/python.exe"])
        @test STORE_PYTHON[] == "C:/venv/python.exe"
        @test_throws ErrorException parse_cli(["--floor"])
        @test_throws ErrorException parse_cli(["--trials", "x.jsonl", "--bogus"])
    end

    @testset "trials read back with null floats as NaN, grouped by leg in file order" begin
        path = tempname() * ".jsonl"
        # trials.py writes NaN as null.
        line(record) = replace(JSON.json(record; allownan = true), "NaN" => "null")
        lines = [
            line(trial(; kind = "warm", transfers = [], dt = nothing, atol = 1e-5)),
            line(trial(; dt = 0.5, atol = nothing, leg = "b")),
            line(trial(; n = 32, ordinal = 1)),
        ]
        write(path, join(lines, "\n") * "\n\n")
        back = read_trials(path)
        rm(path)
        @test length(back) == 3
        @test isnan(back[1]["dt"]) && back[1]["atol"] == 1e-5 && back[1]["transfers"] == []
        @test back[2]["dt"] == 0.5 && isnan(back[2]["atol"])
        @test back[3]["n"] == 32 && isnan(back[3]["dt_min"])
        legs = by_leg(back)
        @test [leg for (leg, _) in legs] == ["lorenz/{}/tsit5/fixed/float32/n", "b"]
        @test length(legs[1][2]) == 2 && length(legs[2][2]) == 1
    end

    @testset "the progress file names the trial" begin
        path = tempname()
        write_progress(path, trial(; trial_id = "feedfacefeedface"))
        progress = JSON.parse(read(path, String))
        rm(path)
        @test progress["trial_id"] == "feedfacefeedface"
        @test endswith(progress["started_utc"], "Z")
    end

    @testset "rejections" begin
        @test reject_reason(trial()) === nothing
        @test reject_reason(trial(; controller = "default", atol = 1e-5, rtol = 1e-5)) === nothing
        @test reject_reason(trial(; controller = "pi")) == "error: unknown controller pi"
        @test reject_reason(trial(; precision = "float64")) == "error: unsupported precision float64"
        @test startswith(reject_reason(trial(; dt_min = 1e-6)), "error: unsupported dt_min")
        @test startswith(reject_reason(trial(; dt_max = 0.1)), "error: unsupported dt_max")
        @test reject_reason(trial(; gains = "{\"beta1\":0.7}")) == "error: unsupported gains {\"beta1\":0.7}"
    end

    @testset "the abandon rule marks one transfers leg from the failing ordinal on" begin
        state = LegState(Dict{String, String}(), NaN, "")
        abandon!(state, "both", failed("error", "error: x"), 2)
        @test isempty(state.abandoned)
        abandon!(state, "none", failed("timeout", "timeout: run exceeded 120.0s"), 2)
        @test state.abandoned == Dict("none" => "abandoned: timeout at ordinal 2")
        abandon!(state, "none", failed("oom", "oom: y"), 3)
        @test state.abandoned["none"] == "abandoned: timeout at ordinal 2"
        abandon!(state, "both", failed("oom", "oom: y"), 3)
        @test state.abandoned["both"] == "abandoned: oom at ordinal 3"
    end

    @testset "outcome classification and texts" begin
        kind, reason = classify(CUDA.OutOfGPUMemoryError())
        @test kind == "oom" && reason == "oom: OutOfGPUMemoryError: Out of GPU memory"
        kind, reason = classify(ErrorException("CUDA_ERROR_OUT_OF_MEMORY while launching"))
        @test kind == "oom" && startswith(reason, "oom: ErrorException: ")
        kind, reason = classify(ArgumentError("bad " * "x"^300))
        @test kind == "error" && startswith(reason, "error: ArgumentError: ArgumentError: bad ")
        @test length(reason) <= length("error: ArgumentError: ") + 200
        ok = timed(() -> (1, 2), "leg")
        @test ok.kind == "ok" && ok.result == (1, 2) && ok.min_ms == 1.5 && ok.samples == [2.0, 1.5]
        bad = timed(() -> error("boom"), "leg")
        @test bad.kind == "error" && bad.reason == "error: ErrorException: boom" && isnan(bad.min_ms)
    end

    @testset "states come from the construction parameters" begin
        @test trial_states(trial(; problem = "lorenz96", system_params = "{\"states\":48}")) == 48
    end
end
