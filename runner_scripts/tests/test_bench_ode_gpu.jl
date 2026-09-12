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
    watchdogged_min_ms(f, on_breach, repeats; cap_s = 120.0) = (1.5, [2.0, 1.5], f())
    run_watchdogged(f, on_breach; budget_s = 270.0) = f()
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
        "finals" => false, "transfers" => ["both", "none"], "cold" => false, "optimize" => nothing,
        "watchdog_s" => 120.0, "timed" => true, "sets" => ["perf"])
    for (name, value) in overrides
        record[String(name)] = value
    end
    return record
end

@testset "bench_ode_gpu.jl helpers" begin
    @testset "the CLI" begin
        cli = parse_cli(["--trials", "x.jsonl", "--floor"])
        @test cli.trials == "x.jsonl" && cli.floor
        @test !parse_cli(["--trials", "x.jsonl"]).floor
        parse_cli(["--trials", "x.jsonl", "--store-python", "C:/venv/python.exe"])
        @test STORE_PYTHON[] == "C:/venv/python.exe"
        @test_throws ErrorException parse_cli(["--floor"])
        @test_throws ErrorException parse_cli(["--trials", "x.jsonl", "--bogus"])
    end

    @testset "trials read back with null floats as NaN in file order" begin
        path = tempname() * ".jsonl"
        # trials.py writes NaN as null.
        line(record) = replace(JSON.json(record; allownan = true), "NaN" => "null")
        lines = [
            line(trial(; transfers = [], dt = nothing, atol = 1e-5)),
            line(trial(; dt = 0.5, atol = nothing)),
            line(trial(; n = 32)),
        ]
        write(path, join(lines, "\n") * "\n\n")
        back = read_trials(path)
        rm(path)
        @test length(back) == 3
        @test isnan(back[1]["dt"]) && back[1]["atol"] == 1e-5 && back[1]["transfers"] == []
        @test back[2]["dt"] == 0.5 && isnan(back[2]["atol"])
        @test back[3]["n"] == 32 && isnan(back[3]["dt_min"])
        @test back[1]["watchdog_s"] == 120.0
        @test back[1]["optimize"] === nothing
    end

    @testset "the progress file names the trial and its stage" begin
        path = tempname()
        write_progress(path, trial(; trial_id = "feedfacefeedface"), "solve")
        progress = JSON.parse(read(path, String))
        rm(path)
        @test progress["trial_id"] == "feedfacefeedface"
        @test progress["stage"] == "solve"
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

    @testset "the abandon rule gives up the harder runs of the family on the same transfers" begin
        failures = Dict{String, Vector{Tuple{Dict{String, Any}, String}}}()
        base = trial(; n = 32, trial_id = "base")
        note_failure!(failures, base, "both", "error")
        @test isempty(failures)
        note_failure!(failures, base, "none", "timeout")
        @test abandon_reason(trial(; n = 128), "none", failures) == "abandoned: timeout at base"
        @test abandon_reason(trial(; n = 32, dt = 2.0^-12), "none", failures) == "abandoned: timeout at base"
        @test abandon_reason(trial(; n = 128, dt = 0.5), "none", failures) === nothing
        @test abandon_reason(trial(; n = 8), "none", failures) === nothing
        @test abandon_reason(trial(; n = 128), "both", failures) === nothing
        @test abandon_reason(trial(; n = 128, algorithm = "vern7"), "none", failures) === nothing
        @test harder(trial(; n = 32, system_params = "{\"states\":64}"), trial(; n = 32, system_params = "{\"states\":32}"))
        @test !harder(base, base)
        @test trial_label(trial()) == "lorenz tsit5 fixed n=8 dt=0.0009765625"
    end

    @testset "outcome classification and texts" begin
        kind, reason = classify(CUDA.OutOfGPUMemoryError())
        @test kind == "oom" && reason == "oom: OutOfGPUMemoryError: Out of GPU memory"
        kind, reason = classify(ErrorException("CUDA_ERROR_OUT_OF_MEMORY while launching"))
        @test kind == "oom" && startswith(reason, "oom: ErrorException: ")
        kind, reason = classify(ArgumentError("bad " * "x"^300))
        @test kind == "error" && startswith(reason, "error: ArgumentError: ArgumentError: bad ")
        @test length(reason) <= length("error: ArgumentError: ") + 200
        ok = timed(() -> (1, 2), "leg", 120.0)
        @test ok.kind == "ok" && ok.result == (1, 2) && ok.min_ms == 1.5 && ok.samples == [2.0, 1.5]
        bad = timed(() -> error("boom"), "leg", 120.0)
        @test bad.kind == "error" && bad.reason == "error: ErrorException: boom" && isnan(bad.min_ms)
    end

    @testset "states come from the construction parameters" begin
        @test trial_states(trial(; problem = "lorenz96", system_params = "{\"states\":48}")) == 48
    end
end
