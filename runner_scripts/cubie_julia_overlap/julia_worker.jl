#!/usr/bin/env julia
# DiffEqGPU worker for the direct Cubie overlap suite.  Deliberately does not
# instantiate or precompile: environment setup is an explicit prerequisite.

using CUDA
using DiffEqGPU
using SciMLBase: ODEFunction, ODEProblem, remake
using StaticArrays
using CSV
using DelimitedFiles
using Printf

CUDA.allowscalar(false)

const HERE = @__DIR__
const REPO_ROOT = dirname(dirname(HERE))
# Protocol constants; mirrored in common.py.
const FIXED_DT = 2.0^-10
const ADAPTIVE_TOL = 1.0e-8
const PERFORMANCE_REPEATS = 20
const WORK_REPEATS = 20
const N_WP = 32768

function cli_args(args)
    out = Dict{String, String}()
    i = 1
    while i <= length(args)
        startswith(args[i], "--") || error("unexpected argument $(args[i])")
        key = args[i][3:end]
        i < length(args) || error("--$(key) requires a value")
        out[key] = args[i + 1]
        i += 2
    end
    return out
end

const OPT = cli_args(ARGS)
const OUT = abspath(haskey(OPT, "output") ? OPT["output"] : error("--output is required"))
const PROFILE = get(OPT, "profile", "smoke")
const ANALYSIS = get(OPT, "analysis", "all")
const NMAX = get(OPT, "nmax", "16777216")
const FROM_N = parse(Int, get(OPT, "from-n", "0"))
const ALGORITHM = get(OPT, "algorithm", "all")

mkpath(OUT)

# A single value is a sweep ceiling; a comma list is the exact counts.
function parse_ns(spec, from_n)
    if occursin(',', spec)
        values = sort(unique(parse.(Int, filter(!isempty, split(spec, ',')))))
        return filter(n -> n >= max(from_n, 8), values)
    end
    ns, n = Int[], 8
    while n <= parse(Int, spec)
        n >= from_n && push!(ns, n)
        n *= 4
    end
    return ns
end

function protocol()
    ns = parse_ns(NMAX, FROM_N)
    if PROFILE == "smoke"
        ns = filter(n -> n <= 32, ns)
    end
    if PROFILE == "smoke"
        return (performance_ns = ns, performance_repeats = 2,
            ne_n = 32, ne_dts = [2.0^-4, 2.0^-8], ne_tols = [1.0e-3],
            wp_n = 256, wp_dts = [2.0^-6], wp_tols = [1.0e-4],
            work_repeats = 2)
    end
    return (performance_ns = ns, performance_repeats = PERFORMANCE_REPEATS,
        ne_n = 1024, ne_dts = [2.0^-k for k in 1:13],
        ne_tols = [10.0^-k for k in 2:6], wp_n = N_WP,
        wp_dts = [2.0^-k for k in 4:13], wp_tols = [10.0^-k for k in 2:8],
        work_repeats = WORK_REPEATS)
end
const PROTOCOL = protocol()

const TIMING_HEADER = "framework,algorithm,phase,mode,tier,transfers,n,setting_kind,setting,sample,time_ms"
const METRIC_HEADER = "framework,algorithm,phase,mode,tier,n,setting_kind,setting,golden_rmse,finite_trajectories,failed_trajectories,finals_path"
const FAILURE_HEADER = "framework,algorithm,phase,mode,tier,n,setting_kind,setting,error_type,message"
const TIMING_FILE = joinpath(OUT, "julia_timings.csv")
const METRIC_FILE = joinpath(OUT, "julia_metrics.csv")
const FAILURE_FILE = joinpath(OUT, "julia_failures.csv")

# Append-only: the launcher clears the rows a run replaces.
function init_csv(path, header)
    if !isfile(path)
        open(path, "w") do io
            println(io, header)
        end
    end
end
init_csv(TIMING_FILE, TIMING_HEADER)
init_csv(METRIC_FILE, METRIC_HEADER)
init_csv(FAILURE_FILE, FAILURE_HEADER)

clean(value) = replace(replace(string(value), ',' => ';'), '\n' => ' ')
function append_row(path, values...)
    open(path, "a") do io
        println(io, join(clean.(values), ','))
        flush(io)
    end
end

function lorenz(u, p, t)
    return @SVector [10.0f0 * (u[2] - u[1]),
        u[1] * (p[1] - u[3]) - u[2],
        u[1] * u[2] - (8.0f0 / 3.0f0) * u[3]]
end
function lorenz_jac(u, p, t)
    return @SMatrix [-10.0f0 10.0f0 0.0f0;
        p[1] - u[3] -1.0f0 -u[1];
        u[2] u[1] -(8.0f0 / 3.0f0)]
end
lorenz_tgrad(u, p, t) = @SVector [0.0f0, 0.0f0, 0.0f0]

const ODEF = ODEFunction{false}(lorenz; jac = lorenz_jac, tgrad = lorenz_tgrad)
const U0 = @SVector [1.0f0, 0.0f0, 0.0f0]
const TSPAN = (0.0f0, 1.0f0)

const golden_ne_all = readdlm(joinpath(REPO_ROOT, "data", "numerical",
    "golden_ne_lorenz_1024.csv"), ',', Float64)
const golden_wp_all = readdlm(joinpath(REPO_ROOT, "data", "numerical",
    "golden_lorenz_32768.csv"), ',', Float64)

function rho_grid(kind, n)
    if kind == "numerical"
        return Float32.(golden_ne_all[1:n, 1])
    elseif kind == "work_precision"
        return Float32.(range(0.0, 21.0, length = N_WP))[1:n]
    end
    return Float32.(range(0.0, 21.0, length = n))
end

function build_problems(kind, n)
    rhos = rho_grid(kind, n)
    prob = ODEProblem{false}(ODEF, U0, TSPAN, @SVector [rhos[1]])
    probs = map(eachindex(rhos)) do i
        DiffEqGPU.make_prob_compatible(remake(prob, p = @SVector [rhos[i]]))
    end
    # Host vector is returned too so the end-to-end timing can re-upload it.
    return probs, cu(probs), prob
end

function run_solve(probs, prob, alg, mode, setting)
    if mode == "fixed"
        return DiffEqGPU.vectorized_solve(probs, prob, alg; saveat = 1.0f0,
            save_everystep = false, dt = Float32(setting))
    else
        return DiffEqGPU.vectorized_asolve(probs, prob, alg; saveat = 1.0f0,
            save_everystep = false, dt = 0.01f0,
            abstol = Float32(setting), reltol = Float32(setting))
    end
end

"Time one solve including the h2d and d2h transfers; the reshape is untimed."
function solve_end_to_end(probs_host, prob, alg, mode, setting)
    CUDA.synchronize()
    start = time_ns()
    probs = cu(probs_host)
    sol = run_solve(probs, prob, alg, mode, setting)
    host_us = Array(sol[2])
    CUDA.synchronize()
    elapsed_ms = (time_ns() - start) / 1.0e6
    final_vectors = host_us[end, :]
    finals = Matrix{Float32}(undef, length(final_vectors), 3)
    for i in eachindex(final_vectors)
        finals[i, :] .= final_vectors[i]
    end
    size(finals) == (length(probs_host), 3) || error(
        "unexpected final-state size $(size(finals)); expected ($(length(probs_host)), 3)")
    return finals, elapsed_ms
end

"Time one solve with neither transfer: probs already resident, results left there."
function solve_device_only(probs, prob, alg, mode, setting)
    CUDA.synchronize()
    start = time_ns()
    run_solve(probs, prob, alg, mode, setting)
    CUDA.synchronize()
    return (time_ns() - start) / 1.0e6
end

function finite_counts(finals)
    good = [all(isfinite, @view finals[i, :]) for i in axes(finals, 1)]
    return count(identity, good), count(!, good)
end

function golden_rmse(finals, golden)
    good = [all(isfinite, @view finals[i, :]) for i in axes(finals, 1)]
    any(good) || return NaN
    delta = Float64.(finals[good, :]) .- golden[good, :]
    return sqrt(sum(abs2, delta) / length(delta))
end

slug(value) = replace(replace(replace(@sprintf("%.10g", value), "-" => "m"),
    "+" => "p"), "." => "p")
function write_finals(alias, mode, tier, setting_kind, setting, finals)
    relative = joinpath("finals", "julia", alias,
        "$(mode)_$(tier)_$(setting_kind)_$(slug(setting)).csv")
    path = joinpath(OUT, relative)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "traj,x,y,z")
        for i in axes(finals, 1)
            @printf(io, "%d,%.9g,%.9g,%.9g\n", i - 1, finals[i, 1], finals[i, 2], finals[i, 3])
        end
    end
    return replace(relative, '\\' => '/')
end

function record_failure(alias, phase, mode, tier, n, setting_kind, setting, err)
    POINT_FAILURE_COUNT[] += 1
    append_row(FAILURE_FILE, "julia", alias, phase, mode, tier, n,
        setting_kind, setting, nameof(typeof(err)), sprint(showerror, err)[1:min(end, 2000)])
    println("FAILED julia $(alias) $(phase) $(mode) $(setting_kind)=$(setting): $(sprint(showerror, err))")
end

const POINT_FAILURE_COUNT = Ref(0)

table = collect(CSV.File(joinpath(HERE, "algorithms.csv")))
if ALGORITHM != "all"
    table = filter(row -> String(row.cubie_alias) == ALGORITHM, table)
    isempty(table) && error("unknown algorithm '$(ALGORITHM)'; see algorithms.csv")
end
phases = ANALYSIS == "all" ? ("performance", "numerical", "work_precision") :
    (replace(ANALYSIS, "-" => "_"),)

for row in table
    alias = String(row.cubie_alias)
    alg = try
        eval(Meta.parse(String(row.julia_constructor)))
    catch err
        for phase in phases
            record_failure(alias, phase, "all", "julia", 0, "constructor", NaN, err)
        end
        continue
    end
    for phase in phases
        points = Tuple{String, String, Float64, Int}[]
        repeats = 1
        if phase == "performance"
            for n in PROTOCOL.performance_ns
                push!(points, ("fixed", "dt", FIXED_DT, n))
                push!(points, ("adaptive", "tol", ADAPTIVE_TOL, n))
            end
            repeats = PROTOCOL.performance_repeats
        elseif phase == "numerical"
            append!(points, [("fixed", "dt", dt, PROTOCOL.ne_n) for dt in PROTOCOL.ne_dts])
            append!(points, [("adaptive", "tol", tol, PROTOCOL.ne_n) for tol in PROTOCOL.ne_tols])
        else
            append!(points, [("fixed", "dt", dt, PROTOCOL.wp_n) for dt in PROTOCOL.wp_dts])
            append!(points, [("adaptive", "tol", tol, PROTOCOL.wp_n) for tol in PROTOCOL.wp_tols])
            repeats = PROTOCOL.work_repeats
        end

        # Memory need is linear in N, so once a mode OOMs, larger N cannot run.
        oom_ceiling = Dict{String, Int}()

        for (mode, setting_kind, setting, n) in points
            tier = mode == "fixed" ? "fixed" : "julia"
            if haskey(oom_ceiling, mode) && n >= oom_ceiling[mode]
                println("SKIPPED julia $(alias) $(phase) $(mode) $(setting_kind)=$(setting) " *
                        "N=$(n): at or above the N=$(oom_ceiling[mode]) out-of-memory ceiling")
                continue
            end
            try
                probs_host, probs, prob = build_problems(phase, n)
                # Warm both paths.
                solve_end_to_end(probs_host, prob, alg, mode, setting)
                solve_device_only(probs, prob, alg, mode, setting)
                finals = Matrix{Float32}(undef, n, 3)
                for sample in 0:(repeats - 1)
                    finals, elapsed = solve_end_to_end(probs_host, prob, alg, mode, setting)
                    finite, failed = finite_counts(finals)
                    if failed > 0 || finite != n
                        append_row(METRIC_FILE, "julia", alias, phase, mode, tier,
                            n, setting_kind, setting, "", finite, failed, "")
                        error("non-finite result: $(finite)/$(n) trajectories valid")
                    end
                    append_row(TIMING_FILE, "julia", alias, phase, mode, tier,
                        "both", n, setting_kind, setting, sample, elapsed)
                    append_row(TIMING_FILE, "julia", alias, phase, mode, tier,
                        "none", n, setting_kind, setting, sample,
                        solve_device_only(probs, prob, alg, mode, setting))
                end
                finite, failed = finite_counts(finals)
                if phase == "performance"
                    append_row(METRIC_FILE, "julia", alias, phase, mode, tier,
                        n, setting_kind, setting, "", finite, failed, "")
                else
                    golden = phase == "numerical" ? golden_ne_all[1:n, 2:4] : golden_wp_all[1:n, :]
                    finals_path = phase == "numerical" ?
                        write_finals(alias, mode, tier, setting_kind, setting, finals) : ""
                    append_row(METRIC_FILE, "julia", alias, phase, mode, tier, n,
                        setting_kind, setting, golden_rmse(finals, golden),
                        finite, failed, finals_path)
                end
                println("OK julia $(alias) $(phase) $(mode) $(setting_kind)=$(setting) N=$(n)")
            catch err
                record_failure(alias, phase, mode, tier, n, setting_kind, setting, err)
                if phase == "performance" && err isa CUDA.OutOfGPUMemoryError
                    oom_ceiling[mode] = n
                end
            end
        end
    end
end

open(joinpath(OUT, "julia_metadata.json"), "w") do io
    println(io, "{")
    println(io, "  \"framework\": \"DiffEqGPU\",")
    println(io, "  \"julia_version\": \"$(VERSION)\",")
    println(io, "  \"diffeqgpu_version\": \"$(pkgversion(DiffEqGPU))\",")
    println(io, "  \"cuda_runtime\": \"$(CUDA.runtime_version())\",")
    println(io, "  \"profile\": \"$(PROFILE)\"")
    println(io, "}")
end

exit(POINT_FAILURE_COUNT[] == 0 ? 0 : 1)
