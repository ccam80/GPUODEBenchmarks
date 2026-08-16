# Numerical-equivalence (ne) sweeps for raw DifferentialEquations.jl in Float32.
#
# Two sweeps over every algorithm mutually supported by cubie and
# DifferentialEquations.jl (runner_scripts/numerical_equivalence/algorithms.csv):
#
# * fixed:    error-vs-dt convergence study, fixed-step at every dt in the
#             dyadic grid. Isolates the tableau from the controller.
#             erk-family rows are excluded.
# * adaptive: error-vs-tolerance study at atol = rtol in TOLS over the
#             mutual adaptive set (the csv's `adaptive` column), each
#             algorithm under its DEFAULT step-size controller — solver
#             performance under real controller dynamics. Per-trajectory
#             accept/reject counts are recorded, and the resolved controller
#             constants are exported (controller_constants.csv) so the cubie
#             runner can mirror them exactly for its "matched" tier.
#
# The protocol mirrors ne_common.py; keep the two in sync.
#
# Float32 discipline: u0, tspan, dt, tolerances and the parameter vector are
# all Float32 (the rho grid is read from the golden file, whose values are
# exactly representable in Float32), and every trajectory's final state is
# asserted to still be Float32 — a Float64 anywhere means the solve silently
# promoted and the point is recorded as failed.
#
# Outputs (machine independent, CPU), rows are 0-based traj indices:
#   data/numerical_equivalence/julia/<alias>.csv            dt,traj,x,y,z
#   data/numerical_equivalence/julia/<alias>_adaptive.csv   tol,traj,x,y,z,naccept,nreject
#   data/numerical_equivalence/julia/controller_constants.csv
#
# Run from the repo root:
#   julia -t auto --project=. runner_scripts/numerical_equivalence/ne_diffeq.jl [fixed|adaptive|all]

using OrdinaryDiffEq
using OrdinaryDiffEqLowOrderRK, OrdinaryDiffEqHighOrderRK
using OrdinaryDiffEqSDIRK, OrdinaryDiffEqFIRK, OrdinaryDiffEqRosenbrock
using OrdinaryDiffEqExplicitRK, OrdinaryDiffEqVerner
import OrdinaryDiffEqCore
import DiffEqBase
# The slim OrdinaryDiffEq v7 umbrella doesn't re-export the ensemble API.
using SciMLBase: ODEProblem, ODEFunction, EnsembleProblem, EnsembleThreads, remake, init
using CSV
using DelimitedFiles
using Printf

function cli_args(args)
    out = Dict{String, String}()
    i = 1
    while i <= length(args)
        startswith(args[i], "--") || error("unexpected argument $(args[i])")
        i < length(args) || error("$(args[i]) requires a value")
        out[args[i][3:end]] = args[i + 1]
        i += 2
    end
    return out
end
const NE_OPT = cli_args(ARGS)
const MODE = lowercase(get(NE_OPT, "controller", "all"))
const ALGORITHM = get(NE_OPT, "algorithm", "all")
MODE in ("fixed", "adaptive", "all") ||
    error("--controller must be fixed, adaptive or all")

const REPO_ROOT = dirname(dirname(@__DIR__))
include(joinpath(REPO_ROOT, "runner_scripts", "problems.jl"))
include(joinpath(REPO_ROOT, "runner_scripts", "julia_systems.jl"))
const N_NE = 1024
# Adaptive protocol (same as ne_common): tolerance grid and dt pins as
# fractions of the problem duration.
const TOLS_NE = [10.0^-k for k in 2:8]
const DT0_FRACTION = 1.0f-2
const DT_MIN_FRACTION = 1.0f-6
const DT_MAX_FRACTION = 0.5f0

const PROBLEM = get(NE_OPT, "problem", "all")
const PROBLEMS = resolve_problems(PROBLEM, "julia")
isempty(PROBLEMS) && error("no problem matches '$(PROBLEM)'")

# Cash-Karp 5(4) and Fehlberg 4(5) tableaus for the generic ExplicitRK
# stepper. DiffEqDevTools 3.x removed its construct* tableau library, so the
# published coefficients are entered here as rationals (Cash & Karp 1990;
# Fehlberg, NASA TR R-315 1969). Both propagate the 5th-order weights,
# matching cubie's cash-karp-54 / fehlberg-45 tableaus coefficient for
# coefficient.
function construct_cash_karp_54(::Type{T}) where {T}
    A = [
        0 0 0 0 0 0
        1//5 0 0 0 0 0
        3//40 9//40 0 0 0 0
        3//10 -9//10 6//5 0 0 0
        -11//54 5//2 -70//27 35//27 0 0
        1631//55296 175//512 575//13824 44275//110592 253//4096 0
    ]
    c = [0; 1 // 5; 3 // 10; 3 // 5; 1; 7 // 8]
    α = [37 // 378; 0; 250 // 621; 125 // 594; 0; 512 // 1771]
    αEEst = [2825 // 27648; 0; 18575 // 48384; 13525 // 55296; 277 // 14336; 1 // 4]
    return DiffEqBase.ExplicitRKTableau(map(T, A), map(T, c), map(T, α), 5;
        αEEst = map(T, αEEst), adaptiveorder = 4)
end

function construct_fehlberg_45(::Type{T}) where {T}
    A = [
        0 0 0 0 0 0
        1//4 0 0 0 0 0
        3//32 9//32 0 0 0 0
        1932//2197 -7200//2197 7296//2197 0 0 0
        439//216 -8 3680//513 -845//4104 0 0
        -8//27 2 -3544//2565 1859//4104 -11//40 0
    ]
    c = [0; 1 // 4; 3 // 8; 12 // 13; 1; 1 // 2]
    α = [16 // 135; 0; 6656 // 12825; 28561 // 56430; -9 // 50; 2 // 55]
    αEEst = [25 // 216; 0; 1408 // 2565; 2197 // 4104; -1 // 5; 0]
    return DiffEqBase.ExplicitRKTableau(map(T, A), map(T, c), map(T, α), 5;
        αEEst = map(T, αEEst), adaptiveorder = 4)
end


const TABLE_ALL = collect(CSV.File(joinpath(@__DIR__, "algorithms.csv")))
const TABLE = ALGORITHM == "all" ? TABLE_ALL :
    filter(row -> String(row.cubie_alias) == ALGORITHM, TABLE_ALL)
isempty(TABLE) && error("unknown algorithm '$(ALGORITHM)'; see algorithms.csv")

failures = Tuple{String, String, Float64, String}[]

"Golden reference, ensemble problem and output directory for one problem."
function setup(problem)
    name = problem["problem"]
    nstates = problem["states"]
    golden_path = joinpath(REPO_ROOT, "data", "numerical",
        "golden_ne_$(name)_$(N_NE).csv")
    isfile(golden_path) || error(
        "$(golden_path) not found - generate it first with `julia -t auto " *
        "--project=. runner_scripts/numerical_equivalence/generate_golden_ne.jl " *
        "--problem $(name)`")
    golden = readdlm(golden_path, ',')
    size(golden) == (N_NE, nstates + 1) || error(
        "golden ne reference has size $(size(golden)), expected " *
        "($(N_NE), $(nstates + 1))")
    sweep32 = Float32.(golden[:, 1])
    # The swept values are float32-rounded, so the cast back is exact.
    all(Float64.(sweep32) .== golden[:, 1]) || error(
        "golden parameter column is not exactly representable in Float32")

    system = julia_system(problem)
    duration = Float32(problem["duration"])
    u0 = Vector{Float32}(system.u0)
    f = system.mass_matrix === nothing ? ODEFunction{true}(system.rhs!) :
        ODEFunction{true}(system.rhs!; mass_matrix = system.mass_matrix)
    prob = ODEProblem{true}(f, u0, (0.0f0, duration), Float32[sweep32[1]])
    # SciMLBase's current ensemble API passes an EnsembleContext (with .sim_id)
    # as the second argument of prob_func/output_func.
    eprob = EnsembleProblem(prob;
        prob_func = (pr, ctx) -> remake(pr, p = Float32[sweep32[ctx.sim_id]]),
        output_func = (sol, ctx) -> ((sol.u[end], sol.retcode,
            sol.stats.naccept, sol.stats.nreject), false),
        safetycopy = false)

    outdir = joinpath(REPO_ROOT, "data", "numerical_equivalence", "julia", name)
    mkpath(outdir)
    return (name = name, nstates = nstates, golden_states = golden[:, 2:end],
        prob = prob, eprob = eprob, outdir = outdir,
        dts = problem_dts_ne(problem),
        dt0 = duration * DT0_FRACTION, dtmin = duration * DT_MIN_FRACTION,
        dtmax = duration * DT_MAX_FRACTION)
end

problem_dts_ne(problem) = problem_ne_dts(problem)

"Final states, step counts and retcodes of one ensemble solve."
function collect_finals(sim, nstates)
    finals = Matrix{Float32}(undef, N_NE, nstates)
    naccept = Vector{Int}(undef, N_NE)
    nreject = Vector{Int}(undef, N_NE)
    converged = Vector{Bool}(undef, N_NE)
    n_bad = 0
    for i in 1:N_NE
        u_end, retcode, na, nr = sim.u[i]
        eltype(u_end) === Float32 || error(
            "Float32 discipline violated: trajectory $(i) returned " *
            "eltype $(eltype(u_end))")
        finals[i, :] .= u_end
        naccept[i] = na
        nreject[i] = nr
        ok = retcode == SciMLBase.ReturnCode.Success
        converged[i] = ok
        ok || (n_bad += 1)
    end
    return finals, naccept, nreject, n_bad, converged
end

ensemble_err(finals, golden_states) =
    sqrt(sum(abs2, Float64.(finals) .- golden_states) / length(golden_states))

"Comma-separated final state of one trajectory."
state_fields(finals, j) = join([@sprintf("%.9g", finals[j, s])
                                for s in 1:size(finals, 2)], ",")

state_header(nstates) = join(["s$(s)" for s in 1:nstates], ",")

# ---------------------------------------------------------------------------
# Fixed-step error-vs-dt sweep
# ---------------------------------------------------------------------------
function run_fixed(ctx)
    for row in TABLE
        alias = String(row.cubie_alias)
        expr = String(row.julia_expr)
        if String(row.family) == "erk"
            println("=== fixed $(alias): skipped (no fixed sweep for erk)")
            continue
        end
        println("=== $(ctx.name) fixed $(alias) -> $(expr) (order $(row.order)) ===")
        alg = try
            eval(Meta.parse(expr))
        catch err
            msg = sprint(showerror, err)[1:min(end, 200)]
            println("  constructor FAILED: $(msg)")
            push!(failures, (ctx.name, alias, NaN, "constructor: $(msg)"))
            continue
        end

        io = IOBuffer()
        println(io, "dt,traj,$(state_header(ctx.nstates)),converged")
        wrote_any = false
        for dt in ctx.dts
            try
                # abstol/reltol are pinned to the OrdinaryDiffEq defaults
                # rather than left implicit: with adaptive=false they only
                # control the implicit solvers' Newton termination, and the
                # cubie runner's inner-tolerance pin is derived from them.
                sim = solve(ctx.eprob, alg, EnsembleThreads();
                    trajectories = N_NE, dt = Float32(dt), adaptive = false,
                    abstol = 1.0f-6, reltol = 1.0f-3,
                    save_everystep = false, save_start = false, dense = false)
                finals, _, _, n_bad, converged = collect_finals(sim, ctx.nstates)
                err = ensemble_err(finals, ctx.golden_states)
                note = n_bad == 0 ? "" : " ($(n_bad) non-Success retcodes)"
                @printf("  dt=%-12g err=%.6e%s\n", dt, err, note)
                for j in 1:N_NE
                    @printf(io, "%.10g,%d,%s,%d\n", dt, j - 1,
                        state_fields(finals, j), converged[j] ? 1 : 0)
                end
                wrote_any = true
            catch err
                msg = sprint(showerror, err)[1:min(end, 200)]
                @printf("  dt=%-12g FAILED: %s\n", dt, msg)
                push!(failures, (ctx.name, alias, dt, msg))
            end
        end

        if wrote_any
            outfile = joinpath(ctx.outdir, "$(alias).csv")
            open(outfile, "w") do f
                write(f, take!(io))
            end
            println("  wrote $(outfile)")
        else
            println("  no successful dt points; nothing written")
        end
    end
end

# ---------------------------------------------------------------------------
# Adaptive error-vs-tolerance sweep (default controllers) + constants export
# ---------------------------------------------------------------------------
function run_adaptive(ctx)
    const_io = IOBuffer()
    println(const_io,
        "cubie_alias,controller,beta1,beta2,qmin,qmax,gamma,order")

    for row in TABLE
        alias = String(row.cubie_alias)
        expr = String(row.julia_expr)
        # Only the mutual adaptive set runs.
        if lowercase(string(row.adaptive)) != "true"
            println("=== adaptive $(alias): skipped (not in the mutual " *
                    "adaptive set)")
            continue
        end
        alg = try
            eval(Meta.parse(expr))
        catch err
            msg = sprint(showerror, err)[1:min(end, 200)]
            println("=== adaptive $(alias): constructor FAILED: $(msg)")
            push!(failures, (ctx.name, alias, NaN, "constructor: $(msg)"))
            continue
        end
        if !OrdinaryDiffEqCore.isadaptive(alg)
            println("=== adaptive $(alias): skipped (not adaptive in " *
                    "OrdinaryDiffEq)")
            continue
        end
        println("=== $(ctx.name) adaptive $(alias) -> $(expr) " *
                "(order $(row.order), default controller) ===")

        # Resolve and export the default controller constants so the cubie
        # runner can mirror them ("matched" tier).
        try
            integ = init(ctx.prob, alg; dt = ctx.dt0, abstol = 1.0f-4,
                reltol = 1.0f-4, save_everystep = false)
            ctrl = integ.controller_cache.controller
            basic = ctrl.basic
            cname = string(typeof(ctrl).name.name)
            b1 = hasproperty(ctrl, :beta1) ? string(ctrl.beta1) : ""
            b2 = hasproperty(ctrl, :beta2) ? string(ctrl.beta2) : ""
            println(const_io,
                "$(alias),$(cname),$(b1),$(b2),$(basic.qmin),$(basic.qmax)," *
                "$(basic.gamma),$(row.order)")
            println("  controller: $(cname) beta1=$(b1) beta2=$(b2) " *
                    "qmin=$(basic.qmin) qmax=$(basic.qmax) " *
                    "gamma=$(basic.gamma)")
        catch err
            msg = sprint(showerror, err)[1:min(end, 200)]
            println("  controller-constants export FAILED: $(msg)")
            push!(failures, (ctx.name, alias, NaN, "controller export: $(msg)"))
        end

        io = IOBuffer()
        println(io,
            "tol,traj,$(state_header(ctx.nstates)),naccept,nreject,converged")
        wrote_any = false
        for tol in TOLS_NE
            try
                sim = solve(ctx.eprob, alg, EnsembleThreads();
                    trajectories = N_NE, adaptive = true, dt = ctx.dt0,
                    abstol = Float32(tol), reltol = Float32(tol),
                    dtmin = ctx.dtmin, dtmax = ctx.dtmax,
                    save_everystep = false, save_start = false, dense = false)
                finals, naccept, nreject, n_bad, converged =
                    collect_finals(sim, ctx.nstates)
                err = ensemble_err(finals, ctx.golden_states)
                note = n_bad == 0 ? "" : " ($(n_bad) non-Success retcodes)"
                @printf("  tol=%-8g err=%.6e steps(med)=%d%s\n", tol, err,
                    Int(round(sum(naccept) / N_NE)), note)
                for j in 1:N_NE
                    @printf(io, "%.10g,%d,%s,%d,%d,%d\n", tol, j - 1,
                        state_fields(finals, j), naccept[j], nreject[j],
                        converged[j] ? 1 : 0)
                end
                wrote_any = true
            catch err
                msg = sprint(showerror, err)[1:min(end, 200)]
                @printf("  tol=%-8g FAILED: %s\n", tol, msg)
                push!(failures, (ctx.name, alias, tol, msg))
            end
        end

        if wrote_any
            outfile = joinpath(ctx.outdir, "$(alias)_adaptive.csv")
            open(outfile, "w") do f
                write(f, take!(io))
            end
            println("  wrote $(outfile)")
        else
            println("  no successful tolerance points; nothing written")
        end
    end

    constfile = joinpath(ctx.outdir, "controller_constants.csv")
    open(constfile, "w") do f
        write(f, take!(const_io))
    end
    println("wrote $(constfile)")
end

for problem in PROBLEMS
    ctx = setup(problem)
    MODE in ("fixed", "all") && run_fixed(ctx)
    MODE in ("adaptive", "all") && run_adaptive(ctx)
end

if !isempty(failures)
    println("\n$(length(failures)) failed points:")
    for (problem, alias, setting, msg) in failures
        println("  $(problem) $(alias) @ $(setting): $(msg)")
    end
end
