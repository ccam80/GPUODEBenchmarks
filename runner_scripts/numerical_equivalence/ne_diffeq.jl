# Numerical-equivalence (ne) sweeps for raw DifferentialEquations.jl in Float32.
#
# Two sweeps over every algorithm mutually supported by cubie and
# DifferentialEquations.jl (runner_scripts/numerical_equivalence/algorithms.csv):
#
# * fixed:    error-vs-dt convergence study, fixed-step at every dt in the
#             dyadic grid. Isolates the tableau from the controller.
# * adaptive: error-vs-tolerance study at atol = rtol in TOLS, each algorithm
#             under its DEFAULT step-size controller — solver performance
#             under real controller dynamics. Per-trajectory accept/reject
#             counts are recorded, and the resolved controller constants are
#             exported (controller_constants.csv) so the cubie runner can
#             mirror them exactly for its "matched" tier.
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
using SciMLBase: ODEProblem, EnsembleProblem, EnsembleThreads, remake, init
using CSV
using DelimitedFiles
using Printf

const MODE = isempty(ARGS) ? "all" : lowercase(ARGS[1])
MODE in ("fixed", "adaptive", "all") ||
    error("usage: ne_diffeq.jl [fixed|adaptive|all]")

const REPO_ROOT = dirname(dirname(@__DIR__))
const N_NE = 1024
# Dyadic dt grid, 2^-1 .. 2^-13 (same as ne_common.DTS_NE): coarse steps
# included so high-order methods have truncation error above the float32
# floor somewhere in the sweep.
const DTS_NE = [2.0^-k for k in 1:13]
# Adaptive protocol (same as ne_common): tolerance grid and pinned dt bounds.
const TOLS_NE = [10.0^-k for k in 2:6]
const DT0_NE = 0.01f0
const DT_MIN_NE = 1.0f-6
const DT_MAX_NE = 0.5f0

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

golden_path = joinpath(REPO_ROOT, "data", "numerical",
    "golden_ne_lorenz_1024.csv")
isfile(golden_path) || error(
    "$(golden_path) not found - generate it first with `julia -t auto " *
    "--project=. runner_scripts/numerical_equivalence/generate_golden_ne.jl`")
golden = readdlm(golden_path, ',')
size(golden) == (N_NE, 4) || error(
    "golden ne reference has size $(size(golden)), expected ($(N_NE), 4)")
rhos32 = Float32.(golden[:, 1])
# The rho values are float32-rounded, so the cast back is exact.
all(Float64.(rhos32) .== golden[:, 1]) || error(
    "golden rho column is not exactly representable in Float32")
golden_states = golden[:, 2:4]

function lorenz!(du, u, p, t)
    du[1] = 10.0f0 * (u[2] - u[1])
    du[2] = u[1] * (p[1] - u[3]) - u[2]
    du[3] = u[1] * u[2] - (8.0f0 / 3.0f0) * u[3]
    return nothing
end

u0 = Float32[1.0, 0.0, 0.0]
tspan = (0.0f0, 1.0f0)
prob = ODEProblem{true}(lorenz!, u0, tspan, Float32[rhos32[1]])
# SciMLBase's current ensemble API passes an EnsembleContext (with .sim_id)
# as the second argument of prob_func/output_func.
eprob = EnsembleProblem(prob;
    prob_func = (p, ctx) -> remake(p, p = Float32[rhos32[ctx.sim_id]]),
    output_func = (sol, ctx) -> ((sol.u[end], sol.retcode,
        sol.stats.naccept, sol.stats.nreject), false),
    safetycopy = false)

outdir = joinpath(REPO_ROOT, "data", "numerical_equivalence", "julia")
mkpath(outdir)

table = CSV.File(joinpath(@__DIR__, "algorithms.csv"))
failures = Tuple{String, Float64, String}[]

function collect_finals(sim)
    finals = Matrix{Float32}(undef, N_NE, 3)
    naccept = Vector{Int}(undef, N_NE)
    nreject = Vector{Int}(undef, N_NE)
    n_bad = 0
    for i in 1:N_NE
        u_end, retcode, na, nr = sim.u[i]
        eltype(u_end) === Float32 || error(
            "Float32 discipline violated: trajectory $(i) returned " *
            "eltype $(eltype(u_end))")
        finals[i, :] .= u_end
        naccept[i] = na
        nreject[i] = nr
        if retcode != SciMLBase.ReturnCode.Success
            n_bad += 1
        end
    end
    return finals, naccept, nreject, n_bad
end

ensemble_err(finals) = sqrt(sum(abs2, Float64.(finals) .- golden_states) /
                            length(golden_states))

# ---------------------------------------------------------------------------
# Fixed-step error-vs-dt sweep
# ---------------------------------------------------------------------------
if MODE in ("fixed", "all")
    for row in table
        alias = String(row.cubie_alias)
        expr = String(row.julia_expr)
        println("=== fixed $(alias) -> $(expr) (order $(row.order)) ===")
        alg = try
            eval(Meta.parse(expr))
        catch err
            msg = sprint(showerror, err)[1:min(end, 200)]
            println("  constructor FAILED: $(msg)")
            push!(failures, (alias, NaN, "constructor: $(msg)"))
            continue
        end

        io = IOBuffer()
        println(io, "dt,traj,x,y,z")
        wrote_any = false
        for dt in DTS_NE
            try
                # abstol/reltol are pinned to the OrdinaryDiffEq defaults
                # rather than left implicit: with adaptive=false they only
                # control the implicit solvers' Newton termination (accept
                # when eta*||dz/(abstol + reltol*|u|)|| < kappa = 1/100), and
                # the cubie runner's inner-tolerance pin is derived from
                # these values — see INNER_SOLVER_SETTINGS in
                # GPU_ODE_CUBIE/numerical_equivalence.py.
                sim = solve(eprob, alg, EnsembleThreads();
                    trajectories = N_NE, dt = Float32(dt), adaptive = false,
                    abstol = 1.0f-6, reltol = 1.0f-3,
                    save_everystep = false, save_start = false, dense = false)
                finals, _, _, n_bad = collect_finals(sim)
                err = ensemble_err(finals)
                note = n_bad == 0 ? "" : " ($(n_bad) non-Success retcodes)"
                @printf("  dt=%-12g err=%.6e%s\n", dt, err, note)
                for j in 1:N_NE
                    @printf(io, "%.10g,%d,%.9g,%.9g,%.9g\n", dt, j - 1,
                        finals[j, 1], finals[j, 2], finals[j, 3])
                end
                wrote_any = true
            catch err
                msg = sprint(showerror, err)[1:min(end, 200)]
                @printf("  dt=%-12g FAILED: %s\n", dt, msg)
                push!(failures, (alias, dt, msg))
            end
        end

        if wrote_any
            outfile = joinpath(outdir, "$(alias).csv")
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
if MODE in ("adaptive", "all")
    const_io = IOBuffer()
    println(const_io,
        "cubie_alias,controller,beta1,beta2,qmin,qmax,gamma," *
        "qsteady_min,qsteady_max,order")

    for row in table
        alias = String(row.cubie_alias)
        expr = String(row.julia_expr)
        alg = try
            eval(Meta.parse(expr))
        catch err
            msg = sprint(showerror, err)[1:min(end, 200)]
            println("=== adaptive $(alias): constructor FAILED: $(msg)")
            push!(failures, (alias, NaN, "constructor: $(msg)"))
            continue
        end
        if !OrdinaryDiffEqCore.isadaptive(alg)
            println("=== adaptive $(alias): skipped (not adaptive in " *
                    "OrdinaryDiffEq)")
            continue
        end
        println("=== adaptive $(alias) -> $(expr) (order $(row.order), " *
                "default controller) ===")

        # Resolve and export the default controller constants so the cubie
        # runner can mirror them ("matched" tier).
        try
            integ = init(prob, alg; dt = DT0_NE, abstol = 1.0f-4,
                reltol = 1.0f-4, save_everystep = false)
            ctrl = integ.controller_cache.controller
            basic = ctrl.basic
            cname = string(typeof(ctrl).name.name)
            b1 = hasproperty(ctrl, :beta1) ? string(ctrl.beta1) : ""
            b2 = hasproperty(ctrl, :beta2) ? string(ctrl.beta2) : ""
            println(const_io,
                "$(alias),$(cname),$(b1),$(b2),$(basic.qmin),$(basic.qmax)," *
                "$(basic.gamma),$(basic.qsteady_min),$(basic.qsteady_max)," *
                "$(row.order)")
            println("  controller: $(cname) beta1=$(b1) beta2=$(b2) " *
                    "qmin=$(basic.qmin) qmax=$(basic.qmax) " *
                    "gamma=$(basic.gamma) " *
                    "qsteady=($(basic.qsteady_min),$(basic.qsteady_max))")
        catch err
            msg = sprint(showerror, err)[1:min(end, 200)]
            println("  controller-constants export FAILED: $(msg)")
            push!(failures, (alias, NaN, "controller export: $(msg)"))
        end

        io = IOBuffer()
        println(io, "tol,traj,x,y,z,naccept,nreject")
        wrote_any = false
        for tol in TOLS_NE
            try
                sim = solve(eprob, alg, EnsembleThreads();
                    trajectories = N_NE, adaptive = true, dt = DT0_NE,
                    abstol = Float32(tol), reltol = Float32(tol),
                    dtmin = DT_MIN_NE, dtmax = DT_MAX_NE,
                    save_everystep = false, save_start = false, dense = false)
                finals, naccept, nreject, n_bad = collect_finals(sim)
                err = ensemble_err(finals)
                note = n_bad == 0 ? "" : " ($(n_bad) non-Success retcodes)"
                @printf("  tol=%-8g err=%.6e steps(med)=%d%s\n", tol, err,
                    Int(round(sum(naccept) / N_NE)), note)
                for j in 1:N_NE
                    @printf(io, "%.10g,%d,%.9g,%.9g,%.9g,%d,%d\n", tol, j - 1,
                        finals[j, 1], finals[j, 2], finals[j, 3],
                        naccept[j], nreject[j])
                end
                wrote_any = true
            catch err
                msg = sprint(showerror, err)[1:min(end, 200)]
                @printf("  tol=%-8g FAILED: %s\n", tol, msg)
                push!(failures, (alias, tol, msg))
            end
        end

        if wrote_any
            outfile = joinpath(outdir, "$(alias)_adaptive.csv")
            open(outfile, "w") do f
                write(f, take!(io))
            end
            println("  wrote $(outfile)")
        else
            println("  no successful tolerance points; nothing written")
        end
    end

    constfile = joinpath(outdir, "controller_constants.csv")
    open(constfile, "w") do f
        write(f, take!(const_io))
    end
    println("wrote $(constfile)")
end

if !isempty(failures)
    println("\n$(length(failures)) failed points:")
    for (alias, setting, msg) in failures
        println("  $(alias) @ $(setting): $(msg)")
    end
end
