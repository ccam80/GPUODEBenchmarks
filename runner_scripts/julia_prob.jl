# The DiffEqGPU kernel path shared by bench_ode_gpu.jl, the overlap worker and GPU_ODE_JuliaKernels; every side builds identical types.

"DiffEqGPU kernel solver named by the julia_gpu column of algorithms.csv."
function gpu_solver(algorithm)
    expr = get_algorithm(algorithm)["julia_gpu"]
    isempty(expr) && error("no DiffEqGPU kernel solver for '$(algorithm)'")
    return Base.eval(@__MODULE__, Meta.parse(expr))
end

"The per-problem pieces every sweep size shares."
build_prob(problem) = build_prob_parts(julia_system(problem), problem)

function build_prob_parts(system, problem)
    duration = Float32(problem["duration"])
    f = system.mass_matrix === nothing ?
        ODEFunction{false}(system.rhs; jac = system.jac,
            tgrad = system.tgrad) :
        ODEFunction{false}(system.rhs; jac = system.jac,
            tgrad = system.tgrad, mass_matrix = system.mass_matrix)
    prob = ODEProblem{false}(f, system.u0, (0.0f0, duration),
        @SArray [Float32(problem["sweep_max"])])
    return system, prob, duration
end

"Host and device ensembles over an explicit parameter grid; data only, no new kernels."
function build_ensemble(system, prob, grid::AbstractVector)
    values = Float32.(collect(grid))
    probs_host = map(eachindex(values)) do i
        DiffEqGPU.make_prob_compatible(remake(prob,
            u0 = system.u0_for(values[i]), p = @SVector [values[i]]))
    end
    return probs_host, cu(probs_host)
end

"Host and device ensembles for one sweep size of a problem row."
build_ensemble(system, prob, problem, n) =
    build_ensemble(system, prob, problem_sweep(problem, n))

# The kernel path takes no dtmin: DiffEqGPU's kernels floor dt at 1e-14 themselves.
"One kernel solve on resident problems; adaptive solves start at the timing step."
function gpu_solve(probs, prob, solver, mode, setting, problem;
        saveat = Float32(problem["duration"]))
    if mode == "fixed"
        return DiffEqGPU.vectorized_solve(probs, prob, solver;
            saveat = saveat, save_everystep = false, dt = Float32(setting))
    end
    return DiffEqGPU.vectorized_asolve(probs, prob, solver;
        saveat = saveat, save_everystep = false,
        dt = Float32(problem_timing_dt(problem)),
        abstol = Float32(setting), reltol = Float32(setting))
end

"Device-only solve: inputs resident, results left on the device, synchronised."
gpu_solve_device(probs, prob, solver, mode, setting, problem) =
    CUDA.@sync gpu_solve(probs, prob, solver, mode, setting, problem)

"Resident solve followed by the d2h copies; returns (sol, ts, us)."
function gpu_solve_d2h(probs, prob, solver, mode, setting, problem)
    sol = CUDA.@sync gpu_solve(probs, prob, solver, mode, setting, problem)
    return sol, Array(sol[1]), Array(sol[2])
end

"End-to-end solve: h2d of the host ensemble, solve, d2h; returns (sol, ts, us)."
function gpu_solve_host(probs_host, prob, solver, mode, setting, problem)
    return gpu_solve_d2h(cu(probs_host), prob, solver, mode, setting, problem)
end

"Golden-ordered final state of every trajectory as an (n, states) Float32 matrix."
function final_states(system, us_end)
    m = Matrix{Float32}(undef, length(us_end), length(system.golden_index))
    for i in eachindex(us_end)
        m[i, :] .= us_end[i][system.golden_index]
    end
    return m
end
