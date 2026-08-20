# Constructors shared by bench_ode_gpu.jl and GPU_ODE_JuliaKernels; both sides must build identical types.

"DiffEqGPU kernel solver for an algorithm name; autodiff off matches the overlap suite."
function gpu_solver(algorithm)
    algorithm == "tsit5" && return GPUTsit5()
    algorithm == "rosenbrock23_sciml" && return GPURosenbrock23(autodiff = Val(false))
    algorithm == "kvaerno3" && return GPUKvaerno3(autodiff = Val(false))
    error("no DiffEqGPU kernel solver for '$(algorithm)'")
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

"Host and device ensembles for one sweep size; data only, no new kernels."
function build_ensemble(system, prob, problem, n)
    grid = Float32.(collect(problem_sweep(problem, n)))
    probs_host = map(1:n) do i
        DiffEqGPU.make_prob_compatible(remake(prob,
            u0 = system.u0_for(grid[i]), p = @SVector [grid[i]]))
    end
    return probs_host, cu(probs_host)
end
