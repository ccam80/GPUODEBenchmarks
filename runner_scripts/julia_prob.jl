# The DiffEqGPU kernel path shared by bench_ode_gpu.jl and GPU_ODE_JuliaKernels; every side builds identical types, so the kernels one side compiles serve the other.

"DiffEqGPU kernel solver named by the julia_gpu column of julia_algorithms.csv."
gpu_solver(algorithm) = Base.eval(@__MODULE__, Meta.parse(julia_constructor(algorithm, "julia_gpu")))

"The ODEProblem every trajectory of a system shares: its u0, t from 0 to duration, a one-parameter placeholder."
function build_prob(system, duration)
    f = system.mass_matrix === nothing ?
        ODEFunction{false}(system.rhs; jac = system.jac, tgrad = system.tgrad) :
        ODEFunction{false}(system.rhs; jac = system.jac, tgrad = system.tgrad,
            mass_matrix = system.mass_matrix)
    return ODEProblem{false}(f, system.u0, (0.0f0, Float32(duration)), @SArray [0.0f0])
end

"Host and device ensembles over the parameter values, one trajectory each; data only, no new kernels."
function build_ensemble(system, prob, values::AbstractVector)
    values = Float32.(collect(values))
    probs_host = map(eachindex(values)) do i
        DiffEqGPU.make_prob_compatible(remake(prob,
            u0 = system.u0_for(values[i]), p = @SVector [values[i]]))
    end
    return probs_host, cu(probs_host)
end

# The kernel path has no controller choice, dt floor or dt cap of its own to set: the fixed kernels step at dt, the adaptive ones start at dt (DiffEqGPU's own when NaN) and floor dt at 1e-14 themselves.
"One kernel solve on resident problems, saving the final state only."
function gpu_solve(probs, prob, solver, controller, dt, atol, rtol; saveat = prob.tspan[2])
    if controller == "fixed"
        return DiffEqGPU.vectorized_solve(probs, prob, solver; saveat = saveat,
            save_everystep = false, dt = Float32(dt))
    end
    start = isnan(dt) ? (;) : (; dt = Float32(dt))
    return DiffEqGPU.vectorized_asolve(probs, prob, solver; saveat = saveat,
        save_everystep = false, abstol = Float32(atol), reltol = Float32(rtol), start...)
end

"Device-only solve: inputs resident, results left on the device, synchronised; returns (ts, us)."
gpu_solve_device(probs, prob, solver, controller, dt, atol, rtol) =
    CUDA.@sync gpu_solve(probs, prob, solver, controller, dt, atol, rtol)

"End-to-end solve: h2d of the host ensemble, solve, d2h of the times and states; returns the device (ts, us)."
function gpu_solve_host(probs_host, prob, solver, controller, dt, atol, rtol)
    sol = CUDA.@sync gpu_solve(cu(probs_host), prob, solver, controller, dt, atol, rtol)
    Array(sol[1])
    Array(sol[2])
    return sol
end

"Golden-ordered final state of every trajectory as an (n, states) Float32 matrix."
function final_states(system, us_end)
    m = Matrix{Float32}(undef, length(us_end), length(system.golden_index))
    for i in eachindex(us_end)
        m[i, :] .= us_end[i][system.golden_index]
    end
    return m
end
