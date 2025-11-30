"""
CPU step-by-step Lorenz ODE solver using OrdinaryDiffEq's Tsit5 algorithm.
Solves the Lorenz system with rho=10.0 from the same initial conditions as other scripts.
Uses adaptive time stepping with atol=rtol=1e-8 and prints error norm at each step.

Created for debugging and error analysis purposes.
"""

using OrdinaryDiffEq
using LinearAlgebra

# Lorenz system parameters
const σ = 10.0  # sigma
const ρ = 10.0  # rho (different from benchmark scripts which use 21.0, set to 10.0 per task requirements)
const β = 2.666  # beta (same as other scripts in the project)

# Define the Lorenz system (in-place form for efficiency)
function lorenz!(du, u, p, t)
    x, y, z = u
    du[1] = σ * (y - x)
    du[2] = x * (ρ - z) - y
    du[3] = x * y - β * z
    return nothing
end

# Initial conditions (same as other scripts)
u0 = [1.0, 0.0, 0.0]

# Time span (same as other scripts)
tspan = (0.0, 1.0)

# Create the ODE problem
prob = ODEProblem(lorenz!, u0, tspan)

# Create integrator with adaptive time stepping
integrator = init(prob, Tsit5(); 
    abstol = 1e-8, 
    reltol = 1e-8,
    save_everystep = true
)

println("Solving Lorenz system step-by-step with Tsit5 (CPU)")
println("Parameters: σ=$σ, ρ=$ρ, β=$β")
println("Initial conditions: u0=$u0")
println("Time span: $tspan")
println("Tolerances: atol=1e-8, rtol=1e-8")
println("=" ^ 80)
println()

# Step through the solution using a function to avoid scope issues
function run_integration(integrator, tspan)
    step_count = 0
    while integrator.t < tspan[2]
        step_count += 1
        
        # Get current state before step
        t_before = integrator.t
        u_before = copy(integrator.u)
        
        # Take one step
        step!(integrator)
        
        # Get state after step
        t_after = integrator.t
        u_after = integrator.u
        dt = t_after - t_before
        
        # Get the error estimate if available
        # In OrdinaryDiffEq, the error estimate is stored in integrator.EEst (scaled error)
        # which represents the maximum weighted error across all components
        if hasproperty(integrator, :EEst)
            error_est = integrator.EEst
            println("Step $step_count: t = $(round(t_after, digits=10)), dt = $(round(dt, sigdigits=6)), " *
                    "u = [$(round(u_after[1], sigdigits=8)), $(round(u_after[2], sigdigits=8)), $(round(u_after[3], sigdigits=8))], " *
                    "Error estimate = $(round(error_est, sigdigits=6))")
        else
            println("Step $step_count: t = $(round(t_after, digits=10)), dt = $(round(dt, sigdigits=6)), " *
                    "u = [$(round(u_after[1], sigdigits=8)), $(round(u_after[2], sigdigits=8)), $(round(u_after[3], sigdigits=8))]")
        end
    end
    return step_count
end

step_count = run_integration(integrator, tspan)

println()
println("=" ^ 80)
println("Integration completed!")
println("Total steps: $step_count")
println("Final time: $(integrator.t)")
println("Final state: $(integrator.u)")

# Get the solution object
sol = integrator.sol

println()
println("Solution statistics:")
println("  Return code: $(sol.retcode)")
println("  Number of time points saved: $(length(sol.t))")
