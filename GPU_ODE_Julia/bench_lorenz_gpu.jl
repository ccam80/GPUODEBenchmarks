using Pkg

Pkg.instantiate()
Pkg.precompile()
using DiffEqGPU, BenchmarkTools, StaticArrays, SimpleDiffEq
using CUDA

@show ARGS
#settings

numberOfParameters = isinteractive() ? 8192 : parse(Int64, ARGS[1])

function lorenz(u, p, t)
    du1 = 10.0f0 * (u[2] - u[1])
    du2 = p[1] * u[1] - u[2] - u[1] * u[3]
    du3 = u[1] * u[2] - 2.666f0 * u[3]
    return @SVector [du1, du2, du3]
end

u0 = @SVector [1.0f0; 0.0f0; 0.0f0]
tspan = (0.0f0, 1.0f0)
p = @SArray [21.0f0]
prob = ODEProblem(lorenz, u0, tspan, p)

parameterList = range(0.0f0, stop = 21.0f0, length = numberOfParameters)

lorenzProblem = ODEProblem(lorenz, u0, tspan, p)
prob_func = (prob, i, repeat) -> remake(prob, p = @SArray [parameterList[i]])

ensembleProb = EnsembleProblem(lorenzProblem, prob_func = prob_func)

# Ensure we error on accidental CPU scalar ops with GPU arrays
CUDA.allowscalar(false)

@info "Solving the problem on GPU (fixed dt)"
data = @benchmark begin
    CUDA.@sync solve($ensembleProb, GPUTsit5(), EnsembleGPUKernel(CUDABackend(), 0.0);
                     trajectories = $numberOfParameters,
                     save_everystep = false,
                     dt = 0.001f0)
end

if !isinteractive()
    open(joinpath(dirname(@__DIR__), "data", "Julia", "Julia_times_unadaptive.txt"),
         "a+") do io
        println(io, numberOfParameters, " ", minimum(data.times) / 1e6)
    end
end

# Save numerical output for 32768-trajectory run
if !isinteractive() && numberOfParameters == 32768
    sol = solve(ensembleProb, GPUTsit5(), EnsembleGPUKernel(CUDABackend(), 0.0);
                trajectories = numberOfParameters,
                save_everystep = false,
                dt = 0.001f0)
    
    # Create directory
    mkpath(joinpath(dirname(@__DIR__), "data", "numerical"))
    
    # Extract final state values for each trajectory
    using CSV, DataFrames
    final_states = zeros(Float32, numberOfParameters, 3)
    for i in 1:numberOfParameters
        final_states[i, :] = sol[i].u[end]
    end
    
    # Save to CSV
    CSV.write(joinpath(dirname(@__DIR__), "data", "numerical", "julia.csv"), 
              DataFrame(final_states, :auto), header=false)
end

println("Parameter number: " * string(numberOfParameters))
println("Minimum time: " * string(minimum(data.times) / 1e6) * " ms")
println("Allocs: " * string(data.allocs))

@info "Solving the problem on GPU (adaptive dt)"
data = @benchmark begin
    CUDA.@sync solve($ensembleProb, GPUTsit5(), EnsembleGPUKernel(CUDABackend(), 0.0);
                     trajectories = $numberOfParameters,
                     dt = 0.001f0,
                     reltol = 1.0f-8,
                     abstol = 1.0f-8)
end

if !isinteractive()
    open(joinpath(dirname(@__DIR__), "data", "Julia", "Julia_times_adaptive.txt"),
         "a+") do io
        println(io, numberOfParameters, " ", minimum(data.times) / 1e6)
    end
end

println("Parameter number: " * string(numberOfParameters))
println("Minimum time: " * string(minimum(data.times) / 1e6) * " ms")
println("Allocs: " * string(data.allocs))
