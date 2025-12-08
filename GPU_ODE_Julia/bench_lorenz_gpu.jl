using Pkg

Pkg.instantiate()
Pkg.precompile()

using CUDA
using BenchmarkTools, DiffEqGPU, OrdinaryDiffEq, StaticArrays


@show ARGS
#settings
CUDA.allowscalar(false)
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
prob = ODEProblem{false}(lorenz, u0, tspan, p)

parameterList = range(0.0f0, stop = 21.0f0, length = numberOfParameters)
# parameterList_d = cu(collect(parameterList))          # GPU copy of parameter values

I = 1:numberOfParameters
probs = map(I) do i
    DiffEqGPU.make_prob_compatible(remake(prob,p= @SVector [parameterList[i]]))
    end
# prob_func = (prob, i, repeat) -> remake(prob, p = view(parameterList_d, i:i))
# ensembleProb = EnsembleProblem(prob, prob_func = prob_func, safetycopy=false)



probs = cu(probs)

@info "Solving the problem on GPU (fixed dt)"
data = @benchmark begin
    CUDA.@sync sol = DiffEqGPU.vectorized_solve(probs, prob, GPUTsit5(),
                           save_everystep=false,
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
  
    # Create directory
    mkpath(joinpath(dirname(@__DIR__), "data", "numerical"))
    CUDA.@sync sol = DiffEqGPU.vectorized_solve(probs, prob, GPUTsit5(),
                           save_everystep=false,
                           dt = 0.001f0)
    # Extract final state values for each trajectory
    using CSV, DataFrames
    final_states = zeros(Float32, numberOfParameters, 3)
    for i in 1:numberOfParameters
        final_states[i, :] = sol[i].u[end]
    end
    
    # Save to CSV
    CSV.write(joinpath(dirname(@__DIR__), "data", "numerical", "julia_fixed.csv"), 
              DataFrame(final_states, :auto), header=false)
end

println("Parameter number: " * string(numberOfParameters))
println("Minimum time: " * string(minimum(data.times) / 1e6) * " ms")
println("Allocs: " * string(data.allocs))

@info "Solving the problem on GPU (adaptive dt)"
data = @benchmark CUDA.@sync sol = DiffEqGPU.vectorized_asolve(probs, prob, GPUTsit5(),
#                            save_everystep=false,
                           reltol = 1.0f-8,
                           abstol = 1.0f-8,
                           dt = 0.001f0)

if !isinteractive()
    open(joinpath(dirname(@__DIR__), "data", "Julia", "Julia_times_adaptive.txt"),
         "a+") do io
        println(io, numberOfParameters, " ", minimum(data.times) / 1f6)
    end
end


println("Parameter number: " * string(numberOfParameters))
println("Minimum time: " * string(minimum(data.times) / 1f6) * " ms")
println("Allocs: " * string(data.allocs))

# Save numerical output for 32768-trajectory run
if !isinteractive() && numberOfParameters == 32768
    data = @benchmark CUDA.@sync sol = DiffEqGPU.vectorized_asolve(probs, prob, GPUTsit5(),
                           save_everystep=false,
                           reltol = 1.0f-8,
                           abstol = 1.0f-8,
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
    CSV.write(joinpath(dirname(@__DIR__), "data", "numerical", "julia_adaptive.csv"), 
              DataFrame(final_states, :auto), header=false)
end
