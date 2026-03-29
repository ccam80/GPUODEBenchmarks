using Pkg

Pkg.instantiate()
Pkg.precompile()

using CUDA
using BenchmarkTools, DiffEqGPU, OrdinaryDiffEq, StaticArrays

include(joinpath(@__DIR__, "src", "model_definitions.jl"))

@show ARGS
#settings
CUDA.allowscalar(false)
numberOfParameters = isinteractive() ? 8192 : parse(Int64, ARGS[1])
model_name = (isinteractive() || length(ARGS) < 2) ? "lorenz" : ARGS[2]

# Get model definition
ode_func, u0, p, tspan, dim = get_model(model_name, Float32)
parameterList = get_parameter_range(model_name, numberOfParameters, Float32)

prob = ODEProblem{false}(ode_func, u0, tspan, p)
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
    # From my rookie reading of the DiffEqGPU "solve" wrapper, which causes 
    # the problem to run on CPU right now, the low-level function allocates
    #  output arrays on the device and returns CuArrays. The higher level
    # function calls Array(ts), Array(us) to transfer back to CPU,
    # so we replicate that here to mirror the level of the other packages.
    # One mystery I haven't cracked is when the initial conditions and 
    # parameters get transferred to the GPU. To keep it an even comparison,
    # Let's assume that if it gets transferred earlier, it's made up for 
    # by Cubie pre-allocating the GPU array.

    CUDA.@sync sol = DiffEqGPU.vectorized_solve(probs, prob, GPUTsit5(),
                           saveat=1.0f0,
                           save_everystep=false,
                           dt = 0.001f0)
        ts = Array(sol[1])
        us = Array(sol[2])
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
                           saveat=1.0f0,
                           save_everystep=false,
                           dt = 0.001f0)
    # Extract final state values for each trajectory
    using CSV, DataFrames
    final_states = Array(sol[2][end,:]) #convert to CPU Array
    
    # Save to CSV - determine column names based on dimension
    col_names = dim == 2 ? [:x, :y] : (dim == 3 ? [:x, :y, :z] : [Symbol("state_$i") for i in 1:dim])
    df2 = DataFrame([Tuple(s) for s in final_states], col_names)
    CSV.write(joinpath(dirname(@__DIR__), "data", "numerical", "julia_fixed.csv"), df2, header=false)
    # CSV.write(joinpath(dirname(@__DIR__), "data", "numerical", "julia_fixed.csv"), 
    #           DataFrame(final_states, :auto), header=false)
end

println("Parameter number: " * string(numberOfParameters))
println("Minimum time: " * string(minimum(data.times) / 1e6) * " ms")
println("Allocs: " * string(data.allocs))

@info "Solving the problem on GPU (adaptive dt)"
data = @benchmark begin 
    CUDA.@sync sol = DiffEqGPU.vectorized_asolve(probs, prob, GPUTsit5(),
        saveat=1.0f0,
        save_everystep=false,
        reltol = 1.0f-8,
        abstol = 1.0f-8,
        dt = 0.001f0)
    # The low-level function returns an array of CuArrays. Their higher-level "solve" function calls Array(ts), Array(us) to transfer back 
    # to CPU, so we replicate that here to mirror the level of the other packages.
    ts = Array(sol[1])
    us = Array(sol[2])
end

if !isinteractive()
    open(joinpath(dirname(@__DIR__), "data", "Julia", "Julia_times_adaptive.txt"),
         "a+") do io
        println(io, numberOfParameters, " ", minimum(data.times) / 1f6)
    end
end


println("Parameter number: " * string(numberOfParameters))
println("Minimum time: " * string(minimum(data.times) / 1f6) * " ms")
println("Allocs: " * string(data.allocs))

results = Vector{Any}(undef, 2)

# Save numerical output for 32768-trajectory run
if !isinteractive() && numberOfParameters == 32768
    CUDA.@sync copyto!(results, DiffEqGPU.vectorized_asolve(probs, prob, GPUTsit5(),
                           saveat=1.0f0,
                           save_everystep=false,
                           reltol = 1.0f-8,
                           abstol = 1.0f-8,
                           dt = 0.001f0))
    # Create directory
    mkpath(joinpath(dirname(@__DIR__), "data", "numerical"))
    
    # Extract final state values for each trajectory
    using CSV, DataFrames
    final_states = Array(sol[2][end,:]) #convert to CPU Array
    
    # Save to CSV - determine column names based on dimension
    col_names = dim == 2 ? [:x, :y] : (dim == 3 ? [:x, :y, :z] : [Symbol("state_$i") for i in 1:dim])
    df2 = DataFrame([Tuple(s) for s in final_states], col_names)
    CSV.write(joinpath(dirname(@__DIR__), "data", "numerical", "julia_adaptive.csv"), df2, header=false)
end
