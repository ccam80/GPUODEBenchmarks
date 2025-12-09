using BenchmarkTools, StaticArrays, OrdinaryDiffEq

include(joinpath(@__DIR__, "src", "model_definitions.jl"))

@show ARGS
#settings

numberOfParameters = isinteractive() ? 8192 : parse(Int64, ARGS[1])
model_name = (isinteractive() || length(ARGS) < 2) ? "lorenz" : ARGS[2]

# Get model definition
ode_func, u0, p, tspan, dim = get_model(model_name, Float64)
parameterList = get_parameter_range(model_name, numberOfParameters, Float64)

prob = ODEProblem(ode_func, u0, tspan, p)
prob_func = (prob, i, repeat) -> remake(prob, p = @SArray [parameterList[i]])

ensembleProb = EnsembleProblem(prob, prob_func = prob_func)

@info "Solving the problem"
data = @benchmark solve($ensembleProb, Tsit5(), EnsembleThreads(), dt = 0.001,
                        adaptive = false, save_everystep = false,
                        trajectories = numberOfParameters)

if !isinteractive()
    open(joinpath(dirname(@__DIR__), "data", "CPU", "times_unadaptive.txt"), "a+") do io
        println(io, numberOfParameters, " ", minimum(data.times) / 1e6)
    end
end

println("Parameter number: " * string(numberOfParameters))
println("Minimum time: " * string(minimum(data.times) / 1e6) * " ms")
println("Allocs: " * string(data.allocs))

data = @benchmark solve($ensembleProb, Tsit5(), EnsembleThreads(), dt = 0.001,
                        adaptive = true, save_everystep = false,
                        trajectories = numberOfParameters)

if !isinteractive()
    open(joinpath(dirname(@__DIR__), "data", "CPU", "times_adaptive.txt"), "a+") do io
        println(io, numberOfParameters, " ", minimum(data.times) / 1e6)
    end
end

println("Parameter number: " * string(numberOfParameters))
println("Minimum time: " * string(minimum(data.times) / 1e6) * " ms")
println("Allocs: " * string(data.allocs))
