using Pkg

Pkg.instantiate()
Pkg.precompile()

using CUDA
using BenchmarkTools, DiffEqGPU, OrdinaryDiffEq, StaticArrays


@show ARGS
#settings
CUDA.allowscalar(false)
numberOfParameters = isinteractive() ? 8192 : parse(Int64, ARGS[1])

# Dataset key ("<os>_<gpu>") so output files are keyed per machine and can be
# additively populated across machines without clobbering each other.
include(joinpath(dirname(@__DIR__), "runner_scripts", "bench_key.jl"))
const DATASET_KEY = dataset_key()

function lorenz(u, p, t)
    du1 = 10.0f0 * (u[2] - u[1])
    du2 = p[1] * u[1] - u[2] - u[1] * u[3]
    du3 = u[1] * u[2] - (8.0f0 / 3.0f0) * u[3]
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

# ========================================
# WORK-PRECISION (wp) MODE
# ========================================
# `bench_lorenz_gpu.jl 32768 wp` sweeps fixed dt / adaptive tolerance at
# N=32768 and records "<setting> <time_ms> <error-vs-golden>" per point.
# Grids and protocol mirror runner_scripts/wp_common.py — keep in sync.
if length(ARGS) > 1 && ARGS[2] == "wp"
    using DelimitedFiles

    numberOfParameters == 32768 || error("wp mode must be run with N = 32768")
    golden = readdlm(
        joinpath(dirname(@__DIR__), "data", "numerical",
            "golden_lorenz_32768.csv"), ',', Float64)

    # l2-at-final error over the ensemble, in Float64.
    function ensemble_error(us)
        final = Array(us[end, :])
        m = Matrix{Float64}(undef, length(final), 3)
        for i in eachindex(final)
            m[i, :] .= Float64.(final[i])
        end
        return sqrt(sum(abs2, m .- golden) / length(m))
    end

    DTS = [2.0^-k for k in 4:13]     # 1/16 .. 1/8192
    TOLS = [10.0^-k for k in 2:8]    # 1e-2 .. 1e-8

    outdir = joinpath(dirname(@__DIR__), "data", "Julia")
    mkpath(outdir)

    open(joinpath(outdir, "Julia_wp_fixed_$(DATASET_KEY).txt"), "w") do io
        for dt in DTS
            dt32 = Float32(dt)
            CUDA.@sync sol = DiffEqGPU.vectorized_solve(probs, prob, GPUTsit5();
                saveat = 1.0f0, save_everystep = false, dt = dt32)
            err = ensemble_error(sol[2])
            data = @benchmark begin
                CUDA.@sync sol = DiffEqGPU.vectorized_solve($probs, $prob,
                    GPUTsit5(); saveat = 1.0f0, save_everystep = false,
                    dt = $dt32)
                ts = Array(sol[1])
                us = Array(sol[2])
            end
            t_ms = minimum(data.times) / 1e6
            println(io, dt, " ", t_ms, " ", err)
            println("wp fixed dt=$(dt): $(t_ms) ms, err=$(err)")
        end
    end

    open(joinpath(outdir, "Julia_wp_adaptive_$(DATASET_KEY).txt"), "w") do io
        for tol in TOLS
            tol32 = Float32(tol)
            CUDA.@sync sol = DiffEqGPU.vectorized_asolve(probs, prob,
                GPUTsit5(); saveat = 1.0f0, save_everystep = false,
                reltol = tol32, abstol = tol32, dt = 0.001f0)
            err = ensemble_error(sol[2])
            data = @benchmark begin
                CUDA.@sync sol = DiffEqGPU.vectorized_asolve($probs, $prob,
                    GPUTsit5(); saveat = 1.0f0, save_everystep = false,
                    reltol = $tol32, abstol = $tol32, dt = 0.001f0)
                ts = Array(sol[1])
                us = Array(sol[2])
            end
            t_ms = minimum(data.times) / 1e6
            println(io, tol, " ", t_ms, " ", err)
            println("wp adaptive tol=$(tol): $(t_ms) ms, err=$(err)")
        end
    end

    exit(0)
end

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
    open(joinpath(dirname(@__DIR__), "data", "Julia", "Julia_times_unadaptive_$(DATASET_KEY).txt"),
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
    # final_states = zeros(Float32, numberOfParameters, 3)
    final_states = Array(sol[2][end,:]) #convert to CPU Array
    
    # Save to CSV
    df2 = DataFrame([Tuple(s) for s in final_states], [:x, :y, :z])
    CSV.write(joinpath(dirname(@__DIR__), "data", "numerical", "julia_fixed_$(DATASET_KEY).csv"), df2, header=false)
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
    open(joinpath(dirname(@__DIR__), "data", "Julia", "Julia_times_adaptive_$(DATASET_KEY).txt"),
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
    
    # Save to CSV
    df2 = DataFrame([Tuple(s) for s in final_states], [:x, :y, :z])
    CSV.write(joinpath(dirname(@__DIR__), "data", "numerical", "julia_adaptive_$(DATASET_KEY).csv"), df2, header=false)
end
