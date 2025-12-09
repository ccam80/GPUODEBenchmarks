"""
Model definitions for ODE benchmarks
Contains various ODE systems that can be used for benchmarking
"""

using StaticArrays

"""
    lorenz(u, p, t)

Lorenz system equations:
    dx/dt = σ(y - x)
    dy/dt = ρx - y - xz
    dz/dt = xy - βz

where σ = 10.0, β = 8/3 or 2.666, and ρ is the parameter
"""
function lorenz(u::AbstractArray{T}, p, t) where {T}
    du1 = T(10.0) * (u[2] - u[1])
    du2 = p[1] * u[1] - u[2] - u[1] * u[3]
    du3 = u[1] * u[2] - T(8 // 3) * u[3]
    return @SVector T[du1, du2, du3]
end

"""
    vanderpol(u, p, t)

Van der Pol oscillator equations:
    dx/dt = y
    dy/dt = μ(1 - x²)y - x

where μ is the parameter controlling nonlinearity
"""
function vanderpol(u::AbstractArray{T}, p, t) where {T}
    du1 = u[2]
    du2 = p[1] * (T(1.0) - u[1] * u[1]) * u[2] - u[1]
    return @SVector T[du1, du2]
end

"""
    get_model(model_name::String)

Get the ODE function for the specified model name.

# Arguments
- `model_name::String`: Name of the model ("lorenz", "vanderpol")

# Returns
- Function handle for the ODE system
- Default initial conditions
- Default parameter value
- Time span
- State dimension

# Examples
```julia
ode_func, u0, p, tspan, dim = get_model("lorenz")
```
"""
function get_model(model_name::String, ::Type{T}=Float32) where {T}
    if lowercase(model_name) == "lorenz"
        u0 = @SVector T[1.0, 0.0, 0.0]
        p = @SArray T[21.0]
        tspan = (T(0.0), T(1.0))
        dim = 3
        return lorenz, u0, p, tspan, dim
    elseif lowercase(model_name) == "vanderpol"
        u0 = @SVector T[2.0, 0.0]
        p = @SArray T[1.0]
        tspan = (T(0.0), T(6.283185307179586))  # 2π
        dim = 2
        return vanderpol, u0, p, tspan, dim
    else
        error("Unknown model: $model_name. Available models: lorenz, vanderpol")
    end
end

"""
    get_parameter_range(model_name::String, npoints::Int)

Get the parameter range for the specified model.

# Arguments
- `model_name::String`: Name of the model
- `npoints::Int`: Number of parameter values to generate

# Returns
- Range of parameter values suitable for ensemble simulations
"""
function get_parameter_range(model_name::String, npoints::Int, ::Type{T}=Float32) where {T}
    if lowercase(model_name) == "lorenz"
        return range(T(0.0), stop=T(21.0), length=npoints)
    elseif lowercase(model_name) == "vanderpol"
        return range(T(0.1), stop=T(5.0), length=npoints)
    else
        error("Unknown model: $model_name")
    end
end

export lorenz, vanderpol, get_model, get_parameter_range
