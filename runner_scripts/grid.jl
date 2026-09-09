# The ensemble grid of section 1.2, reproducing runner_scripts/grid.py bit for bit.

const GRID_SCALES = ("linear", "log")

"v[0..n-1] of the 1.2 formula: float64 arithmetic, the last point pinned to grid_max, cast to Float32."
function grid_values(scale, grid_min, grid_max, n)
    scale in GRID_SCALES || throw(ArgumentError("grid_scale '$(scale)' is not linear or log"))
    n = Int(n)
    n >= 2 || throw(ArgumentError("a grid needs n >= 2, got $(n)"))
    lo, hi = Float64(grid_min), Float64(grid_max)
    (isfinite(lo) && isfinite(hi)) || throw(ArgumentError("grid_min and grid_max must be finite"))
    values = Vector{Float64}(undef, n)
    if scale == "linear"
        step = (hi - lo) / (n - 1)
        for i in 0:(n - 1)
            values[i + 1] = lo + i * step
        end
    else
        (lo > 0.0 && hi > 0.0) || throw(ArgumentError("a log grid needs grid_min > 0 and grid_max > 0"))
        a, b = log10(lo), log10(hi)
        step = (b - a) / (n - 1)
        for i in 0:(n - 1)
            values[i + 1] = 10.0^(a + i * step)
        end
    end
    values[n] = hi
    return Float32.(values)
end

"v[index] of the 1.2 formula in Float64, before the cast: the grid_max of a shorter grid that reproduces v[0..index] bit for bit."
function grid_point(scale, grid_min, grid_max, n, index)
    scale in GRID_SCALES || throw(ArgumentError("grid_scale '$(scale)' is not linear or log"))
    n, index = Int(n), Int(index)
    n >= 2 || throw(ArgumentError("a grid needs n >= 2, got $(n)"))
    0 <= index < n || throw(ArgumentError("index $(index) is outside 0..$(n - 1)"))
    lo, hi = Float64(grid_min), Float64(grid_max)
    index == n - 1 && return hi
    scale == "linear" && return lo + index * ((hi - lo) / (n - 1))
    (lo > 0.0 && hi > 0.0) || throw(ArgumentError("a log grid needs grid_min > 0 and grid_max > 0"))
    a, b = log10(lo), log10(hi)
    return 10.0^(a + index * ((b - a) / (n - 1)))
end

"The grid of a run spec (grid_scale, grid_min, grid_max, n, precision) in its precision: the Float32 values, widened back for float64 runs."
function grid(spec)
    values = grid_values(spec["grid_scale"], spec["grid_min"], spec["grid_max"], spec["n"])
    precision = get(spec, "precision", "float32")
    precision == "float32" && return values
    precision == "float64" && return Float64.(values)
    throw(ArgumentError("precision '$(precision)' is not float32 or float64"))
end
