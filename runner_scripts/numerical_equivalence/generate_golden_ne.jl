# Generate the golden reference for the numerical-equivalence (ne) sweeps.
#
# Same problem family as the wp golden (runner_scripts/golden/generate_golden.jl):
# Lorenz with sigma=10, beta=8/3, x0=(1,0,0), t in [0,1], rho swept linearly
# over [0, 21] — but at N=1024 trajectories so the full per-algorithm dt sweep
# stays cheap enough to run on CPU (and eventually in cubie CI).
#
# The reference is computed on the CPU in Float64 with Vern9 at
# abstol=reltol=1e-13. The rho grid is the float64 linspace ROUNDED THROUGH
# FLOAT32 and promoted back to Float64; the rounded rho values are stored in
# the output so every consumer (the Float32 DifferentialEquations.jl runner
# and the cubie runner) integrates bit-identical Float32 parameters instead
# of rebuilding its own linspace.
#
# Output: data/numerical/golden_ne_lorenz_1024.csv — 1024 rows, columns
# rho,x,y,z (rho then final state at t=1), full Float64 precision, no header.
# Machine independent: no os/gpu dataset key.
#
# Run from the repo root:
#   julia -t auto --project=. runner_scripts/numerical_equivalence/generate_golden_ne.jl

using OrdinaryDiffEq
# The slim OrdinaryDiffEq v7 umbrella doesn't re-export the ensemble API.
using SciMLBase: ODEProblem, remake
using DelimitedFiles

const N = 1024

function lorenz(u, p, t)
    du1 = 10.0 * (u[2] - u[1])
    du2 = u[1] * (p[1] - u[3]) - u[2]
    du3 = u[1] * u[2] - (8.0 / 3.0) * u[3]
    return [du1, du2, du3]
end

rhos = Float64.(Float32.(range(0.0, 21.0, length = N)))

u0 = [1.0, 0.0, 0.0]
tspan = (0.0, 1.0)
prob = ODEProblem(lorenz, u0, tspan, [rhos[1]])

@info "Solving $(N)-trajectory Float64 reference (Vern9, tol 1e-13) on $(Threads.nthreads()) threads..."
out = Matrix{Float64}(undef, N, 4)
out[:, 1] .= rhos
@time Threads.@threads for i in 1:N
    sol = solve(remake(prob, p = [rhos[i]]), Vern9();
        abstol = 1e-13, reltol = 1e-13, save_everystep = false,
        save_start = false, dense = false)
    out[i, 2:4] .= sol.u[end]
end

outdir = joinpath(dirname(dirname(@__DIR__)), "data", "numerical")
mkpath(outdir)
outfile = joinpath(outdir, "golden_ne_lorenz_1024.csv")
open(outfile, "w") do io
    writedlm(io, out, ',')
end
@info "Wrote $(outfile)"
