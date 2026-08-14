# Generate the golden reference solution for the Lorenz work-precision
# benchmarks.
#
# The benchmarked problem (identical in every framework): Lorenz with
# sigma=10, beta=8/3, rho swept linearly over [0, 21] across N=131072
# trajectories, x0=(1,0,0), t in [0,1], solved in float32.
#
# The reference is computed on the CPU in Float64 with Vern9 at
# abstol=reltol=1e-13. The rho grid is the float64 linspace ROUNDED THROUGH
# FLOAT32 and promoted back to Float64 — the frameworks integrate the
# float32-rounded parameters, and at the float32 error floor (~1e-6) a ulp of
# rho is visible in the solution. (Frameworks still differ from each other by
# ~1 ulp of rho depending on how they build their linspace; this bounds the
# meaningful error floor of the work-precision curves at roughly 1e-6
# relative, which is inherent to benchmarking float32 solvers.)
#
# Output: data/numerical/golden_lorenz_131072.csv — 131072 rows, columns x,y,z
# (final state at t=1), full Float64 precision, no header. Machine
# independent: no os/gpu dataset key.
#
# Run from the repo root:  julia -t auto --project=. runner_scripts/golden/generate_golden.jl

using OrdinaryDiffEq
# The slim OrdinaryDiffEq v7 umbrella doesn't re-export the ensemble API.
using SciMLBase: EnsembleProblem, EnsembleThreads, ODEProblem, remake
using DelimitedFiles

const N = 131072

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
out = Matrix{Float64}(undef, N, 3)
@time Threads.@threads for i in 1:N
    sol = solve(remake(prob, p = [rhos[i]]), Vern9();
        abstol = 1e-13, reltol = 1e-13, save_everystep = false,
        save_start = false, dense = false)
    out[i, :] .= sol.u[end]
end

outdir = joinpath(dirname(dirname(@__DIR__)), "data", "numerical")
mkpath(outdir)
outfile = joinpath(outdir, "golden_lorenz_131072.csv")
open(outfile, "w") do io
    writedlm(io, out, ',')
end
@info "Wrote $(outfile)"
