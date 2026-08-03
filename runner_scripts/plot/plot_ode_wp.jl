using Plots
using DelimitedFiles
using Dates
using Statistics
using Plots.PlotMeasures

# Plot the Lorenz WORK-PRECISION benchmarks (error vs runtime at N = 32768).
#
# Each framework's `wp` mode sweeps its supported fixed step size and/or
# adaptive tolerance per integration algorithm and writes rows
# "<setting> <time_ms> <error>" to
#   data/<DIR>/<Prefix>_wp_<fixed|adaptive>_<algorithm>_<os>_<gpu>.txt
# where <algorithm> is the cubie-vocabulary method name (euler,
# classical-rk4, tsit5, cash-karp-54) and the error is the ensemble l2 norm
# of the final-state difference against the Float64 golden reference
# (data/numerical/golden_lorenz_32768.csv — see
# runner_scripts/golden/generate_golden.jl and runner_scripts/wp_common.py).
#
# This script discovers those keyed files exactly like plot_ode_comp.jl and
# emits one error-vs-time plot per (group, mode, algorithm) — every curve
# within a figure runs the same integration method (issue #29) — plus an
# "all" overview per group with the algorithm in each label, giving e.g.
# Lorenz_wp_fixed_euler_windows.png, Lorenz_wp_all_RTX-4070-SUPER.png.
#
# Default: use the repo `data/` directory. Optionally pass a custom data directory as ARGS[1].
parent_dir = length(ARGS) != 0 ? ARGS[1] : "data"
base_path = joinpath(dirname(dirname(@__DIR__)), parent_dir)

# display name => (subdirectory, filename prefix)
# Note: MPGOS data files are stored under `CPP/` in the repo's `data/` folder.
frameworks = [
    ("Julia", "Julia", "Julia"),
    ("MPGOS", "CPP", "MPGOS"),
    ("JAX", "JAX", "Jax"),
    ("PYTORCH", "PYTORCH", "Torch"),
    ("CUBIE", "CUBIE", "Cubie"),
    ("CUBIE_MLIR", "CUBIE_MLIR", "Cubie_mlir"),
    ("MYOKIT CUDA", "MYOKIT_CUDA", "Myokit_cuda"),
]

# color/marker choices per framework (same as plot_ode_comp.jl)
colors = Dict("Julia"=>:Green, "MPGOS"=>:Orange, "JAX"=>:Red,
    "PYTORCH"=>:DarkRed, "CUBIE"=>:Blue, "CUBIE_MLIR"=>:Purple,
    "MYOKIT CUDA"=>:Black)
markers = Dict("Julia"=>:circle, "MPGOS"=>:utriangle, "JAX"=>:diamond,
    "PYTORCH"=>:xcross, "CUBIE"=>:star5, "CUBIE_MLIR"=>:hexagon,
    "MYOKIT CUDA"=>:rect)

# One work-precision curve loaded from disk.
struct WPSeries
    display::String
    mode::String       # "fixed" or "adaptive"
    algorithm::String  # cubie-vocabulary method name, e.g. "classical-rk4"
    os::String
    gpu::String
    key::String      # "<os>_<gpu>"
    err::Vector{Float64}
    time_s::Vector{Float64}
end

# Discover every keyed wp file and load it into a WPSeries.
function collect_series(base_path, frameworks)
    series = WPSeries[]
    for (display, dir, prefix) in frameworks
        dpath = joinpath(base_path, dir)
        isdir(dpath) || continue
        # "<prefix>_wp_<fixed|adaptive>_<algorithm>_<os>_<gpu>.txt"; none of
        # algorithm, os and gpu contain underscores, so they are the last
        # three underscore-separated fields.
        pat = Regex("^" * prefix * "_wp_(fixed|adaptive)_([^_]+)_([^_]+)_([^_]+)\\.txt\$")
        for fname in sort(readdir(dpath))
            m = match(pat, fname)
            m === nothing && continue
            mode, algorithm, os, gpu = m.captures
            fpath = joinpath(dpath, fname)
            # readdlm raises on a file with no data rows, so screen those out.
            isempty(strip(read(fpath, String))) && continue
            data = readdlm(fpath)
            isempty(data) && continue
            setting = Float64.(data[:, 1])
            err = Float64.(data[:, 3])
            time_s = Float64.(data[:, 2]) .* 1e-3
            # Drop non-positive errors (log axis). Order points along the
            # sweep (loose -> tight setting) so the float32 roundoff U-turn
            # in the fixed curves is traced rather than folded onto itself.
            keep = err .> 0
            setting, err, time_s = setting[keep], err[keep], time_s[keep]
            isempty(err) && continue
            order = sortperm(setting, rev = true)
            push!(series, WPSeries(display, String(mode), String(algorithm),
                String(os), String(gpu), "$(os)_$(gpu)",
                err[order], time_s[order]))
        end
    end
    return series
end

# Draw and save one plot for the given series subset, or warn if it is empty.
# alg_label == "all" is the overview figure mixing algorithms (labels carry
# the algorithm); otherwise every series in `sel` runs the same algorithm.
function render_plot(sel, group_label, mode_label, alg_label, plots_dir, multikey)
    if isempty(sel)
        println("Skipping empty plot: wp_$(mode_label)_$(alg_label)_$(group_label)")
        return
    end
    gr(size = (810, 540))
    modeword = mode_label == "fixed" ? "fixed dt" :
        mode_label == "adaptive" ? "adaptive tol" : "fixed + adaptive"
    algword = alg_label == "all" ? "all algorithms" : alg_label
    plt = plot(xaxis = :log, yaxis = :log, linewidth = 2,
        ylabel = "Time (s)", xlabel = "Error (ensemble l2, final state)",
        title = "Lorenz WP, N=32768, $(modeword), $(algword) ($(group_label))",
        titlefontsize = 12, legend = :outertopright, dpi = 600)

    for s in sel
        stepword = s.mode == "adaptive" ? "adaptive" : "fixed"
        algpart = alg_label == "all" ? ", $(s.algorithm)" : ""
        keypart = multikey ? " [$(s.key)]" : ""
        label = "$(s.display) ($(stepword)$(algpart))$(keypart)"
        ls = s.mode == "adaptive" ? :dash : :solid
        plot!(plt, s.err, s.time_s, label = label, color = colors[s.display],
            marker = markers[s.display], linestyle = ls)
    end

    isdir(plots_dir) || mkpath(plots_dir)
    algpart = alg_label == "all" ? "" : "_$(alg_label)"
    outfile = joinpath(plots_dir, "Lorenz_wp_$(mode_label)$(algpart)_$(group_label).png")
    savefig(plt, outfile)
    println("Saved $(outfile)")
end

function main()
    series = collect_series(base_path, frameworks)
    if isempty(series)
        println("Warning: no keyed wp files found under $(base_path). Nothing to plot.")
        println("Expected files like <Prefix>_wp_adaptive_<algorithm>_<os>_<gpu>.txt (run the wp benchmarks first).")
        return
    end

    plots_dir = joinpath(dirname(dirname(@__DIR__)), "plots")

    oses = sort(unique(s.os for s in series))
    gpus = sort(unique(s.gpu for s in series))
    groups = Tuple{String, Vector{WPSeries}}[]
    push!(groups, ("all", series))
    for os in oses
        push!(groups, (os, filter(s -> s.os == os, series)))
    end
    for gpu in gpus
        push!(groups, (gpu, filter(s -> s.gpu == gpu, series)))
    end

    for (label, sel) in groups
        multikey = length(unique(s.key for s in sel)) > 1
        for mode in ("fixed", "adaptive")
            msel = filter(s -> s.mode == mode, sel)
            for alg in sort(unique(s.algorithm for s in msel))
                render_plot(filter(s -> s.algorithm == alg, msel),
                            label, mode, alg, plots_dir, multikey)
            end
        end
        render_plot(sel, label, "all", "all", plots_dir, multikey)
    end
end

main()
