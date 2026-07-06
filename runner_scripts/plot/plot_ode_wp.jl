using Plots
using DelimitedFiles
using Dates
using Statistics
using Plots.PlotMeasures

# Plot the Lorenz WORK-PRECISION benchmarks (error vs runtime at N = 32768).
#
# Each framework's `wp` mode sweeps the fixed step size / adaptive tolerance
# and writes rows "<setting> <time_ms> <error>" to
#   data/<DIR>/<Prefix>_wp_<fixed|adaptive>_<os>_<gpu>.txt
# where the error is the ensemble l2 norm of the final-state difference
# against the Float64 golden reference
# (data/numerical/golden_lorenz_32768.csv — see
# runner_scripts/golden/generate_golden.jl and runner_scripts/wp_common.py).
#
# This script discovers those keyed files exactly like plot_ode_comp.jl and
# emits one error-vs-time plot per (group, mode):
#   * groups: "all" (everything combined), one per distinct os, one per distinct gpu
#   * modes:  "fixed", "adaptive", "all" (both)
# giving e.g. Lorenz_wp_fixed_windows.png, Lorenz_wp_all_RTX-4070-SUPER.png.
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
]

# color/marker choices per framework (same as plot_ode_comp.jl)
colors = Dict("Julia"=>:Green, "MPGOS"=>:Orange, "JAX"=>:Red, "PYTORCH"=>:DarkRed, "CUBIE"=>:Blue, "CUBIE_MLIR"=>:Purple)
markers = Dict("Julia"=>:circle, "MPGOS"=>:utriangle, "JAX"=>:diamond, "PYTORCH"=>:xcross, "CUBIE"=>:star5, "CUBIE_MLIR"=>:hexagon)

# One work-precision curve loaded from disk.
struct WPSeries
    display::String
    mode::String     # "fixed" or "adaptive"
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
        # "<prefix>_wp_<fixed|adaptive>_<os>_<gpu>.txt"; gpu contains no
        # underscores, so os and gpu are the last two fields.
        pat = Regex("^" * prefix * "_wp_(fixed|adaptive)_([^_]+)_([^_]+)\\.txt\$")
        for fname in sort(readdir(dpath))
            m = match(pat, fname)
            m === nothing && continue
            mode, os, gpu = m.captures
            data = readdlm(joinpath(dpath, fname))
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
            push!(series, WPSeries(display, String(mode), String(os),
                String(gpu), "$(os)_$(gpu)", err[order], time_s[order]))
        end
    end
    return series
end

# Draw and save one plot for the given series subset, or warn if it is empty.
function render_plot(sel, group_label, mode_label, plots_dir, multikey)
    if isempty(sel)
        println("Skipping empty plot: wp_$(mode_label)_$(group_label)")
        return
    end
    gr(size = (810, 540))
    modeword = mode_label == "fixed" ? "fixed dt" :
        mode_label == "adaptive" ? "adaptive tol" : "fixed + adaptive"
    plt = plot(xaxis = :log, yaxis = :log, linewidth = 2,
        ylabel = "Time (s)", xlabel = "Error (ensemble l2, final state)",
        title = "Lorenz WP, N=32768, $(modeword) ($(group_label))",
        titlefontsize = 12, legend = :outertopright, dpi = 600)

    for s in sel
        stepword = s.mode == "adaptive" ? "adaptive" : "fixed"
        label = multikey ? "$(s.display) ($(stepword)) [$(s.key)]" :
            "$(s.display) ($(stepword))"
        ls = s.mode == "adaptive" ? :dash : :solid
        plot!(plt, s.err, s.time_s, label = label, color = colors[s.display],
            marker = markers[s.display], linestyle = ls)
    end

    isdir(plots_dir) || mkpath(plots_dir)
    outfile = joinpath(plots_dir, "Lorenz_wp_$(mode_label)_$(group_label).png")
    savefig(plt, outfile)
    println("Saved $(outfile)")
end

function main()
    series = collect_series(base_path, frameworks)
    if isempty(series)
        println("Warning: no keyed wp files found under $(base_path). Nothing to plot.")
        println("Expected files like <Prefix>_wp_adaptive_<os>_<gpu>.txt (run the wp benchmarks first).")
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
        render_plot(filter(s -> s.mode == "fixed", sel), label, "fixed", plots_dir, multikey)
        render_plot(filter(s -> s.mode == "adaptive", sel), label, "adaptive", plots_dir, multikey)
        render_plot(sel, label, "all", plots_dir, multikey)
    end
end

main()
