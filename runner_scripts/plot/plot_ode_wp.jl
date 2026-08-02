using Plots
using DelimitedFiles
using Dates
using Statistics
using Plots.PlotMeasures

# Plot the Lorenz work-precision benchmarks (error vs runtime, N = 32768):
# one plot per (group, mode) at plots/<group>/Lorenz_wp_<mode>.png. ARGS[1] overrides "data".
include(joinpath(@__DIR__, "plot_common.jl"))

parent_dir = length(ARGS) != 0 ? ARGS[1] : "data"
base_path = joinpath(dirname(dirname(@__DIR__)), parent_dir)

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

# Discover every wp file under data/<package>/<key>/ and load it.
function collect_series(base_path)
    series = WPSeries[]
    eachkeyedfile(base_path, "_wp_(fixed|adaptive)[.]txt") do display, key, os, gpu, path, mode
        # readdlm raises on a file with no data rows, so screen those out.
        isempty(strip(read(path, String))) && return
        data = readdlm(path)
        isempty(data) && return
        setting = Float64.(data[:, 1])
        err = Float64.(data[:, 3])
        time_s = Float64.(data[:, 2]) .* 1e-3
        # Drop non-positive errors (log axis). Order points along the
        # sweep (loose -> tight setting) so the float32 roundoff U-turn
        # in the fixed curves is traced rather than folded onto itself.
        keep = err .> 0
        setting, err, time_s = setting[keep], err[keep], time_s[keep]
        isempty(err) && return
        order = sortperm(setting, rev = true)
        push!(series, WPSeries(display, mode, os, gpu, key,
                               err[order], time_s[order]))
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
        plot!(plt, s.err, s.time_s, label = label, color = COLORS[s.display],
            marker = MARKERS[s.display], linestyle = ls)
    end

    outdir = joinpath(plots_dir, group_label)
    isdir(outdir) || mkpath(outdir)
    outfile = joinpath(outdir, "Lorenz_wp_$(mode_label).png")
    savefig(plt, outfile)
    println("Saved $(outfile)")
end

function main()
    series = collect_series(base_path)
    if isempty(series)
        println("Warning: no keyed wp files found under $(base_path). Nothing to plot.")
        println("Expected files like <package>/<os>_<gpu>/<Prefix>_wp_adaptive.txt (run the wp benchmarks first).")
        return
    end

    plots_dir = joinpath(dirname(dirname(@__DIR__)), "plots")

    for (label, sel) in build_groups(series)
        multikey = length(unique(s.key for s in sel)) > 1
        render_plot(filter(s -> s.mode == "fixed", sel), label, "fixed", plots_dir, multikey)
        render_plot(filter(s -> s.mode == "adaptive", sel), label, "adaptive", plots_dir, multikey)
        render_plot(sel, label, "all", plots_dir, multikey)
    end
end

main()
