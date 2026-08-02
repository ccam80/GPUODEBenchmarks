using Plots
using DelimitedFiles
using Dates
using Statistics
using Plots.PlotMeasures

# Plot the Lorenz timing benchmarks: one plot per (group, mode, transfers)
# at plots/<group>/Lorenz_<mode>_<transfers>.png. ARGS[1] overrides "data".
include(joinpath(@__DIR__, "plot_common.jl"))

parent_dir = length(ARGS) != 0 ? ARGS[1] : "data"
base_path = joinpath(dirname(dirname(@__DIR__)), parent_dir)

# One benchmark curve loaded from disk.
struct Series
    display::String
    mode::String       # "fixed" or "adaptive"
    transfers::String  # "both" (h2d + kernel + d2h) or "none"
    os::String
    gpu::String
    key::String      # "<os>_<gpu>"
    x::Vector{Float64}
    y::Vector{Float64}
end

# Discover every timing file under data/<package>/<key>/ and load it.
function collect_series(base_path)
    series = Series[]
    eachkeyedfile(base_path, "_times_(adaptive|unadaptive)[.]txt") do display, key, os, gpu, path, captured
        mode = captured == "adaptive" ? "adaptive" : "fixed"
        data = readdlm(path)
        isempty(data) && return
        size(data, 2) == 3 || error(
            "$(basename(path)) has $(size(data, 2)) columns; expected 3 " *
            "(N, time_with_transfers_ms, time_device_only_ms)")
        order = sortperm(data[:, 1])
        ns = Float64.(data[order, 1])
        push!(series, Series(display, mode, "both", os, gpu, key,
                             ns, data[order, 2] .* 1e-3))
        push!(series, Series(display, mode, "none", os, gpu, key,
                             ns, data[order, 3] .* 1e-3))
    end
    return series
end

# Draw and save one plot for the given series subset, or warn if it is empty.
function render_plot(sel, group_label, mode_label, transfers_label, plots_dir, multikey)
    if isempty(sel)
        println("Skipping empty plot: $(mode_label)_$(transfers_label)_$(group_label)")
        return
    end
    xticks = 10 .^ round.(range(1, 7, length = 13), digits = 2)
    yticks = 10 .^ round.(range(2, -5, length = 15), digits = 2)
    gr(size = (810, 540))
    modeword = mode_label == "fixed" ? "Fixed" : mode_label == "adaptive" ? "Adaptive" : "Adaptive vs Fixed"
    transferword = transfers_label == "both" ? "with h2d+d2h" : "device only"
    plt = plot(xaxis = :log, yaxis = :log, linewidth = 2, ylabel = "Time (s)", xlabel = "Trajectories",
        title = "Lorenz Problem: $(modeword) time-steps, $(transferword) ($(group_label))",
        titlefontsize = 11, legend = :topleft,
        xticks = xticks, yticks = yticks, dpi = 600)

    for s in sel
        stepword = s.mode == "adaptive" ? "adaptive" : "fixed"
        # Only disambiguate by key when the group mixes machines.
        label = multikey ? "$(s.display) ($(stepword)) [$(s.key)]" : "$(s.display) ($(stepword))"
        ls = s.mode == "adaptive" ? :dash : :solid
        plot!(plt, s.x, s.y, label = label, color = COLORS[s.display], marker = MARKERS[s.display], linestyle = ls)
    end

    outdir = joinpath(plots_dir, group_label)
    isdir(outdir) || mkpath(outdir)
    outfile = joinpath(outdir, "Lorenz_$(mode_label)_$(transfers_label).png")
    savefig(plt, outfile)
    println("Saved $(outfile)")
end

function main()
    series = collect_series(base_path)
    if isempty(series)
        println("Warning: no keyed timing files found under $(base_path). Nothing to plot.")
        println("Expected files like <package>/<os>_<gpu>/<Prefix>_times_adaptive.txt (run the benchmarks first).")
        return
    end

    plots_dir = joinpath(dirname(dirname(@__DIR__)), "plots")

    for (label, sel) in build_groups(series)
        multikey = length(unique(s.key for s in sel)) > 1
        for transfers in sort(unique(s.transfers for s in sel))
            tsel = filter(s -> s.transfers == transfers, sel)
            render_plot(filter(s -> s.mode == "fixed", tsel), label, "fixed", transfers, plots_dir, multikey)
            render_plot(filter(s -> s.mode == "adaptive", tsel), label, "adaptive", transfers, plots_dir, multikey)
            render_plot(tsel, label, "all", transfers, plots_dir, multikey)
        end
    end
end

main()
