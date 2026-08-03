using Plots
using DelimitedFiles
using Dates
using Statistics
using Plots.PlotMeasures

# Plot the Lorenz timing benchmarks. Data files are named
#   data/<DIR>/<Prefix>_times_<fixed|adaptive>_<algorithm>_<os>_<gpu>.txt
# where <algorithm> is the cubie-vocabulary method name (euler, classical-rk4,
# tsit5, cash-karp-54) and "<os>_<gpu>" is the per-machine key (see
# runner_scripts/bench_key.*), e.g.
#   data/Julia/Julia_times_adaptive_tsit5_linux_RTX-2060-SUPER.txt
# so the same repo can be populated additively across machines. This script
# discovers those files, parses (framework, mode, algorithm, os, gpu) from
# the names, and emits:
#   * one algorithm-matched plot per (group, mode, algorithm) — every curve
#     within a figure runs the same integration method (issue #29), and
#   * one "all" overview per group with the algorithm in each label.
# Groups: "all" (everything combined), one per distinct os, one per distinct
# gpu; each is drawn per transfer variant (with h2d+d2h / device only),
# giving e.g. Lorenz_fixed_classical-rk4_both_linux.png.
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

# color/marker choices per framework, kept identical across every subset
# figure so a package stays recognisable from plot to plot.
colors = Dict("Julia"=>:Green, "MPGOS"=>:Orange, "JAX"=>:Red,
    "PYTORCH"=>:DarkRed, "CUBIE"=>:Blue, "CUBIE_MLIR"=>:Purple,
    "MYOKIT CUDA"=>:Black)
markers = Dict("Julia"=>:circle, "MPGOS"=>:utriangle, "JAX"=>:diamond,
    "PYTORCH"=>:xcross, "CUBIE"=>:star5, "CUBIE_MLIR"=>:hexagon,
    "MYOKIT CUDA"=>:rect)

# One benchmark curve loaded from disk.
struct Series
    display::String
    mode::String       # "fixed" or "adaptive"
    algorithm::String  # cubie-vocabulary method name, e.g. "classical-rk4"
    transfers::String  # "both" (h2d + kernel + d2h) or "none"
    os::String
    gpu::String
    key::String      # "<os>_<gpu>"
    x::Vector{Float64}
    y::Vector{Float64}
end

# Discover every keyed timing file and load it into a Series.
function collect_series(base_path, frameworks)
    series = Series[]
    for (display, dir, prefix) in frameworks
        dpath = joinpath(base_path, dir)
        isdir(dpath) || continue
        # Match "<prefix>_times_<fixed|adaptive>_<algorithm>_<os>_<gpu>.txt".
        # None of algorithm, os and gpu contain underscores, so they are the
        # last three underscore-separated fields.
        pat = Regex("^" * prefix * "_times_(fixed|adaptive)_([^_]+)_([^_]+)_([^_]+)\\.txt\$")
        for fname in sort(readdir(dpath))
            m = match(pat, fname)
            m === nothing && continue
            mode, algorithm, os, gpu = m.captures
            data = readdlm(joinpath(dpath, fname))
            isempty(data) && continue
            order = sortperm(data[:, 1])
            size(data, 2) == 3 || error(
                "$(fname) has $(size(data, 2)) columns; expected 3 " *
                "(N, time_with_transfers_ms, time_device_only_ms)")
            ns = Float64.(data[order, 1])
            push!(series, Series(display, String(mode), String(algorithm),
                                 "both", os, gpu, "$(os)_$(gpu)",
                                 ns, data[order, 2] .* 1e-3))
            push!(series, Series(display, String(mode), String(algorithm),
                                 "none", os, gpu, "$(os)_$(gpu)",
                                 ns, data[order, 3] .* 1e-3))
        end
    end
    return series
end

# Draw and save one plot for the given series subset, or warn if it is empty.
# alg_label == "all" is the overview figure mixing algorithms (labels carry
# the algorithm); otherwise every series in `sel` runs the same algorithm.
function render_plot(sel, group_label, mode_label, alg_label, transfers_label, plots_dir, multikey)
    if isempty(sel)
        println("Skipping empty plot: $(mode_label)_$(alg_label)_$(transfers_label)_$(group_label)")
        return
    end
    xticks = 10 .^ round.(range(1, 7, length = 13), digits = 2)
    yticks = 10 .^ round.(range(2, -5, length = 15), digits = 2)
    gr(size = (810, 540))
    modeword = mode_label == "fixed" ? "Fixed" : mode_label == "adaptive" ? "Adaptive" : "Adaptive vs Fixed"
    algword = alg_label == "all" ? "all algorithms" : alg_label
    transferword = transfers_label == "both" ? "with h2d+d2h" : "device only"
    plt = plot(xaxis = :log, yaxis = :log, linewidth = 2, ylabel = "Time (s)", xlabel = "Trajectories",
        title = "Lorenz Problem: $(modeword) time-steps, $(algword), $(transferword) ($(group_label))",
        titlefontsize = 11, legend = :topleft,
        xticks = xticks, yticks = yticks, dpi = 600)

    for s in sel
        stepword = s.mode == "adaptive" ? "adaptive" : "fixed"
        # Name the algorithm in the label whenever the figure mixes them.
        algpart = alg_label == "all" ? ", $(s.algorithm)" : ""
        # Only disambiguate by key when the group mixes machines.
        keypart = multikey ? " [$(s.key)]" : ""
        label = "$(s.display) ($(stepword)$(algpart))$(keypart)"
        ls = s.mode == "adaptive" ? :dash : :solid
        plot!(plt, s.x, s.y, label = label, color = colors[s.display], marker = markers[s.display], linestyle = ls)
    end

    isdir(plots_dir) || mkpath(plots_dir)
    algpart = alg_label == "all" ? "" : "_$(alg_label)"
    outfile = joinpath(plots_dir, "Lorenz_$(mode_label)$(algpart)_$(transfers_label)_$(group_label).png")
    savefig(plt, outfile)
    println("Saved $(outfile)")
end

function main()
    series = collect_series(base_path, frameworks)
    if isempty(series)
        println("Warning: no keyed timing files found under $(base_path). Nothing to plot.")
        println("Expected files like <Prefix>_times_adaptive_<algorithm>_<os>_<gpu>.txt (run the benchmarks first).")
        return
    end

    plots_dir = joinpath(dirname(dirname(@__DIR__)), "plots")

    # Build the output groups: everything combined, one per os, one per gpu.
    oses = sort(unique(s.os for s in series))
    gpus = sort(unique(s.gpu for s in series))
    groups = Tuple{String, Vector{Series}}[]
    push!(groups, ("all", series))
    for os in oses
        push!(groups, (os, filter(s -> s.os == os, series)))
    end
    for gpu in gpus
        push!(groups, (gpu, filter(s -> s.gpu == gpu, series)))
    end

    # Emit one algorithm-matched plot per (mode, algorithm) plus an "all"
    # overview, per group and transfer variant.
    for (label, sel) in groups
        multikey = length(unique(s.key for s in sel)) > 1
        for transfers in sort(unique(s.transfers for s in sel))
            tsel = filter(s -> s.transfers == transfers, sel)
            for mode in ("fixed", "adaptive")
                msel = filter(s -> s.mode == mode, tsel)
                for alg in sort(unique(s.algorithm for s in msel))
                    render_plot(filter(s -> s.algorithm == alg, msel),
                                label, mode, alg, transfers, plots_dir, multikey)
                end
            end
            render_plot(tsel, label, "all", "all", transfers, plots_dir, multikey)
        end
    end
end

main()
