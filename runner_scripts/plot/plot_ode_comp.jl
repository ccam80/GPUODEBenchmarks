using Plots
using DelimitedFiles
using Dates
using Statistics
using Plots.PlotMeasures

# Reads data/<package>/<os>_<gpu>/<Prefix>_times_<fixed|adaptive>_<algorithm>.txt
# and emits one plot per (group, mode, algorithm, transfer variant) plus an
# "all" overview per group into plots/<group>/. ARGS[1] overrides the data dir.
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

# color/marker choices per framework, identical across every figure
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

# Discover every timing file under data/<package>/<key>/ and load it.
function collect_series(base_path, frameworks)
    series = Series[]
    for (display, dir, prefix) in frameworks
        dpath = joinpath(base_path, dir)
        isdir(dpath) || continue
        pat = Regex("^" * prefix * "_times_(fixed|adaptive)_([^_]+)[.]txt" * "\$")
        for key in sort(readdir(dpath))
            kpath = joinpath(dpath, key)
            isdir(kpath) || continue
            parts = split(key, '_')
            length(parts) == 2 || continue
            os, gpu = String(parts[1]), String(parts[2])
            for fname in sort(readdir(kpath))
                m = match(pat, fname)
                m === nothing && continue
                mode = String(m.captures[1])
                algorithm = String(m.captures[2])
                data = readdlm(joinpath(kpath, fname))
                isempty(data) && continue
                order = sortperm(data[:, 1])
                size(data, 2) == 3 || error(
                    "$(fname) has $(size(data, 2)) columns; expected 3 " *
                    "(N, time_with_transfers_ms, time_device_only_ms)")
                ns = Float64.(data[order, 1])
                push!(series, Series(display, mode, algorithm, "both", os, gpu,
                                     key, ns, data[order, 2] .* 1e-3))
                push!(series, Series(display, mode, algorithm, "none", os, gpu,
                                     key, ns, data[order, 3] .* 1e-3))
            end
        end
    end
    return series
end

# Draw one plot; alg_label "all" mixes algorithms and labels them per series.
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

    outdir = joinpath(plots_dir, group_label)
    isdir(outdir) || mkpath(outdir)
    algpart = alg_label == "all" ? "" : "_$(alg_label)"
    outfile = joinpath(outdir, "Lorenz_$(mode_label)$(algpart)_$(transfers_label).png")
    savefig(plt, outfile)
    println("Saved $(outfile)")
end

function main()
    series = collect_series(base_path, frameworks)
    if isempty(series)
        println("Warning: no keyed timing files found under $(base_path). Nothing to plot.")
        println("Expected files like <package>/<os>_<gpu>/<Prefix>_times_adaptive_<algorithm>.txt (run the benchmarks first).")
        return
    end

    plots_dir = joinpath(dirname(dirname(@__DIR__)), "plots")

    # Build the output groups: everything combined, one per os, one per gpu.
    # Most specific first; a group repeating an earlier group's keys is dropped.
    groups = Tuple{String, Vector{Series}}[]
    seen = Set{Set{String}}()
    function add_group!(label, sel)
        isempty(sel) && return
        ks = Set(s.key for s in sel)
        ks in seen && return
        push!(seen, ks)
        push!(groups, (label, sel))
    end
    for key in sort(unique(s.key for s in series))
        add_group!(key, filter(s -> s.key == key, series))
    end
    for os in sort(unique(s.os for s in series))
        add_group!(os, filter(s -> s.os == os, series))
    end
    for gpu in sort(unique(s.gpu for s in series))
        add_group!(gpu, filter(s -> s.gpu == gpu, series))
    end
    add_group!("all", series)
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
