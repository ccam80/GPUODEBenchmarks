using Plots
using DelimitedFiles
using Dates
using Statistics
using Plots.PlotMeasures
include(joinpath(dirname(@__DIR__), "errored.jl"))
include(joinpath(dirname(@__DIR__), "results.jl"))

# Reads the `times` rows of every data/<package>/<os>_<gpu>/results.csv and
# emits one plot per (group, problem, mode, algorithm, transfer variant) plus an
# "all" overview into plots/<group>/<problem>/. ARGS[1] overrides the data dir.
parent_dir = length(ARGS) != 0 ? ARGS[1] : "data"
base_path = joinpath(dirname(dirname(@__DIR__)), parent_dir)

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
    problem::String
    mode::String       # "fixed" or "adaptive"
    algorithm::String  # cubie-vocabulary method name, e.g. "classical-rk4"
    transfers::String  # "both" (h2d + kernel + d2h) or "none"
    os::String
    gpu::String
    key::String      # "<os>_<gpu>"
    x::Vector{Float64}
    y::Vector{Float64}
end

# One curve per (package, key, problem, mode, algorithm, transfers) from the store's times rows.
function collect_series(base_path)
    groups = Dict{NTuple{6, String}, Vector{Tuple{Float64, Float64}}}()
    meta = Dict{NTuple{6, String}, Tuple{String, String}}()
    for row in result_rows_under(base_path)
        row["analysis"] == "times" || continue
        # Drop rows past the errored bar.
        within_error_budget(_result_float(row["errored_pct"])) || continue
        id = (RESULT_DISPLAY[row["package"]], row["problem"], row["mode"],
              row["algorithm"], row["transfers"], row["key"])
        push!(get!(groups, id, Tuple{Float64, Float64}[]),
              (parse(Float64, row["n"]), _result_float(row["min_ms"]) * 1e-3))
        meta[id] = (row["os"], row["gpu"])
    end
    series = Series[]
    for (id, points) in groups
        sort!(points)
        display, problem, mode, algorithm, transfers, key = id
        os, gpu = meta[id]
        push!(series, Series(display, problem, mode, algorithm, transfers, os,
            gpu, key, first.(points), last.(points)))
    end
    return sort(series, by = s -> (s.display, s.problem, s.mode, s.algorithm,
        s.transfers, s.key))
end

# Draw one plot; alg_label "all" mixes algorithms and labels them per series.
function render_plot(sel, group_label, problem, mode_label, alg_label, transfers_label, plots_dir, multikey)
    if isempty(sel)
        println("Skipping empty plot: $(problem)_$(mode_label)_$(alg_label)_$(transfers_label)_$(group_label)")
        return
    end
    xticks = 10 .^ round.(range(1, 7, length = 13), digits = 2)
    yticks = 10 .^ round.(range(2, -5, length = 15), digits = 2)
    gr(size = (810, 540))
    modeword = mode_label == "fixed" ? "Fixed" : mode_label == "adaptive" ? "Adaptive" : "Adaptive vs Fixed"
    algword = alg_label == "all" ? "all algorithms" : alg_label
    transferword = transfers_label == "both" ? "with h2d+d2h" : "device only"
    plt = plot(xaxis = :log, yaxis = :log, linewidth = 2, ylabel = "Time (s)", xlabel = "Trajectories",
        title = "$(problem): $(modeword) time-steps, $(algword), $(transferword) ($(group_label))",
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

    outdir = joinpath(plots_dir, group_label, problem)
    isdir(outdir) || mkpath(outdir)
    algpart = alg_label == "all" ? "" : "_$(alg_label)"
    outfile = joinpath(outdir, "times_$(mode_label)$(algpart)_$(transfers_label).png")
    savefig(plt, outfile)
    println("Saved $(outfile)")
end

function main()
    series = collect_series(base_path)
    if isempty(series)
        println("Warning: no times rows found under $(base_path). Nothing to plot.")
        println("Expected <package>/<os>_<gpu>/results.csv files (run the benchmarks first).")
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
        for problem in sort(unique(s.problem for s in sel))
            psel = filter(s -> s.problem == problem, sel)
            for transfers in sort(unique(s.transfers for s in psel))
                tsel = filter(s -> s.transfers == transfers, psel)
                for mode in ("fixed", "adaptive")
                    msel = filter(s -> s.mode == mode, tsel)
                    for alg in sort(unique(s.algorithm for s in msel))
                        render_plot(filter(s -> s.algorithm == alg, msel),
                                    label, problem, mode, alg, transfers,
                                    plots_dir, multikey)
                    end
                end
                render_plot(tsel, label, problem, "all", "all", transfers,
                            plots_dir, multikey)
            end
        end
    end
end

main()
