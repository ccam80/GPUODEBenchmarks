using Plots
using DelimitedFiles
using Dates
using Statistics
using Plots.PlotMeasures
include(joinpath(dirname(@__DIR__), "errored.jl"))
include(joinpath(dirname(@__DIR__), "results.jl"))

# Reads the `wp` rows of every data/<package>/<os>_<gpu>/results.csv and emits
# one error-vs-time plot per (group, problem, mode, algorithm) plus an "all"
# overview into plots/<group>/<problem>/. ARGS[1] overrides the data dir.
parent_dir = length(ARGS) != 0 ? ARGS[1] : "data"
base_path = joinpath(dirname(dirname(@__DIR__)), parent_dir)

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
    problem::String
    mode::String       # "fixed" or "adaptive"
    algorithm::String  # cubie-vocabulary method name, e.g. "classical-rk4"
    os::String
    gpu::String
    key::String      # "<os>_<gpu>"
    err::Vector{Float64}
    time_s::Vector{Float64}
end

# One curve per (package, key, problem, mode, algorithm) from the store's wp rows.
function collect_series(base_path)
    groups = Dict{NTuple{5, String}, Vector{NTuple{3, Float64}}}()
    meta = Dict{NTuple{5, String}, Tuple{String, String}}()
    for row in result_rows_under(base_path)
        row["analysis"] == "wp" || continue
        # Work-precision compares resident solves; rows timed with transfers predate that protocol.
        row["transfers"] == "none" || continue
        err = _result_float(row["error"])
        # Drop non-positive errors (log axis) and rows past the errored bar.
        (isfinite(err) && err > 0) || continue
        within_error_budget(_result_float(row["errored_pct"])) || continue
        id = (RESULT_DISPLAY[row["package"]], row["problem"], row["mode"],
              row["algorithm"], row["key"])
        push!(get!(groups, id, NTuple{3, Float64}[]),
              (_result_float(row["setting"]), err,
               _result_float(row["min_ms"]) * 1e-3))
        meta[id] = (row["os"], row["gpu"])
    end
    series = WPSeries[]
    for (id, points) in groups
        # Loose -> tight setting, so the float32 roundoff U-turn is traced.
        sort!(points, by = first, rev = true)
        display, problem, mode, algorithm, key = id
        os, gpu = meta[id]
        push!(series, WPSeries(display, problem, mode, algorithm, os, gpu, key,
            [p[2] for p in points], [p[3] for p in points]))
    end
    return sort(series, by = s -> (s.display, s.problem, s.mode, s.algorithm, s.key))
end

# Draw one plot; alg_label "all" mixes algorithms and labels them per series.
function render_plot(sel, group_label, problem, mode_label, alg_label, plots_dir, multikey)
    if isempty(sel)
        println("Skipping empty plot: $(problem)_wp_$(mode_label)_$(alg_label)_$(group_label)")
        return
    end
    gr(size = (810, 540))
    modeword = mode_label == "fixed" ? "fixed dt" :
        mode_label == "adaptive" ? "adaptive tol" : "fixed + adaptive"
    algword = alg_label == "all" ? "all algorithms" : alg_label
    plt = plot(xaxis = :log, yaxis = :log, linewidth = 2,
        ylabel = "Time (s)", xlabel = "Error (ensemble l2, final state)",
        title = "$(problem) WP, N=131072, $(modeword), $(algword) ($(group_label))",
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

    outdir = joinpath(plots_dir, group_label, problem)
    isdir(outdir) || mkpath(outdir)
    algpart = alg_label == "all" ? "" : "_$(alg_label)"
    outfile = joinpath(outdir, "wp_$(mode_label)$(algpart).png")
    savefig(plt, outfile)
    println("Saved $(outfile)")
end

function main()
    series = collect_series(base_path)
    if isempty(series)
        println("Warning: no wp rows found under $(base_path). Nothing to plot.")
        println("Expected <package>/<os>_<gpu>/results.csv files (run the wp benchmarks first).")
        return
    end

    plots_dir = joinpath(dirname(dirname(@__DIR__)), "plots")

    # Most specific first; a group repeating an earlier group's keys is dropped.
    groups = Tuple{String, Vector{WPSeries}}[]
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
            for mode in ("fixed", "adaptive")
                msel = filter(s -> s.mode == mode, psel)
                for alg in sort(unique(s.algorithm for s in msel))
                    render_plot(filter(s -> s.algorithm == alg, msel),
                                label, problem, mode, alg, plots_dir, multikey)
                end
            end
            render_plot(psel, label, problem, "all", "all", plots_dir, multikey)
        end
    end
end

main()
