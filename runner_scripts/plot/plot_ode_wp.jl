using Plots
using DelimitedFiles
using Dates
using Statistics
using Plots.PlotMeasures

# Reads data/<package>/<os>_<gpu>/<problem>/<Prefix>_wp_<fixed|adaptive>_<algorithm>.txt
# and emits one error-vs-time plot per (group, problem, mode, algorithm) plus an
# "all" overview into plots/<group>/<problem>/. ARGS[1] overrides the data dir.
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
    problem::String
    mode::String       # "fixed" or "adaptive"
    algorithm::String  # cubie-vocabulary method name, e.g. "classical-rk4"
    os::String
    gpu::String
    key::String      # "<os>_<gpu>"
    err::Vector{Float64}
    time_s::Vector{Float64}
end

# Discover every wp file under data/<package>/<key>/ and load it.
function collect_series(base_path, frameworks)
    series = WPSeries[]
    for (display, dir, prefix) in frameworks
        dpath = joinpath(base_path, dir)
        isdir(dpath) || continue
        pat = Regex("^" * prefix * "_wp_(fixed|adaptive)_([^_]+)[.]txt" * "\$")
        for key in sort(readdir(dpath))
            kpath = joinpath(dpath, key)
            isdir(kpath) || continue
            parts = split(key, '_')
            length(parts) == 2 || continue
            os, gpu = String(parts[1]), String(parts[2])
            for problem in sort(readdir(kpath))
                ppath = joinpath(kpath, problem)
                isdir(ppath) || continue
                for fname in sort(readdir(ppath))
                    m = match(pat, fname)
                    m === nothing && continue
                    mode = String(m.captures[1])
                    algorithm = String(m.captures[2])
                    fpath = joinpath(ppath, fname)
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
                    push!(series, WPSeries(display, problem, mode, algorithm,
                                           os, gpu, key, err[order],
                                           time_s[order]))
                end
            end
        end
    end
    return series
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
        title = "$(problem) WP, N=32768, $(modeword), $(algword) ($(group_label))",
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
    series = collect_series(base_path, frameworks)
    if isempty(series)
        println("Warning: no keyed wp files found under $(base_path). Nothing to plot.")
        println("Expected files like <package>/<os>_<gpu>/<Prefix>_wp_adaptive_<algorithm>.txt (run the wp benchmarks first).")
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
