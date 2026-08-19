using Plots
using DelimitedFiles
using Plots.PlotMeasures

# Reads data/<package>/<os>_<gpu>/lorenz96/<Prefix>_states_<mode>_<algorithm>.txt
# (rows: states t_ms t_dev_ms build_s) and emits one figure per
# (key, mode, algorithm) into plots/states/<key>/: solid = run time, dashed =
# compile time, each trace ending at its last finite value. ARGS[1] overrides
# the data dir.
parent_dir = length(ARGS) != 0 ? ARGS[1] : "data"
base_path = joinpath(dirname(dirname(@__DIR__)), parent_dir)

frameworks = [
    ("Julia", "Julia", "Julia"),
    ("MPGOS", "CPP", "MPGOS"),
    ("JAX", "JAX", "Jax"),
    ("PYTORCH", "PYTORCH", "Torch"),
    ("CUBIE", "CUBIE", "Cubie"),
    ("CUBIE_MLIR", "CUBIE_MLIR", "Cubie_mlir"),
    ("MYOKIT CUDA", "MYOKIT_CUDA", "Myokit_cuda"),
]

colors = Dict("Julia"=>:Green, "MPGOS"=>:Orange, "JAX"=>:Red,
    "PYTORCH"=>:DarkRed, "CUBIE"=>:Blue, "CUBIE_MLIR"=>:Purple,
    "MYOKIT CUDA"=>:Black)
markers = Dict("Julia"=>:circle, "MPGOS"=>:utriangle, "JAX"=>:diamond,
    "PYTORCH"=>:xcross, "CUBIE"=>:star5, "CUBIE_MLIR"=>:hexagon,
    "MYOKIT CUDA"=>:rect)

struct StatesSeries
    display::String
    mode::String
    algorithm::String
    key::String
    states::Vector{Float64}
    run_s::Vector{Float64}
    build_s::Vector{Float64}
end

function collect_series(base_path, frameworks)
    series = StatesSeries[]
    for (display, dir, prefix) in frameworks
        dpath = joinpath(base_path, dir)
        isdir(dpath) || continue
        pat = Regex("^" * prefix * "_states_(fixed|adaptive)_(.+)[.]txt\$")
        for key in sort(readdir(dpath))
            ppath = joinpath(dpath, key, "lorenz96")
            isdir(ppath) || continue
            for file in sort(readdir(ppath))
                m = match(pat, file)
                m === nothing && continue
                raw = readdlm(joinpath(ppath, file), Float64)
                size(raw, 2) >= 4 || continue
                push!(series, StatesSeries(display, String(m[1]),
                    String(m[2]), key, raw[:, 1], raw[:, 2] ./ 1000.0,
                    raw[:, 4]))
            end
        end
    end
    return series
end

# One figure per (key, mode, algorithm): solid run trace, dashed compile trace.
function plot_states(series)
    outroot = joinpath(dirname(dirname(@__DIR__)), "plots", "states")
    groups = Dict{Tuple{String, String, String}, Vector{StatesSeries}}()
    for s in series
        push!(get!(groups, (s.key, s.mode, s.algorithm), StatesSeries[]), s)
    end
    for ((key, mode, algorithm), members) in sort(collect(groups), by = first)
        plt = plot(xscale = :log2, yscale = :log10,
            xlabel = "states", ylabel = "time (s)",
            title = "lorenz96 states sweep, $(mode) $(algorithm)",
            legend = :topleft, dpi = 300, left_margin = 10px)
        drew = false
        for s in sort(members, by = m -> m.display)
            run_keep = findall(isfinite, s.run_s)
            if !isempty(run_keep)
                plot!(plt, s.states[run_keep], s.run_s[run_keep],
                    label = "$(s.display) run", color = colors[s.display],
                    marker = markers[s.display], linestyle = :solid)
                drew = true
            end
            build_keep = findall(i -> isfinite(s.build_s[i]) &&
                                      s.build_s[i] > 0, eachindex(s.build_s))
            if !isempty(build_keep)
                plot!(plt, s.states[build_keep], s.build_s[build_keep],
                    label = "$(s.display) compile",
                    color = colors[s.display], linestyle = :dash)
                drew = true
            end
        end
        drew || continue
        outdir = joinpath(outroot, key)
        mkpath(outdir)
        savefig(plt, joinpath(outdir, "states_$(mode)_$(algorithm).png"))
        println("plots/states/$(key)/states_$(mode)_$(algorithm).png")
    end
end

plot_states(collect_series(base_path, frameworks))
