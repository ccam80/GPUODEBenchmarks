using Plots
using DelimitedFiles
using Plots.PlotMeasures
include(joinpath(dirname(@__DIR__), "errored.jl"))
include(joinpath(dirname(@__DIR__), "results.jl"))

# Reads the `states` rows of every data/<package>/<os>_<gpu>/results.csv and
# emits one figure per (key, mode, algorithm) into plots/states/<key>/: solid =
# run time, dashed = compile time, each trace ending at its last finite value.
# ARGS[1] overrides the data dir.
parent_dir = length(ARGS) != 0 ? ARGS[1] : "data"
base_path = joinpath(dirname(dirname(@__DIR__)), parent_dir)

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

# One trace per (package, key, mode, algorithm) from the store's states rows, host-path leg.
function collect_series(base_path)
    groups = Dict{NTuple{4, String}, Vector{NTuple{3, Float64}}}()
    for row in result_rows_under(base_path)
        (row["analysis"] == "states" && row["transfers"] == "both") || continue
        # Drop rows past the errored bar.
        within_error_budget(_result_float(row["errored_pct"])) || continue
        id = (RESULT_DISPLAY[row["package"]], row["mode"], row["algorithm"],
              row["key"])
        push!(get!(groups, id, NTuple{3, Float64}[]),
              (parse(Float64, row["states"]), _result_float(row["min_ms"]) / 1000.0,
               _result_float(row["build_s"])))
    end
    series = StatesSeries[]
    for (id, points) in groups
        sort!(points, by = first)
        display, mode, algorithm, key = id
        push!(series, StatesSeries(display, mode, algorithm, key,
            first.(points), [p[2] for p in points], [p[3] for p in points]))
    end
    return sort(series, by = s -> (s.display, s.mode, s.algorithm, s.key))
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

plot_states(collect_series(base_path))
