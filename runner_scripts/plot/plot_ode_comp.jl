using Plots
using DelimitedFiles
using Dates
using Statistics
using Plots.PlotMeasures

# Default: use the repo `data/` directory. Optionally pass a custom data directory as ARGS[1].
parent_dir = length(ARGS) != 0 ? ARGS[1] : "data"
base_path = joinpath(dirname(dirname(@__DIR__)), parent_dir)

# Map of display name -> (subdirectory, filename prefix)
# Note: MPGOS data files are stored under `CPP/` in the repo's `data/` folder.
frameworks = Dict(
    "Julia" => ("Julia", "Julia"),
    "MPGOS" => ("CPP", "MPGOS"),
    "JAX" => ("JAX", "Jax"),
    "PYTORCH" => ("PYTORCH", "Torch"),
    "CUBIE" => ("CUBIE", "Cubie"),
)

# color/marker choices per framework
colors = Dict("Julia"=>:Green, "MPGOS"=>:Orange, "JAX"=>:Red, "PYTORCH"=>:DarkRed, "CUBIE"=>:Blue)
markers = Dict("Julia"=>:circle, "MPGOS"=>:utriangle, "JAX"=>:diamond, "PYTORCH"=>:xcross, "CUBIE"=>:star5)

function main()
    # Collect available series and first Ns
    Ns = nothing
    series_plots = []

    for (display, tuple) in frameworks
        dir, prefix = tuple
        unadaptive_path = joinpath(base_path, dir, "$(prefix)_times_unadaptive.txt")
        adaptive_path = joinpath(base_path, dir, "$(prefix)_times_adaptive.txt")

        if isfile(unadaptive_path)
            data = readdlm(unadaptive_path)
            if Ns === nothing
                Ns = data[:, 1]
            end
            push!(series_plots, (label = "$(display) (fixed)", x = Ns, y = data[:, 2] .* 1e-3, color = colors[display], marker = markers[display]))
        else
            println("Warning: missing unadaptive file for $(display) at $(unadaptive_path)")
        end

        if isfile(adaptive_path)
            data = readdlm(adaptive_path)
            if Ns === nothing
                Ns = data[:, 1]
            end
            push!(series_plots, (label = "$(display) (adaptive)", x = Ns, y = data[:, 2] .* 1e-3, color = colors[display], marker = markers[display], linestyle = :dash))
        else
            println("Warning: missing adaptive file for $(display) at $(adaptive_path)")
        end
    # end of per-framework processing
    end
    if Ns === nothing
        println("Warning: no data files found under $(base_path). Nothing to plot.")
    else
        # proceed to plot available series
        xticks = 10 .^ round.(range(1, 7, length = 13), digits = 2)
        yticks = 10 .^ round.(range(2, -5, length = 15), digits = 2)
        gr(size = (810, 540))
        plt = plot(xaxis = :log, yaxis = :log, linewidth = 2, ylabel = "Time (s)", xlabel = "Trajectories",
            title = "Lorenz Problem: Adaptive vs Fixed time-steps (per-framework)", legend = :topleft, xticks = xticks, yticks = yticks, dpi = 600)

        for s in series_plots
            ls = get(s, :linestyle, :solid)
            plot!(plt, s.x, s.y, label = s.label, color = s.color, marker = s.marker, linestyle = ls)
        end

        plots_dir = joinpath(dirname(dirname(@__DIR__)), "plots")
        isdir(plots_dir) || mkdir(plots_dir)
        savefig(plt, joinpath(plots_dir, "Lorenz_adaptive_vs_unadaptive_per_framework_$(Dates.value(Dates.now())).png"))
        println("Saved plot to $(plots_dir)")
    end
end

main()
