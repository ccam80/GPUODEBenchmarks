# The benchmark protocol, loaded from runner_scripts/protocol.csv — the single
# source of truth shared with runner_scripts/protocol.py and the MPGOS binary.
# Include with:
#   include(joinpath(<repo root>, "runner_scripts", "protocol.jl"))
# which defines the same names as protocol.py.

let path = joinpath(@__DIR__, "protocol.csv")
    table = Dict{String, Vector{Float64}}()
    for line in readlines(path)[2:end]
        isempty(strip(line)) && continue
        name, values = split(line, ','; limit = 2)
        table[name] = [parse(Float64, v) for v in split(values)]
    end
    global const PROTOCOL_REPEATS = Int(table["repeats"][1])
    global const PROTOCOL_N_WP = Int(table["n_wp"][1])
    global const PROTOCOL_N_NE = Int(table["n_ne"][1])
    global const PROTOCOL_PERF_FIXED_DT = table["perf_fixed_dt"][1]
    global const PROTOCOL_PERF_ADAPTIVE_TOL = table["perf_adaptive_tol"][1]
    global const PROTOCOL_WP_DTS = table["wp_dts"]
    global const PROTOCOL_WP_TOLS = table["wp_tols"]
    global const PROTOCOL_NE_DTS = table["ne_dts"]
    global const PROTOCOL_NE_TOLS = table["ne_tols"]
end
