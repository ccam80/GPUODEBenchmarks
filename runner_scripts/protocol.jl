# Benchmark protocol constants from protocol.toml, as in protocol.py; safe to include more than once.

if !isdefined(@__MODULE__, :PROTOCOL)
    using TOML

    const PROTOCOL = TOML.parsefile(joinpath(@__DIR__, "protocol.toml"))

    const N_WP = Int(PROTOCOL["ensemble"]["n_wp"])
    const N_NE = Int(PROTOCOL["ensemble"]["n_ne"])
    const STATES_N = Int(PROTOCOL["ensemble"]["n_states"])
    const N_MIN = Int(PROTOCOL["ensemble"]["n_min"])
    const N_STEP = Int(PROTOCOL["ensemble"]["n_step"])
    const NMAX_DEFAULT = Int(PROTOCOL["ensemble"]["nmax_default"])

    const TIMING_DT_K = Int(PROTOCOL["fixed"]["timing_k"])
    const WP_K = Tuple(Int.(PROTOCOL["fixed"]["wp_k"]))
    const EULER_K = Tuple(Int.(PROTOCOL["fixed"]["euler_k"]))
    const NE_K = Tuple(Int.(PROTOCOL["fixed"]["ne_k"]))

    const TIMING_TOL = Float64(PROTOCOL["adaptive"]["timing_tol"])
    const OVERLAP_TOL = Float64(PROTOCOL["adaptive"]["overlap_tol"])
    const TOL_K = Tuple(Int.(PROTOCOL["adaptive"]["tol_k"]))
    const TOLS = [10.0^-k for k in TOL_K[1]:TOL_K[2]]
    const DT_MIN_FRACTION = Float64(PROTOCOL["adaptive"]["dt_min_fraction"])

    # Fixed-step Newton termination scale; adaptive solves use the step tolerance.
    const NEWTON_ATOL = Float64(PROTOCOL["newton"]["atol"])
    const NEWTON_RTOL = Float64(PROTOCOL["newton"]["rtol"])

    const REPEAT_CAP = Int(PROTOCOL["repeats"]["cap"])
    const REPEAT_SCHEDULE = Tuple((Float64(row[1]), Int(row[2]), Int(row[3]))
                                  for row in PROTOCOL["repeats"]["schedule"])
    const REPEAT_SPREAD = Float64(PROTOCOL["repeats"]["spread"])

    # BENCH_WATCHDOG_SECONDS overrides the per-run wall-clock ceiling.
    const WATCHDOG_SECONDS = parse(Float64, get(ENV, "BENCH_WATCHDOG_SECONDS",
        string(PROTOCOL["watchdog"]["seconds"])))
    const WATCHDOG_EXIT_CODE = Cint(PROTOCOL["watchdog"]["exit_code"])

    const MAX_ERRORED_PCT = Float64(PROTOCOL["plots"]["max_errored_pct"])

    "dt grid duration * 2^-k over the inclusive exponent range."
    fixed_dts(duration, k_range) = [duration * 2.0^-k for k in k_range[1]:k_range[2]]

    "N sweep n_min * n_step^k up to nmax, from from_n up."
    function performance_ns(nmax = NMAX_DEFAULT, from_n = 0)
        values, n = Int[], N_MIN
        while n <= nmax
            n >= from_n && push!(values, n)
            n *= N_STEP
        end
        return values
    end

    "A single value is a sweep ceiling; a comma list is the exact counts."
    function parse_ns(spec, from_n = 0)
        text = string(spec)
        occursin(',', text) || return performance_ns(parse(Int, text), from_n)
        values = sort(unique(parse.(Int, filter(!isempty, split(text, ',')))))
        return filter(n -> n >= max(from_n, N_MIN), values)
    end
end
