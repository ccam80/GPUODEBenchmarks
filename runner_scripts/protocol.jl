# Run-time constants from protocol.toml ([repeats], [watchdog], [optimize]), as in protocol.py; safe to include more than once.

if !isdefined(@__MODULE__, :PROTOCOL)
    using TOML

    const PROTOCOL = TOML.parsefile(joinpath(@__DIR__, "protocol.toml"))

    const REPEAT_CAP = Int(PROTOCOL["repeats"]["cap"])
    const REPEAT_SCHEDULE = Tuple((Float64(row[1]), Int(row[2]), Int(row[3]))
                                  for row in PROTOCOL["repeats"]["schedule"])
    const REPEAT_SPREAD = Float64(PROTOCOL["repeats"]["spread"])

    const WATCHDOG_SECONDS = Float64(PROTOCOL["watchdog"]["seconds"])
    const WATCHDOG_EXIT_CODE = Cint(PROTOCOL["watchdog"]["exit_code"])

    const OPTIMIZE_N = Int(PROTOCOL["optimize"]["n"])
    const OPTIMIZE_PER_POINT_FAMILIES = Tuple(String.(PROTOCOL["optimize"]["per_point_families"]))
end
