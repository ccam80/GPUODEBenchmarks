# Run-time constants from protocol.toml ([repeats], [watchdog], [traces]), as in protocol.py; safe to include more than once.

if !isdefined(@__MODULE__, :PROTOCOL)
    using TOML

    const PROTOCOL = TOML.parsefile(joinpath(@__DIR__, "protocol.toml"))

    const REPEAT_CAP = Int(PROTOCOL["repeats"]["cap"])
    const REPEAT_SCHEDULE = Tuple((Float64(row[1]), Int(row[2]), Int(row[3]))
                                  for row in PROTOCOL["repeats"]["schedule"])
    const REPEAT_SPREAD = Float64(PROTOCOL["repeats"]["spread"])

    const WATCHDOG_SECONDS = Float64(PROTOCOL["watchdog"]["seconds"])
    const WATCHDOG_EXIT_CODE = Cint(PROTOCOL["watchdog"]["exit_code"])

    const TRACE_ROWS = Int(PROTOCOL["traces"]["rows"])
    const TRACE_EVERY_S = Float64(PROTOCOL["traces"]["every"])
    const TRACE_SPAN_S = Float64(PROTOCOL["traces"]["span"])
    const TRACE_SAMPLES = round(Int, TRACE_SPAN_S / TRACE_EVERY_S)
end
