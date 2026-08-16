# The adaptive protocol every framework runs, mirrored by wp_common.py.

const PROTOCOL_CSV = joinpath(@__DIR__, "protocol.csv")

"Adaptive settings as a name to value dictionary."
function load_protocol()
    lines = filter(!isempty, strip.(readlines(PROTOCOL_CSV)))
    settings = Dict{String, Float64}()
    for line in lines[2:end]
        name, value = split(line, ',')
        settings[String(name)] = parse(Float64, value)
    end
    return settings
end
