# Per-repeat timing log for the Julia GPU writers, as in runner_scripts/wp_common.py.

using Printf

const SAMPLE_FIELDS = ["analysis", "problem", "algorithm", "mode", "transfers",
    "setting_kind", "setting", "n", "states", "repeat", "ms"]

# Windows creates the file read-only unless the mode is passed explicitly.
const SAMPLE_MODE = 0o644
const SAMPLE_WRITE = Base.Filesystem.JL_O_WRONLY | Base.Filesystem.JL_O_CREAT

"Path of the per-repeat timing log beside its reduced output file."
function samples_outfile(repo_root, package, key, prefix, analysis, mode,
        algorithm, problem)
    return joinpath(data_dir(repo_root, package, key, problem),
        "$(prefix)_samples_$(analysis)_$(mode)_$(algorithm).csv")
end

"Per-repeat timing rows for one (analysis, mode, algorithm) leg; `truncate` matches the sibling output file's open mode."
struct SampleLog
    io::IO
end

function SampleLog(path::AbstractString; truncate::Bool = false)
    header = join(SAMPLE_FIELDS, ",") * "\n"
    if truncate
        io = Base.Filesystem.open(path,
            SAMPLE_WRITE | Base.Filesystem.JL_O_TRUNC, SAMPLE_MODE)
        write(io, header)
        return SampleLog(io)
    end
    # Exclusive create: one header even when sibling processes open the log.
    try
        fresh = Base.Filesystem.open(path,
            SAMPLE_WRITE | Base.Filesystem.JL_O_EXCL, SAMPLE_MODE)
        try
            write(fresh, header)
        finally
            close(fresh)
        end
    catch err
        # Another process created it first.
        isa(err, Base.IOError) || rethrow()
    end
    # O_APPEND: sibling processes' rows cannot overwrite each other.
    return SampleLog(Base.Filesystem.open(path,
        SAMPLE_WRITE | Base.Filesystem.JL_O_APPEND, SAMPLE_MODE))
end

"A `sink(repeat, ms)` callable for one timed point."
function sample_sink(log::SampleLog, analysis, problem, algorithm, mode,
        transfers, n, states; setting_kind = "none", setting = NaN)
    # Lowercase "nan", as the C and Python writers' %.10g produces.
    setting_text = isnan(setting) ? "nan" : @sprintf("%.10g", setting)
    head = string(analysis, ",", problem, ",", algorithm, ",", mode, ",",
        transfers, ",", setting_kind, ",", setting_text, ",", n, ",", states)
    # One unbuffered write per row, so a row reaches the file whole.
    return (repeat, ms) -> write(log.io, string(head, ",", repeat, ",",
        @sprintf("%.6f", ms), "\n"))
end

Base.close(log::SampleLog) = close(log.io)
