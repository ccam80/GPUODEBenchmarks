# Per-repeat timing log for the Julia GPU writers, as in runner_scripts/wp_common.py.

using Printf

const SAMPLE_FIELDS = ["analysis", "problem", "algorithm", "mode", "transfers",
    "setting_kind", "setting", "n", "states", "repeat", "ms"]

"Path of the per-repeat timing log beside its reduced output file."
function samples_outfile(repo_root, package, key, prefix, analysis, mode,
        algorithm, problem)
    return joinpath(data_dir(repo_root, package, key, problem),
        "$(prefix)_samples_$(analysis)_$(mode)_$(algorithm).csv")
end

"The identity of one timed point, shared by its timed legs."
function sample_point(analysis, problem, algorithm, mode, n, states;
        setting_kind = "none", setting = NaN)
    return (analysis = analysis, problem = problem, algorithm = algorithm,
        mode = mode, setting_kind = setting_kind, setting = setting, n = n,
        states = states)
end

"Drop a leg's log, for the sweeps whose reduced file is rewritten."
reset_samples(path) = isfile(path) && rm(path)

"Append one row per attempt of one timed leg, warm-up as repeat 0."
function append_samples(path, point, transfers, samples)
    header = !isfile(path)
    # Lowercase "nan", as the C and Python writers' %.10g produces.
    setting = isnan(point.setting) ? "nan" : @sprintf("%.10g", point.setting)
    head = string(point.analysis, ",", point.problem, ",", point.algorithm,
        ",", point.mode, ",", transfers, ",", point.setting_kind, ",",
        setting, ",", point.n, ",", point.states)
    # O_APPEND: sibling processes' rows cannot overwrite each other.
    io = Base.Filesystem.open(path, Base.Filesystem.JL_O_WRONLY |
        Base.Filesystem.JL_O_CREAT | Base.Filesystem.JL_O_APPEND, 0o644)
    try
        header && write(io, join(SAMPLE_FIELDS, ",") * "\n")
        for (index, ms) in enumerate(samples)
            write(io, string(head, ",", index - 1, ",",
                @sprintf("%.6f", ms), "\n"))
        end
    finally
        close(io)
    end
end
