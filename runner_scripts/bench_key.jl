# Shared dataset-key helper for the Julia benchmark writer.
#
# Produces a key "<os>_<gpu>" identifying the current machine so benchmark output
# files can be additively populated across machines. The GPU name comes from
# nvidia-smi (the single source of truth shared by every framework) and is
# sanitised identically everywhere: tokenise on non-alphanumeric characters, drop
# the "NVIDIA"/"GeForce" vendor words, and join the rest with '-'.
# e.g. "NVIDIA GeForce RTX 2060 SUPER" -> "RTX-2060-SUPER".

function _gpu_name_raw()
    try
        raw = read(`nvidia-smi --query-gpu=name --format=csv,noheader`, String)
        for line in split(raw, '\n')
            s = strip(line)
            isempty(s) || return String(s)
        end
    catch
    end
    return ""
end

function _os_key()
    Sys.iswindows() && return "windows"
    Sys.isapple()   && return "macos"
    Sys.islinux()   && return "linux"
    return "unknown"
end

function _sanitize_gpu(raw)
    tokens = filter(t -> !isempty(t) && t != "NVIDIA" && t != "GeForce",
                    split(raw, r"[^A-Za-z0-9]+"))
    isempty(tokens) ? "unknown-gpu" : join(tokens, "-")
end

"Return \"<os>_<gpu>\" for this machine."
dataset_key() = string(_os_key(), "_", _sanitize_gpu(_gpu_name_raw()))
