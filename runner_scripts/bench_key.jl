# The dataset key "<os>_<gpu>" of this machine, as bench_key.py computes it: the GPU
# name from nvidia-smi, tokenised on non-alphanumeric characters, the "NVIDIA"/"GeForce"
# vendor words dropped, the rest joined with '-'.
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
