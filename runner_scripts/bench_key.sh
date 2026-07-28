#!/bin/bash
# Emit a dataset key "<os>_<gpu>" identifying this machine so benchmark output
# files can be additively populated across machines without clobbering.
#
# The GPU name comes from nvidia-smi (the single source of truth shared by every
# framework's key helper) and is sanitised identically everywhere: tokenise on
# non-alphanumeric characters, drop the "NVIDIA"/"GeForce" vendor words, and join
# the rest with '-'. e.g. "NVIDIA GeForce RTX 2060 SUPER" -> "RTX-2060-SUPER".

case "$(uname -s)" in
    Linux*)               os=linux ;;
    Darwin*)              os=macos ;;
    MINGW*|MSYS*|CYGWIN*) os=windows ;;
    *)                    os=unknown ;;
esac

# Check nvidia-smi's exit status rather than just capturing its output: when
# the driver is unusable it prints its diagnostic ("Failed to initialize NVML:
# ...") on stdout, which would otherwise be sanitised into a bogus GPU name and
# silently key the whole dataset to it. On failure fall through to
# "unknown-gpu", matching bench_key.py.
raw=""
if out="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"; then
    raw="$(printf '%s\n' "$out" | head -n1)"
fi
gpu="$(printf '%s' "$raw" | tr -c 'A-Za-z0-9' ' ' \
      | awk '{for(i=1;i<=NF;i++) if($i!="NVIDIA" && $i!="GeForce") printf "%s%s", (n++?"-":""), $i}')"
[ -z "$gpu" ] && gpu=unknown-gpu

printf '%s_%s\n' "$os" "$gpu"
