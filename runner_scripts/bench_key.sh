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

raw="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1)"
gpu="$(printf '%s' "$raw" | tr -c 'A-Za-z0-9' ' ' \
      | awk '{for(i=1;i<=NF;i++) if($i!="NVIDIA" && $i!="GeForce") printf "%s%s", (n++?"-":""), $i}')"
[ -z "$gpu" ] && gpu=unknown-gpu

printf '%s_%s\n' "$os" "$gpu"
