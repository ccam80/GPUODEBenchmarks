#!/bin/bash
# Forwards to `bench.py -a numerical`; -p all|julia|cubie, --controller, --algorithm and -s map onto its flags.
cd "$(dirname "$0")" || exit 1
ARGS=(-a numerical --no-lock-clocks)
PKG=cubie,julia
while [ $# -gt 0 ]; do
    case "$1" in
        -p|--package)
            case "$2" in
                all) PKG=cubie,julia;;
                julia|cubie) PKG=$2;;
                *) echo "Unknown package '$2' (all|julia|cubie)" >&2; exit 1;;
            esac
            shift 2;;
        --controller) ARGS+=(--controller "$2"); shift 2;;
        --algorithm) ARGS+=(-g "$2"); shift 2;;
        -s|--problem) ARGS+=(-s "$2"); shift 2;;
        -h|--help) sed -n '2,2p' "$0" | sed 's/^# \?//'; exit 0;;
        *) echo "Unknown option $1" >&2; exit 1;;
    esac
done
exec python3 ./bench.py "${ARGS[@]}" -p "$PKG"
