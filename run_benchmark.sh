#!/bin/bash
# Forwards to bench.py with the same flags; -p accepts hyphenated package names.
cd "$(dirname "$0")" || exit 1
PASS=()
while [ $# -gt 0 ]; do
    case "$1" in
        -p|--package) PASS+=(-p "${2//-/_}"); shift 2;;
        *) PASS+=("$1"); shift;;
    esac
done
exec python3 ./bench.py --no-lock-clocks "${PASS[@]}"
