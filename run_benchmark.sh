#!/bin/bash
# GPU ODE runs forward to bench.py with the same flags; -d cpu and -m sde reach the legacy runners under runner_scripts/<device>/.
cd "$(dirname "$0")" || exit 1
DEVICE=gpu
MODEL=ode
PACKAGE=
PASS=()
while [ $# -gt 0 ]; do
    case "$1" in
        -d|--device) DEVICE=$2; shift 2;;
        -m|--model) MODEL=$2; shift 2;;
        -p|--package) PACKAGE=${2//-/_}; PASS+=(-p "$PACKAGE"); shift 2;;
        *) PASS+=("$1"); shift;;
    esac
done
if [ "$DEVICE" == "gpu" ] && [ "$MODEL" == "ode" ]; then
    exec python3 ./bench.py --no-lock-clocks "${PASS[@]}"
fi
[ -n "$PACKAGE" ] || { echo "-p/--package is required" >&2; exit 1; }
RUNNER="./runner_scripts/${DEVICE}/run_${MODEL}_${PACKAGE}.sh"
[ -f "$RUNNER" ] || { echo "Ensemble ${MODEL} on ${DEVICE} with ${PACKAGE} is not supported." >&2; exit 1; }
LEGACY=()
for ((i = 0; i < ${#PASS[@]}; i++)); do
    if [ "${PASS[$i]}" == "-p" ]; then i=$((i + 1)); continue; fi
    LEGACY+=("${PASS[$i]}")
done
exec bash "$RUNNER" "${LEGACY[@]}"
