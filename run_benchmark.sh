#!/bin/bash
# Generate benchmark data for one package and one analysis.
#
# Usage: ./run_benchmark.sh -p <package> [-a <analysis>] [-n <nmax>] [-g <algorithm>] [-d <device>] [-m <model>]
#   -p, --package   julia | cpp | pytorch | jax | cubie | cubie_mlir | myokit_cuda
#   -a, --analysis  performance (default) | work-precision
#   -n, --nmax      sweep ceiling (runs 8, 32, ... <= n; default 16777216),
#                   or a comma list of exact trajectory counts (e.g. 8388608,134217728)
#   -g, --algorithm all (default) | comma list of euler | classical-rk4 | tsit5 | cash-karp-54
#   -d, --device    gpu (default) | cpu
#   -m, --model     ode (default) | sde

# Run from the repo root regardless of the caller's working directory
cd "$(dirname "$0")" || exit 1

PACKAGE=
ANALYSIS=performance
NMAX=16777216
ALGORITHM=all
DEVICE=gpu
MODEL=ode

usage() {
    sed -n '2,11p' "$0" | sed 's/^# \?//'
    exit "${1:-0}"
}

while [ $# -gt 0 ]; do
    case "$1" in
        -p|--package)   PACKAGE=$2; shift 2;;
        -a|--analysis)  ANALYSIS=$2; shift 2;;
        -n|--nmax)      NMAX=$2; shift 2;;
        -g|--algorithm) ALGORITHM=$2; shift 2;;
        -d|--device)    DEVICE=$2; shift 2;;
        -m|--model)     MODEL=$2; shift 2;;
        -h|--help)      usage 0;;
        *) echo "Unknown option $1" >&2; usage 1;;
    esac
done

# Accept hyphenated aliases (e.g. cubie-mlir) by normalizing to underscores
PACKAGE=${PACKAGE//-/_}

[ -n "$PACKAGE" ] || { echo "-p/--package is required" >&2; usage 1; }
case "$ANALYSIS" in
    performance|work-precision) ;;
    *) echo "Unknown analysis '$ANALYSIS' (performance|work-precision)" >&2; exit 1;;
esac
# -g accepts "all" or a comma list of algorithms. The charset check keeps the
# unquoted token split below free of glob and shell metacharacters.
case "$ALGORITHM" in
    ''|*[!a-z0-9,-]*)
        echo "Unknown algorithm '$ALGORITHM' (all|euler|classical-rk4|tsit5|cash-karp-54)" >&2
        exit 1;;
esac
ALG_LIST=
ALG_HAS_ALL=false
for alg in ${ALGORITHM//,/ }; do
    case "$alg" in
        all) ALG_HAS_ALL=true;;
        euler|classical-rk4|tsit5|cash-karp-54) ALG_LIST="$ALG_LIST $alg";;
        *) echo "Unknown algorithm '$alg' (all|euler|classical-rk4|tsit5|cash-karp-54)" >&2; exit 1;;
    esac
done
$ALG_HAS_ALL && ALG_LIST=all
[ -n "$ALG_LIST" ] || { echo "-g/--algorithm requires a value" >&2; exit 1; }
case ",$NMAX," in
    *[!0-9,]*|*,,*)
        echo "-n/--nmax must be a positive integer or a comma list of them, got '$NMAX'" >&2
        exit 1;;
esac

case "$PACKAGE" in
    julia)       DATA_DIR=Julia;;
    cpp)         DATA_DIR=CPP;;
    jax)         DATA_DIR=JAX;;
    pytorch)     DATA_DIR=PYTORCH;;
    cubie)       DATA_DIR=CUBIE;;
    cubie_mlir)  DATA_DIR=CUBIE_MLIR;;
    myokit_cuda) DATA_DIR=MYOKIT_CUDA;;
    *) echo "Unknown package: ${PACKAGE}. Supported: julia, cpp, jax, pytorch, cubie, cubie_mlir, myokit_cuda." >&2
       exit 1;;
esac

RUNNER="./runner_scripts/${DEVICE}/run_${MODEL}_${PACKAGE}.sh"
if [ ! -f "$RUNNER" ]; then
    echo "Ensemble ${MODEL} on ${DEVICE} with ${PACKAGE} is not supported." >&2
    exit 1
fi

if [ "$DEVICE" == "gpu" ] && [ "$MODEL" == "ode" ]; then
    DATASET_KEY="$(bash ./runner_scripts/bench_key.sh)"
    mkdir -p "./data/${DATA_DIR}/${DATASET_KEY}"
fi

# One runner invocation per requested algorithm; a failure does not stop the rest.
STATUS=0
for ALG in $ALG_LIST; do
    echo "Benchmarking ${PACKAGE} ${DEVICE} ensemble ${MODEL} solvers (${ANALYSIS}, ${ALG})..."

    # Clear this machine's appended files for the analysis and algorithm being run.
    if [ "$DEVICE" == "gpu" ] && [ "$MODEL" == "ode" ]; then
        if [ "$ALG" == "all" ]; then
            ALG_GLOB="*"
        else
            ALG_GLOB="*_${ALG}"
        fi
        if [ "$ANALYSIS" == "work-precision" ]; then
            rm -f "./data/${DATA_DIR}/${DATASET_KEY}"/*_wp_${ALG_GLOB}.txt
        else
            rm -f "./data/${DATA_DIR}/${DATASET_KEY}"/*_times_${ALG_GLOB}.txt
        fi
    fi

    bash "$RUNNER" -a "$ANALYSIS" -n "$NMAX" -g "$ALG" || STATUS=1
done
exit $STATUS
