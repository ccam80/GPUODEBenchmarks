#!/bin/bash
# Generate benchmark data for one package and one analysis.
#
# Usage: ./run_benchmark.sh -p <package> [-a <analysis>] [-n <nmax>] [-d <device>] [-m <model>]
#   -p, --package   julia | cpp | pytorch | jax | cubie | cubie_mlir | myokit_cuda
#   -a, --analysis  performance (default) | work-precision
#   -n, --nmax      largest trajectory count for a performance sweep (default 16777216)
#   -d, --device    gpu (default) | cpu
#   -m, --model     ode (default) | sde

# Run from the repo root regardless of the caller's working directory
cd "$(dirname "$0")" || exit 1

PACKAGE=
ANALYSIS=performance
NMAX=16777216
DEVICE=gpu
MODEL=ode

usage() {
    sed -n '2,9p' "$0" | sed 's/^# \?//'
    exit "${1:-0}"
}

while [ $# -gt 0 ]; do
    case "$1" in
        -p|--package)  PACKAGE=$2; shift 2;;
        -a|--analysis) ANALYSIS=$2; shift 2;;
        -n|--nmax)     NMAX=$2; shift 2;;
        -d|--device)   DEVICE=$2; shift 2;;
        -m|--model)    MODEL=$2; shift 2;;
        -h|--help)     usage 0;;
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

echo "Benchmarking ${PACKAGE} ${DEVICE} ensemble ${MODEL} solvers (${ANALYSIS})..."

# Timing files are appended across the N sweep, so clear this machine's files
# for the analysis being run; other machines' keyed files stay.
if [ "$DEVICE" == "gpu" ] && [ "$MODEL" == "ode" ]; then
    DATASET_KEY="$(bash ./runner_scripts/bench_key.sh)"
    mkdir -p "./data/${DATA_DIR}"
    if [ "$ANALYSIS" == "work-precision" ]; then
        rm -f "./data/${DATA_DIR}"/*_wp_*_"${DATASET_KEY}".txt
    else
        rm -f "./data/${DATA_DIR}"/*_times_*_"${DATASET_KEY}".txt
    fi
fi

bash "$RUNNER" -a "$ANALYSIS" -n "$NMAX"
