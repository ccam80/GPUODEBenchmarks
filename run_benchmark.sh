#!/bin/bash
# Generate benchmark data for one package and one analysis.
#
# Usage: ./run_benchmark.sh -p <package> [-a <analysis>] [-n <nmax>] [-g <algorithm>] [-d <device>] [-m <model>]
#   -p, --package   julia | cpp | pytorch | jax | cubie | cubie_mlir | myokit_cuda
#   -a, --analysis  performance (default) | work-precision | states | warm
#   -n, --nmax      sweep ceiling (8, 32, ... <= n; default 16777216) or comma list of exact Ns
#   -g, --algorithm all (default) | comma list of the names in runner_scripts/algorithms.csv
#   -s, --problem   all (default) | comma list of names from runner_scripts/problems.csv
#   -d, --device    gpu (default) | cpu
#   -m, --model     ode (default) | sde

# Run from the repo root regardless of the caller's working directory
cd "$(dirname "$0")" || exit 1

PACKAGE=
ANALYSIS=performance
NMAX=16777216
ALGORITHM=all
PROBLEM=all
DEVICE=gpu
MODEL=ode

usage() {
    sed -n '2,10p' "$0" | sed 's/^# \?//'
    exit "${1:-0}"
}

while [ $# -gt 0 ]; do
    case "$1" in
        -p|--package)   PACKAGE=$2; shift 2;;
        -a|--analysis)  ANALYSIS=$2; shift 2;;
        -n|--nmax)      NMAX=$2; shift 2;;
        -g|--algorithm) ALGORITHM=$2; shift 2;;
        -s|--problem)   PROBLEM=$2; shift 2;;
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
    performance|work-precision|states|warm) ;;
    *) echo "Unknown analysis '$ANALYSIS' (performance|work-precision|states|warm)" >&2; exit 1;;
esac
# -g: "all" or a comma list; charset-check before the unquoted split.
case "$ALGORITHM" in
    ''|*[!a-z0-9,-]*)
        echo "-g/--algorithm must be 'all' or a comma list of algorithm names" >&2
        exit 1;;
esac
ALG_LIST=
ALG_HAS_ALL=false
# The bench scripts reject unknown algorithm names.
for alg in ${ALGORITHM//,/ }; do
    case "$alg" in
        all) ALG_HAS_ALL=true;;
        *) ALG_LIST="$ALG_LIST $alg";;
    esac
done
$ALG_HAS_ALL && ALG_LIST=all
[ -n "$ALG_LIST" ] || { echo "-g/--algorithm requires a value" >&2; exit 1; }
# Problem names are validated by the frameworks against problems.csv.
case "$PROBLEM" in
    ''|*[!a-z0-9_,-]*)
        echo "-s/--problem takes names from runner_scripts/problems.csv, got '$PROBLEM'" >&2
        exit 1;;
esac
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
    echo "Benchmarking ${PACKAGE} ${DEVICE} ensemble ${MODEL} solvers (${ANALYSIS}, ${ALG}, ${PROBLEM})..."

    # Clear this machine's appended files for the analysis, algorithm and problems being run.
    if [ "$DEVICE" == "gpu" ] && [ "$MODEL" == "ode" ]; then
        if [ "$ALG" == "all" ]; then
            ALG_GLOB="*"
        else
            ALG_GLOB="*_${ALG}"
        fi
        if [ "$PROBLEM" == "all" ]; then
            PROBLEM_DIRS=("./data/${DATA_DIR}/${DATASET_KEY}"/*/)
        else
            PROBLEM_DIRS=()
            for P in ${PROBLEM//,/ }; do
                PROBLEM_DIRS+=("./data/${DATA_DIR}/${DATASET_KEY}/${P}")
            done
        fi
        for PDIR in "${PROBLEM_DIRS[@]}"; do
            if [ "$ANALYSIS" == "work-precision" ]; then
                rm -f "${PDIR%/}"/*_wp_${ALG_GLOB}.txt
            elif [ "$ANALYSIS" == "states" ]; then
                rm -f "${PDIR%/}"/*_states_${ALG_GLOB}.txt
            elif [ "$ANALYSIS" == "warm" ]; then
                :
            else
                rm -f "${PDIR%/}"/*_times_${ALG_GLOB}.txt
            fi
        done
    fi

    bash "$RUNNER" -a "$ANALYSIS" -n "$NMAX" -g "$ALG" -s "$PROBLEM" || STATUS=1
done
exit $STATUS
