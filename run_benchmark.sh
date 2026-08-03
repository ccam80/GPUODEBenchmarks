#!/bin/bash
# Run from the repo root regardless of the caller's working directory
cd "$(dirname "$0")" || exit 1
has_n_option=false
wp=false
alg=all
while getopts l:d:m:n:g:w flag
do
    case "${flag}" in
        l) lang=${OPTARG};;
        d) dev=${OPTARG};;
        m) model=${OPTARG};;
        n) nmax=${OPTARG};has_n_option=true;;
        g) alg=${OPTARG};;
        w) wp=true;;
        \?) echo "Unknown option -$OPTARG"; exit 1;;
    esac
done
if ! $has_n_option; then
    nmax=$((2**24))
fi
# Work-precision mode (-w): pass "wp" to the runner instead of nmax; the
# runner sweeps its supported step size and/or tolerance controls at N=32768
# against the golden reference.
if $wp; then
    nmax="wp"
fi
# Accept hyphenated aliases (e.g. cubie-mlir) by normalizing to underscores
lang=${lang//-/_}

# Per-machine dataset key ("<os>_<gpu>"). Timing files are appended across the
# N-sweep, so we clear only *this machine's* files before a run; other machines'
# keyed files are left in place so data accumulates additively across machines.
DATASET_KEY="$(bash ./runner_scripts/bench_key.sh)"

if [ -z "$lang" ] || [ -z "$dev" ] || [ -z "$model" ]; then
    echo "Usage: $0 -l <language> -d <device> -m <model> [-n nmax] [-g algorithm] [-w]"
    exit 1
fi

# -g runs one algorithm (default "all"); the pre-run wipe narrows to match.
if [ "$alg" == "all" ]; then
    times_glob="*_times_*_${DATASET_KEY}.txt"
    wp_glob="*_wp_*_${DATASET_KEY}.txt"
else
    times_glob="*_times_*_${alg}_${DATASET_KEY}.txt"
    wp_glob="*_wp_*_${alg}_${DATASET_KEY}.txt"
fi

echo "$lang"
if [ "$lang" == "julia" ]; then
    echo "Benchmarking ${lang^} ${dev^^} accelerated ensemble ${model^^} solvers..."
    if [ "$dev" == "cpu" ];then
        bash "./runner_scripts/${dev}/run_${model}_${lang}.sh" "${nmax}"
    elif [ "$model" == "sde" ];then
        bash "./runner_scripts/${dev}/run_${model}_${lang}.sh" "${nmax}"
    else
        mkdir -p "./data/${lang^}"
        # Clear only this machine's files for the mode being run: timing
        # files for an N sweep, wp files for a wp sweep.
        if $wp; then
            rm -f "./data/${lang^}"/$wp_glob
        else
            rm -f "./data/${lang^}"/$times_glob
        fi
        bash "./runner_scripts/${dev}/run_${model}_${lang}.sh" "${nmax}" "${alg}"
    fi
elif [[ $lang == "jax" || $lang == "pytorch" || $lang == "cpp" || $lang == "cubie" || $lang == "cubie_mlir" || $lang == "myokit_cuda" ]]; then
    if [[ $model != "ode" || $dev != "gpu" ]]; then
        echo "The benchmarking of ensemble ${model^^} solvers on ${dev^^} with ${lang} is not supported. Please use -m flag with \"ode\" and -d with \"gpu\"."
        exit 1
    else
        echo "Benchmarking ${lang^^} ${dev^^} accelerated ensemble ${model^^} solvers..."
        mkdir -p "./data/${lang^^}"
        if $wp; then
            rm -f "./data/${lang^^}"/$wp_glob
        else
            rm -f "./data/${lang^^}"/$times_glob
        fi
        bash "./runner_scripts/${dev}/run_${model}_${lang}.sh" "${nmax}" "${alg}"
    fi
else
    echo "Unknown language: ${lang}. Supported: julia, cpp, jax, pytorch, cubie, cubie_mlir, myokit_cuda."
    exit 1
fi
