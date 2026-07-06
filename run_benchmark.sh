#!/bin/bash
# Run from the repo root regardless of the caller's working directory
cd "$(dirname "$0")" || exit 1
has_n_option=false
while getopts l:d:m:n: flag
do
    case "${flag}" in
        l) lang=${OPTARG};;
        d) dev=${OPTARG};;
        m) model=${OPTARG};;
        n) nmax=${OPTARG};has_n_option=true;;
        \?) echo "Unknown option -$OPTARG"; exit 1;;
    esac
done
if ! $has_n_option; then
    nmax=$((2**24))
fi
# Accept hyphenated aliases (e.g. cubie-mlir) by normalizing to underscores
lang=${lang//-/_}

# Per-machine dataset key ("<os>_<gpu>"). Timing files are appended across the
# N-sweep, so we clear only *this machine's* files before a run; other machines'
# keyed files are left in place so data accumulates additively across machines.
DATASET_KEY="$(bash ./runner_scripts/bench_key.sh)"

if [ -z "$lang" ] || [ -z "$dev" ] || [ -z "$model" ]; then
    echo "Usage: $0 -l <language> -d <device> -m <model> [-n nmax]"
    exit 1
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
        rm -f "./data/${lang^}"/*_"${DATASET_KEY}".txt
        bash "./runner_scripts/${dev}/run_${model}_${lang}.sh" "${nmax}"
    fi
elif [[ $lang == "jax" || $lang == "pytorch" || $lang == "cpp" || $lang == "cubie" || $lang == "cubie_mlir" ]]; then
    if [[ $model != "ode" || $dev != "gpu" ]]; then
        echo "The benchmarking of ensemble ${model^^} solvers on ${dev^^} with ${lang} is not supported. Please use -m flag with \"ode\" and -d with \"gpu\"."
        exit 1
    else
        echo "Benchmarking ${lang^^} ${dev^^} accelerated ensemble ${model^^} solvers..."
        mkdir -p "./data/${lang^^}"
        rm -f "./data/${lang^^}"/*_"${DATASET_KEY}".txt
        bash "./runner_scripts/${dev}/run_${model}_${lang}.sh" "${nmax}"
    fi
else
    echo "Unknown language: ${lang}. Supported: julia, cpp, jax, pytorch, cubie, cubie_mlir."
    exit 1
fi
