#!/bin/bash
# Linux MPGOS runner: run_ode_cpp.sh --trials <jsonl> [--floor]. Builds every binary the trial file needs with nvcc, then runs its solve trials in file order through GPU_ODE_MPGOS/Bench.exe, which records each one through the store. Exit 0 when the loop finished, the watchdog exit code when a trial never returned, 1 otherwise.
set -e

TRIALS=""
FLOOR=""
while [ $# -gt 0 ]; do
	case "$1" in
		--trials) TRIALS="$2"; shift 2;;
		--floor) FLOOR=1; shift;;
		*) echo "run_ode_cpp.sh: unknown argument '$1'"; exit 1;;
	esac
done
if [ -z "$TRIALS" ]; then echo "run_ode_cpp.sh --trials <jsonl> [--floor]"; exit 1; fi
TRIALS="$(cd "$(dirname "$TRIALS")" && pwd)/$(basename "$TRIALS")"

# Load modules eagerly so the first-launch cubin load stays out of timed regions.
export CUDA_MODULE_LOADING=EAGER

cd "$(dirname "$0")/../.."

# The suite interpreter carries pyarrow and duckdb for the store.
PYTHON=python3
if [ -x GPU_ODE_CUBIE/venv/bin/python3 ]; then PYTHON="$PWD/GPU_ODE_CUBIE/venv/bin/python3"; fi

CONTEXT="$("$PYTHON" runner_scripts/mpgos_trials.py context)"
context_value() { printf '%s\n' "$CONTEXT" | awk -F= -v k="$1" '$1 == k { sub(/^[^=]*=/, ""); print; exit }'; }
DATASET_KEY=$(context_value key)
SRC_HASH=$(context_value source_hash)
PACKAGE_VERSION=$(context_value package_version)
SUITE_REV=$(context_value suite_rev)
WATCHDOG_EXIT=$(context_value watchdog_exit)
CACHE_DIR="GPU_ODE_MPGOS/build_cache/${DATASET_KEY}"

# Binaries are cached per source hash, machine and build constants.
exe_path() {
	local problem=$1 solver=$2 nt=$3 sd=$4 precision=$5 sdtag=""
	[ "$sd" != "-" ] && sdtag="_SD$sd"
	echo "${CACHE_DIR}/Bench_${problem}_${solver}_NT${nt}${sdtag}_${precision}_${SRC_HASH}.exe"
}

# nvcc_build <exe> <problem> <solver> <nt> <sd> <precision>
nvcc_build() {
	local exe=$1 problem=$2 solver=$3 nt=$4 sd=$5 precision=$6 type=float
	[ "$precision" = "float64" ] && type=double
	nvcc -o "$exe" GPU_ODE_MPGOS/Bench.cu \
		-IGPU_ODE_MPGOS/SourceCodes -IGPU_ODE_MPGOS \
		-DPROBLEM_HEADER="\"problems/$problem.cuh\"" -DSOLVER_CHOICE="$solver" \
		-DNT_VALUE="$nt" -DPRECISION_TYPE="$type" $( [ "$sd" != "-" ] && echo "-DPROBLEM_SD=$sd" ) \
		-O3 -std=c++17 --ptxas-options=-v --gpu-architecture=native \
		-lineinfo -maxrregcount=128
}

# warm_build <problem> <solver> <nt> <sd> <precision>: nvcc straight into the cache, quietly.
warm_build() {
	local exe
	exe=$(exe_path "$@")
	[ -f "$exe" ] && return 0
	echo "building $(basename "$exe")"
	nvcc_build "$exe" "$@" > /dev/null 2>&1 || { rm -f "$exe"; echo "FAILED $(basename "$exe")"; }
}

# cold_build <problem> <solver> <nt> <sd> <precision>: a fresh serial build; prints its wall seconds.
cold_build() {
	local exe t0
	exe=$(exe_path "$@")
	rm -f "$exe"
	echo "cold build $(basename "$exe")" >&2
	t0=$(date +%s.%N)
	nvcc_build "$exe" "$@" >&2 || { rm -f "$exe"; echo "FAILED $(basename "$exe")" >&2; }
	echo "$t0 $(date +%s.%N)" | awk '{printf "%.3f", $2 - $1}'
}

mkdir -p "$CACHE_DIR"
BUILDS="$("$PYTHON" runner_scripts/mpgos_trials.py builds "$TRIALS")"
POINTS="$("$PYTHON" runner_scripts/mpgos_trials.py points "$TRIALS")"

# Every missing warm target builds in parallel; cold targets build serially and time the leg's build_s.
JOBS=8
while IFS=$'\t' read -r problem solver nt sd precision cold leg; do
	[ -z "$problem" ] && continue
	[ "$cold" = "true" ] && continue
	while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n; done
	warm_build "$problem" "$solver" "$nt" "$sd" "$precision" &
done <<< "$BUILDS"
wait
declare -A BUILD_SECONDS
while IFS=$'\t' read -r problem solver nt sd precision cold leg; do
	[ "$cold" = "true" ] || continue
	BUILD_SECONDS["$leg"]=$(cold_build "$problem" "$solver" "$nt" "$sd" "$precision")
done <<< "$BUILDS"

# nan_rows <trial_id> <leg> <transfers> <reason>: record NaN rows for a point the script could not run.
nan_rows() {
	local trial_id=$1 leg=$2 transfers=$3 reason=$4
	local extra=()
	[ -n "$FLOOR" ] && extra+=(--floor)
	[ -n "${BUILD_SECONDS[$leg]:-}" ] && extra+=(--build-s "${BUILD_SECONDS[$leg]}")
	"$PYTHON" runner_scripts/mpgos_trials.py nan "$TRIALS" "$trial_id" "$DATASET_KEY" "$transfers" "$reason" "${extra[@]}"
	echo "cpp $leg $transfers: $reason"
}

# (leg|transfers) pairs a timeout or out-of-memory outcome abandoned; Bench.exe recorded their rows.
declare -A ABANDONED
OUTCOME="$TRIALS.outcome"

while IFS=$'\t' read -r trial_id leg ordinal problem solver nt sd precision transfers finals reason; do
	[ -z "$trial_id" ] && continue
	wanted=""
	for t in ${transfers//,/ }; do
		[ -n "${ABANDONED[$leg|$t]:-}" ] && continue
		wanted="${wanted:+$wanted,}$t"
	done
	[ -z "$wanted" ] && continue
	if [ -n "$reason" ]; then
		nan_rows "$trial_id" "$leg" "$wanted" "$reason"
		continue
	fi
	exe=$(exe_path "$problem" "$solver" "$nt" "$sd" "$precision")
	if [ ! -f "$exe" ]; then
		nan_rows "$trial_id" "$leg" "$wanted" "error: BuildError: nvcc failed for $(basename "$exe")"
		continue
	fi
	rm -f "$OUTCOME"
	bench_args=(--trials "$TRIALS" --trial "$trial_id" --key "$DATASET_KEY" --transfers "$wanted"
		--python "$PYTHON" --package-version "$PACKAGE_VERSION" --suite-rev "$SUITE_REV" --outcome "$OUTCOME")
	[ -n "$FLOOR" ] && bench_args+=(--floor)
	[ -n "${BUILD_SECONDS[$leg]:-}" ] && bench_args+=(--build-s "${BUILD_SECONDS[$leg]}")
	echo "cpp $leg ordinal $ordinal n=$nt ($wanted)"
	rc=0
	"$exe" "${bench_args[@]}" || rc=$?
	if [ "$rc" -eq "$WATCHDOG_EXIT" ]; then
		# The driver records the abandoned rows from the progress file and re-invokes this script.
		exit "$WATCHDOG_EXIT"
	fi
	done_legs=""
	if [ -f "$OUTCOME" ]; then
		while read -r which result; do
			[ -z "$which" ] && continue
			done_legs="$done_legs $which"
			case "$result" in timeout|oom) ABANDONED["$leg|$which"]=1;; esac
		done < "$OUTCOME"
	fi
	if [ "$rc" -ne 0 ]; then
		echo "FAILED $leg ordinal $ordinal: Bench.exe exit $rc"
		missing=""
		for t in ${wanted//,/ }; do
			case " $done_legs " in *" $t "*) ;; *) missing="${missing:+$missing,}$t";; esac
		done
		[ -n "$missing" ] && nan_rows "$trial_id" "$leg" "$missing" "error: ProcessError: Bench.exe exit $rc"
	fi
done <<< "$POINTS"

exit 0
