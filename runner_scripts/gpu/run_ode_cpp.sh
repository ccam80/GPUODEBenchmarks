#!/bin/bash
# run_ode_cpp.sh --trials <jsonl> [--floor]: builds the binaries the file needs, runs its trials through Bench.exe; exits the watchdog code when a trial never returned.
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

# The suite interpreter runs the store.
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

# Warm targets build in parallel; cold targets build serially and time build_s.
JOBS=8
while IFS=$'\t' read -r problem solver nt sd precision cold; do
	[ -z "$problem" ] && continue
	[ "$cold" = "true" ] && continue
	while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n; done
	warm_build "$problem" "$solver" "$nt" "$sd" "$precision" &
done <<< "$BUILDS"
wait
declare -A BUILD_SECONDS
while IFS=$'\t' read -r problem solver nt sd precision cold; do
	[ "$cold" = "true" ] || continue
	BUILD_SECONDS["$(exe_path "$problem" "$solver" "$nt" "$sd" "$precision")"]=$(cold_build "$problem" "$solver" "$nt" "$sd" "$precision")
done <<< "$BUILDS"

# build_seconds <exe> <cold>: the cold build time a point carries, empty for a warm point.
build_seconds() {
	[ "$2" = "true" ] || return 0
	echo "${BUILD_SECONDS[$1]:-}"
}

# nan_rows <trial_id> <label> <transfers> <reason> [<build_s>]: record NaN rows for a point the script could not run.
nan_rows() {
	local trial_id=$1 label=$2 transfers=$3 reason=$4 seconds=${5:-}
	local extra=()
	[ -n "$FLOOR" ] && extra+=(--floor)
	[ -n "$seconds" ] && extra+=(--build-s "$seconds")
	"$PYTHON" runner_scripts/mpgos_trials.py nan "$TRIALS" "$trial_id" "$DATASET_KEY" "$transfers" "$reason" "${extra[@]}"
	echo "cpp $label $transfers: $reason"
}

# (trial_id|transfers) pairs the abandon rule gives up after a timeout or oom, with the reason.
declare -A ABANDONED
OUTCOME="$TRIALS.outcome"

while IFS=$'\t' read -r trial_id problem solver nt sd precision transfers finals cold timed reason; do
	[ -z "$trial_id" ] && continue
	label="$problem $solver n=$nt sd=$sd"
	exe=$(exe_path "$problem" "$solver" "$nt" "$sd" "$precision")
	seconds=$(build_seconds "$exe" "$cold")
	wanted=""
	for t in ${transfers//,/ }; do
		if [ -n "${ABANDONED[$trial_id|$t]:-}" ]; then
			nan_rows "$trial_id" "$label" "$t" "${ABANDONED[$trial_id|$t]}" "$seconds"
			continue
		fi
		wanted="${wanted:+$wanted,}$t"
	done
	[ -z "$wanted" ] && continue
	if [ -n "$reason" ]; then
		nan_rows "$trial_id" "$label" "$wanted" "$reason" "$seconds"
		continue
	fi
	if [ ! -f "$exe" ]; then
		nan_rows "$trial_id" "$label" "$wanted" "error: BuildError: nvcc failed for $(basename "$exe")" "$seconds"
		continue
	fi
	rm -f "$OUTCOME"
	bench_args=(--trials "$TRIALS" --trial "$trial_id" --key "$DATASET_KEY" --transfers "$wanted"
		--python "$PYTHON" --package-version "$PACKAGE_VERSION" --suite-rev "$SUITE_REV" --outcome "$OUTCOME")
	[ -n "$FLOOR" ] && bench_args+=(--floor)
	[ "$timed" = "false" ] && bench_args+=(--untimed)
	[ -n "$seconds" ] && bench_args+=(--build-s "$seconds")
	echo "cpp $label ($wanted)"
	rc=0
	"$exe" "${bench_args[@]}" || rc=$?
	if [ "$rc" -eq "$WATCHDOG_EXIT" ]; then
		exit "$WATCHDOG_EXIT"
	fi
	done_transfers=""
	if [ -f "$OUTCOME" ]; then
		while read -r which result; do
			[ -z "$which" ] && continue
			done_transfers="$done_transfers $which"
			case "$result" in timeout|oom)
				while read -r id; do
					[ -n "$id" ] && ABANDONED["$id|$which"]="abandoned: $result at $trial_id"
				done <<< "$("$PYTHON" runner_scripts/mpgos_trials.py harder "$TRIALS" "$trial_id")";;
			esac
		done < "$OUTCOME"
	fi
	if [ "$rc" -ne 0 ]; then
		echo "FAILED $label: Bench.exe exit $rc"
		missing=""
		for t in ${wanted//,/ }; do
			case " $done_transfers " in *" $t "*) ;; *) missing="${missing:+$missing,}$t";; esac
		done
		[ -n "$missing" ] && nan_rows "$trial_id" "$label" "$missing" "error: ProcessError: Bench.exe exit $rc" "$seconds"
	fi
done <<< "$POINTS"

exit 0
