#!/bin/bash
set -e
. "$(dirname "$0")/../parse_args.sh" "$@"

# Load modules eagerly so the first-launch cubin load stays out of timed regions.
export CUDA_MODULE_LOADING=EAGER

# MPGOS solvers: RK4 (classical-rk4, fixed) and RKCK45 (cash-karp-54, adaptive).
SOLVERS=""
case "$ALGORITHM" in
    all) SOLVERS="RK4 RKCK45";;
    classical-rk4) SOLVERS="RK4";;
    cash-karp-54) SOLVERS="RKCK45";;
    *) echo "MPGOS does not support algorithm '$ALGORITHM'; skipping."; exit 0;;
esac

DATASET_KEY=$(bash ./runner_scripts/bench_key.sh)
# Built binaries are cached per source hash, machine and build constants.
SRC_HASH=$( (cat GPU_ODE_MPGOS/Bench.cu GPU_ODE_MPGOS/makefile; \
             find GPU_ODE_MPGOS/problems GPU_ODE_MPGOS/SourceCodes -type f | sort | xargs cat) \
            | sha256sum | cut -c1-12)
CACHE_DIR="GPU_ODE_MPGOS/build_cache/${DATASET_KEY}"

# build <problem> <solver> <NT> [SD]: reuse the cached binary or run nvcc.
build() {
	local exe="${CACHE_DIR}/Bench_$1_$2_NT$3${4:+_SD$4}_${SRC_HASH}.exe"
	if [ -f "$exe" ]; then
		cp "$exe" GPU_ODE_MPGOS/Bench.exe
		echo "Cached build: $(basename "$exe")"
		return
	fi
	make clean --directory=./GPU_ODE_MPGOS/
	make --directory=./GPU_ODE_MPGOS/ PROBLEM="$1" SOLVER="$2" NT="$3" ${4:+SD="$4"}
	mkdir -p "$CACHE_DIR"
	cp GPU_ODE_MPGOS/Bench.exe "$exe"
}

if [ "$ANALYSIS" == "states" ]; then
	# -n (when set) overrides the states-sweep ensemble size.
	STATES_N=131072
	[ "$NMAX" != "16777216" ] && STATES_N=$NMAX
	GRID=$(python3 ./runner_scripts/problems.py --states-grid)
	rm -f "./data/CPP/${DATASET_KEY}/lorenz96/MPGOS_states_"*.txt
	for solver in $SOLVERS
	do
		for n in $GRID
		do
			echo "lorenz96 states = $n ($solver, N=$STATES_N)"
			T0=$(date +%s.%N)
			build lorenz96 "$solver" "$STATES_N" "$n"
			BUILD_S=$(echo "$T0 $(date +%s.%N)" | awk '{printf "%.3f", $2 - $1}')
			./GPU_ODE_MPGOS/Bench.exe states "$BUILD_S"
		done
	done
	exit 0
fi

PROBLEMS=$(python3 ./runner_scripts/mpgos_problems.py "$PROBLEM")
if [ -z "$PROBLEMS" ]; then
	echo "MPGOS runs none of the requested problems; skipping."
	exit 0
fi

if [ "$ANALYSIS" == "warm" ]; then
	JOBS=${BENCH_WARM_JOBS:-8}
	mkdir -p "$CACHE_DIR"
	# warm_build <problem> <solver> <NT> [SD]: nvcc straight into the cache.
	warm_build() {
		local exe="${CACHE_DIR}/Bench_$1_$2_NT$3${4:+_SD$4}_${SRC_HASH}.exe"
		[ -f "$exe" ] && return 0
		echo "building $(basename "$exe")"
		nvcc -o "$exe" GPU_ODE_MPGOS/Bench.cu \
			-IGPU_ODE_MPGOS/SourceCodes -IGPU_ODE_MPGOS \
			-DPROBLEM_HEADER="\"problems/$1.cuh\"" -DSOLVER_CHOICE="$2" \
			-DNT_VALUE="$3" ${4:+-DPROBLEM_SD=$4} \
			-O3 -std=c++11 --ptxas-options=-v --gpu-architecture=native \
			-lineinfo -maxrregcount=128 > /dev/null 2>&1 \
			|| { rm -f "$exe"; echo "FAILED $(basename "$exe")"; }
	}
	NTS=$(echo "$NLIST 131072" | tr ' ' '\n' | sort -un)
	for problem in $PROBLEMS; do
		for solver in $SOLVERS; do
			for a in $NTS; do
				while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n; done
				warm_build "$problem" "$solver" "$a" &
			done
		done
	done
	case " $PROBLEMS " in *" lorenz96 "*)
		for solver in $SOLVERS; do
			for sd in $(python3 ./runner_scripts/problems.py --states-grid); do
				while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n; done
				warm_build lorenz96 "$solver" 131072 "$sd" &
			done
		done;;
	esac
	wait
	echo "MPGOS warm build cache populated."
	exit 0
fi

for problem in $PROBLEMS
do
	if [ "$ANALYSIS" == "work-precision" ]; then
		for solver in $SOLVERS
		do
			build "$problem" "$solver" 131072
			./GPU_ODE_MPGOS/Bench.exe wp
		done
		continue
	fi
	# NT is a compile-time constant, so every uncached point is a rebuild.
	for solver in $SOLVERS
	do
		for a in $NLIST
		do
			echo "No. of trajectories = $a ($problem, $solver)"
			build "$problem" "$solver" "$a"
			./GPU_ODE_MPGOS/Bench.exe
		done
	done
done
