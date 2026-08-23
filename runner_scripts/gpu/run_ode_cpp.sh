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

# BENCH_RESUME / BENCH_NO_OVERWRITE / BENCH_RESUME_FROM: skip covered points via runner_scripts/resume.py.
RESUME_ACTIVE=""
[ -n "${BENCH_RESUME:-}${BENCH_NO_OVERWRITE:-}${BENCH_RESUME_FROM:-}" ] && RESUME_ACTIVE=1

mode_for() { if [ "$1" == "RK4" ]; then echo fixed; else echo adaptive; fi; }
alg_for() { if [ "$1" == "RK4" ]; then echo classical-rk4; else echo cash-karp-54; fi; }

# resume_skip <times|states|wp> <problem> <solver> [N]: true when covered.
resume_skip() {
	[ -n "$RESUME_ACTIVE" ] || return 1
	local kind=$1 problem=$2 solver=$3 n=${4:-}
	local mode alg outfile
	mode=$(mode_for "$solver")
	alg=$(alg_for "$solver")
	outfile="./data/CPP/${DATASET_KEY}/${problem}/MPGOS_${kind}_${mode}_${alg}.txt"
	if [ "$kind" == "wp" ]; then
		[ "$(python3 ./runner_scripts/resume.py leg "$problem" "$alg" "$mode" "$outfile")" == "skip" ]
	else
		[ "$(python3 ./runner_scripts/resume.py point "$problem" "$alg" "$mode" "$n" "$outfile")" == "skip" ]
	fi
}

# resume_prune <times|states> <problem> <solver> <N>: drop a retried point's stale rows.
resume_prune() {
	[ -n "$RESUME_ACTIVE" ] || return 0
	local kind=$1 problem=$2 solver=$3 n=$4
	local mode alg outfile
	mode=$(mode_for "$solver")
	alg=$(alg_for "$solver")
	outfile="./data/CPP/${DATASET_KEY}/${problem}/MPGOS_${kind}_${mode}_${alg}.txt"
	python3 ./runner_scripts/resume.py prune "$n" "$outfile"
}

# Built binaries are cached per source hash, machine and build constants.
SRC_HASH=$( (cat GPU_ODE_MPGOS/Bench.cu GPU_ODE_MPGOS/makefile; \
             find GPU_ODE_MPGOS/problems GPU_ODE_MPGOS/SourceCodes -type f | sort | xargs cat) \
            | sha256sum | cut -c1-12)
CACHE_DIR="GPU_ODE_MPGOS/build_cache/${DATASET_KEY}"

# build_fresh <problem> <solver> <NT> [SD]: always run nvcc, no cache.
build_fresh() {
	make clean --directory=./GPU_ODE_MPGOS/
	make --directory=./GPU_ODE_MPGOS/ PROBLEM="$1" SOLVER="$2" NT="$3" ${4:+SD="$4"}
}

# build <problem> <solver> <NT> [SD]: reuse the cached binary or run nvcc.
build() {
	local exe="${CACHE_DIR}/Bench_$1_$2_NT$3${4:+_SD$4}_${SRC_HASH}.exe"
	if [ -f "$exe" ]; then
		cp "$exe" GPU_ODE_MPGOS/Bench.exe
		echo "Cached build: $(basename "$exe")"
		return
	fi
	build_fresh "$@"
	mkdir -p "$CACHE_DIR"
	cp GPU_ODE_MPGOS/Bench.exe "$exe"
}

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

# warm_nt_builds: every (problem, solver, NT) binary, in parallel.
warm_nt_builds() {
	local jobs=${BENCH_WARM_JOBS:-8}
	mkdir -p "$CACHE_DIR"
	local nts
	nts=$(echo "$NLIST 131072" | tr ' ' '\n' | sort -un)
	for problem in $PROBLEMS; do
		for solver in $SOLVERS; do
			for a in $nts; do
				while [ "$(jobs -rp | wc -l)" -ge "$jobs" ]; do wait -n; done
				warm_build "$problem" "$solver" "$a" &
			done
		done
	done
	wait
}

if [ "$ANALYSIS" == "states" ]; then
	STATES_N=131072
	GRID=$(python3 ./runner_scripts/problems.py --states-grid)
	# A resumed run appends to what earlier runs recorded.
	if [ -z "$RESUME_ACTIVE" ]; then
		rm -f "./data/CPP/${DATASET_KEY}/lorenz96/MPGOS_states_"*.txt
	fi
	for solver in $SOLVERS
	do
		for n in $GRID
		do
			if resume_skip states lorenz96 "$solver" "$n"; then
				echo "-- resume: skipping lorenz96 states=$n ($solver) (already covered)"
				continue
			fi
			resume_prune states lorenz96 "$solver" "$n"
			echo "lorenz96 states = $n ($solver, N=$STATES_N)"
			T0=$(date +%s.%N)
			build_fresh lorenz96 "$solver" "$STATES_N" "$n"
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
	warm_nt_builds
	echo "MPGOS warm build cache populated."
	exit 0
fi

# All binaries compile in parallel before anything is timed.
[ "$ANALYSIS" == "performance" ] && warm_nt_builds

for problem in $PROBLEMS
do
	if [ "$ANALYSIS" == "work-precision" ]; then
		for solver in $SOLVERS
		do
			if resume_skip wp "$problem" "$solver"; then
				echo "-- resume: skipping wp $problem ($solver) (already covered)"
				continue
			fi
			build "$problem" "$solver" 131072
			./GPU_ODE_MPGOS/Bench.exe wp
		done
		continue
	fi
	for solver in $SOLVERS
	do
		for a in $NLIST
		do
			if resume_skip times "$problem" "$solver" "$a"; then
				echo "-- resume: skipping N=$a ($problem, $solver) (already covered)"
				continue
			fi
			resume_prune times "$problem" "$solver" "$a"
			echo "No. of trajectories = $a ($problem, $solver)"
			build "$problem" "$solver" "$a"
			./GPU_ODE_MPGOS/Bench.exe
		done
	done
done
