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

# Problem, solver and trajectory count are compile-time constants, so each
# point is a rebuild.
build() {
	make clean --directory=./GPU_ODE_MPGOS/
	make --directory=./GPU_ODE_MPGOS/ PROBLEM="$1" SOLVER="$2" NT="$3"
}

PROBLEMS=$(python3 ./runner_scripts/mpgos_problems.py "$PROBLEM")
if [ -z "$PROBLEMS" ]; then
	echo "MPGOS runs none of the requested problems; skipping."
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
	# One solver walks the whole N sweep before the next builds; NT is a
	# compile-time constant, so each point is still a rebuild.
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
