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

# Solver and trajectory count are compile-time constants, so each point is a rebuild.
set_solver() {
	sed -i "15d" ./GPU_ODE_MPGOS/Lorenz.cu
	sed -i "15 i #define SOLVER $1" ./GPU_ODE_MPGOS/Lorenz.cu
}
set_nt() {
	sed -i "17d" ./GPU_ODE_MPGOS/Lorenz.cu
	sed -i "17 i const int NT = $1;" ./GPU_ODE_MPGOS/Lorenz.cu
}
rebuild() {
	make clean --directory=./GPU_ODE_MPGOS/
	make --directory=./GPU_ODE_MPGOS/
}

if [ "$ANALYSIS" == "work-precision" ]; then
	set_nt 131072
	for solver in $SOLVERS
	do
		set_solver "$solver"
		rebuild
		./GPU_ODE_MPGOS/Lorenz.exe 131072 wp
	done
	exit 0
fi

for a in $NLIST
do
	echo "No. of trajectories = $a"
	set_nt "$a"
	for solver in $SOLVERS
	do
		set_solver "$solver"
		rebuild
		./GPU_ODE_MPGOS/Lorenz.exe $a
	done
done
