#!/bin/bash
set -e
. "$(dirname "$0")/../parse_args.sh" "$@"

# Load modules eagerly so the first-launch cubin load stays out of timed regions.
export CUDA_MODULE_LOADING=EAGER

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
	set_nt 32768
	for solver in RK4 RKCK45
	do
		set_solver "$solver"
		rebuild
		./GPU_ODE_MPGOS/Lorenz.exe 32768 wp
	done
	exit 0
fi

a=8
while [ $a -le $NMAX ]
do
	echo "No. of trajectories = $a"
	set_nt "$a"
	for solver in RK4 RKCK45
	do
		set_solver "$solver"
		rebuild
		./GPU_ODE_MPGOS/Lorenz.exe $a
	done
	a=$((a*4))
done
