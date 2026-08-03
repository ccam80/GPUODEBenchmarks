#!/bin/bash
set -e

# Load modules eagerly so the first-launch cubin load stays out of timed regions.
export CUDA_MODULE_LOADING=EAGER

# Usage: run_ode_cpp.sh <max-trajectories>|wp [algorithm|all]
#
# MPGOS ships exactly two explicit solvers (issue #29): RK4 (classical-rk4,
# fixed dt) and RKCK45 (cash-karp-54, adaptive). Any other requested
# algorithm skips cleanly so orchestrated sweeps keep going.
ALG=${2:-all}
RUN_RK4=false
RUN_RKCK45=false
case "$ALG" in
    all) RUN_RK4=true; RUN_RKCK45=true;;
    classical-rk4) RUN_RK4=true;;
    cash-karp-54) RUN_RKCK45=true;;
    *) echo "MPGOS does not support algorithm '$ALG'; skipping."; exit 0;;
esac

# Lorenz.cu's config block is rewritten by absolute line number (see the
# warning at GPU_ODE_MPGOS/Lorenz.cu:32-34).
set_solver() {  # $1 = RK4|RKCK45
	sed -i "15d" ./GPU_ODE_MPGOS/Lorenz.cu
	sed -i "15 i #define SOLVER $1" ./GPU_ODE_MPGOS/Lorenz.cu
}
set_nt() {  # $1 = NT
	sed -i "17d" ./GPU_ODE_MPGOS/Lorenz.cu
	sed -i "17 i const int NT = $1;" ./GPU_ODE_MPGOS/Lorenz.cu
}
build() {
	make clean --directory=./GPU_ODE_MPGOS/
	make --directory=./GPU_ODE_MPGOS/
}

# Work-precision mode: `run_ode_cpp.sh wp` builds the requested solvers once
# at NT=32768 and runs the dt/tolerance sweeps ("Lorenz.exe 32768 wp")
# instead of the N sweep.
if [ "$1" == "wp" ]; then
	set_nt 32768
	if $RUN_RK4; then
		set_solver RK4
		build
		./GPU_ODE_MPGOS/Lorenz.exe 32768 wp
	fi
	if $RUN_RKCK45; then
		set_solver RKCK45
		build
		./GPU_ODE_MPGOS/Lorenz.exe 32768 wp
	fi
	exit 0
fi

a=8
# max_a=$((2**24))
max_a=$1
while [ $a -le $max_a ]
do
	echo $a
	set_nt $a
	if $RUN_RK4; then
		set_solver RK4
		build
		./GPU_ODE_MPGOS/Lorenz.exe $a
	fi
	if $RUN_RKCK45; then
		set_solver RKCK45
		build
		./GPU_ODE_MPGOS/Lorenz.exe $a
	fi
	# increment the value
	a=$((a*4))
done
