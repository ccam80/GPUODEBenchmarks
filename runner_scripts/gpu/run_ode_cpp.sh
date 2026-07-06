#!/bin/bash
set -e

# Work-precision mode: `run_ode_cpp.sh wp` builds RK4 and RKCK45 once at
# NT=32768 and runs the dt/tolerance sweeps ("Lorenz.exe 32768 wp") instead
# of the N sweep.
if [ "$1" == "wp" ]; then
	sed -i "17d" ./GPU_ODE_MPGOS/Lorenz.cu
	sed -i "17 i const int NT = 32768;" ./GPU_ODE_MPGOS/Lorenz.cu

	sed -i "15d" ./GPU_ODE_MPGOS/Lorenz.cu
	sed -i "15 i #define SOLVER RK4" ./GPU_ODE_MPGOS/Lorenz.cu
	make clean --directory=./GPU_ODE_MPGOS/
	make --directory=./GPU_ODE_MPGOS/
	./GPU_ODE_MPGOS/Lorenz.exe 32768 wp

	sed -i "15d" ./GPU_ODE_MPGOS/Lorenz.cu
	sed -i "15 i #define SOLVER RKCK45" ./GPU_ODE_MPGOS/Lorenz.cu
	make clean --directory=./GPU_ODE_MPGOS/
	make --directory=./GPU_ODE_MPGOS/
	./GPU_ODE_MPGOS/Lorenz.exe 32768 wp
	exit 0
fi

a=8
# max_a=$((2**24))
max_a=$1
while [ $a -le $max_a ]
do
    echo $a
	sed -i "15d" ./GPU_ODE_MPGOS/Lorenz.cu
	sed -i "15 i #define SOLVER RK4" ./GPU_ODE_MPGOS/Lorenz.cu
	sed -i "17d" ./GPU_ODE_MPGOS/Lorenz.cu
	sed -i "17 i const int NT = $a;" ./GPU_ODE_MPGOS/Lorenz.cu

	make clean --directory=./GPU_ODE_MPGOS/
	make --directory=./GPU_ODE_MPGOS/
	./GPU_ODE_MPGOS/Lorenz.exe $a

	sed -i "15d" ./GPU_ODE_MPGOS/Lorenz.cu
	sed -i "15 i #define SOLVER RKCK45" ./GPU_ODE_MPGOS/Lorenz.cu

	make clean --directory=./GPU_ODE_MPGOS/
	make --directory=./GPU_ODE_MPGOS/
	./GPU_ODE_MPGOS/Lorenz.exe $a
	# increment the value
	a=$((a*4))
done
