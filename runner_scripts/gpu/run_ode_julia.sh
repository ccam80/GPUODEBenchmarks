#!/bin/bash
set -e
unset LD_LIBRARY_PATH
# Work-precision mode: `run_ode_julia.sh wp` sweeps dt/tolerance at N=32768.
if [ "$1" == "wp" ]; then
    julia --project=. ./GPU_ODE_Julia/bench_lorenz_gpu.jl 32768 wp
    exit 0
fi
a=8
max_a=$1
while [ $a -le $max_a ]
do
	# Print the values
	echo $a
	julia --project=. ./GPU_ODE_Julia/bench_lorenz_gpu.jl $a
	# increment the value
	a=$((a*4))
done
