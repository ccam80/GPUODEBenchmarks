#!/bin/bash
set -e
unset LD_LIBRARY_PATH

# Algorithm request (issue #29): forwarded to the bench script, which runs
# every supported algorithm for "all" and skips cleanly when unsupported.
ALG=${2:-all}
# Work-precision mode: `run_ode_julia.sh wp` sweeps dt/tolerance at N=32768.
if [ "$1" == "wp" ]; then
    julia --project=. ./GPU_ODE_Julia/bench_lorenz_gpu.jl 32768 wp "$ALG"
    exit 0
fi
a=8
max_a=$1
while [ $a -le $max_a ]
do
	# Print the values
	echo $a
	julia --project=. ./GPU_ODE_Julia/bench_lorenz_gpu.jl $a "$ALG"
	# increment the value
	a=$((a*4))
done
