#!/bin/bash
set -e
a=8
max_a=$1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Algorithm filter; "all" runs every algorithm this framework supports.
ALG=${2:-all}
source ./GPU_ODE_JAX/venv/bin/activate
# Work-precision mode: `run_ode_jax.sh wp` sweeps dt/tolerance at N=32768.
if [ "$1" == "wp" ]; then
    python3 ./GPU_ODE_JAX/bench_diffrax.py 32768 wp "$ALG"
    deactivate
    exit 0
fi
while [ $a -le $max_a ]
do
    	# Print the values
    	echo "No. of trajectories = $a"
		python3 ./GPU_ODE_JAX/bench_diffrax.py $a "$ALG"	
    	# increment the value
    	a=$((a*4))
done
deactivate
