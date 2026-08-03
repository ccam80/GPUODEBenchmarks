#!/bin/bash
set -e
a=8
max_a=$1
source ./GPU_ODE_PyTorch/venv/bin/activate

# Algorithm request (issue #29): forwarded to the bench script, which runs
# every supported algorithm for "all" and skips cleanly when unsupported.
ALG=${2:-all}
# Work-precision mode: `run_ode_pytorch.sh wp` sweeps dt at N=32768 (fixed
# only: torchdiffeq adaptive solvers are incompatible with torch.vmap).
if [ "$1" == "wp" ]; then
    python3 ./GPU_ODE_PyTorch/bench_torchdiffeq.py 32768 wp "$ALG"
    deactivate
    exit 0
fi
while [ $a -le $max_a ]
do
    	# Print the values
    	echo "No. of trajectories = $a"
		python3 ./GPU_ODE_PyTorch/bench_torchdiffeq.py $a "$ALG"	
    	# increment the value
    	a=$((a*4))
done
deactivate
