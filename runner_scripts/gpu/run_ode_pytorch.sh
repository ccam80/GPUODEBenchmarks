#!/bin/bash
set -e
a=8
max_a=$1
source ./GPU_ODE_PyTorch/venv/bin/activate
# Work-precision mode: `run_ode_pytorch.sh wp` sweeps dt at N=32768 (fixed
# only: torchdiffeq adaptive solvers are incompatible with torch.vmap).
if [ "$1" == "wp" ]; then
    python3 ./GPU_ODE_PyTorch/bench_torchdiffeq.py 32768 wp
    deactivate
    exit 0
fi
while [ $a -le $max_a ]
do
    	# Print the values
    	echo "No. of trajectories = $a"
		python3 ./GPU_ODE_PyTorch/bench_torchdiffeq.py $a	
    	# increment the value
    	a=$((a*4))
done
deactivate
