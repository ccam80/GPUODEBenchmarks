#!/bin/bash
# Activate venv
source ./GPU_ODE_CUBIE/venv_cubie/bin/activate

a=8
max_a=$1
while [ $a -le $max_a ]
do
    echo "No. of trajectories = $a"
    python3 ./GPU_ODE_CUBIE/bench_cubie.py $a
    a=$((a*4))
done

# Deactivate venv
deactivate
