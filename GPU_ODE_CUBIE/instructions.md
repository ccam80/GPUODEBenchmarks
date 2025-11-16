# Cubie Benchmark Implementation Guide

This document provides complete instructions for implementing Cubie support in a personal fork of the GPUODEBenchmarks repository.

## Objective

Add Cubie as a benchmarked GPU ODE solver to GPUODEBenchmarks, enabling direct performance comparisons with Julia/DiffEqGPU, JAX/Diffrax, PyTorch/torchdiffeq, and C++/MPGOS implementations.

## Important Note: Conda vs Python venv

**The original GPUODEBenchmarks repository uses conda environments**, but the environment.yml files for JAX and PyTorch may fail with modern CUDA setups due to outdated dependency specifications. 

**Recommendation**: Use Python venv instead of conda for Cubie (and optionally for JAX/PyTorch/

## Repository Structure Overview

The GPUODEBenchmarks repository follows this pattern for each solver:

```
GPUODEBenchmarks/
├── GPU_ODE_<Language>/          # Solver-specific directory
│   ├── environment.yml          # Conda environment spec
│   └── bench_*.py              # Benchmark script
├── runner_scripts/
│   └── gpu/
│       └── run_ode_<language>.sh  # Runner script
├── run_benchmark.sh             # Main orchestration script
└── data/
    └── <LANGUAGE>/              # Output directory (created at runtime)
```

## Implementation Tasks

### Task 1: Populate GPU_ODE_Cubie Directory

**Location**: `GPU_ODE_Cubie/`

Create the following files in this directory:

#### File 1.1: Python venv Setup

**`GPU_ODE_Cubie/setup_venv.sh`**:
```bash
#!/bin/bash
# Setup Python venv for Cubie benchmarking

# Create venv
python -m venv venv_cubie

# Activate venv
source venv_cubie/bin/activate

# Upgrade pip
pip install --upgrade pip
pip install uv

# AGENT INSTRUCTION
# clone cubie source into a subfolder in venv from https://github.com/ccam80/cubie.git
# install cubie from source using uv pip install .[dev]

# Verify installation
python -c "import cubie; print('Cubie installed successfully')"
python -c "import numba.cuda; print('CUDA available:', numba.cuda.is_available())"

# Deactivate
deactivate

echo "Cubie environment setup complete. Activate with: source GPU_ODE_Cubie/venv_cubie/bin/activate"
```

Make executable and run:
```bash
chmod +x GPU_ODE_Cubie/setup_venv.sh
./GPU_ODE_Cubie/setup_venv.sh
```

#### File 1.2: `GPU_ODE_Cubie/bench_cubie.py`

This is the main benchmark script. It must:
1. Accept trajectory count as command-line argument
2. Define the Lorenz system matching the reference specification
3. Run both fixed and adaptive time-stepping benchmarks
4. Save results in the expected format

```python
#!/usr/bin/env python
# coding: utf-8
"""
Benchmarking Cubie ODE solvers for ensemble problems.
The Lorenz ODE is integrated with fixed and adaptive time-stepping.

Created for GPUODEBenchmarks integration
"""

import sys
import timeit
import numpy as np
import cubie as qb

# Get number of trajectories from command line
numberOfParameters = int(sys.argv[1])

# ========================================
# LORENZ SYSTEM DEFINITION
# ========================================
# Mathematical definition:
#   dx/dt = sigma * (y - x)
#   dy/dt = x * (rho - z) - y
#   dz/dt = x * y - beta * z
#
# Where:
#   sigma = 10.0 (fixed)
#   beta = 8/3 (fixed)  
#   rho = parameter varied from 0 to 21

precision = np.float32

lorenz_system = qb.create_ODE_system(
    """
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    """,
    states={'x': 1.0, 'y': 0.0, 'z': 0.0},
    parameters={'rho': 21.0},
    constants={'sigma': 10.0, 'beta': 8.0/3.0},
    name="Lorenz",
    precision=precision
)

# ========================================
# PARAMETER SWEEP SETUP
# ========================================
# Create linear space from 0 to 21 for rho parameter
parameterList = np.linspace(0.0, 21.0, numberOfParameters)

# Build parameter dictionary for batch solve
# All parameters except rho are scalar (same for all trajectories)
# rho varies across the ensemble
parameters = {
    'rho': parameterList
}

# Initial conditions (same for all trajectories)
initial_conditions = {
    'x': 1.0,
    'y': 0.0,
    'z': 0.0
}

fixed_solver = qb.solver(
    lorenz_system,
    algorithm='RK4',
    dt_save=1.0,
    dt=0.001,
    step_controller='fixed',
    output_types=['state'],
    time_logging_level='verbose',
)

adaptive_solver = qb.solver(
    lorenz_system,
    algorithm='tsit5',
    dt_save=1.0,
    atol=1e-08,
    rtol=1e-08,
    dt_save=1.0,
    dt_min=1e-9,
    dt_max=0.1,
    step_controller='fixed',
    output_types=['state'],
    time_logging_level='verbose',
)

# ========================================
# FIXED TIME-STEPPING BENCHMARK
# ========================================
print(f"Running {numberOfParameters} trajectories with fixed time-stepping...")

def solve_fixed(blocksize=256):
    """Solve with fixed time step (unadaptive)."""
    solution = fixed_solver.solve(
        initial_values=initial_conditions,
        parameters=parameters,
        blocksize=blocksize,
    )
    return solution

def solve_adaptive(blocksize):
    """Solve with adaptive time step."""
    solution = adaptive_solver.solve_ivp(
        initial_values=initial_conditions,
        parameters=parameters,
        blocksize=blocksize,
    )
    return solution
# Warm-up run (JIT compilation)
_ = solve_fixed()
_ = solve_adaptive

# Benchmark with 100 repetitions
res = timeit.repeat(lambda: solve_fixed(), repeat=100, number=1)

best_time = min(res) * 1000  # Convert to milliseconds
print(f"{numberOfParameters} ODE solves with fixed time-stepping completed in {best_time:.1f} ms")

# Save results
import os
os.makedirs("./data/CUBIE", exist_ok=True)
with open("./data/CUBIE/Cubie_times_unadaptive.txt", "a+") as file:
    file.write(f'{numberOfParameters} {best_time}\n')

# ========================================
# ADAPTIVE TIME-STEPPING BENCHMARK
# ========================================
print(f"Running {numberOfParameters} trajectories with adaptive time-stepping...")

# Warm-up run (JIT compilation)
_ = solve_adaptive()

# Benchmark with 100 repetitions
res = timeit.repeat(lambda: solve_adaptive(), repeat=100, number=1)

best_time = min(res) * 1000  # Convert to milliseconds
print(f"{numberOfParameters} ODE solves with adaptive time-stepping completed in {best_time:.1f} ms")

# Save results
with open("./data/CUBIE/Cubie_times_adaptive.txt", "a+") as file:
    file.write(f'{numberOfParameters} {best_time}\n')
```

### Task 2: Create Runner Script

**Location**: `runner_scripts/gpu/run_ode_cubie.sh`


```bash
#!/bin/bash
# Activate venv
source ./GPU_ODE_Cubie/venv_cubie/bin/activate

a=8
max_a=$1
while [ $a -le $max_a ]
do
    echo "No. of trajectories = $a"
    python3 ./GPU_ODE_Cubie/bench_cubie.py $a
    a=$((a*4))
done

# Deactivate venv
deactivate
```

**Make executable**:
```bash
chmod +x runner_scripts/gpu/run_ode_cubie.sh
```

### Task 3: Modify Main Orchestration Script

**Location**: `run_benchmark.sh`

**Modification**: Add cubie to the language options

Find this section (around line 38):
```bash
elif [[ $lang == "jax"  ||  $lang == "pytorch" || $lang == "cpp" ]]; then
```

Change to:
```bash
elif [[ $lang == "jax"  ||  $lang == "pytorch" || $lang == "cpp" || $lang == "cubie" ]]; then
```

**Complete modified section**:
```bash
elif [[ $lang == "jax"  ||  $lang == "pytorch" || $lang == "cpp" || $lang == "cubie" ]]; then
    if [[ $model != "ode" || $dev != "gpu" ]]; then
        echo "The benchmarking of ensemble ${model^^} solvers on ${dev^^} with ${lang} is not supported. Please use -m flag with \"ode\" and -d with \"gpu\"."
        exit 1
    else
        echo "Benchmarking ${lang^^} ${dev^^} accelerated ensemble ${model^^} solvers..."
        if [ -d "./data/${lang^^}" ] 
        then
            rm -rf "./data/${lang^^}"/*
            mkdir -p "./data/${lang^^}"
        else
            mkdir -p "./data/${lang^^}"
        fi
        cmd="./runner_scripts/${dev}/run_${model}_${lang}.sh ${nmax}"
        eval "$cmd"
    fi
fi
```

### Task 4: Create Data Output Directory Structure

The script will create this automatically, but for reference:

```
data/
└── CUBIE/
    ├── Cubie_times_unadaptive.txt
    └── Cubie_times_adaptive.txt
```

**Output Format**: Space-delimited, one line per trajectory count
```
<num_trajectories> <time_milliseconds>
```

Example:
```
8 0.5
32 1.2
128 4.8
512 18.6
```

### Task 5: Add Plotting Support

To include Cubie in comparison plots, modify plotting scripts.

**Location**: `runner_scripts/plot/plot_ode_comp.jl`

Copy existing format and syntax, and add cubie data to the plot_ode_comp.jl and plot_cpu_comp.jl files. Steps include:

1. Read Cubie data files
2. Add Cubie series to plots
3. Use appropriate color/marker for Cubie

Example addition:
```julia
# Read Cubie data
cubie_unadaptive = readdlm("../data/CUBIE/Cubie_times_unadaptive.txt")
cubie_adaptive = readdlm("../data/CUBIE/Cubie_times_adaptive.txt")

# Add to plot
plot!(cubie_unadaptive[:,1], cubie_unadaptive[:,2], 
      label="Cubie", marker=:diamond, linewidth=2)
```

## Verification Checklist

Before considering the implementation complete:

- [ ] Environment file created and tested (`conda env create -f ...`)
- [ ] Benchmark script runs without errors for small trajectory count
- [ ] Runner script executes and iterates through trajectory counts
- [ ] Main script recognizes `cubie` as valid language option
- [ ] Output files generated in correct format
- [ ] Results comparable to other implementations (within expected range)


## Success Criteria

The implementation is successful when:

1. **Correctness**: Solutions match expected Lorenz system behavior
2. **Integration**: Seamlessly works with `run_benchmark.sh -l cubie -d gpu -m ode`
3. **Output Format**: Generates correctly formatted data files
4. **Scalability**: Successfully runs across trajectory range (8 to max feasible)
5. **Comparison**: Can be directly compared with Julia/JAX/PyTorch/MPGOS results
