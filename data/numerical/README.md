# Numerical Comparison of Integration Packages

This directory contains the final state arrays from the 32768-trajectory runs of each integration package.

## Files

Each package saves its results to a CSV file:
- `cubie.csv` - Results from CUBIE solver
- `jax.csv` - Results from JAX/Diffrax solver
- `pytorch.csv` - Results from PyTorch/torchdiffeq solver
- `julia.csv` - Results from Julia/DiffEqGPU solver
- `mpgos.csv` - Results from MPGOS (C++) solver

## Format

Each CSV file contains the final state values for all 32768 trajectories. The format is:
- Each row represents one trajectory
- Each column represents one state variable (x, y, z for the Lorenz system)
- No headers are included in the CSV files

For the Lorenz system with 32768 trajectories:
- Shape: (32768, 3)
- Column 0: x final state
- Column 1: y final state
- Column 2: z final state

## How the data is generated

The data files are automatically generated when running the benchmarks with exactly 32768 trajectories. For example:

```bash
# Run CUBIE benchmark with 32768 trajectories
source ./GPU_ODE_CUBIE/venv_cubie/bin/activate
python3 ./GPU_ODE_CUBIE/bench_cubie.py 32768
deactivate

# Run JAX benchmark with 32768 trajectories
source ./GPU_ODE_JAX/venv_jax/bin/activate
python3 ./GPU_ODE_JAX/bench_diffrax.py 32768
deactivate

# Run PyTorch benchmark with 32768 trajectories
source ./GPU_ODE_PyTorch/venv_torch/bin/activate
python3 ./GPU_ODE_PyTorch/bench_torchdiffeq.py 32768
deactivate

# Run Julia benchmark with 32768 trajectories
julia --project=. ./GPU_ODE_Julia/bench_lorenz_gpu.jl 32768

# Run MPGOS benchmark with 32768 trajectories (requires modification of NT in Lorenz.cu)
cd GPU_ODE_MPGOS && make && ./lorenz.exe
```

## Comparing results

Use the `compare_numerical_results.py` script in the root directory to compare the results:

```bash
# Activate the cubie venv (which has numpy)
source ./GPU_ODE_CUBIE/venv_cubie/bin/activate

# Run comparison
python3 ../compare_numerical_results.py

# Deactivate
deactivate
```

The comparison script will:
1. Load all available CSV files from this directory
2. Perform pairwise comparisons using `numpy.allclose()`
3. Display detailed statistics including:
   - Maximum, mean, minimum, and standard deviation of absolute differences
   - Maximum, mean, minimum, and standard deviation of relative differences
   - Per-state statistics
   - Top 5 worst mismatches
4. Summarize the comparison results

## Tolerance settings

The comparison uses:
- `rtol=1e-5` (relative tolerance)
- `atol=1e-8` (absolute tolerance)

These can be adjusted in the `compare_numerical_results.py` script if needed.
