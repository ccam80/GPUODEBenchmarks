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
from cubie.time_logger import _default_timelogger

_default_timelogger.set_verbosity(None)

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

fixed_solver = qb.Solver(
    lorenz_system,
    algorithm='classical-rk4',
    dt=0.001,
    dt_save=1.0,
    step_controller='fixed',
    output_types=['state'],
    time_logging_level=None,
)

adaptive_solver = qb.Solver(
    lorenz_system,
    algorithm='tsit5',
    atol=1e-08,
    rtol=1e-08,
    dt_save=1.0,
    dt_min=1e-9,
    dt_max=0.1,
    step_controller='i',
    output_types=['state'],
    time_logging_level=None,
)

fixed_solver.set_stride_order(("time", "variable", "run"))
adaptive_solver.set_stride_order(("time", "variable", "run"))

initials_array, parameter_array = fixed_solver.grid_builder(
        states=initial_conditions, params=parameters)
# ========================================
# FIXED TIME-STEPPING BENCHMARK
# ========================================
print(f"Running {numberOfParameters} trajectories with fixed time-stepping...")

def solve_fixed(blocksize=64):
    """Solve with fixed time step (unadaptive)."""
    solution = fixed_solver.solve(
        initial_values=initials_array,
        parameters=parameter_array,
        blocksize=blocksize,
        results_type='raw',
        duration=1.001 # step one past final time - last point is otherwise not saved
    )
    return solution

def solve_adaptive(blocksize=64):
    """Solve with adaptive time step."""
    solution = adaptive_solver.solve(
        initial_values=initials_array,
        parameters=parameter_array,
        blocksize=blocksize,
        results_type='raw',
        duration=1.0 
    )
    return solution

# Warm-up run (JIT compilation)
_ = solve_fixed()

# Benchmark with 100 repetitions
res = timeit.repeat(lambda: solve_fixed(), setup='gc.enable()', repeat=100, number=1)

best_time = min(res) * 1000  # Convert to milliseconds
print(f"{numberOfParameters} ODE solves with fixed time-stepping completed in {best_time:.1f} ms")

# Save results
import os
os.makedirs("./data/CUBIE", exist_ok=True)
with open("./data/CUBIE/Cubie_times_unadaptive.txt", "a+") as file:
    file.write(f'{numberOfParameters} {best_time}\n')

# Save numerical output for 32768-trajectory run
if numberOfParameters == 32768:
    os.makedirs("./data/numerical", exist_ok=True)
    solution = solve_fixed()
    # Extract final state values
    final_states = solution['state'][-1, :, :]  # shape: (trajectories, states)
    np.savetxt("./data/numerical/cubie_unadaptive.csv", final_states, delimiter=',')

# ========================================
# ADAPTIVE TIME-STEPPING BENCHMARK
# ========================================
print(f"Running {numberOfParameters} trajectories with adaptive time-stepping...")

# Warm-up run (JIT compilation)
_ = solve_adaptive()

# Benchmark with 100 repetitions
res = timeit.repeat(lambda: solve_adaptive(), setup='gc.enable()', repeat=100, number=1)

best_time = min(res) * 1000  # Convert to milliseconds
print(f"{numberOfParameters} ODE solves with adaptive time-stepping completed in {best_time:.1f} ms")

# Save results
with open("./data/CUBIE/Cubie_times_adaptive.txt", "a+") as file:
    file.write(f'{numberOfParameters} {best_time}\n')

if numberOfParameters == 32768:
    os.makedirs("./data/numerical", exist_ok=True)
    solution = solve_fixed()
    # Extract final state values
    final_states = solution['state'][-1, :, :]  # shape: (trajectories, states)
    np.savetxt("./data/numerical/cubie_adaptive.csv", final_states, delimiter=',')