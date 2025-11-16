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
    dt=0.001,
    dt_save=1.0,
    step_controller='fixed',
    output_types=['state'],
    time_logging_level='verbose',
)

adaptive_solver = qb.solver(
    lorenz_system,
    algorithm='tsit5',
    atol=1e-08,
    rtol=1e-08,
    dt_save=1.0,
    dt_min=1e-9,
    dt_max=0.1,
    step_controller='adaptive',
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

def solve_adaptive(blocksize=256):
    """Solve with adaptive time step."""
    solution = adaptive_solver.solve_ivp(
        initial_values=initial_conditions,
        parameters=parameters,
        blocksize=blocksize,
    )
    return solution

# Warm-up run (JIT compilation)
_ = solve_fixed()

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
