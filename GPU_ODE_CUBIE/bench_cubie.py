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

# Add parent directory to path for model_definitions
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_definitions import get_cubie_system, get_model_params, get_parameter_list

_default_timelogger.set_verbosity(None)

# Get number of trajectories and model name from command line
numberOfParameters = int(sys.argv[1])
model_name = sys.argv[2] if len(sys.argv) > 2 else "lorenz"

# ========================================
# MODEL SYSTEM DEFINITION
# ========================================
precision = np.float32

# Get model definition
system_string, initial_conditions, param_name, constants = get_cubie_system(model_name, precision)
model_params = get_model_params(model_name, precision)

# Create ODE system
ode_system = qb.create_ODE_system(
    system_string,
    states=initial_conditions,
    parameters={param_name: model_params['parameter_default']},
    constants=constants,
    name=model_params['name'],
    precision=precision
)

# ========================================
# PARAMETER SWEEP SETUP
# ========================================
# Get parameter list for the model
parameterList = get_parameter_list(model_name, numberOfParameters, precision)

# Build parameter dictionary for batch solve
parameters = {
    param_name: parameterList
}

fixed_solver = qb.Solver(
    ode_system,
    algorithm='classical-rk4',
    dt=0.001,
    dt_save=1.0,
    step_controller='fixed',
    output_types=['state'],
    time_logging_level=None,
)

adaptive_solver = qb.Solver(
    ode_system,
    algorithm='tsit5',
    atol=1e-08,
    rtol=1e-08,
    dt_save=1.0,
    dt_min=1e-12,
    dt_max=1e3,
    step_controller='pid',
    kp=6/5,
    kd=0.0,
    ki=0.0,
    max_gain=5.0,
    min_gain=0.1,
    output_types=['state'],
    time_logging_level=None,
)

initials_array, parameter_array = fixed_solver.grid_builder(
        states=initial_conditions, params=parameters)

# Get duration from model parameters
duration = model_params['tspan'][1] - model_params['tspan'][0]
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
        duration=duration
    )
    return solution

def solve_adaptive(blocksize=64):
    """Solve with adaptive time step."""
    solution = adaptive_solver.solve(
        initial_values=initials_array,
        parameters=parameter_array,
        blocksize=blocksize,
        results_type='raw',
        duration=duration
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
    final_states = solution['state'][-1, :, :].T  # shape: (trajectories, states)
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
    solution = solve_adaptive()
    # Extract final state values
    final_states = solution['state'][-1, :, :].T  # shape: (trajectories, states)
    np.savetxt("./data/numerical/cubie_adaptive.csv", final_states, delimiter=',')