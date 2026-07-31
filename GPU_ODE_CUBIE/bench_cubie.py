#!/usr/bin/env python
# coding: utf-8
"""
Benchmarking Cubie ODE solvers for ensemble problems.
The Lorenz ODE is integrated with fixed and adaptive time-stepping.

Created for GPUODEBenchmarks integration
"""

import gc
import os
import sys
import timeit
import numpy as np
import cubie as qb
from numba import cuda
from cubie.time_logger import default_timelogger

default_timelogger.set_verbosity(None)

# Get number of trajectories from command line
numberOfParameters = int(sys.argv[1])
# Timed repeats per point; min is reported.
REPEATS = 20


# Dataset key ("<os>_<gpu>") so output files are keyed per machine and can be
# additively populated across machines without clobbering each other.
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runner_scripts"))
from bench_key import dataset_key, data_dir
DATASET_KEY = dataset_key()

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
    save_every=1.0,
    step_controller='fixed',
    output_types=['state'],
    time_logging_level=None,
)

initials_array, parameter_array = fixed_solver.build_grid(
        initial_values=initial_conditions, parameters=parameters)

# ========================================
# WORK-PRECISION (wp) MODE
# ========================================
# `bench_cubie.py 32768 wp` sweeps fixed dt / adaptive tolerance at N=32768
# and records "<setting> <time_ms> <error-vs-golden>" per point. Protocol and
# sweep grids live in runner_scripts/wp_common.py.
if len(sys.argv) > 2 and sys.argv[2] == "wp":
    from wp_common import (DTS, TOLS, N_WP, load_golden, ensemble_error,
                           wp_outfile)

    if numberOfParameters != N_WP:
        sys.exit("wp mode must be run with N = {0}".format(N_WP))
    golden = load_golden()
    # Grid built; one solver at a time from here.
    fixed_solver.close()

    def bench_solver(solver, repeats=20):
        def run():
            return solver.solve(
                initial_values=initials_array,
                parameters=parameter_array,
                blocksize=64,
                duration=1.0,
            )
        solution = run()  # warm-up (JIT compilation) + numerical result
        final_states = solution.state[-1, :, :].T
        err = ensemble_error(final_states, golden)
        res = timeit.repeat(run, setup='gc.enable()', repeat=repeats, number=1)
        return min(res) * 1000, err

    with open(wp_outfile("CUBIE", "Cubie", "fixed", DATASET_KEY), "w") as f:
        for dt in DTS:
            solver = qb.Solver(
                lorenz_system, algorithm='classical-rk4', dt=dt, save_every=1.0,
                step_controller='fixed', output_types=['state'],
                time_logging_level=None)
            t_ms, err = bench_solver(solver)
            print(f"wp fixed dt={dt:g}: {t_ms:.2f} ms, err={err:.3e}")
            f.write(f"{dt:.10g} {t_ms} {err:.10e}\n")
            solver.close()

    with open(wp_outfile("CUBIE", "Cubie", "adaptive", DATASET_KEY), "w") as f:
        for tol in TOLS:
            solver = qb.Solver(
                lorenz_system, algorithm='tsit5', atol=tol, rtol=tol,
                save_every=1.0, dt_min=1e-12, dt_max=1e3,
                step_controller='pid', kp=6/5, kd=0.0, ki=0.0,
                max_gain=5.0, min_gain=0.1, output_types=['state'],
                time_logging_level=None)
            t_ms, err = bench_solver(solver)
            print(f"wp adaptive tol={tol:g}: {t_ms:.2f} ms, err={err:.3e}")
            f.write(f"{tol:.10g} {t_ms} {err:.10e}\n")
            solver.close()

    sys.exit(0)

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
        duration=1.0
    )
    return solution

def solve_fixed_on_device(blocksize=64):
    """Solve with neither transfer: device arrays in, results left on device."""
    solution = fixed_solver.solve(
        initial_values=d_initials,
        parameters=d_parameters,
        blocksize=blocksize,
        duration=1.0,
        on_device=True
    )
    cuda.synchronize()
    return solution

# Uploaded once so the device-only timing excludes the h2d.
d_initials = cuda.to_device(initials_array)
d_parameters = cuda.to_device(parameter_array)

# Warm-up runs (JIT compilation), one per timed path
_ = solve_fixed()
_ = solve_fixed_on_device()

res = timeit.repeat(lambda: solve_fixed(), setup='gc.enable()', repeat=REPEATS, number=1)
res_dev = timeit.repeat(lambda: solve_fixed_on_device(), setup='gc.enable()', repeat=REPEATS, number=1)

best_time = min(res) * 1000  # Convert to milliseconds
best_time_dev = min(res_dev) * 1000
print(f"{numberOfParameters} ODE solves with fixed time-stepping completed in {best_time:.1f} ms "
      f"({best_time_dev:.1f} ms without transfers)")

# Save results
os.makedirs("./data/CUBIE", exist_ok=True)
with open(os.path.join(data_dir("CUBIE", DATASET_KEY), "Cubie_times_unadaptive.txt"), "a+") as file:
    file.write(f'{numberOfParameters} {best_time} {best_time_dev}\n')

# Save numerical output for 32768-trajectory run
if numberOfParameters == 32768:
    os.makedirs("./data/numerical", exist_ok=True)
    solution = solve_fixed()
    # Extract final state values
    final_states = solution.state[-1, :, :].T  # shape: (trajectories, states)
    np.savetxt(os.path.join(data_dir("numerical", DATASET_KEY), "cubie_unadaptive.csv"), final_states, delimiter=',')

# ========================================
# ADAPTIVE TIME-STEPPING BENCHMARK
# ========================================

# One solver at a time: the fixed solver is released first.
fixed_solver.close()
del fixed_solver
gc.collect()

adaptive_solver = qb.Solver(
    lorenz_system,
    algorithm='tsit5',
    atol=1e-08,
    rtol=1e-08,
    save_every=1.0,
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

def solve_adaptive(blocksize=64):
    """Solve with adaptive time step."""
    solution = adaptive_solver.solve(
        initial_values=initials_array,
        parameters=parameter_array,
        blocksize=blocksize,
        duration=1.0
    )
    return solution

def solve_adaptive_on_device(blocksize=64):
    """Solve with neither transfer: device arrays in, results left on device."""
    solution = adaptive_solver.solve(
        initial_values=d_initials,
        parameters=d_parameters,
        blocksize=blocksize,
        duration=1.0,
        on_device=True
    )
    cuda.synchronize()
    return solution

print(f"Running {numberOfParameters} trajectories with adaptive time-stepping...")

# Warm-up runs (JIT compilation), one per timed path
_ = solve_adaptive()
_ = solve_adaptive_on_device()

res = timeit.repeat(lambda: solve_adaptive(), setup='gc.enable()', repeat=REPEATS, number=1)
res_dev = timeit.repeat(lambda: solve_adaptive_on_device(), setup='gc.enable()', repeat=REPEATS, number=1)

best_time = min(res) * 1000  # Convert to milliseconds
best_time_dev = min(res_dev) * 1000
print(f"{numberOfParameters} ODE solves with adaptive time-stepping completed in {best_time:.1f} ms "
      f"({best_time_dev:.1f} ms without transfers)")

# Save results
with open(os.path.join(data_dir("CUBIE", DATASET_KEY), "Cubie_times_adaptive.txt"), "a+") as file:
    file.write(f'{numberOfParameters} {best_time} {best_time_dev}\n')

if numberOfParameters == 32768:
    os.makedirs("./data/numerical", exist_ok=True)
    solution = solve_adaptive()
    # Extract final state values
    final_states = solution.state[-1, :, :].T  # shape: (trajectories, states)
    np.savetxt(os.path.join(data_dir("numerical", DATASET_KEY), "cubie_adaptive.csv"), final_states, delimiter=',')