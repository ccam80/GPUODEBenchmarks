#!/usr/bin/env python
# coding: utf-8
"""
Benchmarking Cubie ODE solvers for ensemble problems, once per algorithm:
euler/classical-rk4/tsit5 fixed, tsit5/cash-karp-54 adaptive (PID).

Usage: bench_cubie.py <N> [wp] [algorithm|all]
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

# Dataset key ("<os>_<gpu>") so output files are keyed per machine and can be
# additively populated across machines without clobbering each other.
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runner_scripts"))
from bench_key import dataset_key, data_dir
from wp_common import parse_bench_args, times_outfile

DATASET_KEY = dataset_key()

FIXED_ALGORITHMS = ("euler", "classical-rk4", "tsit5")
ADAPTIVE_ALGORITHMS = ("tsit5", "cash-karp-54")
SUPPORTED = ("euler", "classical-rk4", "tsit5", "cash-karp-54")

numberOfParameters, WP_MODE, ALGORITHMS = parse_bench_args(sys.argv[1:], SUPPORTED)
# Timed repeats per point; min is reported.
REPEATS = 20

FRAMEWORK_DIR = "CUBIE"
FRAMEWORK_PREFIX = "Cubie"
NUMERICAL_TAG = "cubie"

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

# rho varies across the ensemble; everything else is scalar.
parameters = {
    'rho': parameterList
}

# Initial conditions (same for all trajectories)
initial_conditions = {
    'x': 1.0,
    'y': 0.0,
    'z': 0.0
}


def make_fixed_solver(algorithm, dt=0.001):
    return qb.Solver(
        lorenz_system,
        algorithm=algorithm,
        dt=dt,
        save_every=1.0,
        step_controller='fixed',
        output_types=['state'],
        time_logging_level=None,
    )


def make_adaptive_solver(algorithm, tol=1e-08):
    return qb.Solver(
        lorenz_system,
        algorithm=algorithm,
        atol=tol,
        rtol=tol,
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


# Grid built once; one solver at a time from here.
grid_solver = make_fixed_solver('classical-rk4')
initials_array, parameter_array = grid_solver.build_grid(
        initial_values=initial_conditions, parameters=parameters)
grid_solver.close()
del grid_solver
gc.collect()

# ========================================
# WORK-PRECISION (wp) MODE
# ========================================
# Sweeps dt / tolerance per algorithm at N=32768; see runner_scripts/wp_common.py.
if WP_MODE:
    from wp_common import (dts_for, TOLS, N_WP, load_golden, ensemble_error,
                           wp_outfile)

    if numberOfParameters != N_WP:
        sys.exit("wp mode must be run with N = {0}".format(N_WP))
    golden = load_golden()

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

    for algorithm in ALGORITHMS:
        if algorithm in FIXED_ALGORITHMS:
            outfile = wp_outfile(FRAMEWORK_DIR, FRAMEWORK_PREFIX, "fixed",
                                 algorithm, DATASET_KEY)
            with open(outfile, "w") as f:
                for dt in dts_for(algorithm):
                    solver = make_fixed_solver(algorithm, dt)
                    t_ms, err = bench_solver(solver)
                    print(f"wp fixed {algorithm} dt={dt:g}: {t_ms:.2f} ms, "
                          f"err={err:.3e}")
                    f.write(f"{dt:.10g} {t_ms} {err:.10e}\n")
                    solver.close()

        if algorithm in ADAPTIVE_ALGORITHMS:
            outfile = wp_outfile(FRAMEWORK_DIR, FRAMEWORK_PREFIX, "adaptive",
                                 algorithm, DATASET_KEY)
            with open(outfile, "w") as f:
                for tol in TOLS:
                    solver = make_adaptive_solver(algorithm, tol)
                    t_ms, err = bench_solver(solver)
                    print(f"wp adaptive {algorithm} tol={tol:g}: {t_ms:.2f} ms, "
                          f"err={err:.3e}")
                    f.write(f"{tol:.10g} {t_ms} {err:.10e}\n")
                    solver.close()

    sys.exit(0)

# ========================================
# N-SWEEP TIMING BENCHMARK
# ========================================
# Uploaded once so the device-only timing excludes the h2d.
d_initials = cuda.to_device(initials_array)
d_parameters = cuda.to_device(parameter_array)


def bench_times(solver):
    """Best-of-REPEATS (with_transfers_ms, device_only_ms, solution)."""
    def with_transfers(blocksize=64):
        return solver.solve(
            initial_values=initials_array,
            parameters=parameter_array,
            blocksize=blocksize,
            duration=1.0
        )

    def device_only(blocksize=64):
        solution = solver.solve(
            initial_values=d_initials,
            parameters=d_parameters,
            blocksize=blocksize,
            duration=1.0,
            on_device=True
        )
        cuda.synchronize()
        return solution

    # Warm-up runs (JIT compilation), one per timed path
    solution = with_transfers()
    _ = device_only()

    res = timeit.repeat(with_transfers, setup='gc.enable()',
                        repeat=REPEATS, number=1)
    res_dev = timeit.repeat(device_only, setup='gc.enable()',
                            repeat=REPEATS, number=1)
    return min(res) * 1000, min(res_dev) * 1000, solution


def save_numerical(solution, name):
    """Final states for the 32768-run numerical cross-check."""
    final_states = solution.state[-1, :, :].T  # shape: (trajectories, states)
    np.savetxt(os.path.join(data_dir("numerical", DATASET_KEY), name),
               final_states, delimiter=',')


def release(solver):
    """One solver at a time: close and free before the next is built."""
    solver.close()
    gc.collect()


for algorithm in ALGORITHMS:
    if algorithm in FIXED_ALGORITHMS:
        print(f"Running {numberOfParameters} trajectories, fixed dt, "
              f"{algorithm}...")
        solver = make_fixed_solver(algorithm)
        best, best_dev, solution = bench_times(solver)
        print(f"{numberOfParameters} ODE solves ({algorithm}, fixed) completed "
              f"in {best:.1f} ms ({best_dev:.1f} ms without transfers)")
        outfile = times_outfile(FRAMEWORK_DIR, FRAMEWORK_PREFIX, "fixed",
                                algorithm, DATASET_KEY)
        with open(outfile, "a+") as file:
            file.write(f'{numberOfParameters} {best} {best_dev}\n')
        # The pairwise numerical cross-check reads this fixed CSV name.
        if numberOfParameters == 32768 and algorithm == "classical-rk4":
            save_numerical(solution, NUMERICAL_TAG + "_unadaptive.csv")
        release(solver)

    if algorithm in ADAPTIVE_ALGORITHMS:
        print(f"Running {numberOfParameters} trajectories, adaptive dt, "
              f"{algorithm}...")
        solver = make_adaptive_solver(algorithm)
        best, best_dev, solution = bench_times(solver)
        print(f"{numberOfParameters} ODE solves ({algorithm}, adaptive) "
              f"completed in {best:.1f} ms ({best_dev:.1f} ms without transfers)")
        outfile = times_outfile(FRAMEWORK_DIR, FRAMEWORK_PREFIX, "adaptive",
                                algorithm, DATASET_KEY)
        with open(outfile, "a+") as file:
            file.write(f'{numberOfParameters} {best} {best_dev}\n')
        if numberOfParameters == 32768 and algorithm == "tsit5":
            save_numerical(solution, NUMERICAL_TAG + "_adaptive.csv")
        release(solver)
