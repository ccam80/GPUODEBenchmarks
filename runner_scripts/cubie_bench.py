#!/usr/bin/env python

"""Cubie ensemble benchmark shared by the CUBIE and CUBIE_MLIR suites; the backend comes from CUBIE_CUDA_BACKEND."""

import gc
import os
import sys
import timeit

import numpy as np
from numba import cuda

from algorithms import supported_for
from bench_key import dataset_key, data_dir
from cubie_systems import (build_system, final_states, output_types,
                           sweep_parameters)
from wp_common import parse_bench_args, times_outfile

# Timed repeats per point; min is reported.
REPEATS = 20

PRECISION = np.float32


def _make_fixed_solver(system, problem, algorithm, dt=None):
    import cubie as qb
    return qb.Solver(
        system,
        algorithm=algorithm,
        dt=problem.timing_dt if dt is None else dt,
        save_every=problem["duration"],
        step_controller='fixed',
        output_types=output_types(system),
        time_logging_level=None,
    )


def _make_adaptive_solver(system, problem, algorithm, tol=1e-08):
    import cubie as qb
    return qb.Solver(
        system,
        algorithm=algorithm,
        atol=tol,
        rtol=tol,
        save_every=problem["duration"],
        dt_min=1e-12,
        dt_max=1e3,
        step_controller='pid',
        kp=6/5,
        kd=0.0,
        ki=0.0,
        max_gain=5.0,
        min_gain=0.1,
        output_types=output_types(system),
        time_logging_level=None,
    )


def _release(solver):
    """One solver at a time: close and free before the next is built."""
    solver.close()
    gc.collect()


def _run_problem(problem, opts):
    """Every requested algorithm for one problem."""
    duration = problem["duration"]
    n = opts["n"]
    dataset = opts["dataset_key"]
    system, initial_conditions = build_system(
        problem, PRECISION, name_suffix=opts["name_suffix"])
    parameters = sweep_parameters(problem, n, PRECISION)

    # Grid built once; one solver at a time from here.
    grid_solver = _make_fixed_solver(system, problem, 'euler')
    initials_array, parameter_array = grid_solver.build_grid(
        initial_values=initial_conditions, parameters=parameters)
    _release(grid_solver)

    if opts["wp"]:
        _run_wp(problem, opts, system, initials_array, parameter_array)
        return

    _run_times(problem, opts, system, initials_array, parameter_array)


def _failed(exc, what):
    """An algorithm that cannot run this system is a NaN row, not an abort."""
    print("FAILED {0}: {1}".format(what, exc))
    return float("nan"), float("nan")


def _run_wp(problem, opts, system, initials_array, parameter_array):
    """dt / tolerance sweep at N = N_WP; see runner_scripts/wp_common.py."""
    from wp_common import (dts_for, TOLS, N_WP, load_golden, ensemble_error,
                           wp_outfile)

    if opts["n"] != N_WP:
        sys.exit("wp mode must be run with N = {0}".format(N_WP))
    duration = problem["duration"]
    golden = load_golden(problem)

    def bench_solver(solver, repeats=REPEATS):
        def run():
            return solver.solve(
                initial_values=initials_array,
                parameters=parameter_array,
                blocksize=64,
                duration=duration,
            )
        solution = run()  # warm-up (JIT compilation) + numerical result
        err = ensemble_error(final_states(system, solution, problem),
                             golden)
        res = timeit.repeat(run, setup='gc.enable()', repeat=repeats, number=1)
        return min(res) * 1000, err

    for algorithm in opts["algorithms"]:
        if algorithm in opts["fixed"]:
            outfile = wp_outfile(opts["framework_dir"], opts["prefix"],
                                 "fixed", algorithm, opts["dataset_key"],
                                 problem)
            with open(outfile, "w") as f:
                for dt in dts_for(algorithm, problem):
                    solver = None
                    try:
                        solver = _make_fixed_solver(system, problem,
                                                    algorithm, dt)
                        t_ms, err = bench_solver(solver)
                    except Exception as exc:
                        t_ms, err = _failed(
                            exc, f"{problem.name} fixed {algorithm} dt={dt:g}")
                    print(f"wp {problem.name} fixed {algorithm} dt={dt:g}: "
                          f"{t_ms:.2f} ms, err={err:.3e}")
                    f.write(f"{dt:.10g} {t_ms} {err:.10e}\n")
                    if solver is not None:
                        _release(solver)

        if algorithm in opts["adaptive"]:
            outfile = wp_outfile(opts["framework_dir"], opts["prefix"],
                                 "adaptive", algorithm, opts["dataset_key"],
                                 problem)
            with open(outfile, "w") as f:
                for tol in TOLS:
                    solver = None
                    try:
                        solver = _make_adaptive_solver(system, problem,
                                                       algorithm, tol)
                        t_ms, err = bench_solver(solver)
                    except Exception as exc:
                        t_ms, err = _failed(
                            exc,
                            f"{problem.name} adaptive {algorithm} tol={tol:g}")
                    print(f"wp {problem.name} adaptive {algorithm} "
                          f"tol={tol:g}: {t_ms:.2f} ms, err={err:.3e}")
                    f.write(f"{tol:.10g} {t_ms} {err:.10e}\n")
                    if solver is not None:
                        _release(solver)


def _run_times(problem, opts, system, initials_array, parameter_array):
    """N-sweep timing: one row per (problem, mode, algorithm)."""
    duration = problem["duration"]
    n = opts["n"]
    dataset = opts["dataset_key"]

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
                duration=duration
            )

        def device_only(blocksize=64):
            solution = solver.solve(
                initial_values=d_initials,
                parameters=d_parameters,
                blocksize=blocksize,
                duration=duration,
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
        np.savetxt(os.path.join(
            data_dir("numerical", dataset, problem=problem), name),
            final_states(system, solution, problem), delimiter=',')

    for algorithm in opts["algorithms"]:
        for mode in ("fixed", "adaptive"):
            if algorithm not in opts[mode]:
                continue
            print(f"Running {problem.name}, {n} trajectories, {mode} dt, "
                  f"{algorithm}...")
            solver, solution = None, None
            try:
                solver = (_make_fixed_solver(system, problem, algorithm)
                          if mode == "fixed"
                          else _make_adaptive_solver(system, problem,
                                                     algorithm))
                best, best_dev, solution = bench_times(solver)
                print(f"{n} ODE solves ({algorithm}, {mode}) completed in "
                      f"{best:.1f} ms ({best_dev:.1f} ms without transfers)")
            except Exception as exc:
                best, best_dev = _failed(
                    exc, f"{problem.name} {mode} {algorithm} N={n}")
            outfile = times_outfile(opts["framework_dir"], opts["prefix"],
                                    mode, algorithm, dataset, problem)
            with open(outfile, "a+") as file:
                file.write(f'{n} {best} {best_dev}\n')
            # The pairwise numerical cross-check reads these fixed CSV names.
            if solution is not None and n == 32768:
                if mode == "fixed" and algorithm == "classical-rk4":
                    save_numerical(solution,
                                   opts["numerical_tag"] + "_unadaptive.csv")
                if mode == "adaptive" and algorithm == "tsit5":
                    save_numerical(solution,
                                   opts["numerical_tag"] + "_adaptive.csv")
            if solver is not None:
                _release(solver)


def run(argv, framework, framework_dir, prefix, numerical_tag,
        name_suffix=""):
    """Entry point: parse the CLI and run every requested problem."""
    from cubie.time_logger import default_timelogger
    default_timelogger.set_verbosity(None)

    n, wp, algorithms, problems = parse_bench_args(argv, framework)
    if not problems:
        print("{0} runs none of the requested problems; skipping."
              .format(framework))
        return 0
    opts = {
        "n": n,
        "wp": wp,
        "algorithms": algorithms,
        "framework_dir": framework_dir,
        "prefix": prefix,
        "numerical_tag": numerical_tag,
        "name_suffix": name_suffix,
        "fixed": supported_for(framework, "fixed"),
        "adaptive": supported_for(framework, "adaptive"),
        "dataset_key": dataset_key(),
    }
    for problem in problems:
        _run_problem(problem, opts)
    return 0
