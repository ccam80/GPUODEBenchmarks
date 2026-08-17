#!/usr/bin/env python

"""Cubie ensemble benchmark shared by the CUBIE and CUBIE_MLIR suites; the backend comes from CUBIE_CUDA_BACKEND."""

import gc
import os
import sys

import numpy as np
from numba import cuda

from algorithms import supported_for
from bench_key import dataset_key, data_dir
from cubie_systems import (build_system, final_states, output_types,
                           sweep_parameters)
from wp_common import TIMING_TOL, parse_bench_args, times_outfile

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


def _make_adaptive_solver(system, problem, algorithm, tol=TIMING_TOL):
    """No step controller passed: cubie runs its shipped defaults."""
    import cubie as qb
    return qb.Solver(
        system,
        algorithm=algorithm,
        atol=tol,
        rtol=tol,
        dt=problem.timing_dt,
        save_every=problem["duration"],
        output_types=output_types(system),
        time_logging_level=None,
    )


def _release(solver):
    """One solver at a time: close and free before the next is built."""
    solver.close()
    gc.collect()


def _grid_builder(initial_conditions, parameters):
    """Build the ensemble grid once, from the first solver that constructs."""
    cache = {}

    def build(solver):
        if "arrays" not in cache:
            cache["arrays"] = solver.build_grid(
                initial_values=initial_conditions, parameters=parameters)
        return cache["arrays"]

    return build


def _run_problem(problem, opts):
    """Every requested algorithm for one problem."""
    system, initial_conditions = build_system(
        problem, PRECISION, name_suffix=opts["name_suffix"])
    parameters = sweep_parameters(problem, opts["n"], PRECISION)
    grid = _grid_builder(initial_conditions, parameters)

    if opts["wp"]:
        _run_wp(problem, opts, system, grid)
        return

    _run_times(problem, opts, system, grid)


def _failed(exc, what):
    """An algorithm that cannot run this system is a NaN row, not an abort."""
    print("FAILED {0}: {1}".format(what, exc))
    return float("nan"), float("nan")


def _run_wp(problem, opts, system, grid):
    """dt / tolerance sweep at N = N_WP; see runner_scripts/wp_common.py."""
    from wp_common import (dts_for, TOLS, load_golden, ensemble_error,
                           timed_min_ms, wp_outfile)

    duration = problem["duration"]
    golden = load_golden(problem)

    def bench_solver(solver, repeats=REPEATS):
        """(best_ms, err); best_ms is None when a run breaches the watchdog."""
        initials_array, parameter_array = grid(solver)

        def run():
            return solver.solve(
                initial_values=initials_array,
                parameters=parameter_array,
                blocksize=64,
                duration=duration,
            )
        best_ms, solution = timed_min_ms(run, repeats)
        if best_ms is None:
            return None, float("nan")
        err = ensemble_error(final_states(system, solution, problem),
                             golden)
        return best_ms, err

    def sweep(mode, make_solver, settings):
        outfile = wp_outfile(opts["framework_dir"], opts["prefix"], mode,
                             algorithm, opts["dataset_key"], problem)
        with open(outfile, "w") as f:
            breached = False
            for setting in settings:
                t_ms, err = float("nan"), float("nan")
                solver = None
                if not breached:
                    try:
                        solver = make_solver(setting)
                        t_ms, err = bench_solver(solver)
                    except Exception as exc:
                        t_ms, err = _failed(
                            exc, f"{problem.name} {mode} {algorithm} "
                            f"setting={setting:g}")
                    if t_ms is None:
                        # Later settings are slower, so the leg is abandoned.
                        print(f"WATCHDOG {problem.name} {mode} {algorithm} "
                              f"setting={setting:g}: run exceeded the cap")
                        breached = True
                        t_ms = float("nan")
                print(f"wp {problem.name} {mode} {algorithm} "
                      f"setting={setting:g}: {t_ms:.2f} ms, err={err:.3e}")
                f.write(f"{setting:.10g} {t_ms} {err:.10e}\n")
                f.flush()
                if solver is not None:
                    _release(solver)

    for algorithm in opts["algorithms"]:
        if algorithm in opts["fixed"]:
            sweep("fixed",
                  lambda dt: _make_fixed_solver(system, problem, algorithm,
                                                dt),
                  dts_for(algorithm, problem))
        if algorithm in opts["adaptive"]:
            sweep("adaptive",
                  lambda tol: _make_adaptive_solver(system, problem,
                                                    algorithm, tol),
                  TOLS)


def _run_times(problem, opts, system, grid):
    """N-sweep timing: one row per (problem, mode, algorithm)."""
    duration = problem["duration"]
    n = opts["n"]
    dataset = opts["dataset_key"]
    device = {}

    def bench_times(solver):
        """Best-of-REPEATS (with_transfers_ms, device_only_ms, solution);
        the times are None when a run breaches the watchdog."""
        from wp_common import timed_min_ms

        initials_array, parameter_array = grid(solver)
        if not device:
            # Uploaded once so the device-only timing excludes the h2d.
            device["initials"] = cuda.to_device(initials_array)
            device["parameters"] = cuda.to_device(parameter_array)
        d_initials, d_parameters = device["initials"], device["parameters"]

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

        best, solution = timed_min_ms(with_transfers, REPEATS)
        if best is None:
            return None, None, None
        best_dev, _ = timed_min_ms(device_only, REPEATS)
        return best, best_dev, solution

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
                if best is None or best_dev is None:
                    print(f"WATCHDOG {problem.name} {mode} {algorithm} "
                          f"N={n}: run exceeded the cap")
                    best = best if best is not None else float("nan")
                    best_dev = float("nan") if best_dev is None else best_dev
                else:
                    print(f"{n} ODE solves ({algorithm}, {mode}) completed "
                          f"in {best:.1f} ms ({best_dev:.1f} ms without "
                          "transfers)")
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
