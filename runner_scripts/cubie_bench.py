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


def _grid_builder(problem, initial_conditions):
    """Per-size ensemble grids; only the current size's arrays are held."""
    cache = {}

    def build(solver, n):
        if cache.get("n") != n:
            cache.clear()
            parameters = sweep_parameters(problem, n, PRECISION)
            cache["n"] = n
            cache["arrays"] = solver.build_grid(
                initial_values=initial_conditions, parameters=parameters)
        return cache["arrays"]

    return build


def _run_problem(problem, opts):
    """Every requested algorithm for one problem."""
    system, initial_conditions = build_system(
        problem, PRECISION, name_suffix=opts["name_suffix"])
    grid = _grid_builder(problem, initial_conditions)

    if opts["analysis"] == "wp":
        _run_wp(problem, opts, system, grid)
        return

    _run_times(problem, opts, system, grid)


def _failed(exc, what):
    """An algorithm that cannot run this system is a NaN row, not an abort."""
    print("FAILED {0}: {1}".format(what, exc))
    return float("nan"), float("nan")


def _run_wp(problem, opts, system, grid):
    """dt / tolerance sweep at N = N_WP; see runner_scripts/wp_common.py."""
    from wp_common import (dts_for, TOLS, N_WP, load_golden, ensemble_error,
                           timed_min_ms, wp_outfile)

    duration = problem["duration"]
    golden = load_golden(problem)

    def bench_solver(solver, repeats=REPEATS):
        """(best_ms, err); best_ms is None when a run breaches the watchdog."""
        initials_array, parameter_array = grid(solver, N_WP)

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
        if not problem.runs(opts["framework"], algorithm):
            continue
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
    """N-sweep timing: each (algorithm, mode) leg walks the sizes ascending on one solver."""
    from wp_common import timed_min_ms

    duration = problem["duration"]
    dataset = opts["dataset_key"]
    ns = opts["ns"]

    def bench_times(solver, n):
        """(with_transfers_ms, device_only_ms, solution); times None on a breach."""
        initials_array, parameter_array = grid(solver, n)
        # Uploaded once per size so the device-only timing excludes the h2d.
        d_initials = cuda.to_device(initials_array)
        d_parameters = cuda.to_device(parameter_array)

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

    def nan_rows(file, sizes):
        for n in sizes:
            file.write(f'{n} nan nan\n')
        file.flush()

    for algorithm in opts["algorithms"]:
        if not problem.runs(opts["framework"], algorithm):
            continue
        for mode in ("fixed", "adaptive"):
            if algorithm not in opts[mode]:
                continue
            outfile = times_outfile(opts["framework_dir"], opts["prefix"],
                                    mode, algorithm, dataset, problem)
            solver = None
            try:
                solver = (_make_fixed_solver(system, problem, algorithm)
                          if mode == "fixed"
                          else _make_adaptive_solver(system, problem,
                                                     algorithm))
            except Exception as exc:
                _failed(exc, f"{problem.name} {mode} {algorithm}")
                with open(outfile, "a+") as file:
                    nan_rows(file, ns)
                continue
            with open(outfile, "a+") as file:
                for index, n in enumerate(ns):
                    print(f"Running {problem.name}, {n} trajectories, "
                          f"{mode} dt, {algorithm}...")
                    solution = None
                    try:
                        best, best_dev, solution = bench_times(solver, n)
                        if best is None or best_dev is None:
                            # Larger sizes are slower, so the leg is abandoned.
                            print(f"WATCHDOG {problem.name} {mode} "
                                  f"{algorithm} N={n}: run exceeded the cap")
                            nan_rows(file, ns[index:])
                            break
                        print(f"{n} ODE solves ({algorithm}, {mode}) "
                              f"completed in {best:.1f} ms ({best_dev:.1f} ms "
                              "without transfers)")
                    except Exception as exc:
                        best, best_dev = _failed(
                            exc, f"{problem.name} {mode} {algorithm} N={n}")
                    file.write(f'{n} {best} {best_dev}\n')
                    file.flush()
                    # The pairwise numerical cross-check reads these fixed CSV names.
                    if solution is not None and n == 32768:
                        if mode == "fixed" and algorithm == "classical-rk4":
                            save_numerical(
                                solution,
                                opts["numerical_tag"] + "_unadaptive.csv")
                        if mode == "adaptive" and algorithm == "tsit5":
                            save_numerical(
                                solution,
                                opts["numerical_tag"] + "_adaptive.csv")
                    gc.collect()
            _release(solver)


def _warm_legs(opts, problems):
    """Every (problem, mode, algorithm, setting) compile task, in a
    deterministic order shared by the parent and its shard children."""
    from wp_common import TOLS

    legs = []
    for problem in problems:
        for algorithm in opts["algorithms"]:
            if not problem.runs(opts["framework"], algorithm):
                continue
            if algorithm in opts["fixed"]:
                legs.append((problem.name, "fixed", algorithm, None))
                for dt in problem.dts(algorithm):
                    legs.append((problem.name, "fixed", algorithm, dt))
            if algorithm in opts["adaptive"]:
                legs.append((problem.name, "adaptive", algorithm, None))
                for tol in TOLS:
                    legs.append((problem.name, "adaptive", algorithm, tol))
    return legs


# Legs per shard child before it exits and is respawned; numba dispatchers
# stay resident, so long-lived children grow without bound.
WARM_RECYCLE = 32


def _run_warm(opts, problems, argv):
    """Compile each leg once at a tiny ensemble; BENCH_WARM_JOBS>1 stripes
    the legs across that many shard children, recycled every WARM_RECYCLE
    legs to cap their memory."""
    import subprocess
    from timeit import default_timer

    # Each shard holds several GB of host RAM; 4 fits in 32 GB.
    jobs = int(os.environ.get("BENCH_WARM_JOBS", "4"))
    shard = opts.get("warm_shard")
    legs = _warm_legs(opts, problems)

    if shard is None and jobs > 1 and len(legs) > 1:
        count = min(jobs, len(legs))
        cursors = [0] * count
        stripe_sizes = [len(legs[index::count]) for index in range(count)]
        while any(cursors[i] < stripe_sizes[i] for i in range(count)):
            children = []
            for index in range(count):
                if cursors[index] >= stripe_sizes[index]:
                    continue
                children.append(subprocess.Popen(
                    [sys.executable, sys.argv[0]] + argv
                    + ["--warm-shard",
                       f"{index}/{count}/{cursors[index]}/{WARM_RECYCLE}"]))
                cursors[index] += WARM_RECYCLE
            for child in children:
                child.wait()
        return

    if shard is not None:
        index, count, offset, limit = shard
        legs = legs[index::count][offset:offset + limit]

    rows = {p.name: p for p in problems}
    systems = {}

    def system_for(name):
        if name not in systems:
            systems[name] = build_system(
                rows[name], PRECISION, name_suffix=opts["name_suffix"])
        return systems[name]

    for name, mode, algorithm, setting in legs:
        row = rows[name]
        tag = ("" if setting is None else
               (f" dt={setting:g}" if mode == "fixed" else
                f" tol={setting:g}"))
        label = f"{name} {mode} {algorithm}{tag}"
        solver = None
        started = default_timer()
        try:
            system, conditions = system_for(name)
            if mode == "fixed":
                solver = (_make_fixed_solver(system, row, algorithm)
                          if setting is None else
                          _make_fixed_solver(system, row, algorithm,
                                             setting))
            else:
                solver = (_make_adaptive_solver(system, row, algorithm)
                          if setting is None else
                          _make_adaptive_solver(system, row, algorithm,
                                                setting))
            initials, params = solver.build_grid(
                initial_values=conditions,
                parameters=sweep_parameters(row, 64, PRECISION))
            solver.solve(initial_values=initials, parameters=params,
                         blocksize=64, duration=row["duration"])
            print("warmed {0} in {1:.1f}s".format(
                label, default_timer() - started), flush=True)
        except Exception as exc:
            _failed(exc, "warm {0}".format(label))
        if solver is not None:
            _release(solver)


def _run_states(opts):
    """Runtime-by-states sweep: lorenz96 resized along STATES_GRID, timed at
    one fixed ensemble size."""
    import tempfile
    from timeit import default_timer

    from cubie.cache_root import set_cache_root
    from problems import states_row
    from wp_common import STATES_N, states_outfile, timed_min_ms

    # Throwaway cache root: every states compile runs cold.
    set_cache_root(tempfile.mkdtemp(prefix="cubie_states_"))

    n = STATES_N
    grid = opts["ns"]
    systems = {}

    def system_for(nstates):
        if nstates not in systems:
            systems[nstates] = build_system(
                states_row(nstates), PRECISION,
                name_suffix="{0}_s{1}".format(opts["name_suffix"], nstates))
        return systems[nstates]

    for algorithm in opts["algorithms"]:
        for mode in ("fixed", "adaptive"):
            if algorithm not in opts[mode]:
                continue
            outfile = states_outfile(opts["framework_dir"], opts["prefix"],
                                     mode, algorithm, opts["dataset_key"])
            with open(outfile, "w") as file:
                for index, nstates in enumerate(grid):
                    row = states_row(nstates)
                    duration = row["duration"]
                    print(f"Running lorenz96 states={nstates}, "
                          f"{n} trajectories, {mode} dt, {algorithm}...")
                    t_ms = t_dev = build_s = float("nan")
                    breached = False
                    solver = None
                    try:
                        started = default_timer()
                        system, initial_conditions = system_for(nstates)
                        solver = (_make_fixed_solver(system, row, algorithm)
                                  if mode == "fixed"
                                  else _make_adaptive_solver(system, row,
                                                             algorithm))
                        initials_array, parameter_array = solver.build_grid(
                            initial_values=initial_conditions,
                            parameters=sweep_parameters(row, n, PRECISION))

                        def with_transfers(blocksize=64):
                            return solver.solve(
                                initial_values=initials_array,
                                parameters=parameter_array,
                                blocksize=blocksize,
                                duration=duration,
                            )

                        with_transfers()
                        build_s = default_timer() - started

                        d_initials = cuda.to_device(initials_array)
                        d_parameters = cuda.to_device(parameter_array)

                        def device_only(blocksize=64):
                            solution = solver.solve(
                                initial_values=d_initials,
                                parameters=d_parameters,
                                blocksize=blocksize,
                                duration=duration,
                                on_device=True,
                            )
                            cuda.synchronize()
                            return solution

                        best, _ = timed_min_ms(with_transfers, REPEATS)
                        best_dev = None
                        if best is not None:
                            best_dev, _ = timed_min_ms(device_only, REPEATS)
                        breached = best is None or best_dev is None
                        if not breached:
                            t_ms, t_dev = best, best_dev
                            print(f"{n} ODE solves (lorenz96 "
                                  f"states={nstates}, {algorithm}, {mode}) "
                                  f"completed in {t_ms:.1f} ms ({t_dev:.1f} "
                                  "ms without transfers)")
                    except Exception as exc:
                        _failed(exc, f"lorenz96 states={nstates} {mode} "
                                     f"{algorithm} N={n}")
                    file.write(f'{nstates} {t_ms} {t_dev} {build_s}\n')
                    file.flush()
                    if solver is not None:
                        _release(solver)
                    if breached:
                        # Larger systems are slower, so the leg is abandoned.
                        print(f"WATCHDOG lorenz96 states={nstates} {mode} "
                              f"{algorithm} N={n}: run exceeded the cap")
                        for rest in grid[index + 1:]:
                            file.write(f'{rest} nan nan nan\n')
                        file.flush()
                        break


def run(argv, framework, framework_dir, prefix, numerical_tag,
        name_suffix=""):
    """Entry point: parse the CLI and run every requested problem."""
    from cubie.time_logger import default_timelogger
    default_timelogger.set_verbosity(None)

    argv = list(argv)
    warm_shard = None
    if "--warm-shard" in argv:
        position = argv.index("--warm-shard")
        warm_shard = tuple(int(t) for t in argv[position + 1].split("/"))
        del argv[position:position + 2]

    ns, analysis, algorithms, problems = parse_bench_args(argv, framework)
    if not problems:
        print("{0} runs none of the requested problems; skipping."
              .format(framework))
        return 0
    opts = {
        "ns": ns,
        "analysis": analysis,
        "framework": framework,
        "algorithms": algorithms,
        "framework_dir": framework_dir,
        "prefix": prefix,
        "numerical_tag": numerical_tag,
        "name_suffix": name_suffix,
        "fixed": supported_for(framework, "fixed"),
        "adaptive": supported_for(framework, "adaptive"),
        "dataset_key": dataset_key(),
        "warm_shard": warm_shard,
    }
    if analysis == "warm":
        _run_warm(opts, problems, argv)
        return 0
    if analysis == "states":
        from problems import STATES_PROBLEM
        if not any(p.name == STATES_PROBLEM for p in problems):
            print("{0} does not run {1}; skipping the states sweep."
                  .format(framework, STATES_PROBLEM))
            return 0
        _run_states(opts)
        return 0
    for problem in problems:
        _run_problem(problem, opts)
    return 0
