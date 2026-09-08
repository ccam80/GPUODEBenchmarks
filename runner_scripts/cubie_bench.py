#!/usr/bin/env python

"""Cubie ensemble benchmark shared by the CUBIE and CUBIE_MLIR suites; solvers, backend and optimize rows come from cubie_adapter."""

import gc
import os
import sys

import numpy as np

import cubie_adapter as adapter
from algorithms import supported_for
from bench_key import dataset_key, data_dir
from cubie_systems import final_states, sweep_parameters
from results import Leg
from resume import (active as resume_active, floor_enabled, skip_point,
                    skip_wp_leg)
from wp_common import REPEAT_CAP, errored_pct, parse_bench_args

REPEATS = REPEAT_CAP

PRECISION = np.float32


def _make_solver(opts, system, problem, algorithm, mode, setting=None,
                 states=None):
    return adapter.make_solver(system, problem, algorithm, mode, setting,
                               package=opts["framework"],
                               key=opts["dataset_key"], states=states)


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
    system, initial_conditions = adapter.build_system(
        problem, opts["framework"], PRECISION)
    grid = _grid_builder(problem, initial_conditions)

    if opts["analysis"] == "wp":
        _run_wp(problem, opts, system, grid)
        return

    _run_times(problem, opts, system, grid)


def _failed(exc, what):
    """An algorithm that cannot run this system is a NaN row, not an abort."""
    print("FAILED {0}: {1}".format(what, exc))
    return float("nan")


def _device_leg(solver, duration, repeats):
    """(best_ms, samples) on the resident inputs with results left on the device; best_ms is None on a breach."""
    from wp_common import timed_min_ms

    # Raises after a chunked host leg.
    d_initials = solver.device_initial_values
    d_parameters = solver.device_parameters

    def device_only():
        adapter.solve(solver, d_initials, d_parameters, duration,
                      on_device=True)

    best, _, samples = timed_min_ms(device_only, repeats)
    return best, samples


def _run_wp(problem, opts, system, grid):
    """dt / tolerance sweep at N = N_WP; see runner_scripts/wp_common.py."""
    from wp_common import (dts_for, TOLS, N_WP, append_samples, load_golden,
                           ensemble_error, reset_samples, sample_point,
                           samples_outfile, timed_min_ms)

    duration = problem["duration"]
    golden = load_golden(problem)

    def bench_solver(solver, repeats=REPEATS):
        """(best_ms, err, errored_percent, samples); best_ms is None when a run breaches the watchdog."""
        initials_array, parameter_array = grid(solver, N_WP)

        def run():
            return adapter.solve(solver, initials_array, parameter_array,
                                 duration)
        best_ms, solution, samples = timed_min_ms(run, repeats)
        if best_ms is None:
            return None, float("nan"), 100.0, samples
        view = final_states(system, solution, problem)
        err = ensemble_error(view, golden)
        return best_ms, err, errored_pct(view), samples

    def sweep(mode, settings):
        leg = Leg(opts["framework"], opts["dataset_key"], "wp", problem,
                  algorithm, mode)
        settings = list(settings)
        if skip_wp_leg(leg, settings):
            print(f"-- resume: skipping wp {problem.name} {mode} "
                  f"{algorithm} (already covered)")
            return
        samples_file = samples_outfile(opts["framework_dir"], opts["prefix"],
                                       "wp", mode, algorithm,
                                       opts["dataset_key"], problem)
        setting_kind = "dt" if mode == "fixed" else "tol"
        # --floor merges the new times in; the log gains a fresh series.
        if not floor_enabled():
            reset_samples(samples_file)
        breached = False
        for setting in settings:
            t_ms, err = float("nan"), float("nan")
            pct = 100.0
            samples = None
            solver = None
            if not breached:
                try:
                    solver = _make_solver(opts, system, problem, algorithm,
                                          mode, setting)
                    t_ms, err, pct, samples = bench_solver(solver)
                    append_samples(samples_file, sample_point(
                        "wp", problem.name, algorithm, mode, N_WP,
                        problem["states"], setting_kind, setting),
                        "both", samples)
                except Exception as exc:
                    t_ms = err = _failed(
                        exc, f"{problem.name} {mode} {algorithm} "
                        f"setting={setting:g}")
                    pct = 100.0
                if t_ms is None:
                    # Later settings are slower, so the leg is abandoned.
                    print(f"WATCHDOG {problem.name} {mode} {algorithm} "
                          f"setting={setting:g}: run exceeded the cap")
                    breached = True
                    t_ms = float("nan")
            print(f"wp {problem.name} {mode} {algorithm} "
                  f"setting={setting:g}: {t_ms:.2f} ms, err={err:.3e}, "
                  f"errored={pct:.1f}%")
            leg.record_wp(setting, t_ms, err, pct, samples=samples)
            if solver is not None:
                _release(solver)

    for algorithm in opts["algorithms"]:
        if not problem.supports(opts["framework"]):
            continue
        if algorithm in opts["fixed"]:
            sweep("fixed", dts_for(algorithm, problem))
        if algorithm in opts["adaptive"]:
            sweep("adaptive", TOLS)


def _run_times(problem, opts, system, grid):
    """N-sweep timing: each (algorithm, mode) leg walks the sizes ascending on one solver."""
    from wp_common import (append_samples, sample_point, samples_outfile,
                           timed_min_ms)

    duration = problem["duration"]
    dataset = opts["dataset_key"]
    ns = opts["ns"]
    # The pairwise numerical cross-check reads these fixed CSV names.
    numerical_names = {("fixed", "classical-rk4"): "_unadaptive.csv",
                       ("adaptive", "tsit5"): "_adaptive.csv"}

    def host_leg(solver, n, want_finals):
        """(best_ms, finals, errored_percent, samples) through host arrays; best_ms is None on a breach, finals a copy or None."""
        initials_array, parameter_array = grid(solver, n)

        def with_transfers():
            return adapter.solve(solver, initials_array, parameter_array,
                                 duration)

        best, solution, samples = timed_min_ms(with_transfers, REPEATS)
        finals = None
        pct = 100.0
        if best is not None:
            view = final_states(system, solution, problem)
            pct = errored_pct(view)
            if want_finals:
                # A copy: the view aliases the buffer the next solve reuses.
                finals = np.array(view)
        return best, finals, pct, samples

    def save_numerical(finals, name):
        """Final states for the 32768-run numerical cross-check."""
        np.savetxt(os.path.join(
            data_dir("numerical", dataset, problem=problem), name),
            finals, delimiter=',')

    for algorithm in opts["algorithms"]:
        if not problem.supports(opts["framework"]):
            continue
        for mode in ("fixed", "adaptive"):
            if algorithm not in opts[mode]:
                continue
            leg = Leg(opts["framework"], dataset, "times", problem, algorithm,
                      mode)
            samples_file = samples_outfile(
                opts["framework_dir"], opts["prefix"], "times", mode,
                algorithm, dataset, problem)
            run_ns = [n for n in ns if not skip_point(leg, n)]
            if not run_ns:
                print(f"-- resume: skipping {problem.name} {mode} "
                      f"{algorithm} (already covered)")
                continue
            if len(run_ns) < len(ns):
                print(f"-- resume: {problem.name} {mode} {algorithm} "
                      f"runs N={','.join(str(n) for n in run_ns)}")
            solver = None
            try:
                solver = _make_solver(opts, system, problem, algorithm, mode)
            except Exception as exc:
                _failed(exc, f"{problem.name} {mode} {algorithm}")
                leg.nan_times(run_ns)
                continue
            # A device-only breach abandons that column alone.
            device_breached = False
            for index, n in enumerate(run_ns):
                print(f"Running {problem.name}, {n} trajectories, "
                      f"{mode} dt, {algorithm}...")
                label = f"{problem.name} {mode} {algorithm} N={n}"
                point = sample_point("times", problem.name, algorithm,
                                     mode, n, problem["states"])
                want_finals = (n == 32768
                               and (mode, algorithm) in numerical_names)
                finals = None
                pct = 100.0
                samples_both = samples_none = None
                try:
                    best, finals, pct, samples_both = host_leg(
                        solver, n, want_finals)
                    append_samples(samples_file, point, "both", samples_both)
                except Exception as exc:
                    best = _failed(exc, label)
                if best is None:
                    # Larger sizes are slower, so the leg is abandoned.
                    print(f"WATCHDOG {label}: run exceeded the cap")
                    leg.nan_times(run_ns[index:])
                    break
                # The device leg runs only after a host-path time.
                best_dev = float("nan")
                if np.isnan(best):
                    print(f"SKIP {label} device-only: no host-path time")
                elif device_breached:
                    print(f"SKIP {label} device-only: breached at a "
                          "smaller size")
                else:
                    try:
                        best_dev, samples_none = _device_leg(
                            solver, duration, REPEATS)
                        append_samples(samples_file, point, "none",
                                       samples_none)
                    except Exception as exc:
                        best_dev = _failed(exc, label + " device-only")
                    if best_dev is None:
                        device_breached = True
                        best_dev = float("nan")
                        print(f"WATCHDOG {label} device-only: run "
                              "exceeded the cap")
                if not np.isnan(best):
                    print(f"{n} ODE solves ({algorithm}, {mode}) "
                          f"completed in {best:.1f} ms ({best_dev:.1f} ms "
                          "without transfers)")
                leg.record_times(n, best, best_dev, pct, samples_both,
                                 samples_none)
                if finals is not None:
                    save_numerical(finals, opts["numerical_tag"]
                                   + numerical_names[(mode, algorithm)])
                gc.collect()
            _release(solver)


def _leg_settings(problem, algorithm, mode):
    """The settings a leg is optimised at: the timing setting, plus every wp setting for per-point families."""
    from wp_common import TOLS, dts_for
    settings = [adapter.timing_setting(problem, mode)[1]]
    if adapter.per_point(algorithm):
        settings += (dts_for(algorithm, problem) if mode == "fixed"
                     else list(TOLS))
    return settings


def _optimize_legs(opts, problems):
    """Every (problem, mode, algorithm, setting) to optimise, in sweep order."""
    legs = []
    for problem in problems:
        for algorithm in opts["algorithms"]:
            if not problem.supports(opts["framework"]):
                continue
            for mode in ("fixed", "adaptive"):
                if algorithm not in opts[mode]:
                    continue
                for setting in _leg_settings(problem, algorithm, mode):
                    legs.append((problem, mode, algorithm, setting))
    return legs


def _run_optimize(opts, problems):
    """Solver.optimize on every leg that has no recorded row; the winner is recorded for the sweeps."""
    from timeit import default_timer
    from protocol import OPTIMIZE_N

    package, key = opts["framework"], opts["dataset_key"]
    systems = {}

    def system_for(problem):
        if problem.name not in systems:
            systems[problem.name] = adapter.build_system(problem, package,
                                                         PRECISION)
        return systems[problem.name]

    status = 0
    for problem, mode, algorithm, setting in _optimize_legs(opts, problems):
        label = (f"{problem.name} {mode} {algorithm} "
                 f"{'dt' if mode == 'fixed' else 'tol'}={setting:g}")
        if adapter.load_optimized(package, key, problem, algorithm, mode,
                                  setting) is not None:
            print(f"-- optimize: {label} recorded; skipping")
            continue
        solver = None
        started = default_timer()
        try:
            system, conditions = system_for(problem)
            solver = adapter.make_solver(system, problem, algorithm, mode,
                                         setting, optimized=False)
            initials, params = solver.build_grid(
                initial_values=conditions,
                parameters=sweep_parameters(problem, OPTIMIZE_N, PRECISION))
            row = adapter.optimize_point(solver, problem, initials, params,
                                         package, key, algorithm, mode,
                                         setting)
            print("optimized {0}: {1} in {2:.1f}s".format(
                label, row["label"], default_timer() - started), flush=True)
        except Exception as exc:
            _failed(exc, "optimize {0}".format(label))
            status = 1
        if solver is not None:
            _release(solver)
    return status


def _warm_legs(opts, problems):
    """Every (problem, mode, algorithm, setting) compile task, in the order the shard children share."""
    from wp_common import TOLS

    legs = []
    for problem in problems:
        for algorithm in opts["algorithms"]:
            if not problem.supports(opts["framework"]):
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
    """Compile each leg once at a tiny ensemble, striped across BENCH_WARM_JOBS shard children recycled every WARM_RECYCLE legs."""
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
            systems[name] = adapter.build_system(rows[name],
                                                 opts["framework"], PRECISION)
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
            solver = _make_solver(opts, system, row, algorithm, mode, setting)
            initials, params = solver.build_grid(
                initial_values=conditions,
                parameters=sweep_parameters(row, 64, PRECISION))
            adapter.solve(solver, initials, params, row["duration"])
            print("warmed {0} in {1:.1f}s".format(
                label, default_timer() - started), flush=True)
        except Exception as exc:
            _failed(exc, "warm {0}".format(label))
        if solver is not None:
            _release(solver)


def _run_states(opts):
    """Runtime-by-states sweep: lorenz96 resized along STATES_GRID, built cold, optimised, then timed at one ensemble size."""
    import tempfile
    from timeit import default_timer

    from cubie.cache_root import set_cache_root
    from problems import STATES_PROBLEM, states_row
    from wp_common import (STATES_N, append_samples, reset_samples,
                           sample_point, samples_outfile, timed_min_ms)

    # Throwaway cache root: every states compile runs cold.
    set_cache_root(tempfile.mkdtemp(prefix="cubie_states_"))

    n = STATES_N
    grid = opts["ns"]
    package, key = opts["framework"], opts["dataset_key"]
    systems = {}

    def system_for(nstates):
        if nstates not in systems:
            systems[nstates] = adapter.build_system(
                STATES_PROBLEM, package, PRECISION, states=nstates)
        return systems[nstates]

    for algorithm in opts["algorithms"]:
        for mode in ("fixed", "adaptive"):
            if algorithm not in opts[mode]:
                continue
            leg = Leg(opts["framework"], opts["dataset_key"], "states",
                      STATES_PROBLEM, algorithm, mode)
            run_grid = [s for s in grid if not skip_point(leg, n, s)]
            if not run_grid:
                print(f"-- resume: skipping states {mode} {algorithm} "
                      "(already covered)")
                continue
            samples_file = samples_outfile(
                opts["framework_dir"], opts["prefix"], "states", mode,
                algorithm, opts["dataset_key"], STATES_PROBLEM)
            # A resumed or --floor leg appends to what earlier runs recorded.
            if not (resume_active() or floor_enabled()):
                reset_samples(samples_file)
            # A device-only breach abandons that column alone.
            device_breached = False
            for index, nstates in enumerate(run_grid):
                row = states_row(nstates)
                duration = row["duration"]
                print(f"Running lorenz96 states={nstates}, "
                      f"{n} trajectories, {mode} dt, {algorithm}...")
                label = (f"lorenz96 states={nstates} {mode} {algorithm} "
                         f"N={n}")
                point = sample_point("states", STATES_PROBLEM,
                                     algorithm, mode, n, nstates)
                t_ms = t_dev = build_s = float("nan")
                pct = 100.0
                breached = False
                solver = None
                samples_both = samples_none = None
                try:
                    started = default_timer()
                    system, initial_conditions = system_for(nstates)
                    solver = adapter.make_solver(system, row, algorithm, mode,
                                                 states=nstates,
                                                 optimized=False)
                    initials_array, parameter_array = solver.build_grid(
                        initial_values=initial_conditions,
                        parameters=sweep_parameters(row, n, PRECISION))

                    def with_transfers():
                        return adapter.solve(solver, initials_array,
                                             parameter_array, duration)

                    with_transfers()
                    build_s = default_timer() - started
                    # The cold build is timed above; the sweep times the optimised kernel.
                    setting = adapter.timing_setting(row, mode)[1]
                    if adapter.load_optimized(package, key, row, algorithm,
                                              mode, setting,
                                              states=nstates) is None:
                        adapter.optimize_point(
                            solver, row, initials_array, parameter_array,
                            package, key, algorithm, mode, setting,
                            states=nstates)
                    else:
                        _release(solver)
                        solver = adapter.make_solver(
                            system, row, algorithm, mode, package=package,
                            key=key, states=nstates)
                        initials_array, parameter_array = solver.build_grid(
                            initial_values=initial_conditions,
                            parameters=sweep_parameters(row, n, PRECISION))
                    best, solution, samples_both = timed_min_ms(
                        with_transfers, REPEATS)
                    append_samples(samples_file, point, "both", samples_both)
                    breached = best is None
                    if not breached:
                        t_ms = best
                        pct = errored_pct(
                            final_states(system, solution, row))
                except Exception as exc:
                    _failed(exc, label)
                # The device leg runs only after a host-path time.
                if not np.isnan(t_ms):
                    if device_breached:
                        print(f"SKIP {label} device-only: breached at a "
                              "smaller size")
                    else:
                        try:
                            best_dev, samples_none = _device_leg(
                                solver, duration, REPEATS)
                            append_samples(samples_file, point, "none",
                                           samples_none)
                            if best_dev is None:
                                device_breached = True
                                print(f"WATCHDOG {label} device-only: "
                                      "run exceeded the cap")
                            else:
                                t_dev = best_dev
                        except Exception as exc:
                            _failed(exc, label + " device-only")
                    print(f"{n} ODE solves (lorenz96 "
                          f"states={nstates}, {algorithm}, {mode}) "
                          f"completed in {t_ms:.1f} ms ({t_dev:.1f} "
                          "ms without transfers)")
                leg.record_times(n, t_ms, t_dev, pct, samples_both,
                                 samples_none, build_s=build_s, states=nstates)
                if solver is not None:
                    _release(solver)
                if breached:
                    # Larger systems are slower, so the leg is abandoned.
                    print(f"WATCHDOG {label}: run exceeded the cap")
                    leg.nan_states(run_grid[index + 1:])
                    break


def run(argv, package):
    """Entry point: select the backend, parse the CLI and run every requested problem."""
    adapter.select_backend(package)
    from cubie.time_logger import default_timelogger
    default_timelogger.set_verbosity(None)

    argv = list(argv)
    warm_shard = None
    if "--warm-shard" in argv:
        position = argv.index("--warm-shard")
        warm_shard = tuple(int(t) for t in argv[position + 1].split("/"))
        del argv[position:position + 2]

    ns, analysis, algorithms, problems = parse_bench_args(argv, package)
    if not problems:
        print("{0} runs none of the requested problems; skipping."
              .format(package))
        return 0
    opts = {
        "ns": ns,
        "analysis": analysis,
        "framework": package,
        "algorithms": algorithms,
        "framework_dir": adapter.DATA_DIRS[package],
        "prefix": {"cubie": "Cubie", "cubie_mlir": "Cubie_mlir"}[package],
        "numerical_tag": package,
        "fixed": supported_for(package, "fixed"),
        "adaptive": supported_for(package, "adaptive"),
        "dataset_key": dataset_key(),
        "warm_shard": warm_shard,
    }
    if analysis == "optimize":
        return _run_optimize(opts, problems)
    if analysis == "warm":
        _run_warm(opts, problems, argv)
        return 0
    if analysis == "states":
        from problems import STATES_PROBLEM
        if not any(p.name == STATES_PROBLEM for p in problems):
            print("{0} does not run {1}; skipping the states sweep."
                  .format(package, STATES_PROBLEM))
            return 0
        _run_states(opts)
        return 0
    for problem in problems:
        _run_problem(problem, opts)
    return 0
