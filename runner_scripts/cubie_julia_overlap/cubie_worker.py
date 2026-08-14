#!/usr/bin/env python3
"""Cubie GPU worker for the direct overlap suite.

The worker owns no orchestration policy.  It executes the requested batch,
flushes every successful point immediately, and converts exceptions into
failure rows so one bad algorithm/setting cannot erase the rest of the run.
"""

from __future__ import annotations

import argparse
import csv
import platform
import sys
import time
from pathlib import Path

import numpy as np
import cubie as qb

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from common import (  # noqa: E402 - suite-local bootstrap above
    ADAPTIVE_TOL, FAILURE_FIELDS, golden_ne, scaled_dts,
    ANALYSES, golden_wp, METRIC_FIELDS, N_WP, TIMING_FIELDS, algorithms,
    append_csv, ensure_csv, finite_counts, phases_for, pi_controller,
    point_slug, protocol as suite_protocol, rmse, timing_stats, write_json,
)
from cubie_systems import build_system  # noqa: E402
from problems import get_problem  # noqa: E402

try:
    from cubie.time_logger import default_timelogger
    default_timelogger.set_verbosity(None)
except Exception:
    pass

try:
    from numba import cuda
except Exception:
    cuda = None


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("-a", "--analysis", choices=ANALYSES + ("all",), default="all")
    parser.add_argument("-n", "--nmax", default="16777216")
    parser.add_argument("--from-n", type=int, default=0)
    parser.add_argument("--algorithm", default="all")
    parser.add_argument("--problem", default="lorenz")
    return parser.parse_args()


def sync():
    if cuda is not None:
        cuda.synchronize()


def package_version():
    try:
        from importlib.metadata import version
        return version("cubie")
    except Exception:
        return "unknown"


def make_system(problem):
    return build_system(problem, np.float32, name_suffix="DirectOverlap")


def sweep_grid(problem, kind, n):
    """The ensemble parameter values for one phase."""
    if kind == "numerical":
        return np.loadtxt(golden_ne(problem), delimiter=",", usecols=(0,),
                          dtype=np.float64)[:n]
    if kind == "work_precision":
        return problem.sweep(N_WP, dtype=np.float32)[:n]
    return problem.sweep(n, dtype=np.float32)


def make_solver(system, alias, mode, setting, order, family, tier, pins):
    duration, _, dt0, dt_min, dt_max = pins
    common = dict(algorithm=alias, save_every=duration, output_types=["state"],
                  time_logging_level=None)
    if mode == "fixed":
        return qb.Solver(system, dt=setting, step_controller="fixed", **common)
    settings = {"dt": dt0, "dt_min": dt_min, "dt_max": dt_max,
                "atol": setting, "rtol": setting}
    controller = {} if tier == "default" else pi_controller(order, family)
    if controller:
        settings["step_controller"] = controller.pop("step_controller")
    solver = qb.Solver(system, **settings, **common)
    if controller:
        result = solver.update(controller, silent=True)
        if result is None or not isinstance(result, (set, list, tuple, dict)):
            raise TypeError("Solver.update must return recognized setting names; got {}"
                            .format(type(result).__name__))
        recognised = set(result)
        ignored = set(controller) - recognised
        if ignored:
            raise ValueError("Cubie ignored PI controller settings: " + ", ".join(sorted(ignored)))
    return solver


def solve_once(solver, initials, parameters, duration, nstates):
    """Time one solve including the h2d and d2h transfers."""
    sync()
    start = time.perf_counter()
    solution = solver.solve(initial_values=initials, parameters=parameters,
                            blocksize=64, duration=duration)
    # solve() already returns host buffers; this is a host-side view.
    finals = solution.state[-1, :, :].T
    sync()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    if finals.ndim != 2 or finals.shape[1] != nstates:
        raise ValueError("unexpected final-state shape {!r}".format(finals.shape))
    return finals, elapsed_ms


def solve_once_on_device(solver, d_initials, d_parameters, duration):
    """Time one solve with neither transfer: device arrays in, results left there."""
    sync()
    start = time.perf_counter()
    solver.solve(initial_values=d_initials, parameters=d_parameters,
                 blocksize=64, duration=duration, on_device=True)
    sync()
    return (time.perf_counter() - start) * 1000.0


def to_device_inputs(initials, parameters):
    """Upload the grid once; None when no CUDA handle is available."""
    if cuda is None:
        return None
    return cuda.to_device(initials), cuda.to_device(parameters)


def write_finals(root, algorithm, mode, tier, setting_kind, setting, finals):
    relative = Path("finals") / "cubie" / algorithm / (
        "{}_{}_{}_{}.csv".format(mode, tier, setting_kind, point_slug(setting)))
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["traj"] + ["s{0}".format(s + 1)
                                    for s in range(finals.shape[1])])
        for index, row in enumerate(finals):
            writer.writerow([index] + [repr(float(v)) for v in row])
    return relative.as_posix()


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    timing_file = ensure_csv(args.output / "cubie_timings.csv", TIMING_FIELDS)
    metric_file = ensure_csv(args.output / "cubie_metrics.csv", METRIC_FIELDS)
    failure_file = ensure_csv(args.output / "cubie_failures.csv", FAILURE_FIELDS)
    protocol = suite_protocol(args.nmax, args.from_n)
    problem = get_problem(args.problem)
    duration = problem["duration"]
    nstates = problem["states"]
    write_json(args.output / "cubie_metadata.json", {
        "framework": "cubie", "problem": problem["problem"],
        "cubie_version": package_version(),
        "python": sys.version, "platform": platform.platform(),
        "protocol": protocol,
    })
    fixed_dt, dt0, dt_min, dt_max = scaled_dts(problem)
    pins = (duration, fixed_dt, dt0, dt_min, dt_max)
    system, initial_values = make_system(problem)
    phases = phases_for(args.analysis)
    point_failure_count = 0

    def failure(algorithm, phase, mode, tier, n, setting_kind, setting, exc):
        nonlocal point_failure_count
        point_failure_count += 1
        append_csv(failure_file, FAILURE_FIELDS, {
            "framework": "cubie", "algorithm": algorithm, "phase": phase,
            "mode": mode, "tier": tier, "n": n, "setting_kind": setting_kind,
            "setting": setting, "error_type": type(exc).__name__,
            "message": str(exc).replace("\n", " ")[:2000],
        })
        print("FAILED cubie {} {} {} {}={}: {}".format(algorithm, phase, mode, setting_kind, setting, exc), flush=True)

    for row in algorithms(args.algorithm):
        alias, order, family = row["cubie_alias"], row["order"], row["family"]
        for phase in phases:
            if phase == "performance":
                points = []
                for n in protocol["performance_ns"]:
                    points.extend([("fixed", "fixed", "dt", fixed_dt, n),
                                   ("adaptive", "default", "tol", ADAPTIVE_TOL, n),
                                   ("adaptive", "pi", "tol", ADAPTIVE_TOL, n)])
                repeats = protocol["performance_repeats"]
            elif phase == "numerical":
                n = protocol["ne_n"]
                points = [("fixed", "fixed", "dt", dt * duration, n)
                          for dt in protocol["ne_dts"]]
                points += [("adaptive", tier, "tol", tol, n)
                           for tier in ("default", "pi") for tol in protocol["ne_tols"]]
                repeats = 1
            else:
                n = protocol["wp_n"]
                points = [("fixed", "fixed", "dt", dt * duration, n)
                          for dt in protocol["wp_dts"]]
                points += [("adaptive", tier, "tol", tol, n)
                           for tier in ("default", "pi") for tol in protocol["wp_tols"]]
                repeats = protocol["work_repeats"]

            for mode, tier, setting_kind, setting, n in points:
                try:
                    # Release the previous point before allocating this one.
                    solver = initials = params = finals = device_inputs = None
                    solver = make_solver(system, alias, mode, setting, order,
                                         family, tier, pins)
                    initials, params = solver.build_grid(
                        initial_values=initial_values,
                        parameters={problem["sweep_parameter"]:
                                    sweep_grid(problem, phase, n)})
                    device_inputs = to_device_inputs(initials, params)
                    solve_once(solver, initials, params, duration, nstates)  # JIT/allocation warmup
                    if device_inputs is not None:
                        solve_once_on_device(solver, *device_inputs, duration)  # warmup
                    # Each transfer variant runs as an unbroken block, so one
                    # variant's samples are never separated by the other's
                    # allocation and transfer traffic.
                    end_to_end = []
                    for _ in range(repeats):
                        finals, elapsed = solve_once(solver, initials, params, duration, nstates)
                        finite, failed = finite_counts(finals)
                        if failed or finite != n:
                            append_csv(metric_file, METRIC_FIELDS, {
                                "framework": "cubie", "algorithm": alias,
                                "phase": phase, "mode": mode, "tier": tier,
                                "n": n, "setting_kind": setting_kind,
                                "setting": setting, "golden_rmse": "",
                                "finite_trajectories": finite,
                                "failed_trajectories": failed,
                                "finals_path": "",
                            })
                            raise FloatingPointError(
                                "non-finite result: {}/{} trajectories valid"
                                .format(finite, n))
                        end_to_end.append(elapsed)
                    device_only = ([solve_once_on_device(solver, *device_inputs, duration)
                                    for _ in range(repeats)]
                                   if device_inputs is not None else [])
                    point = {"framework": "cubie", "algorithm": alias,
                             "phase": phase, "mode": mode, "tier": tier, "n": n,
                             "setting_kind": setting_kind, "setting": setting}
                    for transfers, samples in (("both", end_to_end),
                                               ("none", device_only)):
                        if samples:
                            append_csv(timing_file, TIMING_FIELDS,
                                       dict(point, transfers=transfers,
                                            **timing_stats(samples)))
                    finite, failed = finite_counts(finals)
                    if phase == "performance":
                        append_csv(metric_file, METRIC_FIELDS, {
                            "framework": "cubie", "algorithm": alias,
                            "phase": phase, "mode": mode, "tier": tier,
                            "n": n, "setting_kind": setting_kind,
                            "setting": setting, "golden_rmse": "",
                            "finite_trajectories": finite,
                            "failed_trajectories": failed,
                            "finals_path": "",
                        })
                    else:
                        golden = (np.loadtxt(
                            golden_ne(problem), delimiter=",",
                            usecols=tuple(range(1, nstates + 1)))[:n]
                            if phase == "numerical"
                            else np.loadtxt(golden_wp(problem),
                                            delimiter=",")[:n])
                        finals_path = (write_finals(args.output, alias, mode, tier,
                                                   setting_kind, setting, finals)
                                       if phase == "numerical" else "")
                        append_csv(metric_file, METRIC_FIELDS, {
                            "framework": "cubie", "algorithm": alias, "phase": phase,
                            "mode": mode, "tier": tier, "n": n,
                            "setting_kind": setting_kind, "setting": setting,
                            "golden_rmse": rmse(finals, golden),
                            "finite_trajectories": finite, "failed_trajectories": failed,
                            "finals_path": finals_path,
                        })
                    print("OK cubie {} {} {} {} {}={} N={}".format(alias, phase, mode, tier, setting_kind, setting, n), flush=True)
                except Exception as exc:
                    failure(alias, phase, mode, tier, n, setting_kind, setting, exc)

    return 1 if point_failure_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
