#!/usr/bin/env python3
"""Cubie GPU worker for the overlap suite: runs the requested batch, flushes each point as it lands, and records exceptions as failure rows."""

from __future__ import annotations

import argparse
import csv
import platform
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import cubie_adapter as adapter  # noqa: E402
from common import (  # noqa: E402 - suite-local bootstrap above
    ADAPTIVE_TOL, FAILURE_FIELDS, ANALYSES, METRIC_FIELDS, N_WP,
    TIMING_FIELDS, algorithms, append_csv, cubie_ne_adaptive_file,
    cubie_ne_file, ensure_csv, finite_counts, golden_ne_states, golden_wp,
    ne_sweep, phases_for, point_slug,
    protocol as suite_protocol,
    read_ne_csv, read_ne_adaptive_csv, rmse, timing_stats, write_json,
)
from bench_key import dataset_key  # noqa: E402
from wp_common import repeat_bounds, repeats_done  # noqa: E402
from cubie_systems import final_states  # noqa: E402
from problems import get_problem  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("-a", "--analysis", choices=ANALYSES + ("all",), default="all")
    parser.add_argument("-n", "--nmax", default="16777216")
    parser.add_argument("--from-n", type=int, default=0)
    parser.add_argument("--algorithm", default="all")
    parser.add_argument("--problem", default="lorenz")
    parser.add_argument("--package", choices=adapter.PACKAGES, default="cubie",
                        help="Cubie package: backend, system name and optimize rows.")
    return parser.parse_args()


def package_version():
    try:
        from importlib.metadata import version
        return version("cubie")
    except Exception:
        return "unknown"


def sweep_grid(problem, kind, n):
    """The ensemble parameter values for one phase."""
    if kind == "work_precision":
        return problem.sweep(N_WP, dtype=np.float32)[:n]
    if kind == "numerical":
        return ne_sweep(problem)[:n]
    return problem.sweep(n, dtype=np.float32)


def make_solver(system, problem, alias, mode, setting, order, tier, package,
                key):
    """One point's solver; the pi tier applies the DIRK PI defaults to every family."""
    controller = adapter.pi_tier_controller(order) if tier == "pi" else None
    return adapter.make_solver(system, problem, alias, mode, setting,
                               package=package, key=key,
                               controller=controller)


def solve_once(solver, initials, parameters, duration, nstates, system,
               problem):
    """Time one solve including the h2d and d2h transfers; solve() returns synchronised."""
    start = time.perf_counter()
    solution = adapter.solve(solver, initials, parameters, duration)
    # solve() already returns host buffers; this is a host-side view.
    finals = final_states(system, solution, problem)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    if finals.ndim != 2 or finals.shape[1] != nstates:
        raise ValueError("unexpected final-state shape {!r}".format(finals.shape))
    return finals, elapsed_ms


def solve_once_on_device(solver, d_initials, d_parameters, duration):
    """Time one solve with neither transfer: resident inputs in, results left on the device."""
    start = time.perf_counter()
    adapter.solve(solver, d_initials, d_parameters, duration, on_device=True)
    return (time.perf_counter() - start) * 1000.0


def time_device_leg(solver, duration, repeats):
    """Device-only samples on the resident inputs; repeats follow the first run's duration."""
    # Raises after a chunked host leg.
    d_initials = solver.device_initial_values
    d_parameters = solver.device_parameters
    samples = []
    floor = ceiling = None
    while True:
        elapsed = solve_once_on_device(solver, d_initials, d_parameters,
                                       duration)
        samples.append(elapsed)
        if floor is None:
            floor, ceiling = repeat_bounds(elapsed / 1000.0, repeats)
        if repeats_done(samples, floor, ceiling):
            return samples


def import_numerical_from_ne(output, alias, family, problem, metric_file,
                             failure, package):
    """Import the package's ne finals; erk rows import the adaptive default tier only."""
    key = dataset_key()
    golden = golden_ne_states(problem)
    sources = []
    if family != "erk":
        sources.append(("fixed", "fixed", "dt",
                        Path(cubie_ne_file(alias, key, problem, package))))
    sources.append(("adaptive", "default", "tol",
                    Path(cubie_ne_adaptive_file(alias, "default", key, problem,
                                                package))))
    for mode, tier, setting_kind, path in sources:
        if not path.is_file():
            failure(alias, "numerical", mode, tier, 0, setting_kind, "",
                    FileNotFoundError(
                        "{} not found - run `bench.py -a numerical -p {}` "
                        "first".format(path, package)))
            continue
        blocks = (read_ne_csv(path) if mode == "fixed"
                  else {tol: data[0]
                        for tol, data in read_ne_adaptive_csv(path).items()})
        for setting in sorted(blocks, reverse=True):
            finals = blocks[setting]
            finite, failed = finite_counts(finals)
            finals_path = write_finals(output, alias, mode, tier,
                                       setting_kind, setting, finals)
            append_csv(metric_file, METRIC_FIELDS, {
                "framework": "cubie", "algorithm": alias, "phase": "numerical",
                "mode": mode, "tier": tier, "n": len(finals),
                "setting_kind": setting_kind, "setting": setting,
                "golden_rmse": rmse(finals, golden[:len(finals)]),
                "finite_trajectories": finite, "failed_trajectories": failed,
                "finals_path": finals_path,
            })
            print("OK cubie {} numerical {} {} {}={} (imported from ne)"
                  .format(alias, mode, tier, setting_kind, setting),
                  flush=True)


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
    adapter.select_backend(args.package)
    from cubie.time_logger import default_timelogger
    default_timelogger.set_verbosity(None)
    key = dataset_key()
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
        "package": args.package, "backend": adapter.BACKENDS[args.package],
        "cubie_version": package_version(),
        "python": sys.version, "platform": platform.platform(),
        "protocol": protocol,
    })
    fixed_dt = problem.timing_dt
    system, initial_values = adapter.build_system(problem, args.package,
                                                  np.float32)
    phases = phases_for(args.analysis)
    point_failure_count = 0

    def failure(algorithm, phase, mode, tier, n, setting_kind, setting, exc,
                leg=""):
        """One failure row; `leg` names the timed leg when only it failed."""
        nonlocal point_failure_count
        point_failure_count += 1
        message = (leg + ": " if leg else "") + str(exc)
        append_csv(failure_file, FAILURE_FIELDS, {
            "framework": "cubie", "algorithm": algorithm, "phase": phase,
            "mode": mode, "tier": tier, "n": n, "setting_kind": setting_kind,
            "setting": setting, "error_type": type(exc).__name__,
            "message": message.replace("\n", " ")[:2000],
        })
        print("FAILED cubie {} {} {} {}={}: {}".format(algorithm, phase, mode, setting_kind, setting, message), flush=True)

    for row in algorithms(args.algorithm):
        alias, order, family = row["algorithm"], row["order"], row["family"]
        # Skip the pi tier when it resolves to cubie's shipped defaults.
        shipped = adapter.default_controller(alias, family, order)
        if adapter.controllers_equal(adapter.pi_tier_controller(order),
                                     shipped):
            adaptive_tiers = ("default",)
            print("cubie {}: pi tier equals the shipped defaults; skipped"
                  .format(alias), flush=True)
        else:
            adaptive_tiers = ("default", "pi")
        for phase in phases:
            if phase == "numerical":
                # The cubie side comes from the NE suite's outputs.
                import_numerical_from_ne(args.output, alias, family, problem,
                                         metric_file, failure, args.package)
                continue
            if phase == "performance":
                points = []
                for n in protocol["performance_ns"]:
                    points.append(("fixed", "fixed", "dt", fixed_dt, n))
                    points.extend([("adaptive", tier, "tol", ADAPTIVE_TOL, n)
                                   for tier in adaptive_tiers])
                repeats = protocol["performance_repeats"]
            else:
                n = protocol["wp_n"]
                points = [("fixed", "fixed", "dt", dt * duration, n)
                          for dt in protocol["wp_dts"]]
                points += [("adaptive", tier, "tol", tol, n)
                           for tier in adaptive_tiers for tol in protocol["wp_tols"]]
                repeats = protocol["work_repeats"]

            for mode, tier, setting_kind, setting, n in points:
                point = {"framework": "cubie", "algorithm": alias,
                         "phase": phase, "mode": mode, "tier": tier, "n": n,
                         "setting_kind": setting_kind, "setting": setting}
                try:
                    # Release the previous point before allocating this one.
                    solver = initials = params = finals = None
                    solver = make_solver(system, problem, alias, mode, setting,
                                         order, tier, args.package, key)
                    initials, params = solver.build_grid(
                        initial_values=initial_values,
                        parameters={problem["sweep_parameter"]:
                                    sweep_grid(problem, phase, n)})
                    # One warmup carries the compile for both transfer paths.
                    solve_once(solver, initials, params, duration, nstates,
                               system, problem)
                    # Unbroken block per transfer variant; repeats follow the first timed run's duration.
                    end_to_end = []
                    floor = ceiling = None
                    while True:
                        finals, elapsed = solve_once(
                            solver, initials, params, duration, nstates,
                            system, problem)
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
                        if floor is None:
                            floor, ceiling = repeat_bounds(elapsed / 1000.0,
                                                           repeats)
                        if repeats_done(end_to_end, floor, ceiling):
                            break
                    append_csv(timing_file, TIMING_FIELDS,
                               dict(point, transfers="both",
                                    **timing_stats(end_to_end)))
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
                        golden = np.loadtxt(golden_wp(problem),
                                            delimiter=",")[:n]
                        append_csv(metric_file, METRIC_FIELDS, {
                            "framework": "cubie", "algorithm": alias, "phase": phase,
                            "mode": mode, "tier": tier, "n": n,
                            "setting_kind": setting_kind, "setting": setting,
                            "golden_rmse": rmse(finals, golden),
                            "finite_trajectories": finite, "failed_trajectories": failed,
                            "finals_path": "",
                        })
                    print("OK cubie {} {} {} {} {}={} N={}".format(alias, phase, mode, tier, setting_kind, setting, n), flush=True)
                except Exception as exc:
                    failure(alias, phase, mode, tier, n, setting_kind, setting, exc)
                    continue
                # A device-only failure leaves the end-to-end row standing.
                try:
                    device_only = time_device_leg(solver, duration, repeats)
                    append_csv(timing_file, TIMING_FIELDS,
                               dict(point, transfers="none",
                                    **timing_stats(device_only)))
                except Exception as exc:
                    failure(alias, phase, mode, tier, n, setting_kind, setting,
                            exc, leg="device-only")

    return 1 if point_failure_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
