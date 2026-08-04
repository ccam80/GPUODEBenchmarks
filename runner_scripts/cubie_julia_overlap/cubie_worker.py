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
from common import (  # noqa: E402 - suite-local bootstrap above
    ADAPTIVE_TOL, DT0, DT_MAX, DT_MIN, FAILURE_FIELDS, FIXED_DT, GOLDEN_NE,
    ANALYSES, GOLDEN_WP, METRIC_FIELDS, N_WP, TIMING_FIELDS, algorithms,
    append_csv, ensure_csv, finite_counts, phases_for, pi_controller,
    point_slug, profile_protocol, rmse, write_json,
)

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
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("-a", "--analysis", choices=ANALYSES + ("all",), default="all")
    parser.add_argument("-n", "--nmax", default="16777216")
    parser.add_argument("--from-n", type=int, default=0)
    parser.add_argument("--algorithm", default="all")
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


def make_system():
    return qb.create_ODE_system(
        """
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        """,
        states={"x": 1.0, "y": 0.0, "z": 0.0},
        parameters={"rho": 21.0},
        constants={"sigma": 10.0, "beta": 8.0 / 3.0},
        name="LorenzDirectOverlap",
        precision=np.float32,
    )


def rho_grid(kind, n):
    if kind == "numerical":
        return np.loadtxt(GOLDEN_NE, delimiter=",", usecols=(0,), dtype=np.float64)[:n]
    if kind == "work_precision":
        return np.linspace(0.0, 21.0, N_WP, dtype=np.float32)[:n]
    return np.linspace(0.0, 21.0, n, dtype=np.float32)


def make_solver(system, alias, mode, setting, order, tier):
    common = dict(algorithm=alias, save_every=1.0, output_types=["state"],
                  time_logging_level=None)
    if mode == "fixed":
        return qb.Solver(system, dt=setting, step_controller="fixed", **common)
    settings = {"dt": DT0, "dt_min": DT_MIN, "dt_max": DT_MAX,
                "atol": setting, "rtol": setting}
    controller = {} if tier == "default" else pi_controller(order)
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


def solve_once(solver, initials, parameters):
    """Time one solve including the h2d and d2h transfers."""
    sync()
    start = time.perf_counter()
    solution = solver.solve(initial_values=initials, parameters=parameters,
                            blocksize=64, duration=1.0)
    # solve() already returns host buffers; this is a host-side view.
    finals = solution.state[-1, :, :].T
    sync()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    if finals.ndim != 2 or finals.shape[1] != 3:
        raise ValueError("unexpected final-state shape {!r}".format(finals.shape))
    return finals, elapsed_ms


def solve_once_on_device(solver, d_initials, d_parameters):
    """Time one solve with neither transfer: device arrays in, results left there."""
    sync()
    start = time.perf_counter()
    solver.solve(initial_values=d_initials, parameters=d_parameters,
                 blocksize=64, duration=1.0, on_device=True)
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
        writer.writerow(["traj", "x", "y", "z"])
        for index, row in enumerate(finals):
            writer.writerow([index, repr(float(row[0])), repr(float(row[1])), repr(float(row[2]))])
    return relative.as_posix()


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    timing_file = ensure_csv(args.output / "cubie_timings.csv", TIMING_FIELDS)
    metric_file = ensure_csv(args.output / "cubie_metrics.csv", METRIC_FIELDS)
    failure_file = ensure_csv(args.output / "cubie_failures.csv", FAILURE_FIELDS)
    protocol = profile_protocol(args.profile, args.nmax, args.from_n)
    write_json(args.output / "cubie_metadata.json", {
        "framework": "cubie", "cubie_version": package_version(),
        "python": sys.version, "platform": platform.platform(),
        "profile": args.profile, "protocol": protocol,
    })
    system = make_system()
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
        alias, order = row["cubie_alias"], row["order"]
        for phase in phases:
            if phase == "performance":
                points = []
                for n in protocol["performance_ns"]:
                    points.extend([("fixed", "fixed", "dt", FIXED_DT, n),
                                   ("adaptive", "default", "tol", ADAPTIVE_TOL, n),
                                   ("adaptive", "pi", "tol", ADAPTIVE_TOL, n)])
                repeats = protocol["performance_repeats"]
            elif phase == "numerical":
                n = protocol["ne_n"]
                points = [("fixed", "fixed", "dt", dt, n) for dt in protocol["ne_dts"]]
                points += [("adaptive", tier, "tol", tol, n)
                           for tier in ("default", "pi") for tol in protocol["ne_tols"]]
                repeats = 1
            else:
                n = protocol["wp_n"]
                points = [("fixed", "fixed", "dt", dt, n) for dt in protocol["wp_dts"]]
                points += [("adaptive", tier, "tol", tol, n)
                           for tier in ("default", "pi") for tol in protocol["wp_tols"]]
                repeats = protocol["work_repeats"]

            for mode, tier, setting_kind, setting, n in points:
                try:
                    # Release the previous point before allocating this one.
                    solver = initials = params = finals = device_inputs = None
                    solver = make_solver(system, alias, mode, setting, order, tier)
                    initials, params = solver.build_grid(
                        initial_values={"x": 1.0, "y": 0.0, "z": 0.0},
                        parameters={"rho": rho_grid(phase, n)})
                    device_inputs = to_device_inputs(initials, params)
                    solve_once(solver, initials, params)  # JIT/allocation warmup
                    if device_inputs is not None:
                        solve_once_on_device(solver, *device_inputs)  # warmup
                    for sample in range(repeats):
                        finals, elapsed = solve_once(solver, initials, params)
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
                        append_csv(timing_file, TIMING_FIELDS, {
                            "framework": "cubie", "algorithm": alias, "phase": phase,
                            "mode": mode, "tier": tier, "transfers": "both", "n": n,
                            "setting_kind": setting_kind, "setting": setting,
                            "sample": sample, "time_ms": elapsed,
                        })
                        if device_inputs is not None:
                            append_csv(timing_file, TIMING_FIELDS, {
                                "framework": "cubie", "algorithm": alias,
                                "phase": phase, "mode": mode, "tier": tier,
                                "transfers": "none", "n": n,
                                "setting_kind": setting_kind, "setting": setting,
                                "sample": sample,
                                "time_ms": solve_once_on_device(solver, *device_inputs),
                            })
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
                        golden = (np.loadtxt(GOLDEN_NE, delimiter=",", usecols=(1, 2, 3))[:n]
                                  if phase == "numerical" else np.loadtxt(GOLDEN_WP, delimiter=",")[:n])
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
