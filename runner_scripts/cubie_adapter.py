"""Cubie backend, system naming, controller mappings and optimize store for every cubie suite; `cubie_adapter.py clear <package> <key> [algorithm] [problem]` drops optimize rows. A kernel's record serves every line of the kernel, and names the run that recorded it; nothing here compares sources or dates."""

import csv
import json
import os
import sys
from datetime import datetime, timezone

from problems import as_problem
from store import RUN_ENV, _Lock
from trials import shares_dt_optimize

BACKENDS = {"cubie": "numba-cuda", "cubie_mlir": "mlir"}
SYSTEM_SUFFIX = {"cubie": "", "cubie_mlir": "_mlir"}
PACKAGES = tuple(BACKENDS)

OPTIMIZE_FIELDS = ("package", "key", "problem", "states", "precision", "algorithm", "controller",
                   "gains", "stepping", "n", "duration", "label", "best_ms", "blocksize",
                   "resident_blocks", "settings", "run", "recorded_utc")
# The stepping values that compile into a cubie kernel.
STEPPING_FIELDS = ("dt", "dt_min", "dt_max", "atol", "rtol", "newton_atol", "newton_rtol")

# Controller keys a caller may pass; everything else in a defaults table configures the step.
CONTROLLER_KEYS = ("step_controller", "integral_gain", "proportional_gain",
                   "derivative_gain", "safety", "min_step_shrink",
                   "max_step_growth")


# ------------------------------------------------------------------ backend

def select_backend(package):
    """Set CUBIE_CUDA_BACKEND for the package; cubie must not already be imported on another backend."""
    backend = BACKENDS[package]
    loaded = sys.modules.get("cubie.cuda_backend")
    if loaded is not None and loaded.CUDA_BACKEND != backend:
        raise RuntimeError("cubie is already imported on backend {0}; {1} "
                           "needs {2}".format(loaded.CUDA_BACKEND, package,
                                              backend))
    os.environ["CUBIE_CUDA_BACKEND"] = backend
    return backend


def package_for_backend(backend):
    for package, name in BACKENDS.items():
        if name == backend:
            return package
    raise KeyError(backend)


def build_system(problem, package, precision=None, states=None):
    """(system, initial_values) named for the package so both suites share one generated-code cache per backend."""
    import numpy as np
    from cubie_systems import build_system as build
    row = as_problem(problem)
    suffix = SYSTEM_SUFFIX[package]
    if states is not None:
        row = row.resized(states)
        suffix = "{0}_s{1}".format(suffix, states)
    return build(row, np.float32 if precision is None else precision,
                 name_suffix=suffix)


# -------------------------------------------------------------- controllers

# The gains a mapped controller carries, rounded so a set declares one spelling whatever Float64 residue the arithmetic leaves.
GAIN_DIGITS = 12


def julia_controller(constants, order):
    """Cubie settings reproducing Julia's default controller for an algorithm, or None when Julia has none that maps; cubie's PI exponent (I + P) / (2 (order + 1)) on the squared norm matches Julia's beta1 on the norm and P / (order + 1) its beta2, and its Gustafsson controller Julia's PredictiveController."""
    if constants is None:
        return None
    if constants["controller"] == "PIController":
        proportional = constants["beta2"] * (order + 1)
        settings = {
            "step_controller": "pi",
            "integral_gain": constants["beta1"] * (order + 1) - proportional,
            "proportional_gain": proportional,
            "safety": constants["gamma"],
            "min_step_shrink": constants["qmin"],
            "max_step_growth": constants["qmax"],
        }
    elif constants["controller"] == "PredictiveController":
        settings = {
            "step_controller": "gustafsson",
            "safety": constants["gamma"],
            "min_step_shrink": constants["qmin"],
            "max_step_growth": constants["qmax"],
        }
    else:
        return None
    return {key: value if key == "step_controller" else float(round(value, GAIN_DIGITS))
            for key, value in settings.items()}


# ------------------------------------------------------------------ solvers

def solve(solver, initial_values, parameters, duration, on_device=False):
    """One solve at the solver's own launch geometry, keeping the states of failed runs beside their status codes; a device solve is synchronised before returning."""
    result = solver.solve(initial_values=initial_values,
                          parameters=parameters, duration=duration,
                          nan_error_trajectories=False, on_device=on_device)
    if on_device:
        result.stream.synchronize()
    return result


# ------------------------------------------------------------------ optimize

def optimize_path(package, key, root=None):
    """data/key=<key>/package=<pkg>/optimize.csv; the directory is created."""
    directory = os.path.join(root or "data", "key=" + key, "package=" + package)
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, "optimize.csv")


def _text(value):
    """A float as row text; NaN as ''."""
    return "" if value is None or value != value else "{0:.10g}".format(float(value))


def stepping_text(trial):
    """The stepping values of a trial as one text: dt, dt_min, dt_max, atol, rtol, newton_atol, newton_rtol; dt is blank where shares_dt_optimize."""
    shared = shares_dt_optimize(trial)
    return ";".join("{0}={1}".format(field, "" if shared and field == "dt" else _text(trial[field]))
                    for field in STEPPING_FIELDS)


def kernel_ident(trial, key):
    """The optimize row identity of a trial's kernel: package, key, problem, states, precision, algorithm, controller, gains and stepping; every line of a kernel shares its record."""
    params = json.loads(trial["system_params"]) if trial["system_params"] else {}
    states = params.get("states", as_problem(trial["problem"])["states"])
    return {"package": trial["package"], "key": key, "problem": as_problem(trial["problem"]).name,
            "states": str(int(states)), "precision": trial["precision"], "algorithm": trial["algorithm"],
            "controller": trial["controller"] or "", "gains": trial["gains"] or "",
            "stepping": stepping_text(trial)}


def _same(row, ident):
    return all(str(row.get(field, "")) == str(value) for field, value in ident.items())


def _load(path):
    """The kernel rows of a store file; a row whose `per` column reads solve is skipped."""
    if not os.path.isfile(path):
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle)
                if row.get("package") and row.get("per", "") != "solve"]


def _save(path, rows):
    scratch = path + ".partial"
    with open(scratch, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OPTIMIZE_FIELDS,
                                lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(scratch, path)


def _encode(settings):
    from enum import Enum
    return json.dumps({k: (v.name if isinstance(v, Enum) else v)
                       for k, v in settings.items()}, sort_keys=True)


def _decode(text):
    settings = json.loads(text) if text else {}
    if any(k.startswith("unroll_") for k in settings):
        from cubie.CUDAFactory import UnrollChoice
        settings = {k: (UnrollChoice[v] if k.startswith("unroll_") else v)
                    for k, v in settings.items()}
    return settings


def optimize_rows(package, key, root=None):
    """Every optimize.csv row of a package under a key."""
    path = optimize_path(package, key, root)
    with _Lock(path):
        return _load(path)


def find_optimized(rows, ident):
    """The last row matching every field of an identity; None when there is none."""
    matched = [row for row in rows if _same(row, ident)]
    return matched[-1] if matched else None


def load_optimized(trial, key, root=None, run=None):
    """{'settings', 'resident_blocks'} recorded for a line's kernel, or None; with `run`, a record another run wrote is None."""
    row = find_optimized(optimize_rows(trial["package"], key, root), kernel_ident(trial, key))
    if row is None or row.get("label") == "timeout" or (run is not None and row.get("run", "") != run):
        return None
    resident = row.get("resident_blocks", "")
    return {"settings": _decode(row["settings"]),
            "resident_blocks": int(resident) if resident else None}


def apply_optimized(solver, tuned):
    """Apply a recorded point's settings and resident block count to a solver."""
    if tuned["settings"]:
        solver.update(tuned["settings"])
    kernel = getattr(solver, "kernel", None)
    if tuned["resident_blocks"] is not None and kernel is not None:
        kernel.resident_blocks = tuned["resident_blocks"]


def _replace(trial, key, root, row):
    path = optimize_path(trial["package"], key, root)
    ident = kernel_ident(trial, key)
    with _Lock(path):
        rows = [r for r in _load(path) if not _same(r, ident)]
        rows.append(row)
        _save(path, rows)
    return row


def _stamp():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def record_optimized(trial, key, result, root=None):
    """Replace the optimize row of a line's kernel with the result's best launch, on the runs and duration the result timed."""
    best = result.best
    row = dict(kernel_ident(trial, key), n=str(int(result.runs)), duration=_text(result.duration),
               label=best.label, best_ms="{0:.6g}".format(best.best_ms), blocksize=str(best.blocksize),
               resident_blocks="" if best.resident_blocks is None else str(best.resident_blocks),
               settings=_encode(result.applied_settings), run=os.environ.get(RUN_ENV, ""), recorded_utc=_stamp())
    return _replace(trial, key, root, row)


def record_optimize_timeout(trial, key, root=None):
    """Replace the optimize row of a line's kernel with one labelled timeout and no settings."""
    row = dict(kernel_ident(trial, key), n=str(int(trial["n"])), duration="", label="timeout",
               best_ms="nan", blocksize="", resident_blocks="", settings="", run=os.environ.get(RUN_ENV, ""),
               recorded_utc=_stamp())
    return _replace(trial, key, root, row)


def optimize_point(solver, trial, initial_values, parameters, key, root=None, force=False, verbose=True):
    """Run Solver.optimize with cubie sizing the batch and duration itself (auto_size), apply the winner and record it for the line's kernel; `force` varies settings an earlier optimize applied."""
    result = solver.optimize(initial_values, parameters, duration=float(trial["duration"]), verbose=verbose,
                             force=force, auto_size=True)
    if result.best is None:
        raise RuntimeError("optimize timed no launch for {0} {1}".format(
            trial["problem"], trial["algorithm"]))
    return record_optimized(trial, key, result, root=root)


def clear_optimized(package, key, algorithm=None, problem=None, root=None):
    """Drop the optimize rows of an algorithm and problem; returns the count dropped."""
    path = optimize_path(package, key, root)
    ident = {"package": package, "key": key}
    if algorithm and algorithm != "all":
        ident["algorithm"] = algorithm
    if problem and problem != "all":
        ident["problem"] = as_problem(problem).name
    with _Lock(path):
        rows = _load(path)
        kept = [r for r in rows if not _same(r, ident)]
        if len(kept) != len(rows):
            _save(path, kept)
    return len(rows) - len(kept)


def _cli(argv):
    if len(argv) >= 3 and argv[0] == "clear":
        package, key = argv[1], argv[2]
        algorithm = argv[3] if len(argv) > 3 else None
        problem = argv[4] if len(argv) > 4 else None
        print(clear_optimized(package, key, algorithm, problem))
        return 0
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))
