"""Cubie backend, system naming, controller mappings and optimize store for every cubie suite; `cubie_adapter.py clear <package> <key> [algorithm] [problem]` drops optimize rows."""

import csv
import json
import os
import sys
from datetime import datetime, timezone

from problems import as_problem
from store import _Lock

BACKENDS = {"cubie": "numba-cuda", "cubie_mlir": "mlir"}
SYSTEM_SUFFIX = {"cubie": "", "cubie_mlir": "_mlir"}
PACKAGES = tuple(BACKENDS)

OPTIMIZE_FIELDS = ("package", "key", "problem", "algorithm", "mode",
                   "controller", "gains", "setting_kind", "setting", "states",
                   "n", "label", "best_ms", "blocksize", "resident_blocks",
                   "settings", "recorded_utc")

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

def _resolved(value, order):
    return float(value(order)) if callable(value) else value


def default_controller(alias, family, order):
    """Cubie's shipped controller settings for an algorithm; None when the family has no adaptive table."""
    from cubie.integrators.algorithms import (crank_nicolson, generic_dirk,
                                              generic_erk, generic_firk,
                                              generic_rosenbrock_w)
    tables = {
        "dirk": generic_dirk.DIRK_ADAPTIVE_DEFAULTS,
        "erk": generic_erk.ERK_ADAPTIVE_DEFAULTS,
        "firk": generic_firk.FIRK_ADAPTIVE_DEFAULTS,
        "rosenbrock": generic_rosenbrock_w.ROSENBROCK_ADAPTIVE_DEFAULTS,
    }
    if alias == "crank_nicolson":
        table = crank_nicolson.CN_DEFAULTS
    elif family in tables:
        table = tables[family]
    else:
        return None
    settings = {key: _resolved(value, order)
                for key, value in table.settings.items()
                if key in CONTROLLER_KEYS}
    gains = {"i": ("integral_gain",),
             "pi": ("integral_gain", "proportional_gain"),
             "pid": ("integral_gain", "proportional_gain",
                     "derivative_gain")}
    allowed = gains.get(settings.get("step_controller"), ())
    for key in ("integral_gain", "proportional_gain", "derivative_gain"):
        if key not in allowed:
            settings.pop(key, None)
    return settings


def pi_tier_controller(order):
    """The DIRK PI defaults, resolved for an order, applied to any family."""
    from cubie.integrators.algorithms import generic_dirk
    return {
        "step_controller": "pi",
        "integral_gain": generic_dirk.dirk_default_integral_gain(order),
        "proportional_gain": generic_dirk.dirk_default_proportional_gain(
            order),
        "safety": 0.9,
        "min_step_shrink": 0.2,
        "max_step_growth": 10.0,
    }


def matched_controller(constants, order):
    """Cubie settings reproducing Julia's resolved controller, or (None, reason); cubie's PI exponent (I + P) / (2 (order + 1)) on the squared norm matches Julia's beta1 on the norm."""
    if constants is None:
        return None, "no julia controller constants"
    if constants["controller"] == "PIController":
        proportional = constants["beta2"] * (order + 1)
        return {
            "step_controller": "pi",
            "integral_gain": constants["beta1"] * (order + 1) - proportional,
            "proportional_gain": proportional,
            "safety": constants["gamma"],
            "min_step_shrink": constants["qmin"],
            "max_step_growth": constants["qmax"],
        }, None
    if constants["controller"] == "PredictiveController":
        return {"step_controller": "gustafsson",
                "safety": constants["gamma"]}, None
    return None, "unmapped julia controller {0}".format(
        constants["controller"])


def controllers_equal(a, b, rel_tol=1e-9):
    """Same controller name and the same numeric keys within rel_tol."""
    import numpy as np
    if a is None or b is None:
        return False
    if a.get("step_controller") != b.get("step_controller"):
        return False
    keys = (set(a) | set(b)) - {"step_controller"}
    for key in keys:
        if key not in a or key not in b:
            return False
        if not np.isclose(float(a[key]), float(b[key]), rtol=rel_tol,
                          atol=0.0):
            return False
    return True


# ------------------------------------------------------------------ solvers

def solve(solver, initial_values, parameters, duration, on_device=False):
    """One solve at the solver's own launch geometry; a device solve is synchronised before returning."""
    result = solver.solve(initial_values=initial_values,
                          parameters=parameters, duration=duration,
                          on_device=on_device)
    if on_device:
        result.stream.synchronize()
    return result


# ------------------------------------------------------------------ optimize

def optimize_path(package, key, root=None):
    """data/key=<key>/package=<pkg>/optimize.csv; the directory is created."""
    directory = os.path.join(root or "data", "key=" + key, "package=" + package)
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, "optimize.csv")


def _ident(package, key, problem, algorithm, mode, setting, states,
           controller="", gains=""):
    """The optimize row identity; a row recorded without a setting serves every stepping of its leg. `controller` and `gains` separate the controllers one algorithm runs under."""
    row = as_problem(problem)
    kind = "dt" if mode == "fixed" else "tol"
    text = "" if setting is None else "{0:.10g}".format(float(setting))
    return {"package": package, "key": key, "problem": row.name,
            "algorithm": algorithm, "mode": mode,
            "controller": controller or "", "gains": gains or "",
            "setting_kind": kind, "setting": text,
            "states": str(int(row["states"] if states is None else states))}


def _setting_matches(a, b):
    """Two settings name the same stepping: both empty, or equal within a relative 1e-8."""
    import math
    if a == "" or b == "":
        return a == b
    return math.isclose(float(a), float(b), rel_tol=1e-8, abs_tol=0.0)


def _same(row, ident):
    for field, value in ident.items():
        if field == "setting":
            if not _setting_matches(row[field], value):
                return False
        elif str(row[field]) != str(value):
            return False
    return True


def _load(path):
    if not os.path.isfile(path):
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row.get("package")]


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
        from cubie.cuda_simsafe import UnrollChoice
        settings = {k: (UnrollChoice[v] if k.startswith("unroll_") else v)
                    for k, v in settings.items()}
    return settings


def load_optimized(package, key, problem, algorithm, mode, setting,
                   states=None, root=None, controller="", gains=""):
    """{'settings', 'resident_blocks'} recorded for a point, or None."""
    path = optimize_path(package, key, root)
    ident = _ident(package, key, problem, algorithm, mode, setting, states,
                   controller, gains)
    with _Lock(path):
        recorded = _load(path)
    rows = [row for row in recorded if _same(row, ident)]
    if not rows and ident["setting"]:
        rows = [row for row in recorded if _same(row, dict(ident, setting=""))]
    if not rows:
        return None
    row = rows[-1]
    resident = row.get("resident_blocks", "")
    return {"settings": _decode(row["settings"]),
            "resident_blocks": int(resident) if resident else None}


def record_optimized(package, key, problem, algorithm, mode, setting, result,
                     states=None, n=None, root=None, controller="", gains=""):
    """Replace the optimize row for a point with the result's best launch."""
    path = optimize_path(package, key, root)
    ident = _ident(package, key, problem, algorithm, mode, setting, states,
                   controller, gains)
    best = result.best
    row = dict(ident, n="" if n is None else str(int(n)), label=best.label,
               best_ms="{0:.6g}".format(best.best_ms),
               blocksize=str(best.blocksize),
               resident_blocks=("" if best.resident_blocks is None
                                else str(best.resident_blocks)),
               settings=_encode(result.applied_settings),
               recorded_utc=datetime.now(timezone.utc).strftime(
                   "%Y-%m-%dT%H:%M:%SZ"))
    with _Lock(path):
        rows = [r for r in _load(path) if not _same(r, ident)]
        rows.append(row)
        _save(path, rows)
    return row


def optimize_setting(trial):
    """(mode, setting) of the optimize row a trial records: None on an n or states axis, where one row serves the leg, else the trial's dt or tolerance."""
    mode = "fixed" if trial["controller"] == "fixed" else "adaptive"
    if trial["axis"] in ("n", "states"):
        return mode, None
    return mode, float(trial["dt"] if mode == "fixed" else trial["atol"])


def record_optimize_timeout(trial, key, root=None):
    """Replace the optimize row of a trial with one labelled timeout and no settings, so the leg runs at the solver's own geometry."""
    mode, setting = optimize_setting(trial)
    params = json.loads(trial["system_params"]) if trial["system_params"] else {}
    path = optimize_path(trial["package"], key, root)
    ident = _ident(trial["package"], key, trial["problem"], trial["algorithm"], mode, setting,
                   params.get("states"), trial["controller"], trial["gains"])
    row = dict(ident, n=str(int(trial["n"])), label="timeout", best_ms="nan", blocksize="",
               resident_blocks="", settings="",
               recorded_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    with _Lock(path):
        rows = [r for r in _load(path) if not _same(r, ident)]
        rows.append(row)
        _save(path, rows)
    return row


def optimize_point(solver, problem, initial_values, parameters, package, key,
                   algorithm, mode, setting, states=None, verbose=True,
                   root=None, force=False, controller="", gains=""):
    """Run Solver.optimize on the point's batch, apply the winner to the solver and record it; `force` varies settings an earlier optimize applied."""
    row = as_problem(problem)
    result = solver.optimize(initial_values, parameters,
                             duration=row["duration"], verbose=verbose,
                             force=force)
    if result.best is None:
        raise RuntimeError("optimize timed no launch for {0} {1} {2}".format(
            row.name, algorithm, mode))
    return record_optimized(package, key, row, algorithm, mode, setting,
                            result, states=states,
                            n=int(initial_values.shape[1]), root=root,
                            controller=controller, gains=gains)


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
