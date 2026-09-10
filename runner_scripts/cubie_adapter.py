"""Cubie backend, system naming, controller mappings and optimize store for every cubie suite; `cubie_adapter.py clear <package> <key> [algorithm] [problem]` drops optimize rows, `cubie_adapter.py sources <package> <systems.json>` prints the source hash of each listed system under the package's backend."""

import csv
import json
import os
import subprocess
import sys
import timeit
from datetime import datetime, timezone

from problems import as_problem
from protocol import OPTIMIZE_LAUNCH_MS, OPTIMIZE_WAVES
from store import _Lock

BACKENDS = {"cubie": "numba-cuda", "cubie_mlir": "mlir"}
SYSTEM_SUFFIX = {"cubie": "", "cubie_mlir": "_mlir"}
PACKAGES = tuple(BACKENDS)

OPTIMIZE_FIELDS = ("package", "key", "problem", "states", "precision", "algorithm", "controller",
                   "gains", "stepping", "per", "n", "duration", "source", "label", "best_ms", "blocksize",
                   "resident_blocks", "settings", "recorded_utc")
# The stepping values that compile into a cubie kernel.
STEPPING_FIELDS = ("dt", "dt_min", "dt_max", "atol", "rtol", "newton_atol", "newton_rtol")
# Fractions of the problem duration the optimize-duration probes solve, shortest first.
PROBE_FRACTIONS = (0.01, 0.1)

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


def _text(value):
    """A float as row text; NaN as ''."""
    return "" if value is None or value != value else "{0:.10g}".format(float(value))


def stepping_text(trial):
    """The stepping values of a trial as one text: dt, dt_min, dt_max, atol, rtol, newton_atol, newton_rtol."""
    return ";".join("{0}={1}".format(field, _text(trial[field])) for field in STEPPING_FIELDS)


def kernel_ident(trial, key):
    """The optimize row identity of a trial's kernel: package, key, problem, states, precision, algorithm, controller, gains and stepping."""
    params = json.loads(trial["system_params"]) if trial["system_params"] else {}
    states = params.get("states", as_problem(trial["problem"])["states"])
    return {"package": trial["package"], "key": key, "problem": as_problem(trial["problem"]).name,
            "states": str(int(states)), "precision": trial["precision"], "algorithm": trial["algorithm"],
            "controller": trial["controller"] or "", "gains": trial["gains"] or "",
            "stepping": stepping_text(trial)}


def optimize_ident(trial, key):
    """kernel_ident plus the line's optimize policy; a per-solve identity carries the line's n."""
    ident = dict(kernel_ident(trial, key), per=trial["optimize"] or "")
    if trial["optimize"] == "solve":
        ident["n"] = str(int(trial["n"]))
    return ident


def _same(row, ident):
    return all(str(row.get(field, "")) == str(value) for field, value in ident.items())


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


def system_source_hash(system):
    """Identity of the code a system's kernels are built from: its function hash, the installed cubie source and its version."""
    import hashlib
    import cubie
    from cubie._utils import package_source_hash
    text = "|".join((str(getattr(system, "fn_hash", "")), package_source_hash(),
                     str(getattr(cubie, "__version__", ""))))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def source_hash(solver):
    """system_source_hash of a solver's system."""
    return system_source_hash(solver.system)


def system_key(trial):
    """(problem, system_params, precision): the system a trial's kernels are built from."""
    return (trial["problem"], trial["system_params"], trial["precision"])


def _hash_systems(package, systems):
    """The source hash of each (problem, system_params, precision) under the package's backend, built in this process."""
    import numpy as np
    select_backend(package)
    out = []
    for problem, params, precision in systems:
        states = (json.loads(params) if params else {}).get("states")
        system, _ = build_system(problem, package, np.float64 if precision == "float64" else np.float32,
                                 states=states)
        out.append(system_source_hash(system))
    return out


def source_hashes(package, systems):
    """{(problem, system_params, precision): source hash} of the systems under a cubie package, computed by the package's own interpreter; RuntimeError when it cannot."""
    import tempfile
    from launch import venv_python
    systems = sorted(set(tuple(s) for s in systems))
    if not systems:
        return {}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as handle:
        json.dump(systems, handle)
        path = handle.name
    try:
        proc = subprocess.run([venv_python(package), os.path.abspath(__file__), "sources", package, path],
                              capture_output=True, text=True)
    finally:
        os.remove(path)
    if proc.returncode != 0:
        raise RuntimeError("{0} source hashes failed (exit {1}): {2}".format(
            package, proc.returncode, proc.stderr.strip()[-2000:]))
    hashes = json.loads(proc.stdout.strip().splitlines()[-1])
    return dict(zip(systems, hashes))


def optimize_rows(package, key, root=None):
    """Every optimize.csv row of a package under a key."""
    path = optimize_path(package, key, root)
    with _Lock(path):
        return _load(path)


def find_optimized(rows, ident):
    """The last row matching every field of an identity; None when there is none."""
    matched = [row for row in rows if _same(row, ident)]
    return matched[-1] if matched else None


def load_optimized(trial, key, root=None, source=None):
    """{'settings', 'resident_blocks'} recorded for a line's optimize, or None; with `source`, only a row recorded from that source."""
    recorded = optimize_rows(trial["package"], key, root)
    if source is not None:
        recorded = [row for row in recorded if row.get("source", "") == source]
    row = find_optimized(recorded, optimize_ident(trial, key))
    if row is None:
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


def _occupancy(kernel, blocksize, dynamic_shared):
    """The driver's resident blocks per SM of a compiled kernel at a launch geometry."""
    from cubie.backend.utils import active_blocks_per_multiprocessor
    return int(active_blocks_per_multiprocessor(kernel.kernel, blocksize, dynamic_shared))


def _multiprocessors():
    from cubie.backend.utils import device_hardware
    return int(device_hardware().multiprocessor_count)


def optimize_batch(solver, initial_values, parameters, duration, waves=OPTIMIZE_WAVES):
    """The run count filling `waves` occupancy waves of the solver's kernel at its block size; compiles the kernel on the given grid, which holds at least one block of runs."""
    solver.compile(initial_values, parameters, duration=duration)
    kernel = solver.kernel
    blocksize, dynamic = kernel.launch_geometry(kernel.compile_settings.blocksize)
    resident = max(1, _occupancy(kernel, blocksize, dynamic))
    runs_per_block = max(1, int(blocksize) // int(kernel.single_integrator.threads_per_step))
    return int(round(waves * _multiprocessors() * resident * runs_per_block))


def optimize_duration(solver, initial_values, parameters, duration, target_ms=OPTIMIZE_LAUNCH_MS,
                      fractions=PROBE_FRACTIONS):
    """The solve duration whose launch on the batch takes about target_ms: probes of the fractions run until one takes target_ms, and that probe scaled to target_ms, kept between the first fraction and the duration; the first probe runs untimed once from the host and once on the device."""
    duration = float(duration)
    probe = duration * fractions[0]
    solve(solver, initial_values, parameters, probe)
    solve(solver, solver.device_initial_values, solver.device_parameters, probe, on_device=True)
    elapsed_ms = 0.0
    for fraction in fractions:
        probe = duration * fraction
        started = timeit.default_timer()
        solve(solver, solver.device_initial_values, solver.device_parameters, probe, on_device=True)
        elapsed_ms = (timeit.default_timer() - started) * 1000.0
        if elapsed_ms >= target_ms:
            break
    if elapsed_ms <= 0.0:
        return duration
    return min(duration, max(duration * fractions[0], probe * target_ms / elapsed_ms))


def _replace(trial, key, root, row):
    path = optimize_path(trial["package"], key, root)
    ident = optimize_ident(trial, key)
    with _Lock(path):
        rows = [r for r in _load(path) if not _same(r, ident)]
        rows.append(row)
        _save(path, rows)
    return row


def _stamp():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def record_optimized(trial, key, result, n, root=None, source="", duration=None):
    """Replace the optimize row of a line with the result's best launch on a batch of n at a duration."""
    best = result.best
    row = dict(optimize_ident(trial, key), n=str(int(n)), duration=_text(duration), source=source, label=best.label,
               best_ms="{0:.6g}".format(best.best_ms), blocksize=str(best.blocksize),
               resident_blocks="" if best.resident_blocks is None else str(best.resident_blocks),
               settings=_encode(result.applied_settings), recorded_utc=_stamp())
    return _replace(trial, key, root, row)


def record_optimize_timeout(trial, key, root=None):
    """Replace the optimize row of a line with one labelled timeout and no settings."""
    row = dict(optimize_ident(trial, key), n=str(int(trial["n"])), duration="", source="", label="timeout",
               best_ms="nan", blocksize="", resident_blocks="", settings="", recorded_utc=_stamp())
    return _replace(trial, key, root, row)


def optimize_point(solver, trial, initial_values, parameters, key, root=None, force=False,
                   source="", verbose=True, duration=None):
    """Run Solver.optimize on a batch for `duration` (the trial's when None), apply the winner to the solver and record it under `source`; `force` varies settings an earlier optimize applied."""
    duration = float(trial["duration"] if duration is None else duration)
    result = solver.optimize(initial_values, parameters, duration=duration, verbose=verbose, force=force)
    if result.best is None:
        raise RuntimeError("optimize timed no launch for {0} {1}".format(
            trial["problem"], trial["algorithm"]))
    return record_optimized(trial, key, result, int(initial_values.shape[1]), root=root, source=source,
                            duration=duration)


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
    if len(argv) == 3 and argv[0] == "sources":
        with open(argv[2], encoding="utf-8") as handle:
            systems = [tuple(item) for item in json.load(handle)]
        print(json.dumps(_hash_systems(argv[1], systems)))
        return 0
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))
