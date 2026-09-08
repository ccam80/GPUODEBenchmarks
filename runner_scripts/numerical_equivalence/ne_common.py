"""Numerical-equivalence protocol: cubie against DifferentialEquations.jl in Float32, error against dt and tolerance, scored on the Float64 golden reference."""

import csv
import os

import numpy as np

from problems import DEFAULT_PROBLEM, get_problem
from protocol import N_NE, N_WP, NE_LEGACY_GRID, TOLS as TOLS_NE


def _row(problem):
    """Accept a problem row or a problem name."""
    return problem if isinstance(problem, dict) else get_problem(problem)


def dts_ne(problem=DEFAULT_PROBLEM):
    """The fixed-step dt grid for a problem's ne sweep."""
    return _row(problem).ne_dts()


def golden_ne_path(problem=DEFAULT_PROBLEM):
    """Path of a legacy-grid problem's ne golden file."""
    return os.path.join(
        "data", "numerical",
        "golden_ne_{0}_{1}.csv".format(_row(problem)["problem"], N_NE))


def ne_legacy_grid(problem=DEFAULT_PROBLEM):
    """Whether the problem's ne ensemble is the standalone golden_ne grid."""
    return _row(problem)["problem"] in NE_LEGACY_GRID


def ne_sweep(problem=DEFAULT_PROBLEM):
    """The Float32 ne ensemble grid: golden_ne's column for a legacy problem, else the first N_NE points of the N_WP sweep."""
    row = _row(problem)
    if ne_legacy_grid(row):
        return np.float32(np.loadtxt(golden_ne_path(row), delimiter=",",
                                     usecols=(0,)))
    return row.sweep(N_WP, dtype=np.float32)[:N_NE]

NE_DIR = os.path.join("data", "numerical_equivalence")
JULIA_NE_DIR = os.path.join(NE_DIR, "julia")


def ne_keys(package):
    """Dataset keys with ne outputs of a package (julia, cubie or cubie_mlir)."""
    root = os.path.join(NE_DIR, package)
    if not os.path.isdir(root):
        return set()
    return {name for name in os.listdir(root)
            if os.path.isdir(os.path.join(root, name))}


def load_golden_ne(problem=DEFAULT_PROBLEM):
    """(sweep, golden states) of the ne ensemble as float64 arrays; the states come from golden_ne for a legacy problem, else from the first N_NE rows of the wp golden."""
    row = _row(problem)
    if ne_legacy_grid(row):
        path = golden_ne_path(row)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                "{0} not found; it defines the legacy ne grid of {1}"
                .format(path, row["problem"]))
        data = np.loadtxt(path, delimiter=",")
        expected = (N_NE, row["states"] + 1)
        if data.shape != expected:
            raise ValueError("golden ne reference has shape {0}, expected {1}"
                             .format(data.shape, expected))
        return data[:, 0], data[:, 1:]
    from wp_common import load_golden
    return np.float64(ne_sweep(row)), load_golden(row)[:N_NE]


def ensemble_error(final_states, golden_states):
    """l2-at-final error over the ensemble, computed in float64.

    Same metric as the wp sweeps: sqrt(mean((final - golden)**2)) over the
    (N_NE, 3) array.
    """
    diff = np.asarray(final_states, dtype=np.float64) - golden_states
    return float(np.sqrt(np.mean(diff ** 2)))


def ensemble_error_masked(final_states, golden_states, mask):
    """l2-at-final error over a masked subset of the ensemble, in float64.

    ``mask`` is a boolean (N_NE,) array selecting the trajectories to include
    (e.g. the trajectories both stacks converged on). Returns NaN when the
    mask selects nothing, so a fully non-converged point drops out of the
    comparison rather than contaminating it.
    """
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return float("nan")
    diff = (np.asarray(final_states, dtype=np.float64)[mask]
            - np.asarray(golden_states, dtype=np.float64)[mask])
    return float(np.sqrt(np.mean(diff ** 2)))


# Columns that are not part of the final state.
_META_FIELDS = ("dt", "tol", "traj", "naccept", "nreject", "converged")


def _state_fields(rows):
    """The state column names of an ne file, in order."""
    return [name for name in rows[0].keys() if name not in _META_FIELDS]


def _state_array(sel, fields):
    return np.array([[float(row[name]) for name in fields] for row in sel],
                    dtype=np.float64)


def _converged_from_rows(sel, arr):
    """Per-trajectory converged mask for one dt/tol block.

    A trajectory counts as converged iff it BOTH claims success and produced
    a finite final state, so the two stacks are measured the same way. cubie
    has no retcode column and signals a failed solve with NaN, so its flag is
    pure finiteness. The Julia runner writes a ``converged`` column (1/0 from
    each trajectory's SciML retcode); that is AND-ed with finiteness because an
    explicit fixed step can overflow to NaN/Inf while still returning a
    ``Success`` retcode — without the finite check those would be miscounted
    as converged and inflate the cubie-vs-julia non-convergence gap.
    """
    finite = np.isfinite(arr).all(axis=1)
    if sel and "converged" in sel[0] and sel[0]["converged"] not in ("", None):
        flag = np.array([str(r.get("converged", "")).strip() == "1"
                         for r in sel], dtype=bool)
        return flag & finite
    return finite


def read_ne_csv_masked(path):
    """Like :func:`read_ne_csv` but also returns a per-trajectory mask.

    Returns dict {dt: (finals (N_NE, 3) float64, converged (N_NE,) bool)}.
    """
    out = {}
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    fields = _state_fields(rows)
    dts = sorted({float(row["dt"]) for row in rows}, reverse=True)
    for dt in dts:
        sel = [row for row in rows if float(row["dt"]) == dt]
        sel.sort(key=lambda row: int(row["traj"]))
        arr = _state_array(sel, fields)
        out[dt] = (arr, _converged_from_rows(sel, arr))
    return out


def read_ne_adaptive_csv_masked(path):
    """Like :func:`read_ne_adaptive_csv` but also returns a converged mask.

    Returns dict {tol: (finals (N_NE, 3) float64, converged (N_NE,) bool,
    naccept, nreject)} with naccept/nreject as in :func:`read_ne_adaptive_csv`.
    """
    out = {}
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    fields = _state_fields(rows)
    tols = sorted({float(row["tol"]) for row in rows}, reverse=True)
    for tol in tols:
        sel = [row for row in rows if float(row["tol"]) == tol]
        sel.sort(key=lambda row: int(row["traj"]))
        arr = _state_array(sel, fields)
        conv = _converged_from_rows(sel, arr)

        def counts(field):
            vals = [row.get(field, "") for row in sel]
            if any(v not in ("", None) for v in vals):
                return np.array([float(v) if v not in ("", None) else np.nan
                                 for v in vals])
            return None
        out[tol] = (arr, conv, counts("naccept"), counts("nreject"))
    return out


def julia_ne_dir(problem=DEFAULT_PROBLEM, dataset_key=None):
    """Directory of one machine's Julia outputs for a problem; None keys the current machine."""
    if dataset_key is None:
        from bench_key import dataset_key as current_key
        dataset_key = current_key()
    d = os.path.join(JULIA_NE_DIR, dataset_key, _row(problem)["problem"])
    os.makedirs(d, exist_ok=True)
    return d


def cubie_ne_dir(dataset_key, problem=DEFAULT_PROBLEM, package="cubie"):
    """Directory of one machine's outputs of a cubie package for a problem; creates it."""
    d = os.path.join(NE_DIR, package, dataset_key, _row(problem)["problem"])
    os.makedirs(d, exist_ok=True)
    return d


def julia_ne_file(alias, problem=DEFAULT_PROBLEM, dataset_key=None):
    """Path of the per-machine DifferentialEquations.jl fixed-sweep output."""
    return os.path.join(julia_ne_dir(problem, dataset_key),
                        "{0}.csv".format(alias))


def cubie_ne_file(alias, dataset_key, problem=DEFAULT_PROBLEM,
                  package="cubie"):
    """Path of a cubie package's fixed-sweep output for one machine."""
    return os.path.join(cubie_ne_dir(dataset_key, problem, package),
                        "{0}.csv".format(alias))


def julia_ne_adaptive_file(alias, problem=DEFAULT_PROBLEM, dataset_key=None):
    """Julia adaptive-sweep output (rows tol,traj,states...,naccept,nreject)."""
    return os.path.join(julia_ne_dir(problem, dataset_key),
                        "{0}_adaptive.csv".format(alias))


def cubie_ne_adaptive_file(alias, tier, dataset_key,
                           problem=DEFAULT_PROBLEM, package="cubie"):
    """A cubie package's adaptive-sweep output per controller tier: "default" or "matched"."""
    return os.path.join(cubie_ne_dir(dataset_key, problem, package),
                        "{0}_adaptive_{1}.csv".format(alias, tier))


def controller_constants_csv(problem=DEFAULT_PROBLEM, dataset_key=None):
    """Path of a problem's resolved default-controller constants."""
    return os.path.join(julia_ne_dir(problem, dataset_key),
                        "controller_constants.csv")


def load_controller_constants(problem=DEFAULT_PROBLEM, dataset_key=None):
    """One problem's default-controller constants keyed by cubie alias; missing numeric fields are None."""
    path = controller_constants_csv(problem, dataset_key)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "{0} not found - run the Julia adaptive sweep first "
            "(runner_scripts/numerical_equivalence/ne_diffeq.jl)"
            .format(path))
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            entry = {"controller": row["controller"]}
            for key in ("beta1", "beta2", "qmin", "qmax", "gamma", "order"):
                raw = row.get(key, "")
                entry[key] = float(raw) if raw not in ("", None) else None
            out[row["cubie_alias"]] = entry
    return out


def write_ne_csv(path, per_dt_finals):
    """Write an ne output file (mode "w" — one file holds one full sweep).

    ``per_dt_finals`` is a list of (dt, finals) pairs where finals is an
    (N_NE, 3) array of Float32 final states.
    """
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        nstates = per_dt_finals[0][1].shape[1] if per_dt_finals else 0
        w.writerow(["dt", "traj"] + ["s{0}".format(s + 1)
                                     for s in range(nstates)])
        for dt, finals in per_dt_finals:
            for j in range(finals.shape[0]):
                w.writerow(["{0:.10g}".format(dt), j]
                           + [repr(float(v)) for v in finals[j, :]])


def read_ne_csv(path):
    """Read an ne output file; returns dict {dt: (N_NE, 3) float64 array}."""
    out = {}
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        rows = list(r)
    fields = _state_fields(rows)
    dts = sorted({float(row["dt"]) for row in rows}, reverse=True)
    for dt in dts:
        sel = [row for row in rows if float(row["dt"]) == dt]
        sel.sort(key=lambda row: int(row["traj"]))
        out[dt] = _state_array(sel, fields)
    return out


def write_ne_adaptive_csv(path, per_tol_results):
    """Write an adaptive ne output file (one full sweep per file).

    ``per_tol_results`` is a list of (tol, finals, naccept, nreject) where
    finals is (N_NE, 3) float32 and naccept/nreject are per-trajectory int
    arrays or None when the stack does not report step counts.
    """
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        nstates = per_tol_results[0][1].shape[1] if per_tol_results else 0
        w.writerow(["tol", "traj"] + ["s{0}".format(s + 1)
                                      for s in range(nstates)]
                   + ["naccept", "nreject"])
        for tol, finals, naccept, nreject in per_tol_results:
            for j in range(finals.shape[0]):
                w.writerow(["{0:.10g}".format(tol), j]
                           + [repr(float(v)) for v in finals[j, :]]
                           + ["" if naccept is None else int(naccept[j]),
                              "" if nreject is None else int(nreject[j])])


def read_ne_adaptive_csv(path):
    """Read an adaptive ne file.

    Returns dict {tol: (finals, naccept, nreject)} with finals (N_NE, 3)
    float64 and naccept/nreject float64 arrays (NaN where unreported).
    """
    out = {}
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    fields = _state_fields(rows)
    tols = sorted({float(row["tol"]) for row in rows}, reverse=True)
    for tol in tols:
        sel = [row for row in rows if float(row["tol"]) == tol]
        sel.sort(key=lambda row: int(row["traj"]))
        arr = _state_array(sel, fields)
        def counts(field):
            vals = [row.get(field, "") for row in sel]
            if any(v not in ("", None) for v in vals):
                return np.array([float(v) if v not in ("", None) else np.nan
                                 for v in vals])
            return None
        out[tol] = (arr, counts("naccept"), counts("nreject"))
    return out
