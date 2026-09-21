"""Trial records: one line per point, its run spec merged over every declaration of the point into one contract (transfers, finals, traces, cold, optimize, watchdog, timed; compile marks a line whose problem, algorithm and controller timed out compiling), written in difficulty order one JSON object per line."""

import functools
import json
import math
import os

from algorithms import algorithm_facts
from protocol import WATCHDOG_SECONDS
from store import TRIAL_FIELDS, trial_id

TRIAL_KEYS = TRIAL_FIELDS + ("trial_id", "transfers", "finals", "traces", "cold", "optimize", "compile", "watchdog_s",
                             "single_run_s", "timed", "sets")
# A trial line's compile mark: "" or COMPILE_TIMED_OUT (the store holds a compile_timeout row of its compile_key).
COMPILE_TIMED_OUT = "timeout"
TRANSFERS_ORDER = ("both", "none")
# The fields one package build serves; a line whose values differ from the last needs a new build.
BUILD_FIELDS = ("problem", "system_params", "precision", "algorithm", "controller", "gains")
# The fields of one compiled kernel: the build plus every stepping value.
KERNEL_FIELDS = BUILD_FIELDS + ("dt", "dt_min", "dt_max", "atol", "rtol", "newton_atol", "newton_rtol")
# The fields whose kernels share a compile cost; a compile past the watchdog condemns the whole group.
COMPILE_FIELDS = ("package", "problem", "system_params", "precision", "algorithm", "controller")
# The fields the abandon rule compares within: lines differing only in difficulty.
FAMILY_FIELDS = ("package", "problem", "precision", "algorithm", "controller", "gains")


def build_key(trial):
    """The build a trial runs on, as a tuple of BUILD_FIELDS."""
    return tuple(trial[f] for f in BUILD_FIELDS)


@functools.lru_cache(maxsize=None)
def _family(algorithm):
    return algorithm_facts(algorithm)["family"]


def shares_dt_optimize(trial):
    """True for an explicit fixed-step line: one optimize serves every dt of its build."""
    return trial["controller"] == "fixed" and _family(trial["algorithm"]) == "erk"


def kernel_key(trial):
    """The compiled kernel a trial runs, as a tuple of KERNEL_FIELDS with NaN as None."""
    return tuple(None if isinstance(trial[f], float) and math.isnan(trial[f]) else trial[f] for f in KERNEL_FIELDS)


def optimize_key(trial):
    """The optimize a trial shares: its kernel_key, with dt None where shares_dt_optimize."""
    key = kernel_key(trial)
    if not shares_dt_optimize(trial):
        return key
    dt = KERNEL_FIELDS.index("dt")
    return key[:dt] + (None,) + key[dt + 1:]


def compile_key(trial):
    """The kernels a compile timeout condemns, as a tuple of COMPILE_FIELDS; a store row keys the same way."""
    return tuple(trial[f] for f in COMPILE_FIELDS)


def mark_compile_timeouts(trial_list, groups):
    """The trials with every line of a compile_key in `groups` marked COMPILE_TIMED_OUT and its optimize dropped; the list itself when no line changes."""
    if not groups:
        return trial_list
    marked = [dict(t, optimize=False, compile=COMPILE_TIMED_OUT)
              if compile_key(t) in groups and (t["optimize"] or t.get("compile") != COMPILE_TIMED_OUT) else t
              for t in trial_list]
    return marked if any(a is not b for a, b in zip(marked, trial_list)) else trial_list


def family_key(trial):
    """The family the abandon rule compares a trial within, as a tuple of FAMILY_FIELDS."""
    return tuple(trial[f] for f in FAMILY_FIELDS)


def family_parts(trial_list, kernels=None):
    """File-order parts of whole families, each closed at the first family boundary at or past `kernels` distinct kernels; one part when `kernels` is None."""
    if not kernels:
        return [list(trial_list)]
    parts, current, counted = [], [], 0
    for trial in trial_list:
        if counted >= kernels and family_key(trial) != family_key(current[-1]):
            parts.append(current)
            current, counted = [], 0
        current.append(trial)
        counted = len({kernel_key(t) for t in current})
    if current:
        parts.append(current)
    return parts


def states_of(spec):
    """The states of system_params, 0 when it names none."""
    params = json.loads(spec["system_params"]) if spec["system_params"] else {}
    return int(params.get("states", 0))


def _finite_or(value, fallback):
    return value if isinstance(value, (int, float)) and math.isfinite(value) else fallback


def difficulty(spec):
    """(n, states, -dt, -tol): each entry grows with the cost of the run."""
    return (int(spec["n"]), states_of(spec), -_finite_or(spec["dt"], 0.0), -_finite_or(spec["atol"], 0.0))


def harder(a, b):
    """True when a is at least as hard as b in every difficulty entry and harder in one."""
    da, db = difficulty(a), difficulty(b)
    return all(x >= y for x, y in zip(da, db)) and da != db


def order_key(spec):
    """File order: by problem, precision, algorithm, controller and gains, then states, n, dt descending, tolerance descending."""
    return (spec["problem"], spec["precision"], spec["algorithm"], spec["controller"], spec["gains"],
            states_of(spec), spec["system_params"], int(spec["n"]), -_finite_or(spec["dt"], 0.0),
            -_finite_or(spec["atol"], 0.0))


def _budget(spec):
    """The spec's watchdog budget in seconds; the protocol's when the spec carries none."""
    return float(spec.get("watchdog_s", WATCHDOG_SECONDS))


def _entry(spec):
    return {"spec": spec, "transfers": set(), "finals": False, "traces": False, "cold": False, "optimize": False,
            "watchdog_s": 0.0, "single_run_s": math.inf, "timed": False, "sets": set()}


def _fold(entry, spec):
    """Fold one declaration of a point into its entry: transfers union, finals, traces, cold, optimize and timed true over false, the larger watchdog budget, the smaller single-run threshold, the set's name."""
    entry["transfers"] |= set(spec["transfers"])
    entry["finals"] = entry["finals"] or bool(spec["finals"])
    entry["traces"] = entry["traces"] or bool(spec.get("traces", False))
    entry["cold"] = entry["cold"] or spec["build"] == "cold"
    entry["optimize"] = entry["optimize"] or bool(spec["optimize"])
    entry["watchdog_s"] = max(entry["watchdog_s"], _budget(spec))
    entry["single_run_s"] = min(entry["single_run_s"], float(spec.get("single_run_s", math.inf)))
    entry["timed"] = entry["timed"] or bool(spec.get("timed", True))
    if spec.get("set"):
        entry["sets"].add(spec["set"])


def _record(entry):
    spec = entry["spec"]
    record = {field: spec[field] for field in TRIAL_FIELDS}
    record["trial_id"] = trial_id(spec)
    record["transfers"] = [t for t in TRANSFERS_ORDER if t in entry["transfers"]]
    record["finals"] = bool(entry["finals"])
    record["traces"] = bool(entry["traces"])
    record["cold"] = bool(entry["cold"])
    record["optimize"] = bool(entry["optimize"])
    record["compile"] = ""
    record["watchdog_s"] = float(entry["watchdog_s"])
    record["single_run_s"] = float(entry["single_run_s"])
    record["timed"] = bool(entry["timed"])
    record["sets"] = sorted(entry["sets"])
    return record


def build_trials(specs, declared=None):
    """Trial records of the requested specs: specs of one trial_id, in `specs` and among `declared` (the requested ones when None), merge per _fold; records come per package in order_key order."""
    merged = {}
    for spec in specs:
        ident = trial_id(spec)
        if ident not in merged:
            merged[ident] = _entry(spec)
        _fold(merged[ident], spec)
    for spec in (specs if declared is None else declared):
        ident = trial_id(spec)
        if ident in merged:
            _fold(merged[ident], spec)
    packages = list(dict.fromkeys(e["spec"]["package"] for e in merged.values()))
    out = []
    for package in packages:
        entries = [e for e in merged.values() if e["spec"]["package"] == package]
        entries.sort(key=lambda e: order_key(e["spec"]))
        out.extend(_record(e) for e in entries)
    return out


def by_package(trials):
    """Trials grouped by package, in first appearance."""
    groups = {}
    for trial in trials:
        groups.setdefault(trial["package"], []).append(trial)
    return groups


def builds_of(trials):
    """[(build_key, trials)] in file order, each run of one build's consecutive lines."""
    out = []
    for trial in trials:
        key = build_key(trial)
        if out and out[-1][0] == key:
            out[-1][1].append(trial)
        else:
            out.append((key, [trial]))
    return out


def _json_value(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_jsonl(path, trials):
    """One JSON object per line in TRIAL_KEYS order; NaN and inf are written as null."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for trial in trials:
            line = {key: _json_value(trial.get("sets", []) if key == "sets" else trial[key])
                    for key in TRIAL_KEYS}
            handle.write(json.dumps(line, allow_nan=False) + "\n")
    return path


def read_jsonl(path):
    """The trial records of a file; null floats come back as NaN."""
    trials = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            for field in ("duration", "grid_min", "grid_max", "dt", "dt_min", "dt_max", "atol",
                          "rtol", "newton_atol", "newton_rtol"):
                if record.get(field) is None:
                    record[field] = float("nan")
            if record.get("watchdog_s") is None:
                record["watchdog_s"] = WATCHDOG_SECONDS
            if record.get("sets") is None:
                record["sets"] = []
            record["optimize"] = bool(record.get("optimize"))
            record["compile"] = record.get("compile") or ""
            record.setdefault("cold", False)
            record.setdefault("traces", False)
            if record.get("single_run_s") is None:
                record["single_run_s"] = math.inf
            record.setdefault("timed", True)
            trials.append(record)
    return trials


def optimizes_of(trials):
    """The optimize runs of a trial list: one per optimize_key among the lines that optimize."""
    return len({optimize_key(t) for t in trials if t["optimize"]})


def counts(trials):
    """(solve count, optimize count, cold count, build count) of a trial list."""
    return (sum(1 for t in trials if t["transfers"]), optimizes_of(trials),
            sum(1 for t in trials if t["cold"]), len(builds_of(trials)))


def compile_timeouts_of(trials):
    """The compile_key groups of a trial list marked COMPILE_TIMED_OUT."""
    return {compile_key(t) for t in trials if t.get("compile") == COMPILE_TIMED_OUT}
