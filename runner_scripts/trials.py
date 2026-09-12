"""Trial records: one line per point, its run spec merged over every declaration of the point into one contract (transfers, finals, cold, optimize, watchdog, timed), written in difficulty order one JSON object per line."""

import json
import math
import os

from protocol import WATCHDOG_SECONDS
from store import TRIAL_FIELDS, trial_id

TRIAL_KEYS = TRIAL_FIELDS + ("trial_id", "transfers", "finals", "cold", "optimize", "watchdog_s", "timed", "sets")
TRANSFERS_ORDER = ("both", "none")
# The fields one package build serves; a line whose values differ from the last needs a new build.
BUILD_FIELDS = ("problem", "system_params", "precision", "algorithm", "controller", "gains")
# The fields one compiled kernel serves: the build plus every stepping value.
KERNEL_FIELDS = BUILD_FIELDS + ("dt", "dt_min", "dt_max", "atol", "rtol", "newton_atol", "newton_rtol")
# The fields the abandon rule compares within: lines differing only in difficulty.
FAMILY_FIELDS = ("package", "problem", "precision", "algorithm", "controller", "gains")
# A line's optimize policy: once per compiled kernel, or once per line on its own n.
OPTIMIZE_PER = ("kernel", "solve")


def build_key(trial):
    """The build a trial runs on, as a tuple of BUILD_FIELDS."""
    return tuple(trial[f] for f in BUILD_FIELDS)


def kernel_key(trial):
    """The kernel a trial runs on, as a tuple of KERNEL_FIELDS with NaN as None."""
    return tuple(None if isinstance(trial[f], float) and math.isnan(trial[f]) else trial[f]
                 for f in KERNEL_FIELDS)


def family_key(trial):
    """The family the abandon rule compares a trial within, as a tuple of FAMILY_FIELDS."""
    return tuple(trial[f] for f in FAMILY_FIELDS)


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


def canonical_optimize(tables):
    """The optimize policy over every declaration's table: solve over kernel; None without a table."""
    policies = {t["per"] for t in tables if t is not None}
    return "solve" if "solve" in policies else "kernel" if policies else None


def _entry(spec):
    return {"spec": spec, "transfers": set(), "finals": False, "cold": False, "tables": [],
            "watchdog_s": 0.0, "timed": False, "sets": set()}


def _fold(entry, spec):
    """Fold one declaration of a point into its entry: transfers union, finals, cold and timed true over false, every optimize table, the larger watchdog budget, the set's name."""
    entry["transfers"] |= set(spec["transfers"])
    entry["finals"] = entry["finals"] or bool(spec["finals"])
    entry["cold"] = entry["cold"] or spec["build"] == "cold"
    entry["tables"].append(spec["optimize"])
    entry["watchdog_s"] = max(entry["watchdog_s"], _budget(spec))
    entry["timed"] = entry["timed"] or bool(spec.get("timed", True))
    if spec.get("set"):
        entry["sets"].add(spec["set"])


def _record(entry):
    spec = entry["spec"]
    record = {field: spec[field] for field in TRIAL_FIELDS}
    record["trial_id"] = trial_id(spec)
    record["transfers"] = [t for t in TRANSFERS_ORDER if t in entry["transfers"]]
    record["finals"] = bool(entry["finals"])
    record["cold"] = bool(entry["cold"])
    record["optimize"] = canonical_optimize(entry["tables"])
    record["watchdog_s"] = float(entry["watchdog_s"])
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
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def write_jsonl(path, trials):
    """One JSON object per line in TRIAL_KEYS order; NaN is written as null."""
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
            record.setdefault("optimize", None)
            record.setdefault("cold", False)
            record.setdefault("timed", True)
            trials.append(record)
    return trials


def optimizes_of(trials):
    """The optimize runs of a trial list: one per per-solve line, one per kernel among the per-kernel lines."""
    kernels = {kernel_key(t) for t in trials if t["optimize"] == "kernel"}
    return sum(1 for t in trials if t["optimize"] == "solve") + len(kernels)


def counts(trials):
    """(solve count, optimize count, cold count, build count) of a trial list."""
    return (sum(1 for t in trials if t["transfers"]), optimizes_of(trials),
            sum(1 for t in trials if t["cold"]), len(builds_of(trials)))
