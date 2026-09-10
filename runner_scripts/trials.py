"""Trial records: run specs merged by trial_id into one canonical contract over every declaration of the point, grouped into legs, ordered by cost, each leg led by its build line and the optimize lines the contract asks for, written one JSONL line per trial."""

import json
import math
import os

from protocol import WATCHDOG_SECONDS
from store import TRIAL_FIELDS, trial_id

KINDS = ("solve", "warm", "optimize")
AXES = ("n", "dt", "tol", "states")
# The leg a point shared by several axes joins: a states leg is the point's own cold build, an n leg holds one stepping, the swept legs share one.
AXIS_PRIORITY = ("states", "n", "dt", "tol")
TRIAL_KEYS = TRIAL_FIELDS + ("trial_id", "kind", "finals", "transfers", "leg", "axis", "ordinal", "cold",
                             "watchdog_s", "sets")
TRANSFERS_ORDER = ("both", "none")
LEG_FIELDS = ("problem", "system_params", "algorithm", "controller", "precision")


def leg_name(spec, axis):
    """<problem>/<system_params>/<algorithm>/<controller>/<precision>/<axis>."""
    return "/".join([str(spec[f]) for f in LEG_FIELDS] + [axis])


def _states(spec):
    params = json.loads(spec["system_params"]) if spec["system_params"] else {}
    return int(params.get("states", 0))


def _finite_or(value, fallback):
    return value if isinstance(value, (int, float)) and math.isfinite(value) else fallback


def ordinal_key(spec):
    """Cost order within a leg: n ascending, dt descending, tol descending, states ascending."""
    return (int(spec["n"]), -_finite_or(spec["dt"], 0.0), -_finite_or(spec["atol"], 0.0),
            _states(spec))


def _record(spec, kind, transfers, finals, leg, axis, ordinal, cold=False, watchdog_s=None, sets=()):
    record = {field: spec[field] for field in TRIAL_FIELDS}
    record["trial_id"] = trial_id(spec)
    record["kind"] = kind
    record["finals"] = bool(finals)
    record["transfers"] = list(transfers)
    record["leg"] = leg
    record["axis"] = axis
    record["ordinal"] = int(ordinal)
    record["cold"] = bool(cold)
    record["watchdog_s"] = float(_budget(spec) if watchdog_s is None else watchdog_s)
    record["sets"] = sorted(sets)
    return record


def _budget(spec):
    """The spec's watchdog budget in seconds; the protocol's when the spec carries none."""
    return float(spec.get("watchdog_s", WATCHDOG_SECONDS))


def _optimize_record(entry, table, leg, axis, ordinal):
    n = entry["spec"]["n"] if table["n"] == "solve" else int(table["n"])
    return _record(dict(entry["spec"], n=n), "optimize", [], False, leg, axis, ordinal,
                   watchdog_s=entry["watchdog_s"], sets=entry["sets"])


def canonical_optimize(tables):
    """One optimize table from every declaration's: per leg when any says leg, else per solve when any says solve, else None; n the largest integer among the declarations of that per, else "solve"."""
    tables = [t for t in tables if t is not None]
    if not tables:
        return None
    per = "leg" if any(t["per"] == "leg" for t in tables) else "solve"
    counts = [int(t["n"]) for t in tables if t["per"] == per and t["n"] != "solve"]
    return {"n": max(counts) if counts else "solve", "per": per}


def _entry(spec):
    return {"spec": spec, "transfers": set(), "finals": False, "cold": False, "axis": spec["axis"],
            "tables": [], "watchdog_s": 0.0, "sets": set()}


def _fold(entry, spec):
    """Fold one declaration of a point into its entry: transfers union, finals and cold true over false, every optimize table, the larger watchdog budget, the axis of highest priority, the set's name."""
    entry["transfers"] |= set(spec["transfers"])
    entry["finals"] = entry["finals"] or bool(spec["finals"])
    entry["cold"] = entry["cold"] or spec["build"] == "cold"
    entry["tables"].append(spec["optimize"])
    entry["watchdog_s"] = max(entry["watchdog_s"], _budget(spec))
    if AXIS_PRIORITY.index(spec["axis"]) < AXIS_PRIORITY.index(entry["axis"]):
        entry["axis"] = spec["axis"]
    if spec.get("set"):
        entry["sets"].add(spec["set"])


def build_trials(specs, declared=None):
    """Trial records from the requested specs: specs of one trial_id, in `specs` and among `declared` (every set file's specs, the requested ones when None), merge into one contract independent of their order (see _fold); per package leg, a warm line (cold when any of the leg's points builds cold), one optimize line when any point asks per leg, one before every solve that asks per solve, then the solves in cost order."""
    merged = {}
    order = []
    for spec in specs:
        ident = trial_id(spec)
        if ident not in merged:
            merged[ident] = _entry(spec)
            order.append(ident)
        _fold(merged[ident], spec)
    for spec in (specs if declared is None else declared):
        ident = trial_id(spec)
        if ident in merged:
            _fold(merged[ident], spec)
    for entry in merged.values():
        entry["optimize"] = canonical_optimize(entry["tables"])
        entry["leg"] = leg_name(entry["spec"], entry["axis"])
    # A leg name repeats across packages; each package's trial file holds its own legs.
    legs = {}
    for ident in order:
        entry = merged[ident]
        legs.setdefault((entry["spec"]["package"], entry["leg"]), []).append(entry)
    trials = []
    for (_, leg), entries in legs.items():
        entries.sort(key=lambda e: ordinal_key(e["spec"]))
        first = entries[0]
        axis = first["axis"]
        trials.append(_record(first["spec"], "warm", [], False, leg, axis, 0,
                              cold=any(e["cold"] for e in entries),
                              watchdog_s=max(e["watchdog_s"] for e in entries),
                              sets=set().union(*(e["sets"] for e in entries))))
        leg_table = canonical_optimize([e["optimize"] for e in entries
                                        if e["optimize"] is not None and e["optimize"]["per"] == "leg"])
        if leg_table is not None:
            trials.append(_optimize_record(first, leg_table, leg, axis, 0))
        for ordinal, entry in enumerate(entries):
            if entry["optimize"] is not None and entry["optimize"]["per"] == "solve":
                trials.append(_optimize_record(entry, entry["optimize"], leg, axis, ordinal))
            transfers = [t for t in TRANSFERS_ORDER if t in entry["transfers"]]
            trials.append(_record(entry["spec"], "solve", transfers, entry["finals"], leg,
                                  axis, ordinal, watchdog_s=entry["watchdog_s"], sets=entry["sets"]))
    return trials


def by_package(trials):
    """Trials grouped by package, in first appearance."""
    groups = {}
    for trial in trials:
        groups.setdefault(trial["package"], []).append(trial)
    return groups


def legs_of(trials):
    """{(package, leg): trials} in first appearance, each leg's trials in file order."""
    groups = {}
    for trial in trials:
        groups.setdefault((trial["package"], trial["leg"]), []).append(trial)
    return groups


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
            trials.append(record)
    return trials


def counts(trials):
    """{kind: count} and {leg: solve count} of a trial list."""
    kinds = {kind: 0 for kind in KINDS}
    legs = {}
    for trial in trials:
        kinds[trial["kind"]] += 1
        if trial["kind"] == "solve":
            legs[trial["leg"]] = legs.get(trial["leg"], 0) + 1
    return kinds, legs
