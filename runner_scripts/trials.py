"""Trial records: run specs merged by trial_id, grouped into legs, ordered by cost, each leg led by its build line and the optimize lines the set asks for, written one JSONL line per trial."""

import json
import math
import os

from protocol import WATCHDOG_SECONDS
from store import TRIAL_FIELDS, trial_id

KINDS = ("solve", "warm", "optimize")
AXES = ("n", "dt", "tol", "states")
TRIAL_KEYS = TRIAL_FIELDS + ("trial_id", "kind", "finals", "transfers", "leg", "axis", "ordinal", "cold",
                             "watchdog_s")
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


def _record(spec, kind, transfers, finals, leg, axis, ordinal, cold=False, watchdog_s=None):
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
    return record


def _budget(spec):
    """The spec's watchdog budget in seconds; the protocol's when the spec carries none."""
    return float(spec.get("watchdog_s", WATCHDOG_SECONDS))


def _optimize_record(entry, leg, axis, ordinal):
    table = entry["optimize"]
    n = entry["spec"]["n"] if table["n"] == "solve" else int(table["n"])
    return _record(dict(entry["spec"], n=n), "optimize", [], False, leg, axis, ordinal,
                   watchdog_s=entry["watchdog_s"])


def build_trials(specs):
    """Trial records from specs: same trial_id merges (first leg kept, transfers union, finals true wins, the larger watchdog budget); per package leg, a warm line (cold when the set builds cold), optimize lines once per leg or before every solve as the set's optimize table says, then the solves in cost order."""
    merged = {}
    order = []
    for spec in specs:
        ident = trial_id(spec)
        if ident in merged:
            entry = merged[ident]
            entry["transfers"] |= set(spec["transfers"])
            entry["finals"] = entry["finals"] or bool(spec["finals"])
            entry["watchdog_s"] = max(entry["watchdog_s"], _budget(spec))
            continue
        merged[ident] = {"spec": spec, "transfers": set(spec["transfers"]),
                         "finals": bool(spec["finals"]), "axis": spec["axis"],
                         "build": spec["build"], "optimize": spec["optimize"],
                         "watchdog_s": _budget(spec),
                         "leg": leg_name(spec, spec["axis"])}
        order.append(ident)
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
                              cold=first["build"] == "cold", watchdog_s=first["watchdog_s"]))
        if first["optimize"] is not None and first["optimize"]["per"] == "leg":
            trials.append(_optimize_record(first, leg, axis, 0))
        for ordinal, entry in enumerate(entries):
            if entry["optimize"] is not None and entry["optimize"]["per"] == "solve":
                trials.append(_optimize_record(entry, leg, axis, ordinal))
            transfers = [t for t in TRANSFERS_ORDER if t in entry["transfers"]]
            trials.append(_record(entry["spec"], "solve", transfers, entry["finals"], leg,
                                  axis, ordinal, watchdog_s=entry["watchdog_s"]))
    return trials


def by_package(trials):
    """Trials grouped by package, in first appearance."""
    groups = {}
    for trial in trials:
        groups.setdefault(trial["package"], []).append(trial)
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
            line = {key: _json_value(trial[key]) for key in TRIAL_KEYS}
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
