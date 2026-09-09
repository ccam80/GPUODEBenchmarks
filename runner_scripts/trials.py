"""Trial records: run specs merged by trial_id, grouped into legs, ordered by cost, joined by the warm and optimize trials each leg needs, written one JSONL line per trial."""

import json
import math
import os

from algorithms import get_algorithm
from protocol import OPTIMIZE_N, OPTIMIZE_PER_POINT_FAMILIES
from store import TRIAL_FIELDS, trial_id

KINDS = ("solve", "warm", "optimize")
AXES = ("n", "dt", "tol", "states")
TRIAL_KEYS = TRIAL_FIELDS + ("trial_id", "kind", "finals", "transfers", "leg", "axis", "ordinal")
TRANSFERS_ORDER = ("both", "none")
LEG_FIELDS = ("problem", "system_params", "algorithm", "controller", "precision")
STEPPING_FIELDS = ("controller", "dt", "dt_min", "dt_max", "atol", "rtol", "gains",
                   "newton_atol", "newton_rtol")
# Packages whose legs carry Solver.optimize trials.
OPTIMIZE_PACKAGES = ("cubie", "cubie_mlir")


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


def _record(spec, kind, transfers, finals, leg, axis, ordinal):
    record = {field: spec[field] for field in TRIAL_FIELDS}
    record["trial_id"] = trial_id(spec)
    record["kind"] = kind
    record["finals"] = bool(finals)
    record["transfers"] = list(transfers)
    record["leg"] = leg
    record["axis"] = axis
    record["ordinal"] = int(ordinal)
    return record


def _stepping_key(spec):
    return tuple(spec[f] if not (isinstance(spec[f], float) and math.isnan(spec[f])) else "nan"
                 for f in STEPPING_FIELDS)


def build_trials(specs, optimize_n=OPTIMIZE_N, per_point_families=OPTIMIZE_PER_POINT_FAMILIES):
    """Trial records from expanded specs.

    Specs sharing a trial_id merge: the first keeps its leg and axis, transfers
    union in timing order, finals true over false. Legs are walked in first
    appearance; solve trials take ordinals from ordinal_key. A warm-built leg
    gets one warm trial with its cheapest spec; an optimize package's leg gets
    one optimize trial per stepping for the per-point families and one per leg
    otherwise, at optimize_n. Legs on the states axis get neither.
    """
    merged = {}
    order = []
    for spec in specs:
        ident = trial_id(spec)
        if ident in merged:
            entry = merged[ident]
            entry["transfers"] |= set(spec["transfers"])
            entry["finals"] = entry["finals"] or bool(spec["finals"])
            continue
        merged[ident] = {"spec": spec, "transfers": set(spec["transfers"]),
                         "finals": bool(spec["finals"]), "axis": spec["axis"],
                         "build": spec["build"],
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
        axis = entries[0]["axis"]
        first = entries[0]["spec"]
        if axis != "states":
            if entries[0]["build"] == "warm":
                trials.append(_record(first, "warm", [], False, leg, axis, 0))
            if first["package"] in OPTIMIZE_PACKAGES:
                per_point = get_algorithm(first["algorithm"])["family"] in per_point_families
                seen = set()
                for ordinal, entry in enumerate(entries):
                    stepping = _stepping_key(entry["spec"]) if per_point else ()
                    if stepping in seen:
                        continue
                    seen.add(stepping)
                    tuned = dict(entry["spec"], n=int(optimize_n))
                    trials.append(_record(tuned, "optimize", [], False, leg, axis, ordinal))
        for ordinal, entry in enumerate(entries):
            transfers = [t for t in TRANSFERS_ORDER if t in entry["transfers"]]
            trials.append(_record(entry["spec"], "solve", transfers, entry["finals"], leg,
                                  axis, ordinal))
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
