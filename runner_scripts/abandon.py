"""Watchdog hard-exit bookkeeping shared by the run loop and the julia driver: which transfers rows of a trial exist, the abandoned rows a runner's progress file implies, and the trials still to run."""

import json

import store
from problems import get_problem


def states_of(trial):
    """The state count a trial's system will have, from its construction parameters or the catalogue."""
    params = json.loads(trial["system_params"]) if trial["system_params"] else {}
    if "states" in params:
        return int(params["states"])
    return int(get_problem(trial["problem"])["states"])


def run_ids(trial, key):
    """{transfers: run_id} of a solve trial under a key."""
    return {t: store.run_id(dict(trial, transfers=t, key=key)) for t in trial["transfers"]}


def recorded(data, key, trial):
    """{transfers: True when a row exists} of a solve trial."""
    return {t: bool(data.rows(run_id=run)) for t, run in run_ids(trial, key).items()}


def remaining(data, key, trial_list, doomed=()):
    """The solve trials outside `doomed` with a transfers row still missing, and the warm and optimize lines of their legs, in file order."""
    kept = []
    live_legs = set()
    for trial in trial_list:
        if trial["kind"] != "solve" or trial["trial_id"] in doomed:
            continue
        if not all(recorded(data, key, trial).values()):
            kept.append(trial)
            live_legs.add(trial["leg"])
    return [t for t in trial_list if t["kind"] == "solve" and t in kept
            or t["kind"] != "solve" and t["leg"] in live_legs]


def abandon_after_hard_exit(data, key, trial_list, progress_path, suite_rev):
    """Record the leg's ordinals from the one the progress file names as abandoned (every requested transfers row still absent); returns the trials still without a row, or None when the progress file names no trial. A hard exit on an optimize line records nothing and drops that line, so the leg's solves run at the solver's own geometry."""
    try:
        with open(progress_path, encoding="utf-8") as handle:
            progress = json.load(handle)
        current = [t for t in trial_list if t["trial_id"] == progress["trial_id"]]
    except (OSError, ValueError, KeyError):
        current = []
    if not current:
        return None
    kind = progress.get("kind")
    if kind == "optimize":
        return [t for t in remaining(data, key, trial_list)
                if not (t["kind"] == "optimize" and t["trial_id"] == progress["trial_id"])]
    current.sort(key=lambda t: t["kind"] != "solve")
    leg, ordinal = current[0]["leg"], current[0]["ordinal"]
    reason = "abandoned: hard-exit at ordinal {0}".format(ordinal)
    rows = []
    doomed = set()
    for trial in trial_list:
        if trial["kind"] != "solve" or trial["leg"] != leg or trial["ordinal"] < ordinal:
            continue
        doomed.add(trial["trial_id"])
        for transfers, present in recorded(data, key, trial).items():
            if present:
                continue
            spec = {field: trial[field] for field in store.TRIAL_FIELDS}
            rows.append(dict(spec, transfers=transfers, key=key, states=states_of(trial),
                             reason=reason, suite_rev=suite_rev))
    if rows:
        data.record_batch(rows)
    return remaining(data, key, trial_list, doomed)
