"""The abandon rule shared by the runners and the drivers: a run that timed out or ran out of memory abandons every harder run of its family on the same transfers; the rows a hard exit implies; the trials still to run."""

import json

import cubie_adapter
import store
import trials as trials_mod
from problems import get_problem


def states_of(trial):
    """The state count a trial's system will have, from its construction parameters or the catalogue."""
    params = json.loads(trial["system_params"]) if trial["system_params"] else {}
    if "states" in params:
        return int(params["states"])
    return int(get_problem(trial["problem"])["states"])


def run_ids(trial, key):
    """{transfers: run_id} of a trial under a key."""
    return {t: store.run_id(dict(trial, transfers=t, key=key)) for t in trial["transfers"]}


def recorded(data, key, trial):
    """{transfers: True when a row exists} of a trial."""
    return {t: bool(data.rows(run_id=run)) for t, run in run_ids(trial, key).items()}


def abandon_reason(trial, failures):
    """'abandoned: <outcome> at <trial_id>' when a failure of the trial's family is no harder than it, else None; `failures` is [(trial, outcome)]."""
    family = trials_mod.family_key(trial)
    for failed, outcome in failures:
        if trials_mod.family_key(failed) == family and trials_mod.harder(trial, failed):
            return "abandoned: {0} at {1}".format(outcome, failed["trial_id"])
    return None


class History:
    """The timeouts and OOMs seen so far, per transfers."""

    def __init__(self):
        self.failures = {}

    def add(self, trial, transfers, outcome):
        if outcome in ("timeout", "oom"):
            self.failures.setdefault(transfers, []).append((trial, outcome))

    def reason(self, trial, transfers):
        return abandon_reason(trial, self.failures.get(transfers, []))


def failed_runs(data, key, package):
    """[(row as a trial, outcome)] of the package's rows under a key whose reason names a timeout or oom."""
    out = []
    for row in data.rows(key=key, package=package):
        outcome = row["reason"].split(":")[0] if row["reason"] else ""
        if outcome in ("timeout", "oom"):
            out.append((dict(row, transfers=[row["transfers"]]), outcome, row["transfers"]))
    return out


def abandon_from_store(data, key, trial_list, suite_rev):
    """Record as abandoned every transfers of the trials the store's timeouts and OOMs of their family give up, and return the trials with what is left to run."""
    if not trial_list:
        return []
    by_transfers = {}
    for failed, outcome, transfers in failed_runs(data, key, trial_list[0]["package"]):
        by_transfers.setdefault(transfers, []).append((failed, outcome))
    kept = []
    rows = []
    for trial in trial_list:
        live = []
        for transfers in trial["transfers"]:
            reason = abandon_reason(trial, by_transfers.get(transfers, []))
            if reason is None:
                live.append(transfers)
                continue
            spec = {field: trial[field] for field in store.TRIAL_FIELDS}
            rows.append(dict(spec, transfers=transfers, key=key, states=states_of(trial), reason=reason,
                             suite_rev=suite_rev))
        if live or not trial["transfers"]:
            kept.append(dict(trial, transfers=live))
    if rows:
        data.record_batch(rows)
    return kept


def remaining(data, key, trial_list, doomed=()):
    """The trials outside `doomed` that list transfers and still lack a row for one of them, in file order."""
    return [t for t in trial_list if t["transfers"] and t["trial_id"] not in doomed
            and not all(recorded(data, key, t).values())]


def abandon_after_hard_exit(data, key, trial_list, progress_path, suite_rev):
    """The trials still to run after a hard exit, or None when the progress file names no trial. A hard exit while solving records the named trial and every harder one of its family as abandoned (each requested transfers row still absent); one during its optimize records an optimize.csv row labelled timeout and drops the optimize from that line."""
    try:
        with open(progress_path, encoding="utf-8") as handle:
            progress = json.load(handle)
        current = [t for t in trial_list if t["trial_id"] == progress["trial_id"]]
    except (OSError, ValueError, KeyError):
        current = []
    if not current:
        return None
    current = current[0]
    if progress.get("stage") == "optimize":
        if current["package"] in cubie_adapter.PACKAGES:
            cubie_adapter.record_optimize_timeout(current, key, data.root)
        return [dict(t, optimize=None) if t["trial_id"] == current["trial_id"] else t
                for t in remaining(data, key, trial_list)]
    reason = "abandoned: hard-exit at " + current["trial_id"]
    doomed = set()
    rows = []
    for trial in trial_list:
        if trial["trial_id"] != current["trial_id"] and not (
                trials_mod.family_key(trial) == trials_mod.family_key(current)
                and trials_mod.harder(trial, current)):
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
