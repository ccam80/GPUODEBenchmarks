"""The abandon rule shared by the runners and bench.py: a run that timed out or ran out of memory abandons every harder run of its family on the same transfers; the rows a hard exit implies; the trials still to run."""

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


def remaining(trial_list, current, doomed=()):
    """The trials from `current` on in file order, outside `doomed`, that list transfers: a runner takes its file in order, so every line before the hung one has run, whatever rows the store held before the run."""
    index = next(i for i, t in enumerate(trial_list) if t["trial_id"] == current["trial_id"])
    return [t for t in trial_list[index:] if t["transfers"] and t["trial_id"] not in doomed]


def abandon_after_hard_exit(data, key, trial_list, progress_path, suite_rev):
    """The trials still to run after a hard exit, or None when the progress file names no trial: a hard exit while solving abandons the named trial (the rows of the transfers it ran) and every harder one of its family (each transfers row still absent); one during an optimize records a timeout row and drops the optimize from the line and, per kernel, from every line of its kernel."""
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
        kernel = trials_mod.kernel_key(current)

        def dropped(trial):
            return trial["trial_id"] == current["trial_id"] or (
                current["optimize"] == "kernel" and trial["optimize"] == "kernel"
                and trials_mod.kernel_key(trial) == kernel)

        return [dict(t, optimize=None) if dropped(t) else t for t in remaining(trial_list, current)]
    reason = "abandoned: hard-exit at " + current["trial_id"]
    doomed = set()
    rows = []
    for trial in trial_list:
        hung = trial["trial_id"] == current["trial_id"]
        if not hung and not (trials_mod.family_key(trial) == trials_mod.family_key(current)
                             and trials_mod.harder(trial, current)):
            continue
        doomed.add(trial["trial_id"])
        for transfers, present in recorded(data, key, trial).items():
            if present and not (hung and transfers in trial["transfers"]):
                continue
            spec = {field: trial[field] for field in store.TRIAL_FIELDS}
            rows.append(dict(spec, transfers=transfers, key=key, states=states_of(trial),
                             reason=reason, suite_rev=suite_rev))
    if rows:
        data.record_batch(rows)
    return remaining(trial_list, current, doomed)
