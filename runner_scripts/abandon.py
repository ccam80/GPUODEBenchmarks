"""The abandon rule shared by the runners and bench.py: a run that timed out or ran out of memory abandons every harder run of its family on the same transfers; the rows a hard exit implies; the trials still to run; the compile timeouts the store records per compile_key (`compile = compile_timeout` rows), which mark every line of the group in every later plan until `store.py clear` drops them."""

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


def crashed_builds(progress_path):
    """The builds a hard exit's progress file lists under `failed`; [] when it lists none or cannot be read."""
    try:
        with open(progress_path, encoding="utf-8") as handle:
            failed = json.load(handle).get("failed", [])
    except (OSError, ValueError, AttributeError):
        return []
    return [str(name) for name in failed] if isinstance(failed, list) else []


def remaining(trial_list, current, doomed=()):
    """The trials from `current` on in file order, outside `doomed`, that list transfers: a runner takes its file in order, so every line before the hung one has run, whatever rows the store held before the run."""
    index = next(i for i, t in enumerate(trial_list) if t["trial_id"] == current["trial_id"])
    return [t for t in trial_list[index:] if t["transfers"] and t["trial_id"] not in doomed]


def compile_timeouts(data, key, package=None):
    """The compile_key groups the store records a compile timeout of under a key (one package's when named), as a set of tuples."""
    filters = {"key": key, "compile": store.COMPILE_TIMEOUT}
    if package:
        filters["package"] = package
    return {trials_mod.compile_key(row) for row in data.rows(**filters)}


def compile_timed_out(data, key, trial):
    """True when the trial's own results file holds a compile_timeout row of its compile_key; one file read, for a precompile worker's check before each kernel."""
    path = data.results_path(trial["package"], key, trial["problem"], trial["algorithm"])
    group = trials_mod.compile_key(trial)
    return any(row["compile"] == store.COMPILE_TIMEOUT and trials_mod.compile_key(row) == group
               for row in data._read_results(path))


def abandon_compile(data, key, trial_list, timed_out, suite_rev=""):
    """Record the compile timeout of `timed_out` (a line of the kernel the watchdog took) for its compile_key: a NaN row, reason COMPILE_TIMEOUT_REASON, compile compile_timeout, for every transfers of every line of the group in `trial_list` that has no row yet; an existing row stands. Returns the rows written."""
    group = trials_mod.compile_key(timed_out)
    lines = [t for t in trial_list if trials_mod.compile_key(t) == group]
    if not lines:
        return []
    recorded_ids = {row["run_id"] for row in data.rows(key=key, package=timed_out["package"],
                                                       problem=timed_out["problem"],
                                                       algorithm=timed_out["algorithm"])}
    reason = store.COMPILE_TIMEOUT_REASON + timed_out["trial_id"]
    rows = []
    for trial in lines:
        for transfers, run in run_ids(trial, key).items():
            if run in recorded_ids:
                continue
            spec = {field: trial[field] for field in store.TRIAL_FIELDS}
            rows.append(dict(spec, transfers=transfers, key=key, states=states_of(trial), reason=reason,
                             compile=store.COMPILE_TIMEOUT, suite_rev=suite_rev))
    if rows:
        data.record_batch(rows)
    return rows


def abandon_after_hard_exit(data, key, trial_list, progress_path, suite_rev):
    """The trials still to run after a hard exit, or None when the progress file names no trial: a hard exit while solving abandons the named trial (the rows of the transfers it ran) and every harder one of its family (each transfers row still absent); one during an optimize records the kernel's optimize timeout and the compile timeout of its compile_key (abandon_compile), then marks every remaining line of the group, whose optimize goes with it."""
    try:
        with open(progress_path, encoding="utf-8") as handle:
            progress = json.load(handle)
    except (OSError, ValueError):
        return None
    current = next((t for t in trial_list if t["trial_id"] == progress.get("trial_id")), None)
    if current is None:
        return None
    if progress.get("stage") == "optimize":
        if current["package"] in cubie_adapter.PACKAGES:
            cubie_adapter.record_optimize_timeout(current, key, data.root)
        abandon_compile(data, key, trial_list, current, suite_rev)
        return trials_mod.mark_compile_timeouts(remaining(trial_list, current), {trials_mod.compile_key(current)})
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
