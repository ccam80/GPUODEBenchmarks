"""The runner loop shared by the Python packages: a trial file in, one store row per trial and transfers out. A package supplies an adapter with `version()`, `states(trial)`, `build(trial, cold)`, `compile(build, trial, values)`, `optimize(build, trial)` (returns a text for the log), `solve(build, trial, values, transfers)` and `finals(build, result)`, plus a `controllers` tuple and an optional `reset(build, trial, values, transfers)` that runs untimed before every attempt after the first; `main(argv, make_adapter)` is the `--trials <path> [--floor]` entry."""

import argparse
import gc
import json
import os
import sys
import timeit
from datetime import datetime, timezone

import grid as grid_mod
import store as store_mod
import trials as trials_mod
from abandon import History
from bench_key import dataset_key
from protocol import OPTIMIZE_SECONDS, REPEAT_CAP, WATCHDOG_SECONDS
from wp_common import run_watchdogged, timed_min_ms

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(REPO_ROOT, "data")

NAN = float("nan")
OUTCOMES = ("ok", "timeout", "oom", "error")
STAGES = ("build", "optimize", "solve")
# Exception text that names an exhausted device or host: CUDA, numba, XLA and torch.
OOM_MARKERS = ("OUT_OF_MEMORY", "CUDA_ERROR_OUT_OF_MEMORY", "RESOURCE_EXHAUSTED",
               "OutOfMemoryError")
MESSAGE_CHARS = 200


def classify(exc):
    """'oom' when the exception is a MemoryError or its type or message carries an OOM_MARKERS entry, else 'error'."""
    text = type(exc).__name__ + ": " + str(exc)
    if isinstance(exc, MemoryError) or any(marker in text for marker in OOM_MARKERS):
        return "oom"
    return "error"


def failure_reason(outcome, exc=None, elapsed_s=None, cap_s=None):
    """The reason text of a failed (trial, transfers): 'timeout: <s>s over the <cap>s cap', or '<oom|error>: <Type>: <message>'."""
    if outcome == "timeout":
        cap = WATCHDOG_SECONDS if cap_s is None else cap_s
        return "timeout: {0:.1f}s over the {1:g}s cap".format(elapsed_s, cap)
    return "{0}: {1}: {2}".format(outcome, type(exc).__name__, str(exc)[:MESSAGE_CHARS])


def budget_of(trial):
    """The trial's watchdog soft cap in seconds."""
    return float(trial.get("watchdog_s", WATCHDOG_SECONDS))


def write_progress(path, trial, stage):
    """<trials>.progress: the trial under way, its stage (build, optimize, solve) and the start time."""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"trial_id": trial["trial_id"], "stage": stage,
                   "started_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")},
                  handle)


def watchdogged(run, what, budget_s):
    """run() under the watchdog's hard exit after budget_s seconds."""
    def breach():
        print("WATCHDOG hard exit: {0} never returned".format(what), flush=True)

    return run_watchdogged(run, breach, budget_s)


def label(trial, transfers=None):
    text = "{0} {1} {2} n={3}".format(trial["problem"], trial["algorithm"], trial["controller"],
                                      trial["n"])
    if trial["controller"] == "fixed":
        text += " dt={0:g}".format(trial["dt"])
    else:
        text += " tol={0:g}".format(trial["atol"])
    if trial["system_params"] not in ("", "{}"):
        text += " " + trial["system_params"]
    if transfers:
        text += " [" + transfers + "]"
    return text


class Runner:
    """One trial file through an adapter: a build kept while consecutive lines share it, every finished trial recorded before the next starts."""

    def __init__(self, adapter, key, root=DATA_ROOT, floor=False, repeats=REPEAT_CAP):
        self.adapter = adapter
        self.key = key
        self.store = store_mod.Store(root)
        self.floor = floor
        self.repeats = repeats
        self.package_version = adapter.version()
        self.suite_rev = store_mod.suite_rev(REPO_ROOT)
        self.build = None
        self.build_key = None

    # -------------------------------------------------------------- rows
    def record(self, trial, transfers, states, **values):
        spec = {field: trial[field] for field in store_mod.TRIAL_FIELDS}
        row = dict(spec, transfers=transfers, key=self.key, states=states,
                   package_version=self.package_version, suite_rev=self.suite_rev)
        row.update(values)
        return self.store.record(row, floor=self.floor)

    def record_failed(self, trial, reason, states=None):
        """Every requested transfers row of a trial as NaN with one reason."""
        count = self.adapter.states(trial) if states is None else states
        for transfers in trial["transfers"]:
            self.record(trial, transfers, count, reason=reason)
        if trial["transfers"]:
            print("FAILED {0}: {1}".format(label(trial), reason), flush=True)

    # ------------------------------------------------------------- timing
    def time(self, build, trial, values, transfers):
        """(outcome, best_ms, samples, result, exc, elapsed_s) of one (trial, transfers): the untimed warm-up then the repeat schedule, a never-returning run hard-exiting through the watchdog."""
        def run():
            return self.adapter.solve(build, trial, values, transfers)

        def breach():
            print("WATCHDOG hard exit: {0} never returned".format(label(trial, transfers)),
                  flush=True)

        reset = getattr(self.adapter, "reset", None)
        setup = None if reset is None else (lambda: reset(build, trial, values, transfers))
        try:
            best, result, samples = timed_min_ms(run, self.repeats, on_breach=breach, setup=setup,
                                                 cap_s=budget_of(trial))
        except Exception as exc:  # noqa: BLE001 - every failure is a row
            return classify(exc), NAN, [], None, exc, NAN
        if best is None:
            return "timeout", NAN, samples, result, None, samples[-1] / 1000.0
        return "ok", best, samples, result, None, samples[-1] / 1000.0

    def finals(self, build, trial, result):
        """(errored_pct, finals path) of a solve's result; the finals file is written when the trial keeps finals."""
        states, t_final, retcode = self.adapter.finals(build, result)
        pct = store_mod.errored_pct(states, t_final, retcode, trial["duration"])
        path = ""
        if trial["finals"]:
            spec = {field: trial[field] for field in store_mod.TRIAL_FIELDS}
            path = self.store.record_finals(dict(spec, key=self.key), states, t_final, retcode)
        return pct, path

    def run_solve(self, build, trial, history, build_s):
        values = grid_mod.grid(trial)
        pct, finals_path = NAN, ""
        finals_read = False
        for transfers in trial["transfers"]:
            reason = history.reason(trial, transfers)
            if reason is not None:
                self.record(trial, transfers, build.states, reason=reason, build_s=build_s)
                print("SKIP {0}: {1}".format(label(trial, transfers), reason), flush=True)
                continue
            outcome, best, samples, result, exc, elapsed = self.time(build, trial, values, transfers)
            history.add(trial, transfers, outcome)
            if result is not None and not finals_read:
                try:
                    pct, finals_path = self.finals(build, trial, result)
                    finals_read = True
                except Exception as finals_exc:  # noqa: BLE001 - the timing row still stands
                    print("FINALS {0}: {1}: {2}".format(label(trial), type(finals_exc).__name__,
                                                        finals_exc), flush=True)
            result = None
            reason = "" if outcome == "ok" else failure_reason(outcome, exc, elapsed, budget_of(trial))
            self.record(trial, transfers, build.states, min_ms=best, samples_ms=samples,
                        errored_pct=pct, build_s=build_s, finals=finals_path, reason=reason)
            if outcome == "ok":
                print("{0}: {1:.3f} ms over {2} attempts, errored {3:.1f}%".format(
                    label(trial, transfers), best, len(samples), pct), flush=True)
            else:
                print("FAILED {0}: {1}".format(label(trial, transfers), reason), flush=True)
        gc.collect()

    # ------------------------------------------------------------- builds
    def close_build(self):
        if self.build is not None:
            self.build.close()
            self.build = None
            self.build_key = None
            gc.collect()

    def ensure_build(self, trial):
        """(build, build_s): the build the trial runs on, kept from the last line when it serves it; a cold line rebuilds in a fresh cache and its build and compile wall time is build_s."""
        key = trials_mod.build_key(trial)
        cold = bool(trial["cold"])
        if self.build is not None and key == self.build_key and not cold:
            return self.build, NAN
        self.close_build()
        started = timeit.default_timer()
        self.build = self.adapter.build(trial, cold)
        self.build_key = key
        if cold:
            self.adapter.compile(self.build, trial, grid_mod.grid(trial))
            build_s = timeit.default_timer() - started
            print("built {0} cold in {1:.1f}s".format(label(trial), build_s), flush=True)
            return self.build, build_s
        return self.build, NAN

    def run_trial(self, trial, history, progress_path, failed_builds):
        if trial["controller"] not in self.adapter.controllers:
            self.record_failed(trial, "error: unknown controller " + trial["controller"])
            return
        key = trials_mod.build_key(trial)
        if key in failed_builds:
            self.record_failed(trial, failed_builds[key])
            return
        write_progress(progress_path, trial, "build")
        try:
            build, build_s = self.ensure_build(trial)
        except Exception as exc:  # noqa: BLE001 - the trial's rows carry the reason
            failed_builds[key] = failure_reason(classify(exc), exc)
            self.close_build()
            self.record_failed(trial, failed_builds[key])
            return
        if trial["optimize"]:
            # Past OPTIMIZE_SECONDS the watchdog hard-exits; the driver drops the optimize from this line and re-runs it.
            write_progress(progress_path, trial, "optimize")
            started = timeit.default_timer()
            try:
                done = watchdogged(lambda: self.adapter.optimize(build, trial),
                                   "optimize " + label(trial), OPTIMIZE_SECONDS)
                print("optimized {0} per {1}: {2} in {3:.1f}s".format(
                    label(trial), trial["optimize"], done, timeit.default_timer() - started), flush=True)
            except Exception as exc:  # noqa: BLE001 - the solves run at the solver's own geometry
                print("OPTIMIZE {0} failed: {1}".format(
                    label(trial), failure_reason(classify(exc), exc)), flush=True)
        elif not trial["cold"] and not trial["transfers"]:
            self.adapter.compile(build, trial, grid_mod.grid(trial))
        if trial["transfers"]:
            write_progress(progress_path, trial, "solve")
            self.run_solve(build, trial, history, build_s)

    def run_file(self, path):
        """Every trial of a file in order; returns 0."""
        trial_list = trials_mod.read_jsonl(path)
        progress_path = path + ".progress"
        history = History()
        failed_builds = {}
        try:
            for trial in trial_list:
                self.run_trial(trial, history, progress_path, failed_builds)
        finally:
            self.close_build()
        return 0


def parse_args(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--trials", required=True)
    parser.add_argument("--floor", action="store_true")
    return parser.parse_args(argv)


def main(argv, make_adapter, key=None, root=DATA_ROOT):
    """Run a trial file: `make_adapter(key, root)` gives the package adapter; the dataset key is this machine's unless given."""
    args = parse_args(argv)
    key = key or dataset_key()
    adapter = make_adapter(key, root)
    runner = Runner(adapter, key, root, floor=args.floor)
    return runner.run_file(args.trials)


if __name__ == "__main__":
    sys.exit("runner.py is an entry through a package adapter; see cubie_bench.py")
