#!/usr/bin/env python3
"""bench.py plan|run --set <name>[,<name>] [-p pkgs] [-s problems] [-g algorithms] [--mode fixed|adaptive] [--controller names] [-n list] [--tol list] [--dt list] [--resume | --no-overwrite] [--floor] [--cooldown S] [--allow-unknown-gpu] [--lock-clocks SM[,MEM]] [--clock-tolerance MHZ] [--no-sync]

plan writes trials/<key>/<package>.jsonl and prints counts; run writes them under logs/<key>_<stamp>/ and drives each package's runner.
-p -s -g -n --mode --controller --tol --dt narrow the expanded specs; -n names counts of the grids' n lists and exits for a count no grid of the named sets lists; --controller takes a spec controller or a set token such as matched.
A trial is one line per point; a point declared by several set files runs under one contract whatever sets are named: cold, finals and transfers each true over its declarations, optimize true over its declarations (a cubie optimize runs once per compiled kernel, timing the batch and duration cubie sizes itself), the watchdog budget the largest.
Without a flag every selected trial runs, its rows are overwritten and its cubie kernel is optimized again. --resume runs the trials the store lacks rows of, keeping a recorded NaN or error row and a timed-out optimize; --no-overwrite runs every trial without a finite time; under either a trial lacking a requested output (a cold build time, a readable finals file) or its kernel's optimize record (timed out, under --no-overwrite) runs whole, so its timing, build time and finals come from one execution. A recorded row is never rerun for its age or the source it was recorded from. --floor lets runners keep the lower finite time.
run pulls the store into data/ before planning and pushes this key after the runners (sync/sync.py); the pull keeps a local file newer than the box's; a run refuses to start while this key's local partition holds files the box lacks or differs from, until they are pushed or the partition deleted; a machine without the store refuses to run unless --no-sync.
A run locks the GPU clocks to --lock-clocks or the card's row in runner_scripts/gpu_clocks.conf and refuses to start when it cannot (no row, no elevation, driver refusal, a sampler that dies at once); there is no unlocked run. It samples the clocks at 25 Hz into data/clocks/<run>.csv (pushed with the key), every row records the run, driver and lock (GPUODE_RUN, GPUODE_DRIVER, GPUODE_CLOCK_LOCK_MHZ in the runners' environment) and the host stamps around its timing batch (timed_start_utc, timed_end_utc), and after each package the timed rows it recorded get the clocks that window of the log showed (clock_sm_mhz, clock_sm_min_mhz, clock_throttled; a window the sampler did not cover stays NaN).
The push has the box delete this key's clock logs a day or older that no row on the box names, and drops them and their logs/<run>/ dirs here.
Exit 0 when every runner finished; 1 on a runner failure, a locked row that throttled or fell more than --clock-tolerance below the lock, or a failed push.
"""

import argparse
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "runner_scripts"))
sys.path.insert(0, os.path.join(ROOT, "sync"))

import launch  # noqa: E402


def _under_suite_python():
    """Re-run under the suite interpreter when this one lacks the store's dependencies."""
    try:
        import duckdb  # noqa: F401
        import pyarrow  # noqa: F401
    except ImportError:
        python = launch.suite_python()
        if os.path.abspath(python) == os.path.abspath(sys.executable):
            raise
        raise SystemExit(subprocess.call([python] + sys.argv))


_under_suite_python()

import completeness  # noqa: E402
import cubie_adapter  # noqa: E402
import sets  # noqa: E402
import store  # noqa: E402
import sync  # noqa: E402
import trials as trials_mod  # noqa: E402
from abandon import abandon_after_hard_exit, crashed_builds  # noqa: E402
from algorithms import algorithm_names  # noqa: E402
from bench_key import dataset_key  # noqa: E402
from clocks import (TOL_MHZ, ClockError, ClockGuard, configure as configure_clocks,  # noqa: E402
                    driver_version, drifted, load_samples, window_stats)
from problems import problem_names  # noqa: E402
from protocol import WATCHDOG_EXIT_CODE  # noqa: E402

TRIALS_DIR = os.path.join(ROOT, "trials")
LOGS_DIR = os.path.join(ROOT, "logs")
DATA_DIR = os.path.join(ROOT, "data")


def parse_list(text):
    return [tok.strip() for tok in text.split(",") if tok.strip()]


def parse_floats(text, flag):
    try:
        return [float(tok) for tok in parse_list(text)]
    except ValueError:
        raise SystemExit("{0} takes a comma list of numbers, got '{1}'".format(flag, text))


def parse_args(argv):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("command", nargs="?", default="")
    p.add_argument("--set", default="")
    p.add_argument("-p", "--package", default="")
    p.add_argument("-s", "--problem", default="")
    p.add_argument("-g", "--algorithm", default="")
    p.add_argument("--mode", default="")
    p.add_argument("--controller", default="")
    p.add_argument("-n", default="")
    p.add_argument("--tol", default="")
    p.add_argument("--dt", default="")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no-overwrite", action="store_true")
    p.add_argument("--floor", action="store_true")
    p.add_argument("--cooldown", type=int, default=15)
    p.add_argument("--allow-unknown-gpu", action="store_true")
    p.add_argument("--lock-clocks", default="")
    p.add_argument("--clock-tolerance", type=int, default=TOL_MHZ)
    p.add_argument("--no-sync", action="store_true")
    p.add_argument("-h", "--help", action="store_true")
    args = p.parse_args(argv)
    if args.help:
        print(__doc__)
        raise SystemExit(0)
    if args.command not in ("plan", "run"):
        raise SystemExit("bench.py plan|run --set <name>[,<name>] ...; -h for the flags")
    if not args.set:
        raise SystemExit("--set names at least one set under sets/")
    if args.resume and args.no_overwrite:
        raise SystemExit("--resume and --no-overwrite exclude each other")
    return args


def resolve(args):
    """The expansion inputs from the flags, every name validated."""
    plan = {"sets": parse_list(args.set)}
    for name in plan["sets"]:
        try:
            sets.set_path(name)
        except sets.SetError as exc:
            raise SystemExit(str(exc))
    packages = [tok.replace("-", "_") for tok in parse_list(args.package)]
    for package in packages:
        if package not in store.PACKAGES:
            raise SystemExit("Unknown package '{0}' ({1})".format(package, "|".join(store.PACKAGES)))
    plan["packages"] = launch.ordered(list(dict.fromkeys(packages))) or None
    problems = parse_list(args.problem)
    known = problem_names()
    for name in problems:
        if name not in known:
            raise SystemExit("Unknown problem '{0}' ({1})".format(name, "|".join(known)))
    plan["problems"] = problems or None
    algorithms = parse_list(args.algorithm)
    known = algorithm_names()
    for name in algorithms:
        if name not in known:
            raise SystemExit("Unknown algorithm '{0}' ({1})".format(name, "|".join(known)))
    plan["algorithms"] = algorithms or None
    if args.mode and args.mode not in ("fixed", "adaptive"):
        raise SystemExit("--mode takes fixed or adaptive, got '{0}'".format(args.mode))
    plan["mode"] = args.mode or None
    plan["controllers"] = parse_list(args.controller) or None
    try:
        counts = [int(tok) for tok in parse_list(args.n)]
    except ValueError:
        raise SystemExit("-n takes a comma list of integers, got '{0}'".format(args.n))
    if any(count < 2 for count in counts):
        raise SystemExit("-n counts must be at least 2")
    plan["n"] = sorted(set(counts)) or None
    plan["tols"] = parse_floats(args.tol, "--tol") if args.tol else None
    plan["dts"] = parse_floats(args.dt, "--dt") if args.dt else None
    return plan


# ------------------------------------------------------------------ planning

def continue_filter(trial_list, key, root, resume=False, no_overwrite=False):
    """Trials still to run: without a flag every trial; else each keeps the transfers completeness.audit finds lacking (every one when the optimize record, the build time or the finals are lacking, so a trial's outputs come from one execution); a line without transfers never runs again. A recorded row is never rerun for its age or source."""
    if not (resume or no_overwrite):
        return list(trial_list)
    mode = "no_overwrite" if no_overwrite else "resume"
    audits = completeness.audit(trial_list, key, store.Store(root), mode)
    kept = []
    for trial in trial_list:
        missing = audits.get(trial["trial_id"])
        if missing is None or missing.complete():
            continue
        kept.append(dict(trial, transfers=missing.transfers(), finals=missing.wants_finals,
                         optimize=trial["optimize"] and not missing.optimize_timed_out))
    return kept


def canonical_trials(plan, key, root, sets_dir=sets.SETS_DIR):
    """The trials of the flags: the named sets' specs, narrowed, each merged with its declarations in every set file; SystemExit when -n names a count no grid of the named sets lists."""
    unknown = sorted(set(plan["n"] or []) - set(sets.declared_counts(plan["sets"], sets_dir)))
    if unknown:
        raise SystemExit("-n {0}: no grid of {1} lists {2}".format(
            ",".join(str(c) for c in plan["n"]), ",".join(plan["sets"]),
            ", ".join(str(c) for c in unknown)))
    narrowing = dict(packages=plan["packages"], problems=plan["problems"],
                     algorithms=plan["algorithms"], n=plan["n"], sets_dir=sets_dir)
    specs = sets.expand(plan["sets"], key, root, **narrowing)
    specs = sets.narrow(specs, mode=plan["mode"], controllers=plan["controllers"],
                        tols=plan["tols"], dts=plan["dts"])
    return trials_mod.build_trials(specs, sets.declarations(key, root, **narrowing))


def plan_trials(plan, key, root, resume=False, no_overwrite=False):
    """{package: trials} for the flags, in run order."""
    all_trials = continue_filter(canonical_trials(plan, key, root), key, root, resume, no_overwrite)
    groups = trials_mod.by_package(all_trials)
    return {package: groups[package] for package in launch.ordered(list(groups))}


def write_plan(directory, by_package):
    """{package: path} of the trial files written under directory."""
    os.makedirs(directory, exist_ok=True)
    return {package: trials_mod.write_jsonl(os.path.join(directory, package + ".jsonl"), rows)
            for package, rows in by_package.items()}


def print_counts(by_package):
    total = 0
    for package, rows in by_package.items():
        solves, optimizes, colds, builds = trials_mod.counts(rows)
        total += solves
        print("{0}: {1} trials, {2} optimize, {3} cold, {4} builds".format(
            package, solves, optimizes, colds, builds))
        for key, lines in trials_mod.builds_of(rows):
            print("  {0}  {1}".format("/".join(str(k) for k in key), len(lines)))
    print("{0} trials".format(total))


# ---------------------------------------------------------------------- run

class Run:
    """One invocation: log dir, clock guard, manifest, summary and the runner loop; the run name, driver and lock go into the environment every row is recorded under."""

    def __init__(self, args, plan, key=None, data_root=DATA_DIR, logs_root=LOGS_DIR):
        self.args, self.plan = args, plan
        self.key = key or dataset_key()
        if self.key.endswith("_unknown-gpu") and not args.allow_unknown_gpu:
            raise SystemExit("Could not identify the GPU; the dataset key would be '{0}'. Fix the driver "
                             "or pass --allow-unknown-gpu.".format(self.key))
        self.data_root = data_root
        self.store = store.Store(data_root)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run = "{0}_{1}".format(self.key, stamp)
        self.log_dir = os.path.join(logs_root, self.run)
        os.makedirs(self.log_dir, exist_ok=True)
        self.clocks_csv = os.path.join(data_root, "clocks", self.run + ".csv")
        self.summary = os.path.join(self.log_dir, "summary.tsv")
        open(self.summary, "w").close()
        self.clock_failures = 0
        self.clock_lines = []
        self.failures = 0
        self.partials = 0
        # Every run is locked: no target, no elevation or a driver refusal ends it here.
        try:
            sm, mem = configure_clocks(self.key, args.lock_clocks)
            self.clocks = ClockGuard(sm, mem, args.clock_tolerance)
            self.clocks.lock()
        except ClockError as exc:
            raise SystemExit("Clocks      : " + str(exc))
        self.clock_status = self.clocks.status()
        self.driver = driver_version()
        os.environ[store.RUN_ENV] = self.run
        os.environ[store.DRIVER_ENV] = self.driver
        os.environ[store.CLOCK_LOCK_ENV] = str(sm)
        os.environ[store.OVERWRITE_ENV] = "" if args.resume or args.no_overwrite else "1"

    # ------------------------------------------------------------- plumbing
    def record(self, stage, status, detail, code):
        with open(self.summary, "a", encoding="utf-8") as handle:
            handle.write("\t".join((stage, status, str(detail), str(code))) + "\n")
        if status == "FAILED":
            self.failures += 1
        if status == "PARTIAL":
            self.partials += 1

    def step(self, label, logfile, command):
        """Run one command with its output tee'd to a log; returns the exit code."""
        print("=" * 60)
        print("[{0}] {1}".format(datetime.now(timezone.utc).strftime("%H:%M:%SZ"), label))
        print("  " + subprocess.list2cmdline(command.argv))
        print("=" * 60, flush=True)
        start = time.monotonic()
        env = dict(os.environ, **command.env)
        with open(os.path.join(self.log_dir, logfile), "a", encoding="utf-8") as log:
            proc = subprocess.Popen(command.argv, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, errors="replace")
            for line in proc.stdout:
                sys.stdout.write(line)
                log.write(line)
            proc.stdout.close()
            status = proc.wait()
        elapsed = int(time.monotonic() - start)
        if status in command.ok:
            print("OK {0}  ({1}s)".format(label, elapsed))
        else:
            print("X {0} exited {1}  ({2}s)".format(label, status, elapsed))
        sys.stdout.flush()
        return status

    def annotate(self, package):
        """Fill the clock columns of the rows this run recorded for a package from the log so far; a locked row that throttled or fell below the lock counts as drift."""
        samples = load_samples(self.clocks_csv)
        lock = int(self.clocks.sm) if self.clocks.locked else 0
        seen = []

        def stats(start, end):
            window = window_stats(samples, start, end)
            if window is not None:
                seen.append(window)
            return window

        count = self.store.annotate(self.run, stats, package=package, key=self.key)
        drift = sum(1 for window in seen if drifted(window, lock, self.clocks.tol))
        busy = [w["clock_sm_mhz"] for w in seen if w["clock_sm_mhz"] == w["clock_sm_mhz"]]
        line = "{0}: {1} rows annotated".format(package, count)
        if busy:
            line += ", SM median {0:.0f} MHz, low {1:.0f} MHz".format(
                sorted(busy)[len(busy) // 2], min(w["clock_sm_min_mhz"] for w in seen
                                                  if w["clock_sm_min_mhz"] == w["clock_sm_min_mhz"]))
        if lock:
            line += ", {0} drifted".format(drift)
        self.clock_lines.append(line)
        print("Clocks      : " + line, flush=True)
        if drift:
            self.clock_failures += 1

    def cooldown(self):
        if self.args.cooldown > 0:
            time.sleep(self.args.cooldown)

    # ------------------------------------------------------------- packages
    def run_package(self, package, trial_list, path):
        """Drive one runner over its trial file, re-invoking after every watchdog hard exit; a build that crashed before a hard exit fails the package once the relaunches end."""
        logfile = package + ".log"
        hard_exits = 0
        crashed = []
        while True:
            command = launch.runner_command(package, path, floor=self.args.floor)
            status = self.step("{0} ({1} trials)".format(package, len(trial_list)), logfile, command)
            if status == 0:
                self.finish(package, status, hard_exits, crashed)
                return
            if status != WATCHDOG_EXIT_CODE:
                self.record(package, "FAILED", "runner exit {0}".format(status), status)
                return
            hard_exits += 1
            crashed += [name for name in crashed_builds(path + ".progress") if name not in crashed]
            remaining = abandon_after_hard_exit(self.store, self.key, trial_list, path + ".progress",
                                                store.suite_rev(ROOT))
            if remaining is None:
                self.record(package, "FAILED", "hard exit without a progress file", status)
                return
            if not remaining:
                self.finish(package, status, hard_exits, crashed)
                return
            trial_list = remaining
            path = os.path.join(os.path.dirname(path), "{0}.retry{1}.jsonl".format(package, hard_exits))
            trials_mod.write_jsonl(path, trial_list)

    def finish(self, package, status, hard_exits, crashed):
        """Record a package whose relaunches ended: FAILED when a build crashed before a hard exit, PARTIAL after a hard exit, else OK."""
        detail = "{0} hard exit(s)".format(hard_exits) if hard_exits else "-"
        if crashed:
            self.record(package, "FAILED", "{0}; crashed before a hard exit: {1}".format(detail, ", ".join(crashed)),
                        status)
        elif hard_exits:
            self.record(package, "PARTIAL", detail, status)
        else:
            self.record(package, "OK", detail, status)

    # -------------------------------------------------------------- lifecycle
    def manifest(self, by_package, finished=False):
        path = os.path.join(self.log_dir, "run_manifest.txt")
        if finished:
            with open(path, "a", encoding="utf-8") as handle:
                handle.write("finished_utc={0}\n".format(datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")))
            return
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True)
        lines = ["run=" + self.run,
                 "dataset_key=" + self.key,
                 "started_utc=" + datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                 "sets=" + ",".join(self.plan["sets"]),
                 "argv=" + subprocess.list2cmdline(sys.argv[1:]),
                 "packages=" + ",".join(by_package),
                 "trials=" + ";".join("{0}={1}".format(p, len(t)) for p, t in by_package.items()),
                 "suite_rev=" + store.suite_rev(ROOT),
                 "host={0} {1}".format(platform.node(), sys.platform),
                 "driver=" + self.driver,
                 "clocks=" + self.clock_status,
                 "clocks_csv=" + self.clocks_csv]
        lines += ["gpu=" + line.strip() for line in gpu.stdout.splitlines() if line.strip()]
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")

    def summarise(self):
        print("")
        print("=" * 60)
        print("RUN SUMMARY  ({0})".format(self.key))
        print("=" * 60)
        print("{0:<26} {1:<16} {2}".format("PACKAGE", "STATUS", "DETAIL"))
        with open(self.summary, encoding="utf-8") as handle:
            for line in handle:
                cols = line.rstrip("\n").split("\t")
                if len(cols) >= 3:
                    print("{0:<26} {1:<16} {2}".format(cols[0], cols[1], cols[2]))
        print("=" * 60)
        for line in self.clock_lines:
            print("Clocks: " + line)
        print("Logs: " + self.log_dir)
        print("Clocks: {0}  (25 Hz log in {1})".format(self.clock_status, self.clocks_csv))
        if self.partials:
            print("{0} package(s) partial: a watchdog hard exit abandoned part of a build.".format(self.partials))
        if self.clock_failures:
            print("{0} package(s) have rows that drifted under the lock; lower the lock in "
                  "runner_scripts/gpu_clocks.conf and re-run them.".format(self.clock_failures))
        if self.failures:
            print("{0} runner(s) failed outright.".format(self.failures))
        return 1 if (self.failures or self.clock_failures) else 0

    def _execute(self):
        by_package = plan_trials(self.plan, self.key, self.data_root,
                                 self.args.resume, self.args.no_overwrite)
        if any(package.startswith("julia") for package in by_package):
            launch.check_julia_project()
        paths = write_plan(self.log_dir, by_package)
        print("Dataset key : " + self.key)
        print("Sets        : " + ", ".join(self.plan["sets"]))
        print("Packages    : " + ", ".join(by_package))
        print("Log dir     : " + self.log_dir)
        print("Run         : " + self.run)
        print("Clocks      : {0}, driver {1}".format(self.clock_status, self.driver or "unknown"))
        print("", flush=True)
        print_counts(by_package)
        self.manifest(by_package)
        try:
            self.clocks.start_monitor(self.clocks_csv)
        except ClockError as exc:
            raise SystemExit("Clocks      : " + str(exc))
        try:
            for index, (package, rows) in enumerate(by_package.items()):
                if index:
                    self.cooldown()
                self.run_package(package, rows, paths[package])
                self.annotate(package)
            return self.summarise()
        finally:
            self.manifest(by_package, finished=True)

    def execute(self):
        """Plan, run and summarise; the clock lock taken in __init__ is released whatever fails, planning included."""
        try:
            return self._execute()
        finally:
            self.clocks.stop_monitor()
            self.clocks.reset()


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    plan = resolve(args)
    os.chdir(ROOT)
    key = dataset_key()
    if key.endswith("_unknown-gpu") and not args.allow_unknown_gpu:
        raise SystemExit("Could not identify the GPU; the dataset key would be '{0}'. Fix the driver "
                         "or pass --allow-unknown-gpu.".format(key))
    if args.command == "plan":
        by_package = plan_trials(plan, key, DATA_DIR, args.resume, args.no_overwrite)
        paths = write_plan(os.path.join(TRIALS_DIR, key), by_package)
        print_counts(by_package)
        for path in paths.values():
            print(os.path.relpath(path, ROOT))
        return 0
    pull_store(key, skip=args.no_sync)
    code = Run(args, plan, key=key).execute()
    return max(code, push_store(key, skip=args.no_sync))


def pull_store(key, root=DATA_DIR, skip=False):
    """Fill the local mirror from the store before a run; SystemExit when the store is not set up, this key's local partition holds files the box lacks or differs from, or the pull fails."""
    if skip:
        print("Store       : not used (--no-sync)")
        return
    reason = sync.unavailable()
    if reason:
        raise SystemExit("Store       : {0}; pass --no-sync to run without it".format(reason))
    remote = sync.remote_default()
    # The push after the runners needs the box-side pruner.
    reason = sync.box_ready(remote, key=key)
    if reason:
        raise SystemExit("Store       : {0}; pass --no-sync to run without the store".format(reason))
    if sync.partition_has_files(root, key):
        print("Store       : unpushed check of key={0} against {1}".format(key, remote), flush=True)
        if sync.run("unpushed", root, key):
            raise SystemExit(
                "Store       : data/key={0} holds files the box lacks or differs from; `python sync/sync.py push` "
                "keeps them, deleting data/key={0} takes the box's; pass --no-sync to run without the store".format(key))
    print("Store       : pull from " + remote, flush=True)
    if sync.run("pull", root, key):
        raise SystemExit("Store       : pull FAILED; pass --no-sync to run without it")


def push_store(key, root=DATA_DIR, skip=False, logs_root=LOGS_DIR):
    """Copy this key and its clocks files to the store after a run, under the key's lock on the box, and have the box prune the clock logs no row names (their logs/<run>/ dirs go too); 0 when done or skipped, 1 on failure."""
    if skip:
        return 0
    print("Store       : push to " + sync.remote_default(), flush=True)
    code = sync.run("push", root, key, logs_dir=logs_root)
    print("Store       : " + ("pushed" if code == 0 else "push FAILED (exit {0})".format(code)))
    return 1 if code else 0


if __name__ == "__main__":
    sys.exit(main())
