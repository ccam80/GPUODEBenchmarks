#!/usr/bin/env python3
"""bench.py plan|run --set <name>[,<name>] [-p pkgs] [-s problems] [-g algorithms] [--mode fixed|adaptive] [--controller names] [-n list] [--tol list] [--dt list] [--resume | --no-overwrite] [--floor] [--cooldown S] [--allow-unknown-gpu] [--lock-clocks SM[,MEM]] [--no-lock-clocks] [--clock-tolerance MHZ]

plan writes trials/<key>/<package>.jsonl and prints counts; run writes them under logs/<key>_<stamp>/ and drives each package's runner.
-p -s -g --mode --controller --tol --dt narrow the expanded specs; -n replaces every grid's n list; --controller takes a spec controller or a set token such as matched.
--resume drops trials whose every transfers row exists; --no-overwrite those whose rows are all finite; --floor lets runners keep the lower finite time.
Exit 0 when every runner finished; 1 on a runner failure or clock drift.
"""

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "runner_scripts"))

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

import sets  # noqa: E402
import store  # noqa: E402
import trials as trials_mod  # noqa: E402
from abandon import abandon_after_hard_exit, run_ids  # noqa: E402
from algorithms import algorithm_names  # noqa: E402
from bench_key import dataset_key  # noqa: E402
from clocks import ClockGuard, configure as configure_clocks  # noqa: E402
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
    p.add_argument("--no-lock-clocks", action="store_true")
    p.add_argument("--clock-tolerance", type=int, default=None)
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
    """Trials still to run: under resume the solve trials with a missing transfers row, under no_overwrite those with a NaN or missing row; warm and optimize trials follow their leg."""
    if not (resume or no_overwrite):
        return list(trial_list)
    recorded = {}
    for row in store.Store(root).rows(key=key):
        recorded[row["run_id"]] = math.isfinite(row["min_ms"])
    kept = []
    live_legs = set()
    for trial in trial_list:
        if trial["kind"] != "solve":
            continue
        status = [recorded.get(run) for run in run_ids(trial, key).values()]
        covered = all(s is not None for s in status) if resume else all(status)
        if not covered:
            kept.append(trial)
            live_legs.add(trial["leg"])
    return [t for t in trial_list if t["kind"] == "solve" and t in kept
            or t["kind"] != "solve" and t["leg"] in live_legs]


def plan_trials(plan, key, root, resume=False, no_overwrite=False):
    """{package: trials} for the flags, in run order."""
    specs = sets.expand(plan["sets"], key, root, packages=plan["packages"],
                        problems=plan["problems"], algorithms=plan["algorithms"], n=plan["n"])
    specs = sets.narrow(specs, mode=plan["mode"], controllers=plan["controllers"],
                        tols=plan["tols"], dts=plan["dts"])
    all_trials = continue_filter(trials_mod.build_trials(specs), key, root, resume, no_overwrite)
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
        kinds, legs = trials_mod.counts(rows)
        total += kinds["solve"]
        print("{0}: {1} solve, {2} warm, {3} optimize trials in {4} legs".format(
            package, kinds["solve"], kinds["warm"], kinds["optimize"], len(legs)))
        for leg, count in legs.items():
            print("  {0}  {1}".format(leg, count))
    print("{0} solve trials".format(total))


# ---------------------------------------------------------------------- run

class Run:
    """One invocation: log dir, clock guard, manifest, summary and the runner loop."""

    def __init__(self, args, plan, key=None, data_root=DATA_DIR, logs_root=LOGS_DIR):
        self.args, self.plan = args, plan
        self.key = key or dataset_key()
        if self.key.endswith("_unknown-gpu") and not args.allow_unknown_gpu:
            raise SystemExit("Could not identify the GPU; the dataset key would be '{0}'. Fix the driver "
                             "or pass --allow-unknown-gpu.".format(self.key))
        self.data_root = data_root
        self.store = store.Store(data_root)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.log_dir = os.path.join(logs_root, "{0}_{1}".format(self.key, stamp))
        os.makedirs(self.log_dir, exist_ok=True)
        self.summary = os.path.join(self.log_dir, "summary.tsv")
        open(self.summary, "w").close()
        self.clock_failures = 0
        self.failures = 0
        self.partials = 0
        sm = mem = None
        if not args.no_lock_clocks:
            sm, mem = configure_clocks(self.key, args.lock_clocks)
        self.clocks = ClockGuard(sm, mem, args.clock_tolerance or 15)
        self.clock_status = "off"
        if sm:
            self.clock_status = ("locked SM={0}{1}".format(sm, " MEM=" + mem if mem else "")
                                 if self.clocks.lock()
                                 else "unlocked (not elevated) - target was SM={0}".format(sm))
        elif not args.no_lock_clocks:
            self.clock_status = "unlocked (no target configured)"

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
        cstart = self.clocks.stamp()
        env = dict(os.environ, **command.env)
        with open(os.path.join(self.log_dir, logfile), "a", encoding="utf-8") as log:
            proc = subprocess.Popen(command.argv, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, errors="replace")
            for line in proc.stdout:
                sys.stdout.write(line)
                log.write(line)
            proc.stdout.close()
            status = proc.wait()
        cend = self.clocks.stamp(end=True)
        elapsed = int(time.monotonic() - start)
        if status in command.ok:
            print("OK {0}  ({1}s)".format(label, elapsed))
        else:
            print("X {0} exited {1}  ({2}s)".format(label, status, elapsed))
        sys.stdout.flush()
        if not self.clocks.check(cstart, cend, label):
            self.clock_failures += 1
        return status

    def cooldown(self):
        if self.args.cooldown > 0:
            time.sleep(self.args.cooldown)

    # ------------------------------------------------------------- packages
    def run_package(self, package, trial_list, path):
        """Drive one runner over its trial file, re-invoking after every watchdog hard exit."""
        logfile = package + ".log"
        rounds = 0
        hard_exits = 0
        while True:
            command = launch.runner_command(package, path, floor=self.args.floor)
            status = self.step("{0} ({1} trials)".format(package, len(trial_list)), logfile, command)
            if status == 0:
                self.record(package, "PARTIAL" if hard_exits else "OK",
                            "{0} hard exit(s)".format(hard_exits) if hard_exits else "-", status)
                return
            if status != WATCHDOG_EXIT_CODE:
                self.record(package, "FAILED", "runner exit {0}".format(status), status)
                return
            hard_exits += 1
            remaining = abandon_after_hard_exit(self.store, self.key, trial_list, path + ".progress",
                                                store.suite_rev(ROOT))
            if remaining is None:
                self.record(package, "FAILED", "hard exit without a progress file", status)
                return
            if not remaining:
                self.record(package, "PARTIAL", "{0} hard exit(s)".format(hard_exits), status)
                return
            rounds += 1
            if rounds > len(trial_list):
                self.record(package, "FAILED", "hard exits did not converge", status)
                return
            trial_list = remaining
            path = os.path.join(os.path.dirname(path), "{0}.retry{1}.jsonl".format(package, rounds))
            trials_mod.write_jsonl(path, trial_list)

    # -------------------------------------------------------------- lifecycle
    def manifest(self, by_package, finished=False):
        path = os.path.join(self.log_dir, "run_manifest.txt")
        if finished:
            with open(path, "a", encoding="utf-8") as handle:
                handle.write("finished_utc={0}\n".format(datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")))
            return
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True)
        lines = ["dataset_key=" + self.key,
                 "started_utc=" + datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                 "sets=" + ",".join(self.plan["sets"]),
                 "argv=" + subprocess.list2cmdline(sys.argv[1:]),
                 "packages=" + ",".join(by_package),
                 "trials=" + ";".join("{0}={1}".format(p, len(t)) for p, t in by_package.items()),
                 "suite_rev=" + store.suite_rev(ROOT),
                 "host={0} {1}".format(platform.node(), sys.platform),
                 "clocks=" + self.clock_status]
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
        report = self.clocks.report_text()
        if report:
            print(report)
            print("=" * 60)
        print("Logs: " + self.log_dir)
        print("Clocks: {0}  (1 Hz log in {1})".format(self.clock_status, os.path.join(self.log_dir, "clocks.csv")))
        if self.partials:
            print("{0} package(s) partial: a watchdog hard exit abandoned part of a leg.".format(self.partials))
        if self.clock_failures:
            print("{0} runner(s) drifted; lower the lock in runner_scripts/gpu_clocks.conf and re-run them.".format(self.clock_failures))
        if self.failures:
            print("{0} runner(s) failed outright.".format(self.failures))
        return 1 if (self.failures or self.clock_failures) else 0

    def execute(self):
        by_package = plan_trials(self.plan, self.key, self.data_root,
                                 self.args.resume, self.args.no_overwrite)
        paths = write_plan(self.log_dir, by_package)
        print("Dataset key : " + self.key)
        print("Sets        : " + ", ".join(self.plan["sets"]))
        print("Packages    : " + ", ".join(by_package))
        print("Log dir     : " + self.log_dir)
        print("Clocks      : " + self.clock_status)
        print("", flush=True)
        print_counts(by_package)
        self.manifest(by_package)
        self.clocks.start_monitor(os.path.join(self.log_dir, "clocks.csv"))
        try:
            for index, (package, rows) in enumerate(by_package.items()):
                if index:
                    self.cooldown()
                self.run_package(package, rows, paths[package])
            return self.summarise()
        finally:
            self.manifest(by_package, finished=True)
            self.clocks.stop_monitor()
            self.clocks.reset()


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    plan = resolve(args)
    os.chdir(ROOT)
    if args.command == "plan":
        key = dataset_key()
        if key.endswith("_unknown-gpu") and not args.allow_unknown_gpu:
            raise SystemExit("Could not identify the GPU; the dataset key would be '{0}'. Fix the driver "
                             "or pass --allow-unknown-gpu.".format(key))
        by_package = plan_trials(plan, key, DATA_DIR, args.resume, args.no_overwrite)
        paths = write_plan(os.path.join(TRIALS_DIR, key), by_package)
        print_counts(by_package)
        for path in paths.values():
            print(os.path.relpath(path, ROOT))
        return 0
    return Run(args, plan).execute()


if __name__ == "__main__":
    sys.exit(main())
