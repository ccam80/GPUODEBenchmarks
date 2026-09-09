#!/usr/bin/env python3
"""The benchmark entry point: `plan` writes the trials each package's runner executes, `run` drives the runners.

Usage:
  bench.py plan|run [-p pkgs] [-s problems] [-g algorithms] [--mode fixed|adaptive|all]
                    [--for perf,wp,ne,states,overlap] [-n <ceiling|list>] [--setting <list>]
                    [--states <list>] [--transfers both,none] [--tier <list>]
                    [--point <trial id>]... [--resume | --no-overwrite] [--floor]
                    [--cooldown S] [--allow-unknown-gpu] [--lock-clocks SM[,MEM]]
                    [--no-lock-clocks] [--clock-tolerance MHZ]

  plan            write trials/<key>/<package>.jsonl and print the counts per package and leg
  run             write the same under logs/<key>_<stamp>/ and drive every package's runner
  -p, --package   all | comma list of cubie | cubie_mlir | jax | pytorch | myokit_cuda | cpp | julia_gpu | julia_cpu
  -s, --problem   all | comma list of names in runner_scripts/problems.csv
  -g, --algorithm all | comma list of names in runner_scripts/algorithms.csv
  --mode          all | fixed | adaptive
  --for           comma list of the views to expand (default all five)
  -n, --nmax      N sweep ceiling (8, 32, ... <= n) or a comma list of exact Ns
  --setting       keep only the trials at these dt or tol values
  --states        the states grid of the states view
  --transfers     the transfer legs to time (default both,none)
  --tier          keep only these tiers of default | matched | pi
  --point         a trial id, or a leading path of one; repeatable
  --resume        drop solve trials whose every requested transfers row is present
  --no-overwrite  drop solve trials whose rows are all finite
  --floor         runners keep the lower finite time per row
  --cooldown      seconds between packages (default 15)

Exit code: 0 when every runner completed, 1 otherwise. Clock drift in a runner also fails the run.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "runner_scripts"))

import launch  # noqa: E402
import store  # noqa: E402
import trials as trials_mod  # noqa: E402
from algorithms import resolve_modes  # noqa: E402
from bench_key import dataset_key  # noqa: E402
from clocks import ClockGuard, configure as configure_clocks  # noqa: E402
from protocol import NMAX_DEFAULT, STATES_GRID, WATCHDOG_EXIT_CODE, parse_ns  # noqa: E402

NAN = float("nan")


def parse_list(text):
    return [tok.strip() for tok in text.split(",") if tok.strip()]


def package_name(token):
    """A package token as launch.PACKAGES spells it; hyphens are accepted."""
    return token.replace("-", "_")


def parse_args(argv):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("command", nargs="?", choices=("plan", "run"))
    p.add_argument("-p", "--package", default="all")
    p.add_argument("-s", "--problem", default="all")
    p.add_argument("-g", "--algorithm", default="all")
    p.add_argument("--mode", default="all")
    p.add_argument("--for", dest="views", default=",".join(trials_mod.VIEWS))
    p.add_argument("-n", "--nmax", default=str(NMAX_DEFAULT))
    p.add_argument("--setting", default="")
    p.add_argument("--states", default="")
    p.add_argument("--transfers", default="")
    p.add_argument("--tier", default="")
    p.add_argument("--point", action="append", default=[])
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
    if args.command is None:
        raise SystemExit("bench.py takes a command: plan or run (see --help)")
    if args.resume and args.no_overwrite:
        raise SystemExit("--resume and --no-overwrite exclude each other")
    return args


def resolve_packages(text):
    packages = [package_name(tok) for tok in parse_list(text)]
    if "all" in packages:
        packages = list(launch.PACKAGES)
    for pkg in packages:
        if pkg not in launch.PACKAGES:
            raise SystemExit("Unknown package '{0}' (all|{1})".format(pkg, "|".join(launch.PACKAGES)))
    return launch.ordered(list(dict.fromkeys(packages)))


def _floats(text, flag):
    try:
        return [float(tok) for tok in parse_list(text)]
    except ValueError:
        raise SystemExit("{0} takes a comma list of numbers, got '{1}'".format(flag, text))


def _ints(text, flag):
    try:
        return [int(tok) for tok in parse_list(text)]
    except ValueError:
        raise SystemExit("{0} takes a comma list of integers, got '{1}'".format(flag, text))


def build_request(args, key, store_root="data"):
    """The trials.Request the flags describe."""
    try:
        nlist = parse_ns(args.nmax)
    except ValueError:
        raise SystemExit("-n/--nmax must be a positive integer or a comma list of them, got '{0}'".format(args.nmax))
    if not nlist:
        raise SystemExit("-n/--nmax selects no trajectory count of at least 8")
    states_grid = sorted(_ints(args.states, "--states")) if args.states else STATES_GRID
    return trials_mod.Request(
        packages=resolve_packages(args.package), problems=args.problem,
        algorithms=args.algorithm, modes=resolve_modes(args.mode),
        views=parse_list(args.views), nlist=nlist, states_grid=states_grid,
        key=key, store_root=store_root)


def plan_trials(args, key, store_root="data"):
    """Every trial the flags select, filtered against the store under --resume or --no-overwrite."""
    request = build_request(args, key, store_root)
    tiers = parse_list(args.tier) or None
    if tiers:
        for tier in tiers:
            if tier not in trials_mod.TIERS:
                raise SystemExit("--tier takes {0}, got '{1}'".format("|".join(trials_mod.TIERS), tier))
    transfers = parse_list(args.transfers) or None
    if transfers:
        for leg in transfers:
            if leg not in trials_mod.TRANSFERS:
                raise SystemExit("--transfers takes both and none, got '{0}'".format(leg))
    settings = _floats(args.setting, "--setting") if args.setting else None
    index = None
    if args.resume or args.no_overwrite:
        index = trials_mod.StoreIndex(store_root, key)
    expanded = trials_mod.expand(request)
    return trials_mod.apply_filters(expanded, tiers=tiers, transfers=transfers,
                                    settings=settings, points=args.point,
                                    resume=args.resume, no_overwrite=args.no_overwrite,
                                    index=index)


def by_package(selected, packages):
    """{package: [trials]} in run order, packages without trials included as empty lists."""
    grouped = {package: [] for package in packages}
    for trial in selected:
        grouped[trial.package].append(trial)
    return grouped


def write_plan(selected, packages, directory):
    """One JSONL per package under directory; returns {package: path} for the packages with trials."""
    paths = {}
    for package, members in by_package(selected, packages).items():
        if not members:
            continue
        path = os.path.join(directory, package + ".jsonl")
        trials_mod.write_jsonl(path, members)
        paths[package] = path
    return paths


def counts_text(selected):
    lines = []
    for package, entry in trials_mod.counts(selected).items():
        lines.append("{0}: {1} solve, {2} warm, {3} optimize".format(
            package, entry["solve"], entry["warm"], entry["optimize"]))
        for leg, count in entry["legs"].items():
            lines.append("  {0}: {1}".format(leg, count))
    return "\n".join(lines) if lines else "no trials"


def read_progress(path):
    """The runner's `<trials>.progress` record, or None."""
    progress = path + ".progress"
    if not os.path.isfile(progress):
        return None
    try:
        with open(progress, encoding="utf-8") as handle:
            record = json.load(handle)
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) and "id" in record else None


class Run:
    """One invocation: log dir, clock guard, manifest, summary and the per-package runner loop."""

    def __init__(self, args, key, data_root="data", log_root=None):
        self.args, self.key = args, key
        self.data_root = data_root
        self.store = store.Store(data_root)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.log_dir = os.path.join(log_root or os.path.join(ROOT, "logs"),
                                    "{0}_{1}".format(key, stamp))
        os.makedirs(self.log_dir, exist_ok=True)
        self.summary = os.path.join(self.log_dir, "summary.tsv")
        open(self.summary, "w").close()
        self.clock_failures = 0
        self.failures = 0
        sm = mem = None
        if not args.no_lock_clocks:
            sm, mem = configure_clocks(key, args.lock_clocks)
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
        if status in ("FAILED", "UNPORTED"):
            self.failures += 1

    def step(self, label, logfile, argv, env=None):
        """Run one command with its output tee'd to a log; returns the exit code."""
        print("=" * 60)
        print("[{0}] {1}".format(datetime.now(timezone.utc).strftime("%H:%M:%SZ"), label))
        print("  " + subprocess.list2cmdline(argv))
        print("=" * 60, flush=True)
        start = time.monotonic()
        cstart = self.clocks.stamp()
        with open(os.path.join(self.log_dir, logfile), "a", encoding="utf-8") as log:
            proc = subprocess.Popen(argv, cwd=ROOT, env=dict(os.environ, **(env or {})),
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, errors="replace")
            for line in proc.stdout:
                sys.stdout.write(line)
                log.write(line)
            status = proc.wait()
        cend = self.clocks.stamp(end=True)
        elapsed = int(time.monotonic() - start)
        if status == 0:
            print("OK {0}  ({1}s)".format(label, elapsed))
        else:
            print("X {0} exited {1}  ({2}s)".format(label, status, elapsed))
        sys.stdout.flush()
        if not self.clocks.check(cstart, cend, label, True):
            self.clock_failures += 1
        return status

    def cooldown(self):
        if self.args.cooldown > 0:
            time.sleep(self.args.cooldown)

    # ------------------------------------------------------------- runners
    def abandon(self, pending, trial_id):
        """NaN rows for the hard-exited trial and every higher ordinal of its leg whose requested transfers row is absent; returns the count written."""
        hit = [t for t in pending if t.kind == "solve" and t.id == trial_id]
        if not hit:
            return 0
        hit = hit[0]
        reason = "abandoned: hard-exit at ordinal {0}".format(hit.ordinal)
        rev = store.suite_rev(ROOT)
        written = 0
        for trial in pending:
            if trial.kind != "solve" or trial.leg_key != hit.leg_key or trial.ordinal < hit.ordinal:
                continue
            for transfers in trial.transfers:
                ident = trial.identity(self.key, transfers)
                if self.store.status(ident) != "absent":
                    continue
                self.store.record(dict(ident, min_ms=NAN, reason=reason, suite_rev=rev))
                written += 1
        return written

    def unfinished(self, pending):
        """The solve trials with any requested transfers row still absent, plus the services of their legs."""
        kept = []
        for trial in pending:
            if trial.kind != "solve":
                kept.append(trial)
                continue
            if any(self.store.status(trial.identity(self.key, t)) == "absent"
                   for t in trial.transfers):
                kept.append(trial)
        return trials_mod.keep_services(kept)

    def run_package(self, package, members):
        label = "run:" + package
        solves = sum(1 for t in members if t.kind == "solve")
        try:
            argv = launch.runner_argv(package)
        except launch.UnportedPackage as exc:
            self.record(label, "UNPORTED", str(exc), "-")
            return
        env = launch.runner_env(package)
        pending = list(members)
        attempt = 0
        while pending:
            suffix = "" if attempt == 0 else "_{0}".format(attempt)
            path = os.path.join(self.log_dir, package + suffix + ".jsonl")
            trials_mod.write_jsonl(path, pending)
            command = list(argv) + ["--trials", path] + (["--floor"] if self.args.floor else [])
            status = self.step("{0} (attempt {1})".format(label, attempt + 1), package + ".log",
                               command, env)
            if status == 0:
                self.record(label, "OK", "{0} solve trials, {1} attempt(s)".format(solves, attempt + 1), 0)
                return
            if status != WATCHDOG_EXIT_CODE:
                self.record(label, "FAILED", "runner exited {0} on attempt {1}".format(status, attempt + 1), status)
                return
            progress = read_progress(path)
            if progress is None:
                self.record(label, "FAILED", "hard exit without a progress file", status)
                return
            written = self.abandon(pending, progress["id"])
            print("  hard exit at {0}: {1} abandoned row(s) recorded".format(progress["id"], written))
            remaining = self.unfinished(pending)
            if len([t for t in remaining if t.kind == "solve"]) >= len([t for t in pending if t.kind == "solve"]):
                self.record(label, "FAILED", "hard exit at {0} left no row; not re-invoked".format(progress["id"]), status)
                return
            pending = remaining
            attempt += 1
        self.record(label, "OK", "{0} solve trials, {1} attempt(s)".format(solves, attempt), 0)

    # -------------------------------------------------------------- lifecycle
    def manifest(self, args, packages, selected, finished=False):
        path = os.path.join(self.log_dir, "run_manifest.txt")
        if finished:
            with open(path, "a", encoding="utf-8") as handle:
                handle.write("finished_utc={0}\n".format(datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")))
            return
        lines = ["dataset_key=" + self.key,
                 "started_utc=" + datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                 "argv=" + subprocess.list2cmdline(sys.argv[1:]),
                 "packages=" + ",".join(packages),
                 "views=" + args.views, "nmax=" + args.nmax,
                 "trials=" + str(len(selected)),
                 "suite_rev=" + store.suite_rev(ROOT),
                 "host={0} {1}".format(platform.node(), sys.platform),
                 "clocks=" + self.clock_status]
        try:
            gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                  "--format=csv,noheader"], capture_output=True, text=True)
            lines += ["gpu=" + line.strip() for line in gpu.stdout.splitlines() if line.strip()]
        except OSError:
            pass
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")

    def summarise(self):
        print("")
        print("=" * 60)
        print("RUN SUMMARY  ({0})".format(self.key))
        print("=" * 60)
        print("{0:<26} {1:<16} {2}".format("STAGE", "STATUS", "DETAIL"))
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
        if self.clock_failures:
            print("{0} runner(s) drifted; lower the lock in runner_scripts/gpu_clocks.conf and re-run them.".format(self.clock_failures))
        if self.failures:
            print("{0} runner(s) failed.".format(self.failures))
        return 1 if (self.failures or self.clock_failures) else 0

    def execute(self, packages, selected):
        print("Dataset key : " + self.key)
        print("Packages    : " + ", ".join(packages))
        print("Trials      : " + str(len(selected)))
        print("Log dir     : " + self.log_dir)
        print("Clocks      : " + self.clock_status)
        print("", flush=True)
        self.manifest(self.args, packages, selected)
        self.clocks.start_monitor(os.path.join(self.log_dir, "clocks.csv"))
        try:
            for package, members in by_package(selected, packages).items():
                if not members:
                    self.record("run:" + package, "SKIPPED", "no trials", "-")
                    continue
                self.run_package(package, members)
                self.cooldown()
            return self.summarise()
        finally:
            self.manifest(self.args, packages, selected, finished=True)
            self.clocks.stop_monitor()
            self.clocks.reset()


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    os.chdir(ROOT)
    key = dataset_key()
    packages = resolve_packages(args.package)
    if args.command == "run" and key.endswith("_unknown-gpu") and not args.allow_unknown_gpu:
        raise SystemExit("Could not identify the GPU; the dataset key would be '{0}'. Fix the driver "
                         "or pass --allow-unknown-gpu.".format(key))
    selected = plan_trials(args, key)
    if args.command == "plan":
        directory = os.path.join(ROOT, "trials", key)
        paths = write_plan(selected, packages, directory)
        print(counts_text(selected))
        for package, path in paths.items():
            print("wrote " + os.path.relpath(path, ROOT))
        return 0
    return Run(args, key).execute(packages, selected)


if __name__ == "__main__":
    sys.exit(main())
