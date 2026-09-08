#!/usr/bin/env python3
"""The benchmark entry point: one run of any subset of packages, analyses, algorithms, problems, modes and points.

Usage:
  bench.py                                   every analysis, every package
  bench.py -p cubie,julia -a performance     two packages, one analysis
  bench.py -a optimize,warm -p cubie         the cubie tuning and cache stages
  bench.py -n 8388608,134217728 -g euler     exact trajectory counts, one algorithm
  bench.py --mode adaptive -s pollu          one mode, one problem
  bench.py --point times:cubie:lorenz:tsit5:fixed:32768 --point wp:julia:pollu:kvaerno3
  bench.py --points-file retakes.txt         one point per line
  bench.py --resume                          skip every recorded point
  bench.py --resume-from cubie:pollu:tsit5:adaptive:262144

  -p, --package   all | comma list of julia | cpp | pytorch | jax | cubie | cubie_mlir | myokit_cuda
  -a, --analysis  all | comma list of optimize | warm | performance | states | work-precision | numerical | overlap | plots
  -n, --nmax      sweep ceiling (8, 32, ... <= n) or a comma list of exact Ns
  -g, --algorithm all | comma list of names in runner_scripts/algorithms.csv
  -s, --problem   all | comma list of names in runner_scripts/problems.csv
  --mode          all | fixed | adaptive; the timed sweeps and the numerical-equivalence sweeps take it
  --point         <times|wp|states>:<package>:<problem>:<algorithm>[:<fixed|adaptive>][:<N|states>]; repeatable
  --points-file   a file of --point lines
  --resume        skip every recorded point (implies --keep)
  --no-overwrite  skip only points with a finite recorded time (implies --keep)
  --keep          keep existing rows; a run replaces only what it records
  --floor         keep the lower of the recorded and new time per point (implies --keep)
  --resume-from   package[:problem[:algorithm[:fixed|adaptive[:N]]]]; the performance sweep restarts there
  --cooldown      seconds between packages (default 15)
  --allow-unknown-gpu, --lock-clocks SM[,MEM], --no-lock-clocks, --clock-tolerance MHZ

Without --keep a run first drops only the store rows it is about to record.

Exit code: 0 when every stage succeeded, 1 otherwise. Clock drift in a timed stage also fails the run.
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

import launch  # noqa: E402
from algorithms import (MODES, get_algorithm, ne_algorithms,  # noqa: E402
                        overlap_algorithms, resolve_modes)
from bench_key import dataset_key  # noqa: E402
from clocks import ClockGuard, configure as configure_clocks  # noqa: E402
from problems import get_problem  # noqa: E402
from protocol import NMAX_DEFAULT, STATES_GRID, parse_ns  # noqa: E402

ALL_ANALYSES = ("optimize", "warm", "performance", "states", "work-precision",
                "numerical", "overlap", "plots")
DEFAULT_ANALYSES = ("performance", "states", "work-precision", "numerical",
                    "overlap", "plots")
POINT_ANALYSES = ("times", "wp", "states")
STAGE_OF = {"times": "performance", "wp": "work-precision", "states": "states"}


def parse_list(text):
    return [tok.strip() for tok in text.split(",") if tok.strip()]


def package_name(token):
    """A package token as launch.PACKAGES spells it; hyphens are accepted."""
    return token.replace("-", "_")


class Point:
    """One retake: an analysis, package, problem, algorithm, optional mode and N or state count."""

    def __init__(self, spec):
        parts = spec.split(":")
        if len(parts) < 4:
            raise SystemExit("--point takes <times|wp|states>:<package>:<problem>"
                             ":<algorithm>[:<mode>][:<N|states>], got '{0}'".format(spec))
        self.analysis, self.package, self.problem, self.algorithm = parts[:4]
        self.package = package_name(self.package)
        if self.analysis not in POINT_ANALYSES:
            raise SystemExit("--point analysis must be one of {0}, got '{1}'".format(
                "|".join(POINT_ANALYSES), self.analysis))
        if self.package not in launch.PACKAGES:
            raise SystemExit("--point names an unknown package '{0}'".format(self.package))
        get_problem(self.problem)
        get_algorithm(self.algorithm)
        self.mode = "all"
        self.n = None
        for tok in parts[4:]:
            if tok in MODES:
                self.mode = tok
            elif tok.isdigit():
                self.n = int(tok)
            else:
                raise SystemExit("--point '{0}': '{1}' is neither a mode nor a count".format(spec, tok))
        if self.analysis in ("times", "states") and self.n is None:
            raise SystemExit("--point '{0}' needs an N (times) or a state count (states)".format(spec))

    @property
    def stage(self):
        return STAGE_OF[self.analysis]

    def identity(self, key):
        ident = {"package": self.package, "key": key, "analysis": self.analysis,
                 "problem": self.problem, "algorithm": self.algorithm}
        if self.mode != "all":
            ident["mode"] = self.mode
        if self.analysis == "times":
            ident["n"] = str(self.n)
        elif self.analysis == "states":
            ident["states"] = str(self.n)
        return ident


def parse_args(argv):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("-p", "--package", default="all")
    p.add_argument("-a", "--analysis", default="all")
    p.add_argument("-n", "--nmax", default=str(NMAX_DEFAULT))
    p.add_argument("-g", "--algorithm", default="all")
    p.add_argument("-s", "--problem", default="all")
    p.add_argument("--mode", default="all")
    p.add_argument("--point", action="append", default=[])
    p.add_argument("--points-file", default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no-overwrite", action="store_true")
    p.add_argument("--keep", action="store_true")
    p.add_argument("--floor", action="store_true")
    p.add_argument("--resume-from", default="")
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
    if args.resume or args.no_overwrite or args.floor or args.resume_from:
        args.keep = True
    return args


def resolve(args):
    """Validate the axes and expand the points into a plan dict."""
    packages = [package_name(tok) for tok in parse_list(args.package)]
    if "all" in packages:
        packages = list(launch.PACKAGES)
    for pkg in packages:
        if pkg not in launch.PACKAGES:
            raise SystemExit("Unknown package '{0}' (all|{1})".format(pkg, "|".join(launch.PACKAGES)))
    packages = launch.ordered(list(dict.fromkeys(packages)))

    analyses = parse_list(args.analysis)
    if "all" in analyses:
        analyses = list(DEFAULT_ANALYSES)
    for name in analyses:
        if name not in ALL_ANALYSES:
            raise SystemExit("Unknown analysis '{0}' (all|{1})".format(name, "|".join(ALL_ANALYSES)))
    plot_all = analyses == ["plots"]
    if any(a in analyses for a in ("performance", "states", "work-precision")) and "plots" not in analyses:
        analyses.append("plots")

    algorithms = parse_list(args.algorithm)
    if "all" not in algorithms:
        for name in algorithms:
            get_algorithm(name)
    algorithm = "all" if "all" in algorithms else ",".join(algorithms)

    problems = parse_list(args.problem)
    if "all" not in problems:
        for name in problems:
            get_problem(name)
    problem = "all" if "all" in problems else ",".join(problems)

    modes = resolve_modes(args.mode)
    mode = "all" if modes == MODES else modes[0]

    try:
        nlist = parse_ns(args.nmax)
    except ValueError:
        raise SystemExit("-n/--nmax must be a positive integer or a comma list of them, got '{0}'".format(args.nmax))
    if not nlist:
        raise SystemExit("-n/--nmax selects no trajectory count of at least 8")

    points = [Point(spec) for spec in args.point]
    if args.points_file:
        with open(args.points_file, encoding="utf-8") as handle:
            points += [Point(line.strip()) for line in handle
                       if line.strip() and not line.startswith("#")]

    resume_pkg, resume_tail = "", ""
    if args.resume_from:
        resume_pkg, _, resume_tail = args.resume_from.partition(":")
        resume_pkg = package_name(resume_pkg)
        if resume_pkg not in launch.PACKAGES:
            raise SystemExit("--resume-from names an unknown package '{0}'".format(resume_pkg))

    return {"packages": packages, "analyses": analyses, "plot_all": plot_all,
            "algorithm": algorithm, "problem": problem, "mode": mode, "nlist": nlist,
            "nmax": args.nmax, "points": points, "resume_pkg": resume_pkg,
            "resume_tail": resume_tail}


def ne_package(packages):
    """The -p token the NE and overlap suites take: both, one, or none."""
    has_julia = "julia" in packages
    has_cubie = any(p in launch.CUBIE_PACKAGES for p in packages)
    if has_julia and has_cubie:
        return "all"
    if has_julia:
        return "julia"
    if has_cubie:
        return "cubie"
    return ""


def cubie_packages(packages):
    """The cubie packages requested, in run order."""
    return [p for p in packages if p in launch.CUBIE_PACKAGES]


def bench_env(args):
    """The continuation contract the bench scripts read."""
    env = dict(os.environ)
    for name in ("BENCH_RESUME", "BENCH_NO_OVERWRITE", "BENCH_RESUME_FROM", "BENCH_FLOOR"):
        env.pop(name, None)
    if args.resume:
        env["BENCH_RESUME"] = "1"
    if args.no_overwrite:
        env["BENCH_NO_OVERWRITE"] = "1"
    if args.floor:
        env["BENCH_FLOOR"] = "1"
    return env


def clear_identity(package, key, analysis, algorithm, problem, mode, nlist):
    """The store identity a stage is about to record, with comma lists as any-of members."""
    ident = {"package": package, "key": key, "analysis": launch.store_analysis(analysis)}
    if algorithm != "all":
        ident["algorithm"] = algorithm.split(",")
    if mode != "all":
        ident["mode"] = mode
    if analysis == "states":
        ident["states"] = [str(s) for s in STATES_GRID]
    else:
        if problem != "all":
            ident["problem"] = problem.split(",")
        if analysis == "performance":
            ident["n"] = [str(n) for n in nlist]
    return ident


class Run:
    """One invocation: log dir, clock guard, manifest, summary and the stage loop."""

    def __init__(self, args, plan):
        self.args, self.plan = args, plan
        self.key = dataset_key()
        if self.key.endswith("_unknown-gpu") and not args.allow_unknown_gpu:
            raise SystemExit("Could not identify the GPU; the dataset key would be '{0}'. Fix the driver "
                             "or pass --allow-unknown-gpu.".format(self.key))
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.log_dir = os.path.join(ROOT, "logs", "{0}_{1}".format(self.key, stamp))
        os.makedirs(self.log_dir, exist_ok=True)
        self.summary = os.path.join(self.log_dir, "summary.tsv")
        open(self.summary, "w").close()
        self.env = bench_env(args)
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

    def step(self, label, logfile, command, critical=True, check=True):
        """Run one command with its output tee'd to a log; returns the exit code."""
        print("=" * 60)
        print("[{0}] {1}".format(datetime.now(timezone.utc).strftime("%H:%M:%SZ"), label))
        print("  " + subprocess.list2cmdline(command.argv))
        print("=" * 60, flush=True)
        start = time.monotonic()
        cstart = self.clocks.stamp()
        env = dict(self.env, **command.env)
        with open(os.path.join(self.log_dir, logfile), "a", encoding="utf-8") as log:
            proc = subprocess.Popen(command.argv, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, errors="replace")
            for line in proc.stdout:
                sys.stdout.write(line)
                log.write(line)
            status = proc.wait()
        cend = self.clocks.stamp(end=True)
        elapsed = int(time.monotonic() - start)
        if status in command.ok:
            print("OK {0}  ({1}s)".format(label, elapsed))
        else:
            print("X {0} failed with exit {1}  ({2}s) - continuing".format(label, status, elapsed))
        sys.stdout.flush()
        if check and not self.clocks.check(cstart, cend, label, critical):
            self.clock_failures += 1
        return status

    def run_commands(self, label, logfile, commands, critical=True):
        """Run a stage's commands in order; the worst non-ok status is returned."""
        worst = 0
        for command in commands:
            status = self.step("{0}: {1}".format(label, command.label), logfile, command, critical)
            if status not in command.ok:
                worst = worst or status
        return worst

    def clear_rows(self, package, analysis, algorithm, problem, mode, nlist):
        """Drop the store rows a package's analysis is about to record, and nothing beyond them."""
        if analysis == "optimize":
            if package in launch.CUBIE_PACKAGES:
                import cubie_adapter
                cubie_adapter.clear_optimized(package, self.key, algorithm, problem)
            return
        if analysis == "warm":
            return
        import results
        results.clear(results.store_path(package, self.key),
                      **clear_identity(package, self.key, analysis, algorithm, problem, mode, nlist))

    def max_n_reached(self, package, algorithm, problem, mode, nlist):
        """Largest N with a finite time among the rows the stage recorded."""
        import math
        import results
        ident = clear_identity(package, self.key, "performance", algorithm, problem, mode, nlist)
        best = 0
        for row in results.rows_for(results.store_path(package, self.key), **ident):
            try:
                if math.isfinite(float(row["min_ms"])):
                    best = max(best, int(row["n"]))
            except ValueError:
                continue
        return best

    # --------------------------------------------------------------- stages
    def package_stage(self, analysis, package, nlist, algorithm, problem, mode="all",
                      extra_env=None, label=None, logfile=None):
        """One (analysis, package) stage; returns the recorded status."""
        stage = label or "{0}:{1}".format({"performance": "perf", "work-precision": "wp"}.get(analysis, analysis), package)
        logfile = logfile or "{0}_{1}.log".format(analysis.replace("-", "_"), package)
        critical = analysis not in ("warm", "optimize")
        commands = launch.commands(package, analysis, nlist, self.plan_nmax(nlist), algorithm, problem, mode)
        if not commands:
            self.record(stage, "SKIPPED", "no {0} step".format(analysis), "-")
            return "SKIPPED"
        if not self.args.keep:
            self.clear_rows(package, analysis, algorithm, problem, mode, nlist)
        saved = dict(self.env)
        self.env.update(extra_env or {})
        try:
            worst = self.run_commands(stage, logfile, commands, critical)
        finally:
            self.env = saved
        if analysis == "performance":
            reached = self.max_n_reached(package, algorithm, problem, mode, nlist)
            if worst == 0:
                self.record(stage, "OK", "maxN={0}".format(reached), worst)
            elif reached > 0:
                self.record(stage, "PARTIAL", "maxN={0}".format(reached), worst)
            else:
                self.record(stage, "FAILED", "no data", worst)
        else:
            self.record(stage, "OK" if worst == 0 else "FAILED", "-", worst)
        return "OK" if worst == 0 else "FAILED"

    def plan_nmax(self, nlist):
        """The -n value the cpp launcher takes: the ceiling, or the exact list."""
        if len(nlist) == 1:
            return str(nlist[0])
        default = parse_ns(str(max(nlist)))
        return str(max(nlist)) if default == nlist else ",".join(str(n) for n in nlist)

    def julia(self, script, *args):
        return launch.Command(os.path.basename(script),
                             launch.julia_command() + ["-t", "auto", "--project=.", script] + list(args))

    def replot(self, script):
        if "plots" not in self.plan["analyses"]:
            return
        command = launch.Command(script, launch.julia_command() + ["--project=.",
                                 os.path.join("runner_scripts", "plot", script)])
        with open(os.path.join(self.log_dir, "plot_refresh.log"), "a", encoding="utf-8") as log:
            proc = subprocess.run(command.argv, cwd=ROOT, env=self.env, stdout=log,
                                  stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            print("  replot failed; see plot_refresh.log")

    def cooldown(self):
        if self.args.cooldown > 0:
            time.sleep(self.args.cooldown)

    def run_points(self):
        """Every --point as its own narrow stage; only that identity's rows are replaced."""
        import results
        for point in self.plan["points"]:
            if not self.args.keep:
                results.clear(results.store_path(point.package, self.key), **point.identity(self.key))
            extra = {}
            nlist = self.plan["nlist"]
            if point.analysis == "times":
                nlist = [point.n]
            elif point.analysis == "states":
                extra["BENCH_STATES_GRID"] = str(point.n)
            label = "point:{0}:{1}:{2}:{3}".format(point.package, point.problem, point.algorithm, point.mode)
            saved_keep = self.args.keep
            self.args.keep = True
            try:
                self.package_stage(point.stage, point.package, nlist, point.algorithm,
                                   point.problem if point.analysis != "states" else "all",
                                   mode=point.mode, extra_env=extra, label=label,
                                   logfile="points_{0}.log".format(point.package))
            finally:
                self.args.keep = saved_keep
        self.replot("plot_ode_comp.jl")
        self.replot("plot_ode_wp.jl")
        self.replot("plot_states.jl")

    def run_stages(self):
        plan = self.plan
        packages, analyses = plan["packages"], plan["analyses"]
        algorithm, problem, mode, nlist = plan["algorithm"], plan["problem"], plan["mode"], plan["nlist"]
        skipping = bool(plan["resume_pkg"])

        if "optimize" in analyses:
            for package in packages:
                self.package_stage("optimize", package, nlist, algorithm, problem, mode)
        if "warm" in analyses:
            for package in packages:
                self.package_stage("warm", package, nlist, algorithm, problem, mode)
        if "performance" in analyses:
            for package in packages:
                extra = {}
                if skipping:
                    if package == plan["resume_pkg"]:
                        skipping = False
                        if plan["resume_tail"]:
                            extra["BENCH_RESUME_FROM"] = plan["resume_tail"]
                    else:
                        self.record("perf:" + package, "SKIPPED", "before --resume-from", "-")
                        continue
                self.package_stage("performance", package, nlist, algorithm, problem, mode, extra_env=extra)
                self.replot("plot_ode_comp.jl")
                self.cooldown()
        if "states" in analyses:
            for package in packages:
                self.package_stage("states", package, nlist, algorithm, "all", mode)
                self.replot("plot_states.jl")
                self.cooldown()
        if "work-precision" in analyses:
            status = self.step("Golden references for work-precision", "wp_golden.log",
                               self.julia(os.path.join("runner_scripts", "golden", "generate_golden.jl"),
                                          "--problem", problem), critical=False)
            self.record("wp:golden", "OK" if status == 0 else "FAILED",
                        "-" if status == 0 else "wp sweeps cannot score", status)
            for package in packages:
                self.package_stage("work-precision", package, nlist, algorithm, problem, mode)
                self.replot("plot_ode_wp.jl")
                self.cooldown()
        if "numerical" in analyses:
            self.numerical(ne_package(packages), cubie_packages(packages), algorithm, problem, mode)
        if "overlap" in analyses:
            self.overlap(ne_package(packages), cubie_packages(packages), nlist, algorithm, problem)
        if "plots" in analyses:
            self.plots(ne_package(packages), plan["plot_all"])

    def numerical(self, package, cubie_pkgs, algorithm, problem, mode):
        """Golden NE references, the Float32 DifferentialEquations.jl sweep, one cubie sweep per cubie package, and the comparison."""
        if not package:
            self.record("ne", "SKIPPED", "no requested package is in the ne suite", "-")
            return
        if not ne_algorithms(algorithm):
            self.record("ne", "SKIPPED", "no requested algorithm is in the ne suite", "-")
            return
        worst = 0
        if package in ("all", "julia"):
            worst = worst or self.step("ne: golden references", "numerical_equivalence.log",
                                       self.julia(os.path.join("runner_scripts", "golden", "generate_golden.jl"),
                                                  "--problem", problem), critical=False)
            worst = worst or self.step("ne: DifferentialEquations.jl sweeps", "numerical_equivalence.log",
                                       self.julia(os.path.join("runner_scripts", "numerical_equivalence", "ne_diffeq.jl"),
                                                  "--controller", mode, "--algorithm", algorithm,
                                                  "--problem", problem), critical=False)
        # The cubie ne finals are the first N_NE rows of the work-precision solves, so the ne sweep is that leg set; its rows replace by identity.
        for cubie_pkg in cubie_pkgs:
            if worst != 0:
                break
            if "work-precision" in self.plan["analyses"]:
                self.record("ne:" + cubie_pkg, "SKIPPED", "the work-precision stage ran the ne legs", "-")
                continue
            saved_keep = self.args.keep
            self.args.keep = True
            try:
                status = self.package_stage("numerical", cubie_pkg, self.plan["nlist"], algorithm, problem, mode,
                                            label="ne:" + cubie_pkg, logfile="numerical_equivalence.log")
            finally:
                self.args.keep = saved_keep
            worst = 0 if status in ("OK", "SKIPPED") else 1
        if worst != 0:
            self.record("ne", "FAILED", "-", worst)
            return
        status = self.step("ne: comparison", "numerical_equivalence.log", launch.Command(
            "compare", [launch.cubie_python(), os.path.join(ROOT, "compare_numerical_equivalence.py"),
                        "--problem", problem]), critical=False)
        self.record("ne", "OK" if status == 0 else "FAILED",
                    "see plots/<key>/<problem>/numerical_equivalence_*" if status == 0 else "-", status)

    def overlap(self, package, cubie_pkgs, nlist, algorithm, problem):
        """The overlap suite once per requested cubie backend; julia alone runs once."""
        if not package:
            self.record("overlap", "SKIPPED", "no requested package is in the overlap suite", "-")
            return
        if not overlap_algorithms(algorithm):
            self.record("overlap", "SKIPPED", "no requested algorithm is in the overlap suite", "-")
            return
        backends = cubie_pkgs or [None]
        for backend in backends:
            argv = [launch.cubie_python(), os.path.join(ROOT, "run_cubie_julia_overlap.py"),
                    "-a", "all", "-p", package, "-n", ",".join(str(n) for n in nlist),
                    "--algorithm", algorithm, "-s", problem]
            label = "overlap"
            if backend:
                argv += ["--backend", backend]
                label = "overlap:" + backend
            status = self.step("Cubie vs DiffEqGPU overlap", "cubie_julia_overlap.log",
                               launch.Command(label, argv))
            self.record(label, "OK" if status == 0 else "PARTIAL",
                        "-" if status == 0 else "a worker failed; see manifest.json", status)

    def plots(self, package, plot_all):
        analyses = self.plan["analyses"]
        for script, stage, wanted in (("plot_ode_comp.jl", "plot:timing", "performance"),
                                      ("plot_ode_wp.jl", "plot:wp", "work-precision"),
                                      ("plot_states.jl", "plot:states", "states")):
            if not (plot_all or wanted in analyses):
                continue
            status = self.step("Plot: " + script, script.replace(".jl", ".log"), launch.Command(
                script, launch.julia_command() + ["--project=.", os.path.join("runner_scripts", "plot", script)]),
                critical=False, check=False)
            self.record(stage, "OK" if status == 0 else "FAILED", "-", status)
        if package and (plot_all or "overlap" in analyses):
            status = self.step("Overlap plot and report", "cubie_julia_overlap_analyze.log", launch.Command(
                "analyze", [launch.cubie_python(), os.path.join(ROOT, "runner_scripts", "cubie_julia_overlap", "analyze.py")]),
                critical=False, check=False)
            self.record("plot:overlap", "OK" if status == 0 else "FAILED", "-", status)
        status = self.step("Pairwise numerical comparison", "compare_numerical.log", launch.Command(
            "compare", [launch.cubie_python(), os.path.join(ROOT, "compare_numerical_results.py")], ok=(0, 3)),
            critical=False, check=False)
        if status == 0:
            self.record("compare:pairwise", "OK", "-", status)
        elif status == 3:
            self.record("compare:pairwise", "SKIPPED", "needs >=2 keyed datasets", status)
        else:
            self.record("compare:pairwise", "FAILED", "-", status)

    # -------------------------------------------------------------- lifecycle
    def manifest(self, finished=False):
        path = os.path.join(self.log_dir, "run_manifest.txt")
        if finished:
            with open(path, "a", encoding="utf-8") as handle:
                handle.write("finished_utc={0}\n".format(datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")))
            return
        rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT)
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=ROOT)
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True)
        lines = ["dataset_key=" + self.key,
                 "started_utc=" + datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                 "nmax=" + self.plan["nmax"], "algorithm=" + self.plan["algorithm"],
                 "problem=" + self.plan["problem"], "mode=" + self.plan["mode"],
                 "packages=" + ",".join(self.plan["packages"]),
                 "analyses=" + ",".join(self.plan["analyses"]),
                 "points=" + ";".join(repr(p.identity(self.key)) for p in self.plan["points"]),
                 "git_rev=" + (rev.stdout.strip() or "unknown"),
                 "git_dirty=" + ("yes" if dirty.stdout.strip() else "no"),
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
        if self.partials:
            print("{0} stage(s) partial - expected when frameworks OOM at high N.".format(self.partials))
        if self.clock_failures:
            print("{0} timed stage(s) drifted; lower the lock in runner_scripts/gpu_clocks.conf and re-run them.".format(self.clock_failures))
        if self.failures:
            print("{0} stage(s) failed outright.".format(self.failures))
        return 1 if (self.failures or self.clock_failures) else 0

    def execute(self):
        print("Dataset key : " + self.key)
        print("nmax        : " + self.plan["nmax"])
        print("Algorithm   : " + self.plan["algorithm"])
        print("Problems    : " + self.plan["problem"])
        print("Mode        : " + self.plan["mode"])
        print("Packages    : " + ", ".join(self.plan["packages"]))
        print("Analyses    : " + ", ".join(self.plan["analyses"]))
        print("Points      : " + str(len(self.plan["points"])))
        print("Log dir     : " + self.log_dir)
        print("Clocks      : " + self.clock_status)
        print("", flush=True)
        self.manifest()
        self.clocks.start_monitor(os.path.join(self.log_dir, "clocks.csv"))
        try:
            if self.plan["points"]:
                self.run_points()
            else:
                self.run_stages()
            return self.summarise()
        finally:
            self.manifest(finished=True)
            self.clocks.stop_monitor()
            self.clocks.reset()


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    plan = resolve(args)
    os.chdir(ROOT)
    return Run(args, plan).execute()


if __name__ == "__main__":
    sys.exit(main())
