"""Convert the legacy CSV trees under data/ into the parquet store and delete them: convert_legacy.py [--root data] [--dry-run] [--keep-old]; every legacy row becomes a run spec, rows meeting by run_id merge, and the report counts every mapping and drop and the rows no shipped set produces."""

import argparse
import csv
import glob
import json
import math
import os
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import algorithms  # noqa: E402
import problems as problem_catalogue  # noqa: E402
import sets  # noqa: E402
import store  # noqa: E402

NAN = float("nan")

# Legacy results.csv directory -> package; the legacy package column spelt julia_gpu "julia".
LEGACY_DIRS = {"CUBIE": "cubie", "CUBIE_MLIR": "cubie_mlir", "JAX": "jax",
               "PYTORCH": "pytorch", "MYOKIT_CUDA": "myokit_cuda", "CPP": "cpp",
               "Julia": "julia_gpu"}
LEGACY_PACKAGE = {"julia": "julia_gpu"}
LEGACY_TREES = tuple(LEGACY_DIRS) + ("cubie_julia_overlap", "numerical_equivalence",
                                     "numerical")
DROPPED_PACKAGES = ("cpp",)
DROPPED_ALGORITHMS = {"jax": ("kvaerno3",)}
GOLDEN_ONLY_PROBLEMS = ("nand_gate",)

# Legacy pins: dt0 = duration * 2^-10, fixed-step Newton scale 1e-6.
DT0_POW = -10
NEWTON_FIXED = 1e-6
STATES_PROBLEM = problem_catalogue.RESIZABLE_PROBLEM
GRID_DTYPE = "float32"

# Overlap tier -> controller; the performance phase keeps the fixed tier only.
OVERLAP_TIERS = {"fixed": "fixed", "julia": "default"}
OVERLAP_PHASES = ("performance", "work_precision")
OVERLAP_PERFORMANCE_TIERS = ("fixed",)
OVERLAP_KEPT = ("julia_timings.csv", "julia_metrics.csv", "julia_failures.csv",
                "work_precision.csv")

CONTROLLER_TABLE = "controller_constants"
GOLDEN_N = 131072
GOLDEN_KEY = "windows_RTX-4070-SUPER"
GOLDEN_PREFIX = "golden_"
GOLDEN_SUFFIX = "_{0}.csv".format(GOLDEN_N)
GOLDEN_RETCODES = "_{0}_retcodes.csv".format(GOLDEN_N)

SNAP_REL = 1e-8


def _float(text):
    try:
        value = float(text)
    except (TypeError, ValueError):
        return NAN
    return value


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_problems():
    """problem -> the catalogue row's system, grid and golden fields."""
    catalogue = {}
    for row in problem_catalogue.load_problems():
        catalogue[row.name] = dict(
            states=row["states"], duration=row["duration"],
            parameter=row["sweep_parameter"], grid_scale=row["sweep_scale"],
            grid_min=row["sweep_min"], grid_max=row["sweep_max"],
            golden_algorithm=row["golden_algorithm"], golden_tol=row["golden_tol"])
    return catalogue


def load_newton():
    """(package, algorithm) -> whether the catalogue row takes a Newton tolerance."""
    return {(row.package, row.name): bool(row["newton"]) for row in algorithms.load_algorithms()}


def snap_dt(value, duration):
    """The exact duration * 2^-k a legacy dt was printed from, else the value itself."""
    for k in range(0, 64):
        exact = duration * 2.0 ** -k
        if exact == value or abs(exact - value) <= SNAP_REL * exact:
            return exact
    return value


def snap_tol(value):
    """The exact 10^-k a legacy tolerance was printed from, else the value itself."""
    if not value > 0.0:
        return value
    k = int(round(-math.log10(value)))
    exact = float("1e-{0}".format(k)) if k >= 0 else float("1e{0}".format(-k))
    if exact == value or abs(exact - value) <= SNAP_REL * exact:
        return exact
    return value


def _git_stamp(path, repo_root, cache={}):
    """The time of the last commit that added or changed a file, as an ISO UTC string, else its mtime."""
    path = os.path.abspath(path)
    if path in cache:
        return cache[path]
    stamp = None
    try:
        out = subprocess.run(["git", "log", "-1", "--diff-filter=AM", "--format=%cI", "--", path],
                             cwd=repo_root, capture_output=True, text=True, timeout=60)
        if out.returncode == 0 and out.stdout.strip():
            stamp = datetime.fromisoformat(out.stdout.strip()).astimezone(timezone.utc)
    except (OSError, subprocess.SubprocessError, ValueError):
        stamp = None
    if stamp is None:
        stamp = datetime.fromtimestamp(os.path.getmtime(path), timezone.utc)
    cache[path] = stamp.strftime("%Y-%m-%dT%H:%M:%SZ")
    return cache[path]


def _t_final(success, duration):
    """duration where the legacy success flag holds, NaN elsewhere."""
    return np.where(np.asarray(success, dtype=bool), float(duration), NAN)


class Conversion:
    """Rows keyed by run_id, finals written as they are met, counts per rule."""

    def __init__(self, root, dry_run=False, problems=None, newton=None, sets_dir=sets.SETS_DIR):
        self.root = root
        self.dry_run = dry_run
        self.repo_root = os.path.dirname(os.path.abspath(root))
        self.sets_dir = sets_dir
        self.store = store.Store(root)
        self.problems = load_problems() if problems is None else problems
        self.newton = load_newton() if newton is None else newton
        self.rows = {}
        self.sources = {}
        self.counts = Counter()
        self.finals_written = 0
        self.produced = Counter()
        self.unproduced = Counter()

    # ------------------------------------------------------------- specs
    def system(self, problem, states=None, precision="float32"):
        """The system and ensemble fields of a problem from the catalogue."""
        row = self.problems[problem]
        states = row["states"] if states is None else int(states)
        params = {"states": states} if problem == STATES_PROBLEM else {}
        return dict(problem=problem, system_params=params, duration=row["duration"],
                    precision=precision, parameter=row["parameter"],
                    grid_scale=row["grid_scale"], grid_min=row["grid_min"],
                    grid_max=row["grid_max"], grid_dtype=GRID_DTYPE), states

    def takes_newton(self, package, algorithm):
        """True when the (package, algorithm) catalogue row has newton = true."""
        if (package, algorithm) not in self.newton:
            self.counts["rows of ({0}, {1}) without a catalogue row".format(
                package, algorithm)] += 1
            return False
        return self.newton[(package, algorithm)]

    def stepping(self, package, algorithm, mode, setting, duration):
        """The stepping fields of a legacy (mode, setting) for a package."""
        newton = self.takes_newton(package, algorithm)
        if mode == "fixed":
            dt = snap_dt(float(setting), duration)
            scale = NEWTON_FIXED if newton else NAN
            return dict(algorithm=algorithm, controller="fixed", dt=dt, dt_min=NAN,
                        dt_max=NAN, atol=NAN, rtol=NAN, gains={}, newton_atol=scale,
                        newton_rtol=scale)
        if mode != "adaptive":
            raise ValueError("mode '{0}' is not fixed or adaptive".format(mode))
        tol = snap_tol(float(setting))
        scale = tol if newton else NAN
        return dict(algorithm=algorithm, controller="default", dt=duration * 2.0 ** DT0_POW,
                    dt_min=NAN, dt_max=NAN, atol=tol, rtol=tol, gains={},
                    newton_atol=scale, newton_rtol=scale)

    def spec(self, package, key, problem, algorithm, mode, setting, n, states=None):
        """A trial spec plus key (every 1.2 field but transfers) from the legacy axes."""
        system, states = self.system(problem, states)
        spec = dict(system, **self.stepping(package, algorithm, mode, setting,
                                            system["duration"]))
        spec.update(n=int(n), package=package, key=key)
        return spec, states

    # -------------------------------------------------------------- rows
    def add(self, fields, source):
        """Keep one row per run_id: the row with samples wins, then the later stamp; the loser fills the winner's NaN and empty columns."""
        row = store.make_row(**fields)
        key = row["run_id"]
        standing = self.rows.get(key)
        if standing is None:
            self.rows[key] = row
            self.sources[key] = source
            return row
        self.counts["rows merged by run_id ({0} vs {1})".format(
            *sorted((self.sources[key], source)))] += 1
        winner, loser = (row, standing) if self._outranks(row, standing) else (standing, row)
        if winner is row:
            self.sources[key] = source
        for field in store.FLOAT_VALUE_COLUMNS:
            if math.isnan(winner[field]) and not math.isnan(loser[field]):
                winner[field] = loser[field]
        for field in store.TEXT_VALUE_COLUMNS:
            if not winner[field] and loser[field]:
                winner[field] = loser[field]
        if not winner["samples_ms"] and loser["samples_ms"]:
            winner["samples_ms"] = loser["samples_ms"]
        self.rows[key] = winner
        return winner

    @staticmethod
    def _outranks(a, b):
        return ((bool(a["samples_ms"]), a["recorded_utc"])
                > (bool(b["samples_ms"]), b["recorded_utc"]))

    def write_finals(self, spec, finals, t_final, retcode=None):
        """Record the finals; returns (relative path, errored_pct)."""
        self.finals_written += 1
        if retcode is None:
            retcode = [""] * len(t_final)
        pct = store.errored_pct(finals, t_final, retcode, spec["duration"])
        if self.dry_run:
            return store.finals_name(spec), pct
        return self.store.record_finals(spec, finals, t_final, retcode), pct

    # ------------------------------------------------------- results.csv
    def convert_results(self, path, package):
        for row in _read_csv(path):
            legacy = LEGACY_PACKAGE.get(row["package"], row["package"])
            if legacy != package:
                self.counts["results.csv rows under another package's directory dropped"] += 1
                continue
            if package in DROPPED_PACKAGES:
                self.counts["results.csv {0} rows dropped".format(package)] += 1
                continue
            if row["problem"] in GOLDEN_ONLY_PROBLEMS:
                self.counts["results.csv {0} rows dropped".format(row["problem"])] += 1
                continue
            if row["transfers"] not in store.TRANSFERS:
                self.counts["results.csv rows with transfers = {0} dropped".format(
                    row["transfers"])] += 1
                continue
            if row["algorithm"] in DROPPED_ALGORITHMS.get(package, ()):
                self.counts["results.csv {0} {1} rows dropped".format(
                    package, row["algorithm"])] += 1
                continue
            spec, states = self.spec(package, row["key"], row["problem"], row["algorithm"],
                                     row["mode"], row["setting"], row["n"], row["states"])
            samples = [float(v) for v in row.get("samples_ms", "").split(";") if v]
            self.add(dict(spec, transfers=row["transfers"], states=states,
                          min_ms=_float(row["min_ms"]), samples_ms=samples,
                          errored_pct=_float(row.get("errored_pct")),
                          build_s=_float(row.get("build_s")),
                          recorded_utc=row["recorded_utc"]),
                     "results.csv " + row["analysis"])
            self.counts["results.csv rows converted ({0})".format(package)] += 1

    # ----------------------------------------------------------- overlap
    def convert_overlap(self, problem_dir, key, problem):
        timings = os.path.join(problem_dir, "julia_timings.csv")
        if not os.path.isfile(timings):
            return
        if problem in GOLDEN_ONLY_PROBLEMS:
            self.counts["overlap {0} rows dropped".format(problem)] += len(_read_csv(timings))
            return
        version = ""
        metadata = os.path.join(problem_dir, "julia_metadata.json")
        if os.path.isfile(metadata):
            with open(metadata, encoding="utf-8") as handle:
                version = json.load(handle).get("diffeqgpu_version", "")
        stamp = _git_stamp(timings, self.repo_root)
        metrics = {}
        metrics_path = os.path.join(problem_dir, "julia_metrics.csv")
        if os.path.isfile(metrics_path):
            for row in _read_csv(metrics_path):
                metrics[(row["algorithm"], row["phase"], row["mode"], row["tier"],
                         int(row["n"]), float(row["setting"]))] = row
        for row in _read_csv(timings):
            phase = row["phase"]
            if phase not in OVERLAP_PHASES:
                self.counts["overlap {0} phase rows dropped".format(phase)] += 1
                continue
            if row["tier"] not in OVERLAP_TIERS:
                raise SystemExit("{0}: unknown tier '{1}'".format(timings, row["tier"]))
            if phase == "performance" and row["tier"] not in OVERLAP_PERFORMANCE_TIERS:
                self.counts["overlap performance rows of tier {0} dropped".format(
                    row["tier"])] += 1
                continue
            n = int(row["n"])
            metric = metrics.get((row["algorithm"], phase, row["mode"], row["tier"], n,
                                  float(row["setting"])))
            errored = NAN
            if metric is None:
                self.counts["overlap rows without a metrics row (errored_pct NaN)"] += 1
            else:
                finite = _float(metric.get("finite_trajectories"))
                failed = _float(metric.get("failed_trajectories"))
                if finite + failed > 0:
                    errored = 100.0 * failed / (finite + failed)
            spec, states = self.spec("julia_gpu", key, problem, row["algorithm"],
                                     row["mode"], row["setting"], n)
            self.add(dict(spec, transfers=row["transfers"], states=states,
                          min_ms=_float(row["min_ms"]), samples_ms=[], errored_pct=errored,
                          package_version=version, recorded_utc=stamp), "overlap " + phase)
            self.counts["overlap {0} rows converted".format(phase)] += 1
        failures = os.path.join(problem_dir, "julia_failures.csv")
        if os.path.isfile(failures):
            self.counts["overlap failure records dropped"] += len(_read_csv(failures))
        self.counts["overlap finals files dropped"] += len(
            glob.glob(os.path.join(problem_dir, "finals", "*", "*", "*.csv")))

    # ---------------------------------------------------------- NE julia
    def convert_ne_julia(self, problem_dir, key, problem):
        """The controller table moves under julia_cpu; every sweep file and its finals are dropped."""
        for path in sorted(glob.glob(os.path.join(problem_dir, "*.csv"))):
            name = os.path.basename(path)[:-4]
            if name == CONTROLLER_TABLE:
                target = os.path.join(self.store.package_dir("julia_cpu", key),
                                      "controllers", problem + ".csv")
                if not self.dry_run:
                    os.makedirs(os.path.dirname(target), exist_ok=True)
                    shutil.copyfile(path, target)
                self.counts["NE controller tables copied"] += 1
                continue
            kind = "tol" if name.endswith("_adaptive") else "dt"
            settings = set()
            with open(path, newline="", encoding="utf-8") as handle:
                for record in csv.DictReader(handle):
                    settings.add(record[kind])
            self.counts["NE sweep files dropped"] += 1
            self.counts["NE sweep rows dropped"] += len(settings)

    # ------------------------------------------------------------ golden
    def convert_golden(self, path):
        name = os.path.basename(path)
        problem = name[len(GOLDEN_PREFIX):-len(GOLDEN_SUFFIX)]
        if problem not in self.problems:
            self.counts["golden files of unknown problems dropped"] += 1
            return
        catalogue = self.problems[problem]
        algorithm, tol = catalogue["golden_algorithm"], catalogue["golden_tol"]
        finals = np.loadtxt(path, delimiter=",", dtype=np.float64, ndmin=2)
        if finals.shape != (GOLDEN_N, catalogue["states"]):
            self.counts["golden files of the wrong shape dropped"] += 1
            return
        success = np.ones(GOLDEN_N, dtype=bool)
        retcode = [""] * GOLDEN_N
        sidecar = path[:-len(GOLDEN_SUFFIX)] + GOLDEN_RETCODES
        if os.path.isfile(sidecar):
            for row in _read_csv(sidecar):
                success[int(row["row"]) - 1] = False
                retcode[int(row["row"]) - 1] = row["retcode"]
            self.counts["golden retcode sidecar rows carried"] += int((~success).sum())
        system, states = self.system(problem, precision="float64")
        spec = dict(system, algorithm=algorithm, controller="default", dt=NAN, dt_min=NAN,
                    dt_max=NAN, atol=tol, rtol=tol, gains={}, newton_atol=NAN,
                    newton_rtol=NAN, n=GOLDEN_N, package="julia_cpu", key=GOLDEN_KEY)
        relative, errored = self.write_finals(
            spec, finals, _t_final(success, spec["duration"]), retcode)
        self.add(dict(spec, transfers="none", states=states, min_ms=NAN,
                      errored_pct=errored, finals=relative,
                      recorded_utc=_git_stamp(path, self.repo_root)), "golden")
        self.counts["golden rows converted"] += 1

    # ------------------------------------------------------ numerical dirs
    def convert_numerical(self, problem_dir):
        """Every per-key finals file under data/numerical is dropped."""
        for path in sorted(glob.glob(os.path.join(problem_dir, "*.csv"))):
            self.counts["numerical {0} dropped".format(os.path.basename(path))] += 1

    # -------------------------------------------------------------- driver
    def run(self):
        root = self.root
        for directory, package in LEGACY_DIRS.items():
            for path in sorted(glob.glob(os.path.join(root, directory, "*", "results.csv"))):
                self.convert_results(path, package)
        overlap = os.path.join(root, "cubie_julia_overlap")
        for problem_dir in sorted(glob.glob(os.path.join(overlap, "*", "*"))):
            if os.path.isdir(problem_dir):
                self.convert_overlap(problem_dir, os.path.basename(os.path.dirname(problem_dir)),
                                     os.path.basename(problem_dir))
        if os.path.isdir(overlap):
            self.counts["overlap derived tables dropped"] += len(
                [p for p in glob.glob(os.path.join(overlap, "**", "*.csv"), recursive=True)
                 if os.sep + "finals" + os.sep not in p
                 and os.path.basename(p) not in OVERLAP_KEPT])
        ne_julia = os.path.join(root, "numerical_equivalence", "julia")
        for problem_dir in sorted(glob.glob(os.path.join(ne_julia, "*", "*"))):
            if os.path.isdir(problem_dir):
                self.convert_ne_julia(problem_dir, os.path.basename(os.path.dirname(problem_dir)),
                                      os.path.basename(problem_dir))
        numerical = os.path.join(root, "numerical")
        for path in sorted(glob.glob(os.path.join(numerical, GOLDEN_PREFIX + "*" + GOLDEN_SUFFIX))):
            self.convert_golden(path)
        for key_dir in sorted(glob.glob(os.path.join(numerical, "*_*"))):
            if not os.path.isdir(key_dir) or os.path.basename(key_dir).startswith("."):
                continue
            for problem_dir in sorted(glob.glob(os.path.join(key_dir, "*"))):
                if os.path.isdir(problem_dir):
                    self.convert_numerical(problem_dir)
        if not self.dry_run:
            self.store.record_batch(list(self.rows.values()))
        self.match_sets()
        return self

    def match_sets(self):
        """Count the converted rows each shipped set produces (by run_id, under the row's key) and the rows no set produces by (package, controller)."""
        self.produced = Counter()
        self.unproduced = Counter()
        packages = sorted({row["package"] for row in self.rows.values()})
        keys = sorted({row["key"] for row in self.rows.values()})
        produced = {}
        for key in keys:
            for name in sets.set_names(self.sets_dir):
                for spec in sets.expand([name], key, root=self.root, packages=packages,
                                        sets_dir=self.sets_dir):
                    for transfers in spec["transfers"]:
                        run = store.run_id(dict(spec, transfers=transfers, key=key))
                        produced.setdefault(run, set()).add(name)
        for row in self.rows.values():
            names = produced.get(row["run_id"])
            if names:
                for name in sorted(names):
                    self.produced[name] += 1
            else:
                self.unproduced[(row["package"], row["controller"])] += 1
        return self.produced, self.unproduced

    def verify(self):
        """DuckDB counts per (key, package) against the converted rows and every stored row's run_id rehashed from its spec; raises on a mismatch."""
        expected = Counter((row["key"], row["package"]) for row in self.rows.values())
        if self.dry_run:
            return expected, expected
        found = Counter()
        for row in self.store.rows():
            found[(row["key"], row["package"])] += 1
            if row["run_id"] != store.run_id(row) or row["run_id"] not in self.rows:
                raise SystemExit("stored row {0} does not hash its spec".format(row["run_id"]))
        if found != expected:
            raise SystemExit("store counts differ from the converted rows: {0} vs {1}".format(
                dict(found), dict(expected)))
        return expected, found

    def delete_old(self):
        if self.dry_run:
            return
        for name in LEGACY_TREES:
            path = os.path.join(self.root, name)
            if os.path.isdir(path):
                shutil.rmtree(path)

    def report(self, expected):
        lines = ["| rule | count |", "|---|---|"]
        for name, count in sorted(self.counts.items()):
            lines.append("| {0} | {1} |".format(name, count))
        lines.append("| finals files written | {0} |".format(self.finals_written))
        lines.append("| store rows | {0} |".format(len(self.rows)))
        lines += ["", "| key | package | rows |", "|---|---|---|"]
        for (key, package), count in sorted(expected.items()):
            lines.append("| {0} | {1} | {2} |".format(key, package, count))
        lines += ["", "| set | rows it produces |", "|---|---|"]
        for name, count in sorted(self.produced.items()):
            lines.append("| {0} | {1} |".format(name, count))
        lines.append("| any set | {0} |".format(len(self.rows) - sum(self.unproduced.values())))
        lines += ["", "| package | controller | rows no shipped set produces |", "|---|---|---|"]
        for (package, controller), count in sorted(self.unproduced.items()):
            lines.append("| {0} | {1} | {2} |".format(package, controller, count))
        lines.append("| all | all | {0} |".format(sum(self.unproduced.values())))
        return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="data")
    parser.add_argument("--dry-run", action="store_true",
                        help="count only; write and delete nothing")
    parser.add_argument("--keep-old", action="store_true",
                        help="leave the legacy trees in place")
    args = parser.parse_args(argv)
    conversion = Conversion(args.root, dry_run=args.dry_run).run()
    expected, _ = conversion.verify()
    if not args.keep_old:
        conversion.delete_old()
    print(conversion.report(expected))
    return 0


if __name__ == "__main__":
    sys.exit(main())
