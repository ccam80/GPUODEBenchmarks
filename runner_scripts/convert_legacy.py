"""Convert the legacy CSV trees under data/ into the parquet result store, then delete them (docs/unification-plan.md, P3).

    convert_legacy.py [--root data] [--dry-run] [--keep-old]

Rules, one counter each in the report:
  results.csv rows      analysis dropped, julia renamed julia_gpu; every cpp row, wp rows with transfers != none and jax kvaerno3 rows dropped
  overlap julia_timings performance and work_precision phases become julia_gpu rows (tier fixed and julia -> default, pi kept; golden_rmse -> error; no samples); the numerical phase, its finals and the derived tables are dropped
  NE julia trees        julia_cpu finals at n = 1024 with an untimed row per setting; controller_constants.csv -> controllers/<problem>.csv
  numerical finals      jax.csv, pytorch.csv, myokit_cuda.csv and the cubie files attach to the n = 32768 times row; julia_*.csv and mpgos*.csv dropped
Rows that meet by identity collapse to one: the row with samples wins, then the later recorded_utc; NaN and empty value columns of the winner fill from the loser.
"""

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

import store  # noqa: E402
from algorithms import get_algorithm  # noqa: E402
from problems import get_problem  # noqa: E402
from protocol import N_NE, N_WP, TIMING_TOL  # noqa: E402

NAN = float("nan")

LEGACY_DIRS = {"CUBIE": "cubie", "CUBIE_MLIR": "cubie_mlir", "JAX": "jax",
               "PYTORCH": "pytorch", "MYOKIT_CUDA": "myokit_cuda", "CPP": "cpp",
               "Julia": "julia_gpu"}
LEGACY_PACKAGE = {"julia": "julia_gpu"}
OVERLAP_TIERS = {"fixed": "default", "julia": "default", "pi": "pi"}
OVERLAP_PHASES = ("performance", "work_precision")
# Legacy numerical finals file -> (package, algorithm, mode).
NUMERICAL_FILES = {"jax.csv": ("jax", "tsit5", "fixed"),
                   "pytorch.csv": ("pytorch", "classical-rk4", "fixed"),
                   "myokit_cuda.csv": ("myokit_cuda", "euler", "fixed"),
                   "cubie_unadaptive.csv": ("cubie", "classical-rk4", "fixed"),
                   "cubie_adaptive.csv": ("cubie", "tsit5", "adaptive"),
                   "cubie_mlir_unadaptive.csv": ("cubie_mlir", "classical-rk4", "fixed"),
                   "cubie_mlir_adaptive.csv": ("cubie_mlir", "tsit5", "adaptive")}
NUMERICAL_DROPPED = ("julia_fixed.csv", "julia_adaptive.csv", "mpgos.csv",
                     "mpgos_internalsave.csv")
GOLDEN_PREFIX = "golden_"
FINALS_N = 32768


def _float(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return NAN


def _stamp(path, cache={}):
    """The last git commit time of a file as an ISO UTC string, else its mtime."""
    if path in cache:
        return cache[path]
    stamp = None
    try:
        out = subprocess.run(["git", "log", "-1", "--format=%cI", "--", os.path.basename(path)],
                             cwd=os.path.dirname(os.path.abspath(path)),
                             capture_output=True, text=True, timeout=60)
        if out.returncode == 0 and out.stdout.strip():
            stamp = datetime.fromisoformat(out.stdout.strip()).astimezone(timezone.utc)
    except (OSError, subprocess.SubprocessError, ValueError):
        stamp = None
    if stamp is None:
        stamp = datetime.fromtimestamp(os.path.getmtime(path), timezone.utc)
    cache[path] = stamp.strftime("%Y-%m-%dT%H:%M:%SZ")
    return cache[path]


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _rmtree(path, dry_run):
    if os.path.isdir(path) and not dry_run:
        shutil.rmtree(path)


class Conversion:
    """Rows keyed by identity, finals written as they are met, counts per rule."""

    def __init__(self, root, dry_run=False):
        self.root = root
        self.dry_run = dry_run
        self.store = store.Store(root)
        self.rows = {}
        self.sources = {}
        self.counts = Counter()
        self.finals_written = 0
        self.old_paths = []

    # ------------------------------------------------------------ rows
    @staticmethod
    def _key(row):
        return (row["package"], row["key"], row["problem"], row["algorithm"],
                row["mode"], row["setting_kind"], store.format_setting(row["setting"]),
                int(row["n"]), int(row["states"]), row["tier"], row["transfers"])

    def add(self, row, source):
        row = store.make_row(**row)
        key = self._key(row)
        standing = self.rows.get(key)
        if standing is None:
            self.rows[key] = row
            self.sources[key] = source
            return row
        self.counts["identity collisions merged ({0} vs {1})".format(
            *sorted((self.sources[key], source)))] += 1
        winner, loser = (row, standing) if self._outranks(row, standing) else (standing, row)
        if winner is row:
            self.sources[key] = source
        for field in store.FLOAT_COLUMNS:
            if math.isnan(winner[field]) and not math.isnan(loser[field]):
                winner[field] = loser[field]
        for field in store.TEXT_COLUMNS:
            if not winner[field] and loser[field]:
                winner[field] = loser[field]
        self.rows[key] = winner
        return winner

    @staticmethod
    def _outranks(a, b):
        return (bool(a["samples_ms"]), a["recorded_utc"]) > (bool(b["samples_ms"]), b["recorded_utc"])

    def matching(self, **ident):
        return [row for row in self.rows.values() if store.same_identity(row, ident)]

    def write_finals(self, ident, finals, converged):
        self.finals_written += 1
        if self.dry_run:
            return "finals/" + store.finals_name(ident)
        return self.store.record_finals(ident, finals, converged)

    # ------------------------------------------------------- results.csv
    def convert_results(self, path, package):
        rows = _read_csv(path)
        self.old_paths.append(os.path.dirname(path))
        for row in rows:
            package_name = LEGACY_PACKAGE.get(row["package"], row["package"])
            if package_name != package:
                self.counts["results.csv rows under another package's directory dropped"] += 1
                continue
            if package_name == "cpp":
                self.counts["results.csv cpp rows dropped"] += 1
                continue
            if row["analysis"] == "wp" and row["transfers"] != "none":
                self.counts["results.csv wp rows with transfers != none dropped"] += 1
                continue
            if package_name == "jax" and row["algorithm"] == "kvaerno3":
                self.counts["results.csv jax kvaerno3 rows dropped"] += 1
                continue
            samples = [float(v) for v in row.get("samples_ms", "").split(";") if v]
            self.add(dict(
                package=package_name, key=row["key"], problem=row["problem"],
                algorithm=row["algorithm"], mode=row["mode"],
                setting_kind=row["setting_kind"], setting=float(row["setting"]),
                n=int(row["n"]), states=int(row["states"]), tier=row["tier"],
                transfers=row["transfers"], min_ms=_float(row["min_ms"]),
                samples_ms=samples, errored_pct=_float(row["errored_pct"]),
                error=_float(row["error"]), build_s=_float(row["build_s"]),
                recorded_utc=row["recorded_utc"]), "results.csv " + row["analysis"])
            self.counts["results.csv rows converted (" + package_name + ")"] += 1

    # ------------------------------------------------------------ overlap
    def convert_overlap(self, problem_dir, key, problem):
        states = get_problem(problem)["states"]
        timings = os.path.join(problem_dir, "julia_timings.csv")
        if not os.path.isfile(timings):
            return
        version = ""
        metadata = os.path.join(problem_dir, "julia_metadata.json")
        if os.path.isfile(metadata):
            with open(metadata, encoding="utf-8") as handle:
                version = json.load(handle).get("diffeqgpu_version", "")
        stamp = _stamp(timings)
        metrics = {}
        metrics_path = os.path.join(problem_dir, "julia_metrics.csv")
        if os.path.isfile(metrics_path):
            for row in _read_csv(metrics_path):
                metrics[(row["algorithm"], row["phase"], row["mode"], row["tier"],
                         int(row["n"]), store.format_setting(row["setting"]))] = row
        wp = {}
        wp_path = os.path.join(problem_dir, "work_precision.csv")
        if os.path.isfile(wp_path):
            for row in _read_csv(wp_path):
                wp[(row["algorithm"], row["mode"], row["tier"], row["transfers"],
                    int(row["n"]), store.format_setting(row["setting"]))] = row
        for row in _read_csv(timings):
            phase = row["phase"]
            if phase not in OVERLAP_PHASES:
                self.counts["overlap {0} phase rows dropped".format(phase)] += 1
                continue
            if row["tier"] not in OVERLAP_TIERS:
                raise SystemExit("{0}: unknown tier '{1}'".format(timings, row["tier"]))
            n = int(row["n"])
            setting = store.format_setting(row["setting"])
            metric = metrics.get((row["algorithm"], phase, row["mode"], row["tier"], n, setting))
            error = NAN
            if phase == "work_precision":
                if n != N_WP:
                    self.counts["overlap work_precision rows off n = {0} dropped".format(N_WP)] += 1
                    continue
                metric = wp.get((row["algorithm"], row["mode"], row["tier"],
                                 row["transfers"], n, setting)) or metric
                if metric is not None:
                    error = _float(metric.get("golden_rmse"))
            errored = NAN
            if metric is not None:
                finite = _float(metric.get("finite_trajectories"))
                failed = _float(metric.get("failed_trajectories"))
                if finite + failed > 0:
                    errored = 100.0 * failed / (finite + failed)
            self.add(dict(
                package="julia_gpu", key=key, problem=problem,
                algorithm=row["algorithm"], mode=row["mode"],
                setting_kind=row["setting_kind"], setting=float(row["setting"]),
                n=n, states=states, tier=OVERLAP_TIERS[row["tier"]],
                transfers=row["transfers"], min_ms=_float(row["min_ms"]),
                samples_ms=[], errored_pct=errored, error=error,
                package_version=version, recorded_utc=stamp), "overlap " + phase)
            self.counts["overlap {0} rows converted".format(phase)] += 1
        failures = os.path.join(problem_dir, "julia_failures.csv")
        if os.path.isfile(failures):
            self.counts["overlap failure records dropped"] += len(_read_csv(failures))
        self.counts["overlap finals files dropped"] += len(
            glob.glob(os.path.join(problem_dir, "finals", "*", "*", "*.csv")))

    # ----------------------------------------------------------- NE julia
    def convert_ne_julia(self, problem_dir, key, problem):
        states = get_problem(problem)["states"]
        for path in sorted(glob.glob(os.path.join(problem_dir, "*.csv"))):
            name = os.path.basename(path)[:-4]
            if name == "controller_constants":
                target = os.path.join(self.store.package_dir("julia_cpu", key),
                                      "controllers", problem + ".csv")
                if not self.dry_run:
                    os.makedirs(os.path.dirname(target), exist_ok=True)
                    shutil.copyfile(path, target)
                self.counts["NE controller tables copied"] += 1
                continue
            mode = "adaptive" if name.endswith("_adaptive") else "fixed"
            algorithm = name[:-len("_adaptive")] if mode == "adaptive" else name
            try:
                get_algorithm(algorithm)
            except SystemExit:
                self.counts["NE files of unknown algorithms dropped"] += 1
                continue
            kind = "tol" if mode == "adaptive" else "dt"
            stamp = _stamp(path)
            groups = {}
            with open(path, newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                names = sorted((c for c in reader.fieldnames if c[0] == "s" and c[1:].isdigit()),
                               key=lambda c: int(c[1:]))
                for record in reader:
                    groups.setdefault(record[kind], []).append(record)
            if len(names) != states:
                self.counts["NE files with the wrong state count dropped"] += 1
                continue
            for setting_text, records in groups.items():
                if len(records) != N_NE:
                    self.counts["NE settings without {0} trajectories dropped".format(N_NE)] += 1
                    continue
                records.sort(key=lambda r: int(r["traj"]))
                finals = np.array([[float(r[c]) for c in names] for r in records], dtype=np.float32)
                converged = [r["converged"].strip().lower() in ("1", "true", "t", "yes")
                             for r in records]
                ident = dict(package="julia_cpu", key=key, problem=problem,
                             algorithm=algorithm, mode=mode, setting_kind=kind,
                             setting=float(setting_text), n=N_NE, states=states,
                             tier="default")
                relative = self.write_finals(ident, finals, converged)
                self.add(dict(ident, transfers="none", min_ms=NAN, reason="untimed",
                              finals=relative, recorded_utc=stamp), "NE julia")
                self.counts["NE julia_cpu finals converted"] += 1
        self.old_paths.append(problem_dir)

    # ------------------------------------------------------ numerical dirs
    def convert_numerical(self, problem_dir, key, problem):
        row = get_problem(problem)
        for path in sorted(glob.glob(os.path.join(problem_dir, "*.csv"))):
            name = os.path.basename(path)
            if name in NUMERICAL_DROPPED:
                self.counts["numerical {0} dropped".format(name)] += 1
                continue
            if name not in NUMERICAL_FILES:
                self.counts["numerical files of unknown packages dropped"] += 1
                continue
            package, algorithm, mode = NUMERICAL_FILES[name]
            finals = np.loadtxt(path, delimiter=",", dtype=np.float64, ndmin=2)
            if finals.shape != (FINALS_N, row["states"]):
                self.counts["numerical files of the wrong shape dropped"] += 1
                continue
            kind = "dt" if mode == "fixed" else "tol"
            setting = row.timing_dt if mode == "fixed" else TIMING_TOL
            ident = dict(package=package, key=key, problem=problem, algorithm=algorithm,
                         mode=mode, setting_kind=kind, setting=setting, n=FINALS_N,
                         states=row["states"], tier="default")
            carriers = self.matching(**ident)
            if not carriers:
                self.counts["numerical finals without a times row dropped"] += 1
                continue
            converged = np.isfinite(finals).all(axis=1)
            relative = self.write_finals(ident, finals, converged)
            for carrier in carriers:
                carrier["finals"] = relative
            self.counts["numerical finals attached"] += 1
        self.old_paths.append(problem_dir)

    # -------------------------------------------------------------- driver
    def run(self):
        root = self.root
        for directory, package in LEGACY_DIRS.items():
            for path in sorted(glob.glob(os.path.join(root, directory, "*", "results.csv"))):
                self.convert_results(path, package)
            if os.path.isdir(os.path.join(root, directory)):
                self.old_paths.append(os.path.join(root, directory))
        overlap = os.path.join(root, "cubie_julia_overlap")
        for problem_dir in sorted(glob.glob(os.path.join(overlap, "*", "*"))):
            if os.path.isdir(problem_dir):
                self.convert_overlap(problem_dir, os.path.basename(os.path.dirname(problem_dir)),
                                     os.path.basename(problem_dir))
        if os.path.isdir(overlap):
            self.counts["overlap derived tables dropped"] += len(
                [p for p in glob.glob(os.path.join(overlap, "**", "*.csv"), recursive=True)
                 if os.sep + "finals" + os.sep not in p and os.path.basename(p)
                 not in ("julia_timings.csv", "work_precision.csv", "julia_metrics.csv",
                         "julia_failures.csv")])
            self.old_paths.append(overlap)
        ne_julia = os.path.join(root, "numerical_equivalence", "julia")
        for problem_dir in sorted(glob.glob(os.path.join(ne_julia, "*", "*"))):
            if os.path.isdir(problem_dir):
                self.convert_ne_julia(problem_dir, os.path.basename(os.path.dirname(problem_dir)),
                                      os.path.basename(problem_dir))
        if os.path.isdir(ne_julia):
            self.old_paths.append(ne_julia)
        numerical = os.path.join(root, "numerical")
        for key_dir in sorted(glob.glob(os.path.join(numerical, "*_*"))):
            if not os.path.isdir(key_dir) or os.path.basename(key_dir).startswith("."):
                continue
            for problem_dir in sorted(glob.glob(os.path.join(key_dir, "*"))):
                if os.path.isdir(problem_dir):
                    self.convert_numerical(problem_dir, os.path.basename(key_dir),
                                           os.path.basename(problem_dir))
            self.old_paths.append(key_dir)
        if not self.dry_run:
            self.store.record_batch(list(self.rows.values()))
        return self

    def verify(self):
        """DuckDB counts per (key, package) against the converted rows; raises on a mismatch."""
        expected = Counter((row["key"], row["package"]) for row in self.rows.values())
        if self.dry_run:
            return expected, expected
        table = self.store.query(
            "SELECT key, package, count(*) AS c FROM results GROUP BY key, package")
        found = Counter()
        for record in table.to_pylist():
            found[(record["key"], record["package"])] = record["c"]
        if found != expected:
            raise SystemExit("store counts differ from the converted rows: {0} vs {1}".format(
                dict(found), dict(expected)))
        return expected, found

    def delete_old(self):
        for path in sorted(set(self.old_paths), key=len, reverse=True):
            _rmtree(path, self.dry_run)
        numerical = os.path.join(self.root, "numerical_equivalence")
        if os.path.isdir(numerical) and not self.dry_run and not any(
                not d.startswith(".") for d in os.listdir(numerical)):
            shutil.rmtree(numerical)

    def report(self, expected):
        lines = ["| rule | count |", "|---|---|"]
        for name, count in sorted(self.counts.items()):
            lines.append("| {0} | {1} |".format(name, count))
        lines.append("| finals files written | {0} |".format(self.finals_written))
        lines.append("| store rows | {0} |".format(len(self.rows)))
        lines += ["", "| key | package | rows |", "|---|---|---|"]
        for (key, package), count in sorted(expected.items()):
            lines.append("| {0} | {1} | {2} |".format(key, package, count))
        return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="data")
    parser.add_argument("--dry-run", action="store_true", help="count only; write and delete nothing")
    parser.add_argument("--keep-old", action="store_true", help="leave the legacy trees in place")
    args = parser.parse_args(argv)
    conversion = Conversion(args.root, dry_run=args.dry_run).run()
    expected, _ = conversion.verify()
    if not args.keep_old:
        conversion.delete_old()
    print(conversion.report(expected))
    return 0


if __name__ == "__main__":
    sys.exit(main())
