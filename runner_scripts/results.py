"""The result store: one long-form results.csv per package and machine key, one row per timed point and transfer leg, mirrored by results.jl.

CLI: `results.py record <package> <key> <analysis> <problem> <algorithm> <mode> <setting_kind> <setting> <n> <states> <tier> <transfers> [field=value ...] [samples=a;b;c]`,
`results.py nan <package> <key> <analysis> <problem> <algorithm> <mode> <N|states> [build_s]`, `results.py status <package> <key> <analysis> <problem> <algorithm> <mode> <n> <states>`,
`results.py clear <package> <key> [analysis] [algorithm] [problem]`.
"""

import csv
import math
import os
import sys
import time
from datetime import datetime, timezone

from algorithms import NE_PACKAGES, get_algorithm, ne_member
from problems import get_problem
from protocol import N_WP, STATES_N, TIMING_TOL, TOLS

IDENTITY = ("package", "key", "analysis", "problem", "algorithm", "mode",
            "setting_kind", "setting", "n", "states", "tier", "transfers")
# samples_ms is every attempt of the leg in ms, warm-up first, ';'-joined; min_ms is the minimum over the attempts after the warm-up.
VALUES = ("min_ms", "samples_ms", "errored_pct", "error", "build_s",
          "recorded_utc")
FIELDS = IDENTITY + VALUES

# CLI package name -> data directory.
PACKAGE_DIRS = {"cubie": "CUBIE", "cubie_mlir": "CUBIE_MLIR", "julia": "Julia",
                "cpp": "CPP", "jax": "JAX", "pytorch": "PYTORCH",
                "myokit_cuda": "MYOKIT_CUDA"}

NAN = float("nan")
LOCK_TIMEOUT_S = 120.0
LOCK_STALE_S = 300.0


def data_root():
    """The data directory under the working directory, as bench_key.data_dir resolves it."""
    return "data"


def store_path(package, key, root=None):
    """data/<PACKAGE_DIR>/<key>/results.csv; the directory is created."""
    directory = os.path.join(root or data_root(), PACKAGE_DIRS[package], key)
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, "results.csv")


def floor_enabled():
    return os.environ.get("BENCH_FLOOR", "") not in ("", "0")


def _float(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return NAN


def _fmt(value):
    if isinstance(value, float):
        return "nan" if math.isnan(value) else "{0:.10g}".format(value)
    return str(value)


def samples_of(row):
    """The attempts of a row in ms, warm-up first; empty when none were recorded."""
    return [float(v) for v in row.get("samples_ms", "").split(";") if v]


def setting_matches(a, b):
    """Two settings name the same point within a relative 1e-8."""
    a, b = _float(a), _float(b)
    if math.isnan(a) and math.isnan(b):
        return True
    return math.isclose(a, b, rel_tol=1e-8, abs_tol=0.0)


def same_point(row, ident):
    """True when a row carries the identity columns of ident; a list, tuple or set value matches any of its members."""
    for field in IDENTITY:
        if field not in ident:
            continue
        wanted = ident[field]
        if not isinstance(wanted, (list, tuple, set)):
            wanted = (wanted,)
        if field == "setting":
            if not any(setting_matches(row[field], v) for v in wanted):
                return False
        elif str(row[field]) not in {str(v) for v in wanted}:
            return False
    return True


class _Lock:
    """A mkdir lock beside the store; a stale directory is taken over."""

    def __init__(self, path):
        self.path = path + ".lock"

    def __enter__(self):
        deadline = time.monotonic() + LOCK_TIMEOUT_S
        while True:
            try:
                os.mkdir(self.path)
                return self
            except FileExistsError:
                try:
                    age = time.time() - os.stat(self.path).st_mtime
                except OSError:
                    age = 0.0
                if age > LOCK_STALE_S:
                    try:
                        os.rmdir(self.path)
                    except OSError:
                        pass
                    continue
                if time.monotonic() > deadline:
                    raise TimeoutError("result store locked: " + self.path)
                time.sleep(0.05)

    def __exit__(self, *_):
        try:
            os.rmdir(self.path)
        except OSError:
            pass


def load(path):
    """Every row of a store as dicts of strings; a missing file is empty."""
    if not os.path.isfile(path):
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [row for row in reader if row.get("package")]
    return rows


def _save(path, rows):
    scratch = path + ".partial"
    with open(scratch, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n",
                                extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})
    # Windows refuses the swap while a reader still holds the old file.
    for attempt in range(20):
        try:
            os.replace(scratch, path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.05)


def _lower_wins(recorded, new):
    """True when the new row's time beats the recorded one; NaN loses."""
    old, fresh = _float(recorded.get("min_ms")), _float(new.get("min_ms"))
    if math.isnan(old):
        return True
    if math.isnan(fresh):
        return False
    return fresh < old


def make_row(package, key, analysis, problem, algorithm, mode, setting_kind,
             setting, n, states, tier="default", transfers="both",
             min_ms=NAN, samples=None, errored_pct=NAN, error=NAN,
             build_s=NAN):
    """One store row; `samples` is every attempt in ms, warm-up first."""
    return {"package": package, "key": key, "analysis": analysis,
            "problem": problem, "algorithm": algorithm, "mode": mode,
            "setting_kind": setting_kind, "setting": _fmt(float(setting)),
            "n": str(int(n)), "states": str(int(states)), "tier": tier,
            "transfers": transfers, "min_ms": _fmt(float(min_ms)),
            "samples_ms": ";".join(_fmt(float(s)) for s in (samples or [])),
            "errored_pct": _fmt(float(errored_pct)),
            "error": _fmt(float(error)), "build_s": _fmt(float(build_s)),
            "recorded_utc": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ")}


def record(path, row, floor=None):
    """Replace the row with this identity, or under --floor keep whichever has the lower time."""
    floor = floor_enabled() if floor is None else floor
    with _Lock(path):
        rows = load(path)
        replaced = False
        for index, existing in enumerate(rows):
            if same_point(existing, row):
                if not floor or _lower_wins(existing, row):
                    rows[index] = row
                replaced = True
                break
        if not replaced:
            rows.append(row)
        _save(path, rows)


def rows_for(path, **ident):
    """Rows matching the given identity columns."""
    return [row for row in load(path) if same_point(row, ident)]


def point_status(path, **ident):
    """'absent', 'nan' or 'finite' for the rows matching ident."""
    matched = rows_for(path, **ident)
    if not matched:
        return "absent"
    if any(math.isfinite(_float(row["min_ms"])) for row in matched):
        return "finite"
    return "nan"


def clear(path, **ident):
    """Drop every row matching ident; returns the count dropped."""
    with _Lock(path):
        rows = load(path)
        kept = [row for row in rows if not same_point(row, ident)]
        if len(kept) != len(rows):
            _save(path, kept)
        return len(rows) - len(kept)


def timing_setting(problem, mode):
    """(setting_kind, setting) of the N and states sweeps for a problem row."""
    if mode == "fixed":
        return "dt", problem.timing_dt
    return "tol", TIMING_TOL


def wp_settings(problem, algorithm, mode, package):
    """The settings a package's wp leg records: the ne dt grid for its ne members, else the wp grids."""
    if mode == "adaptive":
        return list(TOLS)
    row = problem if isinstance(problem, dict) else get_problem(problem)
    if package in NE_PACKAGES and ne_member(get_algorithm(algorithm), "fixed"):
        return row.ne_dts()
    return row.dts(algorithm)


class Leg:
    """One (package, key, analysis, problem, algorithm, mode) writer with the resume checks."""

    def __init__(self, package, key, analysis, problem, algorithm, mode,
                 root=None):
        self.package, self.key, self.analysis = package, key, analysis
        self.problem = (problem if isinstance(problem, dict)
                        else get_problem(problem))
        self.algorithm, self.mode = algorithm, mode
        self.path = store_path(package, key, root)
        self.setting_kind, self.setting = timing_setting(self.problem, mode)

    def _ident(self, n, states, setting=None, tier=None):
        ident = dict(package=self.package, key=self.key,
                     analysis=self.analysis, problem=self.problem.name,
                     algorithm=self.algorithm, mode=self.mode,
                     setting_kind=self.setting_kind,
                     setting=_fmt(float(self.setting if setting is None
                                        else setting)),
                     n=str(int(n)), states=str(int(states)))
        if tier is not None:
            ident["tier"] = tier
        return ident

    def status(self, n, states=None, setting=None, tier=None):
        """'absent', 'nan' or 'finite' for the point; tier None matches any tier."""
        return point_status(self.path, **self._ident(
            n, self.problem["states"] if states is None else states,
            setting, tier))

    def record(self, n, transfers, states=None, setting=None, tier="default",
               **values):
        states = self.problem["states"] if states is None else states
        row = make_row(self.package, self.key, self.analysis,
                       self.problem.name, self.algorithm, self.mode,
                       self.setting_kind,
                       self.setting if setting is None else setting, n,
                       states, tier=tier, transfers=transfers, **values)
        record(self.path, row)

    def record_times(self, n, t_both, t_none, errored_pct, samples_both=None,
                     samples_none=None, build_s=NAN, states=None):
        """The two transfer legs of one N or states point."""
        self.record(n, "both", states=states, min_ms=t_both,
                    samples=samples_both, errored_pct=errored_pct,
                    build_s=build_s)
        self.record(n, "none", states=states, min_ms=t_none,
                    samples=samples_none, errored_pct=errored_pct,
                    build_s=build_s)

    def record_wp(self, setting, t_ms, error, errored_pct, samples=None,
                  tier="default"):
        """One work-precision point; every package times the resident solve alone."""
        self.record(N_WP, "none", setting=setting, tier=tier, min_ms=t_ms,
                    error=error, errored_pct=errored_pct, samples=samples)

    def nan_times(self, ns):
        for n in ns:
            self.record_times(n, NAN, NAN, 100.0)

    def nan_states(self, sizes, build_s=NAN):
        for nstates in sizes:
            self.record_times(STATES_N, NAN, NAN, 100.0, build_s=build_s,
                              states=nstates)

    def nan_wp(self, settings):
        for setting in settings:
            self.record_wp(setting, NAN, NAN, 100.0)


def _cli(argv):
    if len(argv) >= 13 and argv[0] == "record":
        (package, key, analysis, problem, algorithm, mode, kind, setting, n,
         states, tier, transfers) = argv[1:13]
        values = {}
        for item in argv[13:]:
            name, _, value = item.partition("=")
            if name == "samples":
                values[name] = [_float(v) for v in value.split(";") if v]
            else:
                values[name] = _float(value)
        row = make_row(package, key, analysis, problem, algorithm, mode, kind,
                       _float(setting), int(n), int(states), tier=tier,
                       transfers=transfers, **values)
        record(store_path(package, key), row)
        return 0
    if len(argv) in (8, 9) and argv[0] == "nan":
        package, key, analysis, problem, algorithm, mode, value = argv[1:8]
        build_s = _float(argv[8]) if len(argv) == 9 else NAN
        leg = Leg(package, key, analysis, problem, algorithm, mode)
        if analysis == "states":
            leg.nan_states([int(value)], build_s=build_s)
        else:
            leg.nan_times([int(value)])
        return 0
    if len(argv) == 9 and argv[0] == "status":
        package, key, analysis, problem, algorithm, mode, n, states = argv[1:]
        print(point_status(store_path(package, key), package=package, key=key,
                           analysis=analysis, problem=problem,
                           algorithm=algorithm, mode=mode, n=n,
                           states=states))
        return 0
    if len(argv) >= 3 and argv[0] == "clear":
        package, key = argv[1], argv[2]
        ident = {}
        for name, value in zip(("analysis", "algorithm", "problem"), argv[3:]):
            if value and value != "all":
                ident[name] = value
        print(clear(store_path(package, key), package=package, key=key,
                    **ident))
        return 0
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))
