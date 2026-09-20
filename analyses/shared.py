"""What the analysis shares with its tests: the suite interpreter, the --set/--where flags, the store read in its current form whatever form a row was written in (a cubie PI row within Float32 rounding of the DIRK tier carries the tier's exact gains, and of the rows one run_id then holds the fastest stands), the completeness report of a named set's canonical trials under every key with the compile timeouts the store records marked as a plan marks them, row selection by set or SQL predicate with the ensemble fields ignored under every key, the errored filter, the figure encoding (a colour per package, a marker per controller, a line style per transfers) and CSVs that leave out the columns no row captured."""

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RUNNER_SCRIPTS = os.path.join(ROOT, "runner_scripts")
if RUNNER_SCRIPTS not in sys.path:
    sys.path.insert(0, RUNNER_SCRIPTS)

DATA_DIR = os.path.join(ROOT, "data")
PLOTS_DIR = os.path.join(ROOT, "plots")
SUITE_VENV = os.path.join(ROOT, "GPU_ODE_CUBIE", "venv")

ERRORED_PCT_LIMIT = 10.0
NAN = float("nan")

# The figure encoding: a colour per package, a marker per controller, a line style per transfers.
COLOURS = {
    "cubie": "tab:blue", "cubie_mlir": "tab:purple", "jax": "tab:red", "pytorch": "darkred",
    "myokit_cuda": "black", "cpp": "tab:orange", "julia_gpu": "tab:green", "julia_cpu": "tab:cyan",
}
MARKERS = {"fixed": "s", "default": "o", "pi": "^", "pi matched": "v", "gustafsson": "D"}
LINES = {"both": "-", "none": "--"}


def under_suite_python():
    """Re-run under GPU_ODE_CUBIE/venv when this interpreter lacks the store's or the figures' dependencies."""
    try:
        import duckdb  # noqa: F401
        import matplotlib  # noqa: F401
        import pyarrow  # noqa: F401
        return
    except ImportError:
        pass
    for candidate in (os.path.join(SUITE_VENV, "Scripts", "python.exe"),
                      os.path.join(SUITE_VENV, "bin", "python3"),
                      os.path.join(SUITE_VENV, "bin", "python")):
        if os.path.isfile(candidate) and os.path.abspath(candidate) != os.path.abspath(sys.executable):
            raise SystemExit(subprocess.call([candidate] + sys.argv))
    raise SystemExit("duckdb, pyarrow and matplotlib are required; none found in " + SUITE_VENV)


# ------------------------------------------------------------------- flags

def parser(description):
    """The argument parser: --set (repeatable) or --where, --root and --out."""
    p = argparse.ArgumentParser(description=description,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set", action="append", default=[], metavar="NAME",
                   help="a set under sets/; repeatable")
    p.add_argument("--where", default="", metavar="SQL",
                   help="a SQL predicate over the spec columns of the results view")
    p.add_argument("--root", default=DATA_DIR, help="the store root (default data/)")
    p.add_argument("--out", default=PLOTS_DIR, help="the output root (default plots/)")
    p.add_argument("--no-sync", action="store_true", help="read --root as it is, without pulling the store")
    return p


def check_selection(args):
    if bool(args.set) == bool(args.where):
        raise SystemExit("give --set NAME (repeatable) or --where \"<sql>\", not both")


def pull_store(args):
    """Pull the store into --root before reading it; SystemExit when the store is not set up or the pull fails, nothing under --no-sync."""
    if args.no_sync:
        return
    sys.path.insert(0, os.path.join(ROOT, "sync"))
    import sync
    reason = sync.unavailable()
    if reason:
        raise SystemExit("store: {0}; pass --no-sync to read {1} as it is".format(reason, args.root))
    if sync.run("pull", args.root, ""):
        raise SystemExit("store: pull FAILED; pass --no-sync to read {0} as it is".format(args.root))


# ------------------------------------------------------------------- store

class AnalysisStore:
    """A store whose rows and optimize records read in the current form: the gains of a cubie PI row within Float32 rounding of the DIRK PI tier at its algorithm's order are the tier's exact gains (the form a set declares since the rounding rule), a row's ids hash the rewritten spec, and of the rows one run_id then holds the fastest stands (fastest). Everything else is the underlying Store."""

    def __init__(self, root):
        import store as store_mod
        self._store = store_mod.Store(root)
        self.root = root
        self._tiers = {}

    def __getattr__(self, name):
        return getattr(self._store, name)

    def rows(self, sql_where="", **eq_filters):
        return fastest(normalise_gains(r, self._tiers) for r in self._store.rows(sql_where, **eq_filters))

    def optimize_rows(self, package, key, root=None):
        """Every optimize.csv record of a package under a key, its gains in the current form."""
        import cubie_adapter
        return [normalise_gains(r, self._tiers, ids=False)
                for r in cubie_adapter.optimize_rows(package, key, root or self.root)]


def tier_gains(package, algorithm, cache=None):
    """The DIRK PI tier controller at a cubie algorithm's catalogue order, None when the package has no row of the algorithm; cached per (package, algorithm) in `cache`."""
    import cubie_adapter
    from algorithms import get_algorithm
    ident = (package, algorithm)
    if cache is not None and ident in cache:
        return cache[ident]
    entry = get_algorithm(algorithm, package)
    settings = None
    if entry is not None and entry["order"] is not None:
        settings = dict(cubie_adapter.pi_tier_controller(entry["order"]))
    if cache is not None:
        cache[ident] = settings
    return settings


def tier_gains_json(package, algorithm, cache=None):
    """The tier's gains as the canonical JSON a row carries; None when there is no tier."""
    import store as store_mod
    tier = tier_gains(package, algorithm, cache)
    if tier is None:
        return None
    return store_mod.canonical_json({k: v for k, v in tier.items() if k != "step_controller"})


def normalise_gains(row, cache=None, ids=True):
    """The row (or optimize record) with its gains rewritten to the DIRK PI tier's exact values, and with `ids` its ids rehashed, when it is a cubie PI row within FLOAT32_REL_TOL of the tier; the row itself otherwise."""
    import cubie_adapter
    import store as store_mod
    if row.get("package") not in cubie_adapter.PACKAGES or row.get("controller") != "pi":
        return row
    tier = tier_gains(row["package"], row["algorithm"], cache)
    if tier is None:
        return row
    exact = store_mod.canonical_json({k: v for k, v in tier.items() if k != "step_controller"})
    if row["gains"] == exact:
        return row
    try:
        gains = json.loads(row["gains"] or "{}")
    except ValueError:
        return row
    if not isinstance(gains, dict) or not cubie_adapter.controllers_equal(
            tier, dict(gains, step_controller="pi"), cubie_adapter.FLOAT32_REL_TOL):
        return row
    rewritten = dict(row, gains=exact)
    if ids:
        rewritten.update(store_mod.ids(rewritten))
    return rewritten


def fastest(rows):
    """One row per run_id, in first-appearance order: the lowest finite min_ms; among untimed rows one with its cold build time, then one with finals, then the latest recorded."""
    standing = {}
    for row in rows:
        held = standing.get(row["run_id"])
        if held is None or _rank(row) > _rank(held):
            standing[row["run_id"]] = row
    return list(standing.values())


def _rank(row):
    ms = number(row.get("min_ms"))
    stamp = row.get("recorded_utc")
    return (math.isfinite(ms), -ms if math.isfinite(ms) else 0.0, math.isfinite(number(row.get("build_s"))),
            bool(row.get("finals")), stamp.timestamp() if stamp is not None else float("-inf"))


# --------------------------------------------------------------- selection

def store_keys(store):
    """The dataset keys with results files under the store."""
    keys = set()
    for path in store.results_files():
        parts = os.path.normpath(path).split(os.sep)
        keys.update(p[len("key="):] for p in parts if p.startswith("key="))
    return sorted(keys)


def canonical_trials(store, set_names, key, sets_dir=None):
    """The canonical trials the named sets expand to under a key: every point merged with its declarations in every set file, the lines of a problem, algorithm and controller the store records a compile timeout of marked as a plan marks them (no optimize record wanted)."""
    import abandon
    import sets
    import trials
    sets_dir = sets_dir or sets.SETS_DIR
    trial_list = trials.build_trials(sets.expand(list(set_names), key, store.root, sets_dir=sets_dir),
                                     sets.declarations(key, store.root, sets_dir=sets_dir))
    return trials.mark_compile_timeouts(trial_list, abandon.compile_timeouts(store, key))


INCOMPLETE_COLUMNS = ("key", "package", "trial_id", "problem", "system_params", "precision",
                      "algorithm", "controller", "n", "dt", "atol", "sets", "missing")


def incomplete(store, set_names, sets_dir=None):
    """[(key, Missing)] of every canonical trial of the sets whose rows or artifacts the store lacks under that key, in key and file order."""
    import completeness
    out = []
    records = getattr(store, "optimize_rows", None)
    for key in store_keys(store):
        trial_list = canonical_trials(store, set_names, key, sets_dir)
        audits = completeness.audit(trial_list, key, store, optimize_rows=records)
        out.extend((key, missing) for missing in audits.values() if not missing.complete())
    return out


def incomplete_rows(lacking):
    """INCOMPLETE_COLUMNS rows of incomplete()'s result."""
    rows = []
    for key, missing in lacking:
        trial = missing.trial
        row = {c: trial.get(c) for c in INCOMPLETE_COLUMNS if c in trial}
        row.update(key=key, sets=" ".join(trial.get("sets", [])), missing=" ".join(missing.reasons()))
        rows.append(row)
    return rows


def report_incomplete(store, set_names, out, sets_dir=None):
    """Print what the store lacks of the named sets under every key and write the trials to <out>/<key>/incomplete.csv; returns the count of incomplete trials."""
    import completeness
    lacking = incomplete(store, set_names, sets_dir)
    by_key = {}
    for key, missing in lacking:
        by_key.setdefault(key, {}).setdefault(missing.trial["package"], []).append(missing)
    for key in store_keys(store):
        packages = by_key.get(key, {})
        if not packages:
            print("{0}: complete for {1}".format(key, ", ".join(set_names)))
            continue
        directory = os.path.join(out, key)
        os.makedirs(directory, exist_ok=True)
        path = write_csv(os.path.join(directory, "incomplete.csv"), INCOMPLETE_COLUMNS,
                         incomplete_rows([(k, m) for k, m in lacking if k == key]))
        print("{0}: {1} incomplete trial(s) of {2}; see {3}".format(
            key, sum(len(m) for m in packages.values()), ", ".join(set_names), path))
        for package, missing in sorted(packages.items()):
            counts = completeness.summary({m.trial["trial_id"]: m for m in missing})
            print("  {0}: {1} trial(s) lacking {2}".format(
                package, len(missing), ", ".join("{0} x{1}".format(k, v) for k, v in counts.items())))
    return len(lacking)


def selection_pairs(store, set_names=(), where=""):
    """The (group_id, package) pairs the sets expand to under every key of the store, or the pairs of the rows a SQL predicate matches."""
    import sets
    import store as store_mod
    pairs = set()
    if set_names:
        for key in store_keys(store):
            for spec in sets.expand(list(set_names), key, store.root):
                pairs.add((store_mod.group_id(spec), spec["package"]))
    else:
        for row in store.rows(sql_where=where):
            pairs.add((row["group_id"], row["package"]))
    return pairs


def select_rows(store, set_names=(), where=""):
    """Every row of the store whose (group_id, package) a set or predicate names: the trial identity with the ensemble fields ignored, under every key."""
    pairs = selection_pairs(store, set_names, where)
    return [row for row in store.rows() if (row["group_id"], row["package"]) in pairs]


def number(value):
    """A float from a row value; None and text are NaN."""
    if value is None or isinstance(value, str):
        return NAN
    try:
        return float(value)
    except (TypeError, ValueError):
        return NAN


def timed(row):
    return math.isfinite(number(row.get("min_ms")))


def within_errored_limit(row):
    """False when errored_pct is a number above the limit; an absent or NaN column keeps the row."""
    pct = number(row.get("errored_pct"))
    return math.isnan(pct) or pct <= ERRORED_PCT_LIMIT


def usable(rows):
    return [row for row in rows if within_errored_limit(row)]


# ---------------------------------------------------------------- encoding

def controller_label(row, cache=None):
    """The controller a figure keys its marker by: the row's controller, 'pi matched' for a cubie PI row whose gains are not the DIRK tier's (Julia's matched controller)."""
    import cubie_adapter
    controller = row["controller"]
    if controller == "pi" and row["package"] in cubie_adapter.PACKAGES:
        tier = tier_gains_json(row["package"], row["algorithm"], cache)
        if tier is not None and row.get("gains") != tier:
            return "pi matched"
    return controller


def colour(package):
    return COLOURS.get(package, "gray")


def marker(controller):
    return MARKERS.get(controller, "x")


def line(transfers):
    return LINES.get(transfers, ":")


def dyadic(value):
    """2^-k when value is that power of two, else %g."""
    if value > 0 and math.isfinite(value):
        k = -math.log2(value)
        if k == int(k):
            return "2^{0:d}".format(int(-k))
    return "{0:g}".format(value)


def slug(text):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")


# ------------------------------------------------------------------ output

def cell(value):
    if isinstance(value, float):
        return "nan" if math.isnan(value) else repr(value)
    if isinstance(value, list):
        return ";".join(repr(float(v)) for v in value)
    if value is None:
        return ""
    return str(value)


def captured(value):
    """False for what a row never recorded: None, empty text, NaN, an empty list."""
    if value is None:
        return False
    if isinstance(value, str):
        return value != ""
    if isinstance(value, float):
        return not math.isnan(value)
    if isinstance(value, list):
        return bool(value)
    return True


def captured_columns(columns, rows):
    """The columns some row carries a captured value of, in order; every column when there are no rows."""
    if not rows:
        return list(columns)
    return [c for c in columns if any(captured(row.get(c)) for row in rows)]


def write_csv(path, columns, rows):
    """rows as CSV with LF line ends; NaN written as nan, lists ';'-joined; a column no row captured is left out."""
    kept = captured_columns(columns, rows)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(kept)
        for row in rows:
            writer.writerow([cell(row.get(c)) for c in kept])
    return path


def pyplot():
    """matplotlib.pyplot on the Agg backend."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt
