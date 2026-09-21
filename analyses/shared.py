"""What the analysis shares with its tests: the suite interpreter, the --set/--where flags, the store read with two rows of one run_id refused, the completeness report of a named set's canonical trials under every key with the compile timeouts the store records marked as a plan marks them, row selection by set or SQL predicate with the ensemble fields ignored under every key, the errored filter, the figure encoding (a colour per package, a marker per stepping kind (fixed or adaptive) and card, a line style per transfers: none solid, both dashed) and the display names and CSVs that leave out the columns no row captured."""

import argparse
import csv
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

# The figure encoding: a colour per package, a marker per controller kind (one set per card), a line style per transfers.
COLOURS = {
    "cubie": "tab:blue", "cubie_mlir": "tab:purple", "jax": "tab:red", "pytorch": "darkred",
    "myokit_cuda": "black", "cpp": "tab:orange", "julia_gpu": "tab:green", "julia_cpu": "tab:cyan",
}
PACKAGE_NAMES = {
    "cubie": "Cubie", "cubie_mlir": "Cubie (MLIR)", "jax": "Diffrax", "pytorch": "torchdiffeq",
    "myokit_cuda": "Myokit", "cpp": "MPGOS", "julia_gpu": "DiffEqGPU.jl", "julia_cpu": "DifferentialEquations.jl",
}
CONTROLLER_KINDS = ("fixed", "adaptive")
MARKER_SETS = (("s", "o"), ("P", "v"), ("*", "p"))
LINES = {"both": "--", "none": "-"}


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
    """A store whose rows read one per run_id (two rows of one run_id raise) and whose optimize records come from cubie_adapter; everything else is the underlying Store."""

    def __init__(self, root):
        import store as store_mod
        self._store = store_mod.Store(root)
        self.root = root

    def __getattr__(self, name):
        return getattr(self._store, name)

    def rows(self, sql_where="", **eq_filters):
        return unique(self._store.rows(sql_where, **eq_filters))

    def optimize_rows(self, package, key, root=None):
        """Every optimize.csv record of a package under a key."""
        import cubie_adapter
        return cubie_adapter.optimize_rows(package, key, root or self.root)


def unique(rows):
    """The rows, one per run_id; ValueError naming the run_id when two rows share one."""
    seen = {}
    for row in rows:
        held = seen.get(row["run_id"])
        if held is not None:
            raise ValueError("two rows of run_id {0}: {1}/{2} gains {3!r} and {4!r}".format(
                row["run_id"], row["key"], row["package"], held.get("gains"), row.get("gains")))
        seen[row["run_id"]] = row
    return list(seen.values())


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
    trial_list = trials.build_trials(sets.expand(list(set_names), sets_dir=sets_dir),
                                     sets.declarations(sets_dir=sets_dir))
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
    """The (group_id, package) pairs the sets expand to, or the pairs of the rows a SQL predicate matches."""
    import sets
    import store as store_mod
    pairs = set()
    if set_names:
        for spec in sets.expand(list(set_names)):
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

def controller_kind(row):
    """The stepping kind a figure keys its marker by: fixed or adaptive."""
    return "fixed" if row["controller"] == "fixed" else "adaptive"


def controller_text(name, row=None):
    """'1024 fixed steps' (or 'Fixed-step' without a row) or 'Adaptive steps'."""
    if name == "fixed":
        if row is None:
            return "Fixed-step"
        return "{0:g} fixed steps".format(round(number(row["duration"]) / number(row["dt"])))
    return "Adaptive steps"


def package_name(package):
    return PACKAGE_NAMES.get(package, package)


def problem_name(problem, states=None):
    """The catalogue's display name of a problem without a trailing parenthesis, with ' (<states> states)' when given."""
    import re as re_mod
    from problems import load_problems
    name = problem
    for entry in load_problems():
        if entry["problem"] == problem:
            name = re_mod.sub(r"\s*\([^)]*\)$", "", entry["display"])
    if states is not None:
        name += " ({0} states)".format(states)
    return name


def algorithm_name(algorithm):
    """The catalogue's display name of an algorithm."""
    from algorithms import algorithm_facts
    try:
        return algorithm_facts(algorithm)["display"]
    except SystemExit:
        return algorithm


def key_label(key):
    """'RTX4070-Super (Win)' from 'windows_RTX-4070-SUPER'."""
    system, _, gpu = key.partition("_")
    parts = gpu.split("-")
    name = "".join(parts[:2]) + "".join("-" + p.capitalize() for p in parts[2:])
    return "{0} ({1})".format(name, {"windows": "Win", "linux": "Linux"}.get(system, system))


def colour(package):
    return COLOURS.get(package, "gray")


def marker(name, card=0):
    """The marker of a stepping kind on a card; cards past the sets share the last set."""
    markers = MARKER_SETS[min(card, len(MARKER_SETS) - 1)]
    return markers[CONTROLLER_KINDS.index(name)] if name in CONTROLLER_KINDS else "x"


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
