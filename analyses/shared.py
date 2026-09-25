"""What the analysis shares with its tests: the suite interpreter, the --where/--kind flags, the store read with two rows of one run_id refused, row selection by SQL predicate (every row without one), the errored filter, the figure style and encoding (Encoding) and the display names and CSVs that leave out the columns no row captured."""

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

COLOURS = {
    "cubie": "tab:blue", "jax": "tab:red", "pytorch": "darkred",
    "myokit_cuda": "black", "cpp": "tab:orange", "julia_gpu": "tab:green", "julia_cpu": "tab:cyan",
}
PACKAGE_NAMES = {
    "cubie": "Cubie", "jax": "Diffrax", "pytorch": "torchdiffeq",
    "myokit_cuda": "Myokit", "cpp": "MPGOS", "julia_gpu": "DiffEqGPU.jl", "julia_cpu": "DifferentialEquations.jl",
}
CONTROLLER_KINDS = ("fixed", "adaptive")
# Line style per (steps, transfers); marker shapes in the order algorithms take them.
LINE_STYLES = {("adaptive", "none"): "-", ("fixed", "none"): "--", ("adaptive", "both"): ":", ("fixed", "both"): "-."}
LINE_LABELS = {("adaptive", "none"): "Adaptive", ("fixed", "none"): "Fixed",
               ("adaptive", "both"): "Adaptive, with transfers", ("fixed", "both"): "Fixed, with transfers"}
SHAPES = ("o", "s", "^", "D", "v", "P", "X", "*", "p", "h", "<", ">", "8", "H", "d")
SERIES_STYLE = {"linewidth": 1.5, "markersize": 6, "markeredgecolor": "black", "markeredgewidth": 0.5}
CROSS_STYLE = {"linestyle": "none", "marker": "x", "color": "black", "markersize": 10, "markeredgewidth": 1.5}
CROSSED_LABEL = "over 10% of trajectories errored"
RC = {"axes.grid": True, "axes.grid.which": "both", "grid.alpha": 0.3, "legend.fontsize": 7,
      "axes.titlesize": 9, "figure.titlesize": 12, "savefig.dpi": 150}


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

def parser(description, kinds=()):
    """The argument parser: --where (every row without it), --kind (repeatable, every kind without it), --root, --out and --no-sync."""
    p = argparse.ArgumentParser(description=description,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--where", default="", metavar="SQL",
                   help="a SQL predicate over the columns of the results view; every row without it")
    p.add_argument("--kind", action="append", default=[], choices=kinds, metavar="KIND",
                   help="a figure kind to write, one of {0}; repeatable, every kind without it".format(", ".join(kinds)))
    p.add_argument("--root", default=DATA_DIR, help="the store root (default data/)")
    p.add_argument("--out", default=PLOTS_DIR, help="the output root (default plots/)")
    p.add_argument("--no-sync", action="store_true", help="read --root as it is, without pulling the store")
    return p


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

def select_rows(store, where=""):
    """Every row of the store, one per run_id, that a SQL predicate over the results view matches; every row without one."""
    return store.rows(sql_where=where)


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


def problem_label(problem):
    """'Lorenz 96 (20 states)', 'Ring modulator, index 2 (15 states)' from the catalogue."""
    from problems import load_problems
    entry = next(e for e in load_problems() if e["problem"] == problem)
    name = re.sub(r"\s*\((\d+)\)$", "", entry["display"])
    name = re.sub(r"\s*\((.+)\)$", r", \1", name)
    return "{0} ({1} states)".format(name, entry["states"])


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


def catalogue_order():
    """Every algorithm name in algorithms.csv order."""
    with open(os.path.join(RUNNER_SCRIPTS, "algorithms.csv"), encoding="utf-8") as handle:
        return list(dict.fromkeys(r["algorithm"] for r in csv.DictReader(handle)))


class Encoding:
    """Every figure's visual language: colour is the package, marker shape and fill the algorithm, line style the steps (dashed fixed) and transfers."""

    def __init__(self, series):
        self.order = catalogue_order()
        self.packages = [p for p in COLOURS if any(k[1] == p for k in series)]
        self.algorithms = sorted({self.algorithm(k, pts) for k, pts in series.items()}, key=self._rank)
        self.lines = [s for s in LINE_STYLES if any((k[2], k[3] or "none") == s for k in series)]

    def _rank(self, algorithm):
        return self.order.index(algorithm) if algorithm in self.order else len(self.order)

    @staticmethod
    def algorithm(key, points):
        return key[4] if len(key) > 4 else points[0][2]["algorithm"]

    def marker_of(self, algorithm):
        """(shape, filled): the catalogue's first len(SHAPES) algorithms filled, the rest hollow."""
        rank = self._rank(algorithm) % (2 * len(SHAPES))
        return SHAPES[rank % len(SHAPES)], rank < len(SHAPES)

    def style(self, key, points):
        """matplotlib plot() keywords for one series."""
        shape, filled = self.marker_of(self.algorithm(key, points))
        colour_value = colour(key[1])
        return dict(SERIES_STYLE, color=colour_value, marker=shape,
                    markerfacecolor=colour_value if filled else "white",
                    linestyle=LINE_STYLES[(key[2], key[3] or "none")])

    def legend(self, plt, crossed):
        """(handles, labels, headings): each channel under its heading, only what the figure shows."""
        handles, labels, headings = [], [], []

        def section(heading, entries):
            headings.append(heading)
            handles.append(plt.Line2D([], [], linestyle="none"))
            labels.append(heading)
            for handle, text in entries:
                handles.append(handle)
                labels.append(text)

        def swatch(**kw):
            return plt.Line2D([], [], **dict(SERIES_STYLE, **kw))

        section("Package", [(swatch(color=colour(p)), package_name(p)) for p in self.packages])
        markers = []
        for a in self.algorithms:
            shape, filled = self.marker_of(a)
            markers.append((swatch(color="gray", linestyle="none", marker=shape,
                                   markerfacecolor="gray" if filled else "white"), algorithm_name(a)))
        section("Algorithm", markers)
        section("Steps", [(swatch(color="gray", linestyle=LINE_STYLES[s]), LINE_LABELS[s]) for s in self.lines])
        if crossed:
            handles.append(plt.Line2D([], [], **CROSS_STYLE))
            labels.append(CROSSED_LABEL)
        return handles, labels, headings


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
    """matplotlib.pyplot on the Agg backend with the RC defaults."""
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update(RC)
    import matplotlib.pyplot as plt
    return plt
