"""What the analysis scripts share: the suite interpreter, the --set/--where flags, the completeness report of a named set's canonical trials under every key, row selection by set or SQL predicate with the ensemble fields ignored under every key, the errored filter, row labels and the figure style."""

import argparse
import csv
import hashlib
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

# Colour and marker per package, shared by every figure.
STYLE = {
    "cubie": ("tab:blue", "*"), "cubie_mlir": ("tab:purple", "h"),
    "jax": ("tab:red", "D"), "pytorch": ("darkred", "x"),
    "myokit_cuda": ("black", "s"), "cpp": ("tab:orange", "^"),
    "julia_gpu": ("tab:green", "o"), "julia_cpu": ("tab:cyan", "v"),
}


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
    """The argument parser every script starts from: --set (repeatable) or --where, --root and --out."""
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


# --------------------------------------------------------------- selection

def store_keys(store):
    """The dataset keys with leg files under the store."""
    keys = set()
    for path in store.leg_files():
        parts = os.path.normpath(path).split(os.sep)
        keys.update(p[len("key="):] for p in parts if p.startswith("key="))
    return sorted(keys)


def canonical_trials(store, set_names, key, sets_dir=None):
    """The canonical trials the named sets expand to under a key: every point merged with its declarations in every set file."""
    import sets
    import trials
    sets_dir = sets_dir or sets.SETS_DIR
    return trials.build_trials(sets.expand(list(set_names), key, store.root, sets_dir=sets_dir),
                               sets.declarations(key, store.root, sets_dir=sets_dir))


INCOMPLETE_COLUMNS = ("key", "package", "trial_id", "leg", "problem", "system_params", "precision",
                      "algorithm", "controller", "n", "dt", "atol", "sets", "missing")


def incomplete(store, set_names, sets_dir=None):
    """[(key, Missing)] of every canonical trial of the sets whose rows or artifacts the store lacks under that key, in key and file order."""
    import completeness
    out = []
    for key in store_keys(store):
        trial_list = canonical_trials(store, set_names, key, sets_dir)
        audits = completeness.audit(trial_list, key, store)
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


def within_errored_limit(row):
    """False when errored_pct is a number above the limit; an absent or NaN column keeps the row."""
    pct = number(row.get("errored_pct"))
    return math.isnan(pct) or pct <= ERRORED_PCT_LIMIT


def usable(rows):
    return [row for row in rows if within_errored_limit(row)]


# ------------------------------------------------------------------ labels

def stepping_label(row):
    """'fixed dt=2^-10' or '<controller> tol=1e-05' with the gains and Newton tolerance when set."""
    if row["controller"] == "fixed":
        text = "fixed dt=" + dyadic(number(row["dt"]))
    else:
        text = "{0} tol={1:g}".format(row["controller"], number(row["atol"]))
        if row.get("gains") not in (None, "", "{}"):
            text += " gains=" + row["gains"]
    newton = number(row.get("newton_atol"))
    if not math.isnan(newton):
        text += " newton={0:g}".format(newton)
    return text


def dyadic(value):
    """2^-k when value is that power of two, else %g."""
    if value > 0 and math.isfinite(value):
        k = -math.log2(value)
        if k == int(k):
            return "2^{0:d}".format(int(-k))
    return "{0:g}".format(value)


def system_label(row):
    text = row["problem"]
    params = row.get("system_params") or "{}"
    if params != "{}":
        text += " " + " ".join("{0}={1}".format(k, v) for k, v in json.loads(params).items())
    return text + " " + row["precision"]


def slug(text):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")


def short_hash(values):
    return hashlib.sha1(repr(tuple(values)).encode("utf-8")).hexdigest()[:8]


def output_dir(out, key, problem):
    path = os.path.join(out, key, problem)
    os.makedirs(path, exist_ok=True)
    return path


# ------------------------------------------------------------------ output

def cell(value):
    if isinstance(value, float):
        return "nan" if math.isnan(value) else repr(value)
    if isinstance(value, list):
        return ";".join(repr(float(v)) for v in value)
    if value is None:
        return ""
    return str(value)


def write_csv(path, columns, rows):
    """rows as CSV with LF line ends; NaN written as nan, lists ';'-joined."""
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        for row in rows:
            writer.writerow([cell(row.get(c)) for c in columns])
    return path


def pyplot():
    """matplotlib.pyplot on the Agg backend."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def style(package):
    return STYLE.get(package, ("gray", "."))
