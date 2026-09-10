"""timing.py --x n|error|states (--set NAME)* | --where "<sql>" [--root data] [--out plots]

min_ms against the axis: one figure and CSV per (key, problem, transfers, stepping) under plots/<key>/<problem>/, one series per package. The figure lets one field vary: n, the states of system_params, or the swept dt/tolerance with each row scored against the golden. --x states adds a build_s panel. Dropped: errored_pct above 10, no time, figures with one axis value.
With --set, the store is first checked against the sets' canonical trials under every key: what it lacks is printed and written to plots/<key>/incomplete.csv, and the exit code is 1 while anything is lacking.
"""

import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import shared  # noqa: E402

shared.under_suite_python()

import errors as errors_mod  # noqa: E402
import store as store_mod  # noqa: E402
from trials import difficulty  # noqa: E402

AXES = ("n", "error", "states")
# The columns that vary along each axis and so stay out of the figure identity.
VARYING = {"n": ("n",), "states": ("system_params",),
           "error": ("dt", "atol", "rtol", "newton_atol", "newton_rtol")}
IDENTITY_FIELDS = store_mod.GROUP_FIELDS + store_mod.ENSEMBLE_FIELDS
X_LABEL = {"n": "n (trajectories)", "states": "states",
           "error": "error (RMS of the final state against the golden)"}
CSV_COLUMNS = ("x", "min_ms", "build_s", "error", "errored_pct", "package", "transfers", "key") + \
    tuple(f for f in store_mod.TRIAL_FIELDS if f != "package") + \
    ("run_id", "trial_id", "group_id", "states", "reason", "finals")


def identity(row, axis):
    """The figure identity of a row on an axis: every group and ensemble field but the varying ones, as text so NaN equals NaN."""
    return tuple(shared.cell(row[f]) for f in IDENTITY_FIELDS if f not in VARYING[axis])


def x_value(row, axis, errs):
    if axis == "n":
        return float(row["n"])
    if axis == "states":
        return float(row["states"])
    return errs.error(row)


def timed(row):
    return math.isfinite(shared.number(row.get("min_ms")))


def collect(rows, axis, errs):
    """Two dicts keyed by (key, problem, transfers, identity) over the usable rows: {package: [(x, row)]} of the timed rows (on the error axis those with a finite positive error), and on the states axis {package: [(states, build_s)]} of the rows with a measured cold build."""
    figures, builds = {}, {}
    for row in shared.usable(rows):
        ident = (row["key"], row["problem"], row["transfers"], identity(row, axis))
        if axis == "states":
            build = shared.number(row.get("build_s"))
            if math.isfinite(build):
                builds.setdefault(ident, {}).setdefault(row["package"], []).append(
                    (float(row["states"]), build))
        if not timed(row):
            continue
        x = x_value(row, axis, errs)
        if axis == "error" and not errors_mod.is_finite_positive(x):
            continue
        row = dict(row, x=x, error=x if axis == "error" else shared.NAN)
        figures.setdefault(ident, {}).setdefault(row["package"], []).append((x, row))
    for figure in figures.values():
        for points in figure.values():
            points.sort(key=lambda p: difficulty(p[1]))
    for figure in builds.values():
        for points in figure.values():
            points.sort()
    return figures, builds


def is_curve(series):
    """True when the series span at least two distinct x values; a lone point is not a curve."""
    return len({x for points in series.values() for x, _ in points}) >= 2


def title_of(first, axis):
    if axis == "n":
        stepping = shared.stepping_label(first)
        system = shared.system_label(first)
    elif axis == "states":
        stepping = shared.stepping_label(first)
        system = "{0} {1}".format(first["problem"], first["precision"])
    else:
        stepping = "{0} {1} n={2}".format(first["algorithm"], first["controller"], first["n"])
        system = shared.system_label(first)
    return "{0} | {1} | transfers={2} | {3}".format(system, stepping, first["transfers"], first["key"])


def stem_of(first, axis, ident):
    return "timing_{0}_{1}_{2}_{3}_{4}".format(
        axis, first["transfers"], shared.slug(first["algorithm"]), shared.slug(first["controller"]),
        shared.short_hash(ident))


def render(path, series, axis, title, builds=None):
    """One figure: min_ms against the axis per package, log-log, with a build_s panel when builds are given."""
    plt = shared.pyplot()
    panels = 2 if builds is not None else 1
    fig, axes = plt.subplots(1, panels, figsize=(7.5 * panels, 5.0), squeeze=False)
    main = axes[0][0]
    for package in store_mod.PACKAGES:
        if package not in series:
            continue
        colour, marker = shared.style(package)
        points = series[package]
        main.plot([p[0] for p in points], [p[1]["min_ms"] for p in points], label=package,
                  color=colour, marker=marker, linewidth=1.5)
    main.set_xscale("log")
    main.set_yscale("log")
    main.set_xlabel(X_LABEL[axis])
    main.set_ylabel("time (ms)")
    main.grid(True, which="both", alpha=0.3)
    main.legend(fontsize=8)
    if builds is not None:
        panel = axes[0][1]
        for package in store_mod.PACKAGES:
            if package not in builds:
                continue
            colour, marker = shared.style(package)
            panel.plot([p[0] for p in builds[package]], [p[1] for p in builds[package]], label=package,
                       color=colour, marker=marker, linewidth=1.5)
        panel.set_xscale("log")
        panel.set_yscale("log")
        panel.set_xlabel(X_LABEL[axis])
        panel.set_ylabel("cold build (s)")
        panel.grid(True, which="both", alpha=0.3)
        if builds:
            panel.legend(fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def run(store, axis, set_names=(), where="", out=shared.PLOTS_DIR):
    """Write every figure and CSV; returns (the paths written, the count of lone-point figures skipped)."""
    if axis not in AXES:
        raise ValueError("--x takes one of " + ", ".join(AXES))
    rows = shared.select_rows(store, set_names, where)
    errs = errors_mod.Errors(store)
    figures, builds = collect(rows, axis, errs)
    written = []
    skipped = 0
    for (key, problem, transfers, ident), series in sorted(figures.items()):
        if not is_curve(series):
            skipped += 1
            continue
        first = next(iter(series.values()))[0][1]
        directory = shared.output_dir(out, key, problem)
        stem = stem_of(first, axis, ident)
        panel = builds.get((key, problem, transfers, ident), {}) if axis == "states" else None
        written.append(render(os.path.join(directory, stem + ".png"), series, axis,
                              title_of(first, axis), panel))
        flat = [row for points in series.values() for _, row in points]
        written.append(shared.write_csv(os.path.join(directory, stem + ".csv"), CSV_COLUMNS, flat))
    return written, skipped


def main(argv=None):
    p = shared.parser(__doc__)
    p.add_argument("--x", choices=AXES, required=True, help="the axis")
    args = p.parse_args(argv)
    shared.check_selection(args)
    shared.pull_store(args)
    store = store_mod.Store(args.root)
    lacking = shared.report_incomplete(store, args.set, args.out) if args.set else 0
    written, skipped = run(store, args.x, args.set, args.where, args.out)
    for path in written:
        print(path)
    if not written:
        print("no timed rows selected")
    if skipped:
        print("{0} figure(s) with a single {1} value skipped".format(skipped, args.x))
    return 1 if lacking else 0


if __name__ == "__main__":
    sys.exit(main())
