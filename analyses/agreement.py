"""agreement.py (--set NAME)* | --where "<sql>" [--root data] [--out plots]

Per stepping, each package's error against the golden and the difference between every package pair, from finals paired by grid value. Per (key, problem) under plots/<key>/<problem>/: agreement.csv (one row per trial), agreement_pairs.csv (one row per pair) and one figure per sweep against the swept dt/tolerance. Rows with errored_pct above 10 are dropped.
With --set, the store is first checked against the sets' canonical trials under every key: what it lacks is printed and written to plots/<key>/incomplete.csv, and the exit code is 1 while anything is lacking.
"""

import math
import os
import sys
from itertools import combinations

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import shared  # noqa: E402

shared.under_suite_python()

import errors as errors_mod  # noqa: E402
import store as store_mod  # noqa: E402

# The stepping value a sweep varies; every other group field names the sweep.
SWEPT_FIELDS = ("dt", "atol", "rtol", "newton_atol", "newton_rtol")
SWEEP_FIELDS = tuple(f for f in store_mod.GROUP_FIELDS if f not in SWEPT_FIELDS)
ROW_COLUMNS = ("group_id", "trial_id", "run_id", "package", "n", "states", "error", "errored_pct") + \
    tuple(f for f in store_mod.TRIAL_FIELDS if f not in ("package", "n")) + ("key", "finals")
PAIR_COLUMNS = ("group_id",) + store_mod.GROUP_FIELDS + ("package_a", "n_a", "package_b", "n_b",
                                                          "difference")


def loose_first(row):
    """Sort key from the loosest stepping to the tightest: dt descending, then tolerance descending."""
    dt, tol = shared.number(row["dt"]), shared.number(row["atol"])
    return (-(dt if math.isfinite(dt) else 0.0), -(tol if math.isfinite(tol) else 0.0))


def with_finals(rows):
    """The usable rows that keep finals, one per (key, trial_id): the both and none rows share a finals file."""
    seen = set()
    out = []
    for row in shared.usable(rows):
        if not row.get("finals"):
            continue
        ident = (row["key"], row["trial_id"])
        if ident in seen:
            continue
        seen.add(ident)
        out.append(row)
    return out


def swept_value(row):
    """The dt of a fixed stepping, the tolerance otherwise."""
    return shared.number(row["dt"]) if row["controller"] == "fixed" else shared.number(row["atol"])


def sweep_of(row):
    """The sweep fields of a row as text, so NaN equals NaN."""
    return tuple(shared.cell(row[f]) for f in SWEEP_FIELDS)


def pair_row(group, a, b, difference):
    fields = {c: group[c] for c in PAIR_COLUMNS if c in group}
    fields.update(package_a=a["package"], n_a=a["n"], package_b=b["package"], n_b=b["n"],
                  difference=difference)
    return fields


def analyse(rows, errs):
    """{(key, problem): (trial rows with their error, pair rows)} over the rows with finals."""
    by_problem = {}
    groups = {}
    for row in sorted(with_finals(rows), key=lambda r: (r["key"], r["problem"], r["group_id"],
                                                        r["package"], r["n"])):
        groups.setdefault((row["key"], row["problem"], row["group_id"]), []).append(row)
    for (key, problem, _), members in groups.items():
        trial_rows, pair_rows = by_problem.setdefault((key, problem), ([], []))
        scored = [dict(row, error=errs.error(row)) for row in members]
        trial_rows.extend(scored)
        for a, b in combinations(scored, 2):
            if a["package"] == b["package"]:
                continue
            pair_rows.append(pair_row(a, a, b, errs.compare(a, b)))
    # Sweep by sweep, loosest stepping first, then by package.
    for trial_rows, pair_rows in by_problem.values():
        trial_rows.sort(key=lambda r: (sweep_of(r), loose_first(r), r["package"], r["n"]))
        pair_rows.sort(key=lambda p: (sweep_of(p), loose_first(p), p["package_a"], p["package_b"],
                                      p["n_a"], p["n_b"]))
    return by_problem


def sweeps(trial_rows, pair_rows):
    """{sweep: ({package: [(x, error)]}, {(package_a, package_b): [(x, difference)]})} of one problem's rows, points in loose-to-tight order."""
    out = {}
    for row in sorted(trial_rows, key=loose_first):
        series, _ = out.setdefault(sweep_of(row), ({}, {}))
        if errors_mod.is_finite_positive(row["error"]):
            series.setdefault(row["package"], []).append((swept_value(row), row["error"]))
    for pair in sorted(pair_rows, key=loose_first):
        _, diffs = out.setdefault(sweep_of(pair), ({}, {}))
        if errors_mod.is_finite_positive(pair["difference"]):
            diffs.setdefault((pair["package_a"], pair["package_b"]), []).append(
                (swept_value(pair), pair["difference"]))
    return out


def render(path, sweep, series, diffs, key):
    """Two panels against the swept value: error per package, difference per package pair."""
    plt = shared.pyplot()
    first = dict(zip(SWEEP_FIELDS, sweep))
    fixed = first["controller"] == "fixed"
    x_label = "dt" if fixed else "tolerance (atol = rtol)"
    fig, (left, right) = plt.subplots(1, 2, figsize=(15.0, 5.0))
    for package in store_mod.PACKAGES:
        if package not in series:
            continue
        colour, marker = shared.style(package)
        left.plot([p[0] for p in series[package]], [p[1] for p in series[package]], label=package,
                  color=colour, marker=marker, linewidth=1.5)
    for (a, b), points in sorted(diffs.items()):
        _, marker = shared.style(b)
        colour, _ = shared.style(a)
        right.plot([p[0] for p in points], [p[1] for p in points], label="{0} vs {1}".format(a, b),
                   color=colour, marker=marker, linewidth=1.5)
    for panel, y_label in ((left, "error against the golden"), (right, "difference between packages")):
        panel.set_xscale("log")
        panel.set_yscale("log")
        if fixed:
            panel.invert_xaxis()
        panel.set_xlabel(x_label)
        panel.set_ylabel(y_label)
        panel.grid(True, which="both", alpha=0.3)
        if panel.has_data():
            panel.legend(fontsize=8)
    gains = "" if first["gains"] in ("", "{}") else " gains=" + first["gains"]
    fig.suptitle("{0} | {1} {2}{3} | {4}".format(shared.system_label(first), first["algorithm"],
                                                first["controller"], gains, key), fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def run(store, set_names=(), where="", out=shared.PLOTS_DIR):
    """Write the CSVs and figures of every (key, problem); returns the paths written."""
    rows = shared.select_rows(store, set_names, where)
    errs = errors_mod.Errors(store)
    written = []
    for (key, problem), (trial_rows, pair_rows) in sorted(analyse(rows, errs).items()):
        directory = shared.output_dir(out, key, problem)
        written.append(shared.write_csv(os.path.join(directory, "agreement.csv"), ROW_COLUMNS, trial_rows))
        written.append(shared.write_csv(os.path.join(directory, "agreement_pairs.csv"), PAIR_COLUMNS,
                                        pair_rows))
        for sweep, (series, diffs) in sorted(sweeps(trial_rows, pair_rows).items(), key=lambda item: repr(item[0])):
            if not series and not diffs:
                continue
            first = dict(zip(SWEEP_FIELDS, sweep))
            stem = "agreement_{0}_{1}_{2}".format(shared.slug(first["algorithm"]),
                                                  shared.slug(first["controller"]), shared.short_hash(sweep))
            written.append(render(os.path.join(directory, stem + ".png"), sweep, series, diffs, key))
    return written


def main(argv=None):
    args = shared.parser(__doc__).parse_args(argv)
    shared.check_selection(args)
    shared.pull_store(args)
    store = store_mod.Store(args.root)
    lacking = shared.report_incomplete(store, args.set, args.out) if args.set else 0
    written = run(store, args.set, args.where, args.out)
    for path in written:
        print(path)
    if not written:
        print("no rows with finals selected")
    return 1 if lacking else 0


if __name__ == "__main__":
    sys.exit(main())
