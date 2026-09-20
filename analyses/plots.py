"""plots.py (--set NAME)* | --where "<sql>" [--root data] [--out plots]

Per (key, problem, algorithm) under plots/<key>/<problem>/: runtime_vs_n, error_vs_runtime, error_vs_dt, error_vs_tol and states (runtime and cold build panels) figures, plus one <kind>.csv per problem. A package is a colour, a controller a marker, the transfers a line style; julia_cpu is on the error_vs_dt and error_vs_tol figures only. A series is one (package, controller, transfers) whose rows differ only along the axis; a lone point is dropped and a second curve under one series is labelled by what differs. With --set, what the store lacks of the sets goes to plots/<key>/incomplete.csv and the exit code is 1.
"""

import json
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

GPU_PACKAGES = tuple(p for p in store_mod.PACKAGES if p != "julia_cpu")
STEPPING_FIELDS = ("dt", "atol", "rtol", "newton_atol", "newton_rtol")
# The fields a series' rows must agree on, but for the ones its axis varies; package, controller and gains name the series.
CONTEXT_FIELDS = tuple(f for f in store_mod.TRIAL_FIELDS if f not in ("package", "controller", "gains"))


class Kind:
    """One figure kind: its x and y columns, the spec fields the x axis varies, the packages shown, whether transfers splits the series, the rows it takes (timed, with an error, or with a build), and the axis labels."""

    def __init__(self, name, x, y, varying, packages, by_transfers, needs, x_label, y_label, invert_x=False):
        self.name, self.x, self.y, self.varying = name, x, y, varying
        self.packages, self.by_transfers, self.needs = packages, by_transfers, needs
        self.x_label, self.y_label, self.invert_x = x_label, y_label, invert_x


KINDS = (
    Kind("runtime_vs_n", "n", "min_ms", ("n",), GPU_PACKAGES, True, "timed",
         "n (trajectories)", "time (ms)"),
    Kind("error_vs_runtime", "min_ms", "error", STEPPING_FIELDS, GPU_PACKAGES, True, "timed error",
         "time (ms)", "error (RMS of the final state against the golden)"),
    Kind("error_vs_dt", "dt", "error", ("dt",), store_mod.PACKAGES, False, "fixed error",
         "dt", "error (RMS of the final state against the golden)", invert_x=True),
    Kind("error_vs_tol", "atol", "error", STEPPING_FIELDS, store_mod.PACKAGES, False, "adaptive error",
         "tolerance (atol = rtol)", "error (RMS of the final state against the golden)", invert_x=True),
    Kind("states", "states", "min_ms", ("system_params",), GPU_PACKAGES, True, "timed",
         "states", "time (ms)"),
)
BUILDS = Kind("builds", "states", "build_s", ("system_params",), GPU_PACKAGES, False, "build",
              "states", "cold build (s)")
KIND_NAMES = tuple(k.name for k in KINDS)
CSV_COLUMNS = ("kind", "algorithm", "series", "package", "controller", "transfers", "x", "y",
               "min_ms", "build_s", "error", "errored_pct", "key") + \
    tuple(f for f in store_mod.TRIAL_FIELDS if f not in ("package", "algorithm", "controller")) + \
    ("run_id", "trial_id", "group_id", "states", "reason", "finals")


# ------------------------------------------------------------------ rows

def with_errors(rows, errs):
    """The usable rows, each with its `error` against the golden (NaN without finals or golden) and `controller_label`."""
    cache = {}
    out = []
    for row in shared.usable(rows):
        out.append(dict(row, error=errs.error(row), controller_label=shared.controller_label(row, cache)))
    return out


def one_per_trial(rows):
    """One row per (key, trial_id): the both and none rows of a trial share its finals."""
    seen = {}
    for row in rows:
        ident = (row["key"], row["trial_id"])
        held = seen.get(ident)
        if held is None or _artifacts(row) > _artifacts(held):
            seen[ident] = row
    return list(seen.values())


def _artifacts(row):
    return (bool(row.get("finals")), math.isfinite(shared.number(row.get("build_s"))))


def takes(kind, row):
    """True when a row belongs on a kind: its package is shown, and it has what the axes need."""
    if row["package"] not in kind.packages:
        return False
    if kind.needs == "fixed error" and row["controller"] != "fixed":
        return False
    if kind.needs == "adaptive error" and row["controller"] == "fixed":
        return False
    if "error" in kind.needs and not errors_mod.is_finite_positive(row["error"]):
        return False
    if "timed" in kind.needs and not shared.timed(row):
        return False
    if kind.needs == "build" and not math.isfinite(shared.number(row.get("build_s"))):
        return False
    return True


def x_of(kind, row):
    return float(row[kind.x]) if kind.x != "min_ms" else shared.number(row["min_ms"])


def y_of(kind, row):
    return shared.number(row.get(kind.y))


def loose_first(row):
    """Sort key along a stepping sweep, loosest first: dt descending, then tolerance descending."""
    dt, tol = shared.number(row["dt"]), shared.number(row["atol"])
    return (-(dt if math.isfinite(dt) else 0.0), -(tol if math.isfinite(tol) else 0.0))


def series_key(kind, row):
    return (row["package"], row["controller_label"], row["transfers"] if kind.by_transfers else "")


def context(kind, row):
    return tuple(shared.cell(row[f]) for f in CONTEXT_FIELDS if f not in kind.varying)


def field_label(field, value):
    """'dt=2^-10', 'tol=1e-05', 'states=32' or 'field=value' from a field and its cell text."""
    if field == "dt":
        return "dt=" + shared.dyadic(float(value))
    if field == "atol":
        return "tol={0:g}".format(float(value))
    if field == "system_params":
        return " ".join("{0}={1}".format(k, v) for k, v in json.loads(value or "{}").items()) or "system_params={}"
    return "{0}={1}".format(field, value)


def context_suffix(kind, contexts):
    """{context: ' field=value ...'} naming what tells the surviving contexts of one series apart; '' with one context."""
    if len(contexts) < 2:
        return {c: "" for c in contexts}
    fields = [f for f in CONTEXT_FIELDS if f not in kind.varying]
    differing = [i for i, f in enumerate(fields) if len({c[i] for c in contexts}) > 1]
    # The tolerance names rtol and the Newton tolerances that a set derives from it.
    if "atol" in fields and fields.index("atol") in differing:
        differing = [i for i in differing if fields[i] not in ("rtol", "newton_atol", "newton_rtol")]
    return {c: " " + " ".join(field_label(fields[i], c[i]) for i in differing) for c in contexts}


def series_of(kind, rows):
    """{(package, controller, transfers, suffix): [points]} of one algorithm's rows on a kind; a point is (x, y, row). A series holds one context; a context with fewer than two distinct x values is dropped."""
    grouped = {}
    for row in rows:
        if not takes(kind, row):
            continue
        x = x_of(kind, row)
        if not math.isfinite(x):
            continue
        grouped.setdefault((series_key(kind, row), context(kind, row)), []).append((x, y_of(kind, row), row))
    curves = {k: v for k, v in grouped.items() if len({x for x, _, _ in v}) >= 2}
    by_key = {}
    for (key, ctx), points in curves.items():
        by_key.setdefault(key, {})[ctx] = points
    out = {}
    for key, contexts in by_key.items():
        suffixes = context_suffix(kind, list(contexts))
        for ctx, points in contexts.items():
            if kind.x == "min_ms":
                points.sort(key=lambda p: loose_first(p[2]))
            else:
                points.sort(key=lambda p: p[0])
            out[key + (suffixes[ctx],)] = points
    return dict(sorted(out.items(), key=lambda item: (
        store_mod.PACKAGES.index(item[0][0]), controller_order(item[0][1]), item[0][2], item[0][3])))


def controller_order(controller):
    """Fixed, default, pi, pi matched, gustafsson, then any other by name."""
    known = list(shared.MARKERS)
    return (known.index(controller), "") if controller in known else (len(known), controller)


def series_label(key):
    package, controller, transfers, suffix = key
    return " ".join(p for p in (package, controller, transfers) if p) + suffix


def shared_context(kind, series):
    """'n=131072 states=32 dt=2^-10 tol=1e-05': the displayed fields every point of a figure agrees on."""
    parts = []
    for field in ("n", "system_params", "dt", "atol"):
        if field == kind.x or field in kind.varying:
            continue
        values = {shared.cell(p[2][field]) for points in series.values() for p in points}
        values.discard("nan")
        if len(values) == 1:
            value = values.pop()
            if field == "system_params" and value == "{}":
                continue
            parts.append(field_label(field, value))
    return " ".join(parts)


# --------------------------------------------------------------- figures

def draw(panel, kind, series):
    for key, points in series.items():
        package, controller, transfers, _ = key
        panel.plot([p[0] for p in points], [p[1] for p in points], label=series_label(key),
                   color=shared.colour(package), marker=shared.marker(controller),
                   linestyle=shared.line(transfers) if transfers else "-", linewidth=1.5, markersize=6)
    panel.set_xscale("log")
    panel.set_yscale("log")
    if kind.invert_x:
        panel.invert_xaxis()
    panel.set_xlabel(kind.x_label)
    panel.set_ylabel(kind.y_label)
    panel.grid(True, which="both", alpha=0.3)
    if series:
        panel.legend(fontsize=7)


def title_of(first, algorithm, key, contexts):
    text = "{0} {1} | {2} | {3}".format(first["problem"], first["precision"], algorithm, key)
    shown = " | ".join(c for c in contexts if c)
    return text + ("\n" + shown if shown else "")


def render(path, kind, series, algorithm, key, builds=None):
    """One figure of a kind, with the builds panel beside it on the states kind."""
    plt = shared.pyplot()
    panels = 2 if builds is not None else 1
    fig, axes = plt.subplots(1, panels, figsize=(7.5 * panels, 5.0), squeeze=False)
    draw(axes[0][0], kind, series)
    contexts = [shared_context(kind, series)]
    if builds is not None:
        draw(axes[0][1], BUILDS, builds)
        contexts.append(shared_context(BUILDS, builds))
    first = next(iter((series or builds).values()))[0][2]
    fig.suptitle(title_of(first, algorithm, key, dict.fromkeys(contexts)), fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def csv_rows(kind, algorithm, series):
    out = []
    for key, points in series.items():
        for x, y, row in points:
            out.append(dict(row, kind=kind.name, algorithm=algorithm, series=series_label(key), x=x, y=y,
                            controller=key[1], transfers=key[2] or row["transfers"]))
    return out


def run(store, set_names=(), where="", out=shared.PLOTS_DIR):
    """Write every figure and CSV of the selected rows; returns the paths written."""
    rows = with_errors(shared.select_rows(store, set_names, where), errors_mod.Errors(store))
    by_problem = {}
    for row in rows:
        by_problem.setdefault((row["key"], row["problem"]), {}).setdefault(row["algorithm"], []).append(row)
    written = []
    for (key, problem), algorithms in sorted(by_problem.items()):
        directory = os.path.join(out, key, problem)
        os.makedirs(directory, exist_ok=True)
        tables = {kind.name: [] for kind in KINDS}
        tables[BUILDS.name] = []
        for algorithm, members in sorted(algorithms.items()):
            per_trial = one_per_trial(members)
            for kind in KINDS:
                series = series_of(kind, per_trial if not kind.by_transfers else members)
                builds = series_of(BUILDS, one_per_trial(members)) if kind.name == "states" else None
                tables[kind.name].extend(csv_rows(kind, algorithm, series))
                if builds:
                    tables[BUILDS.name].extend(csv_rows(BUILDS, algorithm, builds))
                if not series and not builds:
                    continue
                path = os.path.join(directory, "{0}_{1}.png".format(kind.name, shared.slug(algorithm)))
                written.append(render(path, kind, series, algorithm, key, builds if kind.name == "states" else None))
        for name, table in tables.items():
            if table:
                written.append(shared.write_csv(os.path.join(directory, name + ".csv"), CSV_COLUMNS, table))
    return written


def main(argv=None):
    args = shared.parser(__doc__).parse_args(argv)
    shared.check_selection(args)
    shared.pull_store(args)
    store = shared.AnalysisStore(args.root)
    lacking = shared.report_incomplete(store, args.set, args.out) if args.set else 0
    written = run(store, args.set, args.where, args.out)
    for path in written:
        print(path)
    if not written:
        print("no rows selected form a curve")
    return 1 if lacking else 0


if __name__ == "__main__":
    sys.exit(main())
