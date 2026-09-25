"""plots.py [--where "<sql>"] [--kind KIND]* [--root data] [--out plots]

plots/<key>/<kind>/<problem>_<algorithm>.png for the kinds runtime_vs_n, error_vs_runtime, interval_error_vs_runtime (the mean absolute inter-beat interval error of the traced first state, ms), trace_errored_vs_runtime (percent of traced trajectories errored by a failure code or a non-finite sample, linear axis), error_vs_dt, error_vs_tol and states (runtime and compile panels), with the points of a problem in <kind>/<problem>.csv; plots/all_cards/ holds the same figures with every key's series together, a marker set per key. Every row of the store is read, or the rows a SQL predicate over the results view matches; every kind is written, or the kinds named. A package is a colour, a stepping kind a marker (fixed, adaptive), the transfers a line style (solid, dashed with the transfer); julia_cpu is on the error_vs_dt and error_vs_tol figures only. A series is one (key, package, controller kind, transfers) and is drawn when it has two or more x values. A figure with one package or no series past three points goes under <kind>/limited_data/. A row over 10% errored trajectories is drawn with a black cross over its marker. runtime_vs_n, error_vs_runtime, interval_error_vs_runtime, trace_errored_vs_runtime and states also get <problem>_algorithms.png (a subplot per algorithm) and <algorithm>_problems.png (a subplot per problem); all but states also <problem>.png, every algorithm on one axis (a colour per algorithm, a marker per package, filled for adaptive steps).
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

GPU_PACKAGES = tuple(p for p in store_mod.PACKAGES if p != "julia_cpu")
STEPPING_FIELDS = ("dt", "atol", "rtol", "newton_atol", "newton_rtol")
# The fields a series' rows must agree on, but for the ones its axis varies; package, controller and gains name the series.
CONTEXT_FIELDS = tuple(f for f in store_mod.TRIAL_FIELDS if f not in ("package", "controller", "gains"))
ALL_CARDS = "all_cards"
LIMITED_DIR = "limited_data"
LIMITED_POINTS = 3
GRID_KINDS = ("runtime_vs_n", "error_vs_runtime", "interval_error_vs_runtime", "trace_errored_vs_runtime", "states")
# The kinds that also draw every algorithm of a problem on one axis, <problem>.png.
COMBINED_KINDS = ("runtime_vs_n", "error_vs_runtime", "interval_error_vs_runtime", "trace_errored_vs_runtime")
ERROR_LABEL = "Mean absolute error at the final time"


class Kind:
    """One figure kind: its x and y columns, the spec fields the x axis varies, the packages shown, whether transfers splits the series, the rows it takes (timed, with an error, or with a build), the axis labels and the name a grid title uses."""

    def __init__(self, name, x, y, varying, packages, by_transfers, needs, x_label, y_label, title,
                 invert_x=False, log_y=True):
        self.name, self.x, self.y, self.varying = name, x, y, varying
        self.packages, self.by_transfers, self.needs = packages, by_transfers, needs
        self.x_label, self.y_label, self.title, self.invert_x = x_label, y_label, title, invert_x
        self.log_y = log_y


KINDS = (
    Kind("runtime_vs_n", "n", "min_ms", ("n",), GPU_PACKAGES, True, "timed", "Trajectories", "Time (s)",
         "Batch time"),
    Kind("error_vs_runtime", "error", "min_ms", STEPPING_FIELDS, GPU_PACKAGES, True, "timed error",
         ERROR_LABEL, "Time (s)", "Work-precision"),
    Kind("interval_error_vs_runtime", "interval_error", "min_ms", STEPPING_FIELDS, GPU_PACKAGES, True, "timed interval",
         "Inter-beat interval error (ms)", "Time (s)", "Interval work-precision"),
    Kind("trace_errored_vs_runtime", "min_ms", "trace_errored_pct", STEPPING_FIELDS, GPU_PACKAGES, True, "timed traced",
         "Time (s)", "Traced trajectories errored (%)", "Trace failures", log_y=False),
    Kind("error_vs_dt", "dt", "error", ("dt",), store_mod.PACKAGES, False, "fixed error",
         "dt", ERROR_LABEL, "Error against step size", invert_x=True),
    Kind("error_vs_tol", "atol", "error", STEPPING_FIELDS, store_mod.PACKAGES, False, "adaptive error",
         "Tolerance", ERROR_LABEL, "Error against tolerance", invert_x=True),
    Kind("states", "states", "min_ms", ("system_params",), GPU_PACKAGES, True, "timed",
         "States", "Time (s)", "State size"),
)
BUILDS = Kind("builds", "states", "build_s", ("system_params",), GPU_PACKAGES, False, "build",
              "States", "Compile time (s)", "State size")
KIND_NAMES = tuple(k.name for k in KINDS)
CSV_COLUMNS = ("kind", "algorithm", "series", "package", "controller", "transfers", "x", "y",
               "min_ms", "build_s", "error", "interval_error", "trace_errored_pct", "errored_pct", "key") + \
    tuple(f for f in store_mod.TRIAL_FIELDS if f not in ("package", "algorithm", "controller")) + \
    ("run_id", "trial_id", "group_id", "states", "reason", "finals", "traces")


# ------------------------------------------------------------------ rows

def with_errors(rows, errs):
    """Every row, with its `error`, `interval_error` and `trace_errored_pct` against the golden (NaN without finals or traces, or without a golden) and `controller_kind`; a row over the errored limit stays and is drawn crossed out."""
    return [dict(row, error=errs.error(row), interval_error=errs.interval_error(row),
                 trace_errored_pct=errs.trace_errored_pct(row), controller_kind=shared.controller_kind(row)) for row in rows]


def one_per_trial(rows):
    """One row per (key, trial_id): the both and none rows of a trial share its finals and traces."""
    seen = {}
    for row in rows:
        ident = (row["key"], row["trial_id"])
        held = seen.get(ident)
        if held is None or _artifacts(row) > _artifacts(held):
            seen[ident] = row
    return list(seen.values())


def _artifacts(row):
    return (bool(row.get("finals")) or bool(row.get("traces")), math.isfinite(shared.number(row.get("build_s"))))


def takes(kind, row):
    """True when a row belongs on a kind: its package is shown and it has what the axes need."""
    if row["package"] not in kind.packages:
        return False
    if kind.needs == "fixed error" and row["controller"] != "fixed":
        return False
    if kind.needs == "adaptive error" and row["controller"] == "fixed":
        return False
    if "error" in kind.needs and not errors_mod.is_finite_positive(row["error"]):
        return False
    if "interval" in kind.needs and not errors_mod.is_finite_positive(row["interval_error"]):
        return False
    if "traced" in kind.needs and not math.isfinite(shared.number(row["trace_errored_pct"])):
        return False
    if "timed" in kind.needs and not shared.timed(row):
        return False
    if kind.needs == "build" and not math.isfinite(shared.number(row.get("build_s"))):
        return False
    return True


def value_of(column, row):
    """A row's value on an axis column; min_ms in seconds."""
    if column == "min_ms":
        return shared.number(row["min_ms"]) / 1000.0
    return float(row[column]) if column in ("n", "states") else shared.number(row.get(column))


def loose_first(row):
    """Sort key along a stepping sweep, loosest first: dt descending, then tolerance descending."""
    dt, tol = shared.number(row["dt"]), shared.number(row["atol"])
    return (-(dt if math.isfinite(dt) else 0.0), -(tol if math.isfinite(tol) else 0.0))


def series_key(kind, row):
    return (row["key"], row["package"], row["controller_kind"], row["transfers"] if kind.by_transfers else "")


def context(kind, row):
    return tuple(shared.cell(row[f]) for f in CONTEXT_FIELDS if f not in kind.varying)


def series_of(kind, rows):
    """{(key, package, controller kind, transfers): [(x, y, row)]} of one algorithm's rows on a kind, in axis order (sweep order on the runtime axis). A series keeps the rows of the context with the most distinct x values and needs two."""
    grouped = {}
    for row in rows:
        if not takes(kind, row):
            continue
        x = value_of(kind.x, row)
        if not math.isfinite(x):
            continue
        grouped.setdefault(series_key(kind, row), {}).setdefault(context(kind, row), []).append(
            (x, value_of(kind.y, row), row))
    out = {}
    for key, contexts in grouped.items():
        points = max(contexts.values(), key=lambda pts: len({x for x, _, _ in pts}))
        if len({x for x, _, _ in points}) < 2:
            continue
        if "min_ms" in (kind.x, kind.y) and kind.varying == STEPPING_FIELDS:
            points.sort(key=lambda p: loose_first(p[2]))
        else:
            points.sort(key=lambda p: p[0])
        out[key] = points
    return ordered(out)


def ordered(series):
    return dict(sorted(series.items(), key=lambda item: (
        item[0][0], store_mod.PACKAGES.index(item[0][1]), shared.CONTROLLER_KINDS.index(item[0][2]), item[0][3])))


def merged(figures):
    """One series dict from several series dicts of the same figure."""
    out = {}
    for series in figures:
        out.update(series)
    return ordered(out)


def series_label(kind, key, points):
    """'Cubie, 1024 fixed steps + transfer' from a series key and its first row; ' + transfer' for the both transfers, no step count where the axis sweeps dt."""
    _, package, controller, transfers = key[:4]
    parts = [shared.package_name(package),
             shared.controller_text(controller, None if "dt" in kind.varying else points[0][2])]
    return ", ".join(parts) + (" + transfer" if transfers == "both" else "")


def limited(series, builds=None):
    """True when a figure compares nothing: one package, or no series past LIMITED_POINTS points."""
    everything = dict(series)
    everything.update(builds or {})
    if len({key[1] for key in everything}) <= 1:
        return True
    return max((len(points) for points in everything.values()), default=0) <= LIMITED_POINTS


# --------------------------------------------------------------- figures

def cards_of(series):
    return sorted({key[0] for key in series})


ERROR_COLUMNS = ("error", "interval_error")


def fit_unflagged(panel, kind, series):
    """Fit an error axis to the points within the errored limit; crossed points beyond it fall off the panel."""
    for column, index, limits in ((kind.x, 0, panel.set_xlim), (kind.y, 1, panel.set_ylim)):
        if column not in ERROR_COLUMNS:
            continue
        values = [p[index] for points in series.values() for p in points
                  if shared.within_errored_limit(p[2]) and math.isfinite(p[index]) and p[index] > 0]
        if values:
            limits(min(values) / 2.0, max(values) * 2.0)


def cross_out(panel, points):
    """A black cross over every point of a series whose row is over the errored limit; True when any point is."""
    flagged = [(x, y) for x, y, row in points if not shared.within_errored_limit(row)]
    if flagged:
        panel.plot([x for x, _ in flagged], [y for _, y in flagged], **shared.CROSS_STYLE)
    return bool(flagged)


def plot_panel(panel, kind, series, encoding):
    """Every series on one panel in the figure's encoding; True when any point is crossed."""
    crossed = False
    for key, points in series.items():
        panel.plot([p[0] for p in points], [p[1] for p in points], **encoding.style(key, points))
        crossed = cross_out(panel, points) or crossed
    panel.set_xscale("log")
    if kind.log_y:
        panel.set_yscale("log")
    fit_unflagged(panel, kind, series)
    if kind.invert_x:
        panel.invert_xaxis()
    panel.set_xlabel(kind.x_label)
    panel.set_ylabel(kind.y_label)
    panel.grid(True, which="both")
    return crossed


def tagged(series):
    """The series with each key's algorithm appended, so several algorithms share one panel."""
    return {key[:4] + (shared.Encoding.algorithm(key, points),): points for key, points in series.items()}


def render_panels(path, kind, panels, title, colour_by="package", columns=None):
    """The one renderer: a subplot per (name, series[, kind]) row-major over `columns` (near-square without), an empty series left blank, one encoding legend at the right."""
    plt = shared.pyplot()
    count = len(panels)
    columns = columns or min(4, math.ceil(math.sqrt(count)))
    rows = math.ceil(count / columns)
    encoding = shared.Encoding(merged(p[1] for p in panels if p[1]), colour_by)
    probe, labels, _ = encoding.legend(plt, True)
    width = 5.0 * columns
    fig, axes = plt.subplots(rows, columns, figsize=(width + 2.6, max(3.8 * rows, 0.17 * len(labels) + 0.8)),
                             squeeze=False)
    crossed = False
    for index, spec in enumerate(panels):
        name, series = spec[:2]
        panel = axes[index // columns][index % columns]
        if not series:
            panel.set_axis_off()
            continue
        crossed = plot_panel(panel, spec[2] if len(spec) > 2 else kind, series, encoding) or crossed
        panel.set_title(name)
    for index in range(count, rows * columns):
        axes[index // columns][index % columns].set_axis_off()
    fig.suptitle(title)
    fig.tight_layout(rect=(0.0, 0.0, width / (width + 2.6), 0.96))
    handles, labels, headings = encoding.legend(plt, crossed)
    box = fig.legend(handles, labels, loc="center left", bbox_to_anchor=(width / (width + 2.6), 0.5))
    for text in box.get_texts():
        if text.get_text() in headings:
            text.set_fontweight("bold")
    fig.savefig(path)
    plt.close(fig)
    return path


def problem_title(series):
    """'Lorenz 96 (32 states), 1s integration time' from the rows of a figure; a range when the state count varies."""
    rows = [p[2] for points in series.values() for p in points]
    states = sorted({r["states"] for r in rows})
    text = "{0}".format(states[0]) if len(states) == 1 else "{0} to {1}".format(states[0], states[-1])
    return "{0}, {1:g}s integration time".format(shared.problem_name(rows[0]["problem"], text),
                                                  shared.number(rows[0]["duration"]))


def title_of(series, algorithm):
    """'Lorenz 96 problem (32 states): 1s integration time, Vern7 algorithm'."""
    problem, _, duration = problem_title(series).partition(", ")
    return "{0} problem{1}: {2}, {3} algorithm".format(
        problem.partition(" (")[0], " (" + problem.partition(" (")[2] if " (" in problem else "", duration,
        shared.algorithm_name(algorithm))


def render(path, kind, series, algorithm, builds=None):
    """One figure of a kind, with the compile panel beside it on the states kind."""
    panels = [("", series)] + ([("", builds, BUILDS)] if builds is not None else [])
    return render_panels(path, kind, panels, title_of(series or builds, algorithm), columns=len(panels))


def render_grid(path, kind, panels, title, columns=None):
    """A subplot per (name, series[, kind]), a colour per package."""
    return render_panels(path, kind, panels, title, "package", columns)


def render_combined(path, kind, panels, title):
    """Every (algorithm name, series) on one axis, a colour per algorithm."""
    return render_panels(path, kind, [("", merged(tagged(s) for _, s in panels))], title, "algorithm", 1)


def render_combined_grid(path, kind, panels, title, columns=None):
    """A subplot per (name, {algorithm: series}), a colour per algorithm."""
    return render_panels(path, kind, [(name, merged(tagged(s) for s in by.values())) for name, by in panels],
                         title, "algorithm", columns)


def csv_rows(kind, algorithm, series):
    out = []
    for key, points in series.items():
        for x, y, row in points:
            out.append(dict(row, kind=kind.name, algorithm=algorithm, series=series_label(kind, key, points), x=x, y=y,
                            controller=key[2], transfers=key[3] or row["transfers"]))
    return out


# ------------------------------------------------------ figure definitions

TRANSFER = Kind("transfer_vs_n", "n", "transfer_s", ("n",), GPU_PACKAGES, False, "timed",
                "Trajectories", "Transfer time (s)", "Transfer time")


class Pane:
    """One panel: the rows it takes ({column: value or tuple of values}), a note after its title, a transform of its series, a kind in place of the figure's."""

    def __init__(self, where, note="", transform=None, kind=None):
        self.where, self.note, self.transform, self.kind = where, note, transform, kind

    def title(self, card_in_title):
        parts = []
        if card_in_title and isinstance(self.where.get("key"), str):
            parts.append(shared.key_label(self.where["key"]))
        if isinstance(self.where.get("problem"), str):
            parts.append(shared.problem_label(self.where["problem"]))
        return ", ".join(parts) + (": " + self.note if self.note else "")


class Figure:
    """A figure as an inclusion list: its file stem, kind, panes, the rows every pane takes, the colour channel, the grid columns, a figure per key or one over every key, and a title."""

    def __init__(self, name, kind, panes, where=None, colour_by="algorithm", columns=None, per_key=True, title=""):
        self.name, self.kind, self.panes, self.where = name, kind, panes, where or {}
        self.colour_by, self.columns, self.per_key, self.title = colour_by, columns, per_key, title


def problem_panes(problems, **pane):
    """A pane per problem."""
    return [Pane(dict(problem=p), **pane) for p in problems]


def matches(row, where):
    return all(row.get(k) in v if isinstance(v, tuple) else row.get(k) == v for k, v in where.items())


def pane_series(kind, rows):
    """Every algorithm's series of the rows, algorithm-tagged, merged."""
    groups = {}
    for row in rows:
        groups.setdefault((row["problem"], row["algorithm"]), []).append(row)
    return merged(tagged(series_of(kind, members)) for members in groups.values())


def transfer_time(series):
    """With transfers minus without, at each trajectory count both have; non-positive differences dropped."""
    out = {}
    for key, points in series.items():
        if key[3] != "both":
            continue
        without = {x: y for x, y, _ in series.get(key[:3] + ("none",) + key[4:], [])}
        diff = [(x, y - without[x], row) for x, y, row in points if x in without and y > without[x]]
        if len(diff) >= 2:
            out[key[:3] + ("",) + key[4:]] = diff
    return out


def best_per_package(series, levels=9, span=100.0):
    """Per package, the curve cheapest at the most of `levels` error levels from its least error to `span` times that; ties to the lowest summed log time."""
    out = {}
    for package in {k[1] for k in series}:
        curves = {k: pts for k, pts in series.items()
                  if k[1] == package and sum(shared.within_errored_limit(r) for _, _, r in pts) >= 3}
        errors = [x for pts in curves.values() for x, _, r in pts if shared.within_errored_limit(r)]
        if not errors:
            continue
        low = min(errors)
        wins, cost = dict.fromkeys(curves, 0), dict.fromkeys(curves, 0.0)
        for i in range(levels):
            level = low * span ** (i / (levels - 1))
            times = {k: min((y for x, y, r in pts if x <= level and shared.within_errored_limit(r)), default=math.nan)
                     for k, pts in curves.items()}
            reached = {k: t for k, t in times.items() if math.isfinite(t)}
            if reached:
                wins[min(reached, key=reached.get)] += 1
            for k, t in times.items():
                cost[k] += math.log(t) if math.isfinite(t) else 50.0
        chosen = max(curves, key=lambda k: (wins[k], -cost[k]))
        out[chosen] = curves[chosen]
    return ordered(out)


def draw_figure(figure, rows, keys, out):
    """Every file of one figure: one per key, or one over every key; a figure with no series is skipped. Returns the paths written."""
    written = []
    for key in (keys if figure.per_key else [None]):
        panels, tables = [], []
        for pane in figure.panes:
            where = dict(figure.where, **pane.where)
            if key is not None:
                where["key"] = key
            elif "key" not in where:
                where["key"] = tuple(keys)
            kind = pane.kind or figure.kind
            series = pane_series(figure.kind, [r for r in rows if matches(r, where)])
            if pane.transform:
                series = pane.transform(series)
            if series:
                panels.append((pane.title(key is None), series, kind))
                tables += csv_rows(kind, "", series)
        if not panels:
            continue
        stem = figure.name + ("_" + shared.slug(key) if key else "")
        title = (shared.key_label(key) + ": " if key else "") + figure.title
        path = os.path.join(out, stem + ".png")
        render_panels(path, figure.kind, panels, title, figure.colour_by, figure.columns)
        shared.write_csv(os.path.join(out, stem + ".csv"), CSV_COLUMNS, tables)
        written.append(path)
    return written


def load_rows(store, cache=""):
    """Every row with its error, controller kind and NaN trace metrics, samples dropped; kept in and read from a pickle at `cache` when given."""
    import pickle
    if cache and os.path.isfile(cache):
        with open(cache, "rb") as handle:
            rows = pickle.load(handle)
    else:
        errs = errors_mod.Errors(store)
        rows = []
        for row in store.rows():
            row = {k: v for k, v in row.items() if k != "samples_ms"}
            try:
                row["error"] = errs.error(row)
            except ValueError:
                row["error"] = math.nan
            errs._finals.clear()
            rows.append(row)
        if cache:
            with open(cache, "wb") as handle:
                pickle.dump(rows, handle)
    return [dict(r, interval_error=math.nan, trace_errored_pct=math.nan, controller_kind=shared.controller_kind(r))
            for r in rows]


def kind_named(name):
    return next(k for k in KINDS if k.name == name)


def write_tree(out, card, figures, tables):
    """The figures, CSVs and grids of one output tree <out>/<card>/ from {(problem, algorithm, kind): (series, builds)} and {(kind, problem): rows}; returns the paths written."""
    written = []
    grids = {}
    for (problem, algorithm, name), (series, builds) in sorted(figures.items()):
        if series and name in GRID_KINDS:
            grids.setdefault(name, {})[(problem, algorithm)] = series
        directory = os.path.join(out, card, name)
        if limited(series, builds):
            directory = os.path.join(directory, LIMITED_DIR)
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, "{0}_{1}.png".format(problem, shared.slug(algorithm)))
        written.append(render(path, kind_named(name), series, algorithm, builds))
    for (name, problem), table in sorted(tables.items()):
        os.makedirs(os.path.join(out, card, name), exist_ok=True)
        written.append(shared.write_csv(os.path.join(out, card, name, problem + ".csv"), CSV_COLUMNS, table))
    for name, panels_by in sorted(grids.items()):
        kind = kind_named(name)
        directory = os.path.join(out, card, name)
        for problem in sorted({p for p, _ in panels_by}):
            panels = [(shared.algorithm_name(a), s) for (p, a), s in sorted(panels_by.items()) if p == problem]
            written.append(render_grid(os.path.join(directory, problem + "_algorithms.png"), kind, panels,
                                       "{0}: {1}, all algorithms".format(
                                           problem_title(merged(s for _, s in panels)), kind.title)))
            if name in COMBINED_KINDS:
                written.append(render_combined(os.path.join(directory, problem + ".png"), kind, panels,
                                               "{0}: {1}".format(problem_title(merged(s for _, s in panels)),
                                                                 kind.title)))
        for algorithm in sorted({a for _, a in panels_by}):
            panels = [(problem_title(s), s) for (p, a), s in sorted(panels_by.items()) if a == algorithm]
            written.append(render_grid(os.path.join(directory, shared.slug(algorithm) + "_problems.png"), kind,
                                       panels, "{0} algorithm: {1}, all problems".format(
                                           shared.algorithm_name(algorithm), kind.title)))
    return written


def run(store, where="", kinds=KIND_NAMES, out=shared.PLOTS_DIR):
    """Write every figure and CSV of the named kinds from the rows a predicate matches (every row without one) under <out>/<key>/<kind>/ and, with several keys, <out>/all_cards/<kind>/; returns the paths written."""
    chosen = [k for k in KINDS if k.name in kinds]
    rows = with_errors(shared.select_rows(store, where), errors_mod.Errors(store))
    by_figure = {}
    for row in rows:
        by_figure.setdefault((row["key"], row["problem"], row["algorithm"]), []).append(row)
    figures, tables = {}, {}
    for (key, problem, algorithm), members in sorted(by_figure.items()):
        per_trial = one_per_trial(members)
        for kind in chosen:
            series = series_of(kind, members if kind.by_transfers else per_trial)
            builds = series_of(BUILDS, per_trial) if kind.name == "states" else None
            table = csv_rows(kind, algorithm, series) + (csv_rows(BUILDS, algorithm, builds) if builds else [])
            if table:
                tables.setdefault((key, kind.name, problem), []).extend(table)
            if series or builds:
                figures[(key, problem, algorithm, kind.name)] = (series, builds)
    written = []
    cards = sorted({key for key, _, _, _ in figures} | {key for key, _, _ in tables})
    for card in cards:
        written.extend(write_tree(
            out, card, {(p, a, k): v for (key, p, a, k), v in figures.items() if key == card},
            {(k, p): t for (key, k, p), t in tables.items() if key == card}))
    if len(cards) > 1:
        everything, all_tables = {}, {}
        for (key, problem, algorithm, name), (series, builds) in figures.items():
            held = everything.setdefault((problem, algorithm, name), [[], []])
            held[0].append(series)
            if builds:
                held[1].append(builds)
        for (key, name, problem), table in tables.items():
            all_tables.setdefault((name, problem), []).extend(table)
        written.extend(write_tree(out, ALL_CARDS, {
            k: (merged(s), merged(b) if b else None) for k, (s, b) in everything.items()}, all_tables))
    return written


def main(argv=None):
    args = shared.parser(__doc__, KIND_NAMES).parse_args(argv)
    shared.pull_store(args)
    store = shared.AnalysisStore(args.root)
    written = run(store, args.where, args.kind or KIND_NAMES, args.out)
    for path in written:
        print(path)
    if not written:
        print("no rows selected form a curve")
    return 0


if __name__ == "__main__":
    sys.exit(main())
