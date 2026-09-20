"""plots.py (--set NAME)* | --where "<sql>" [--root data] [--out plots]

plots/<key>/<kind>/<problem>_<algorithm>.png for the kinds runtime_vs_n, error_vs_runtime, error_vs_dt, error_vs_tol and states (runtime and compile panels), with the points of a problem in <kind>/<problem>.csv; plots/all_cards/ holds the same figures with every key's series together, a marker set per key. A package is a colour, a controller kind a marker (fixed, adaptive, matched, Gustafsson), the transfers a line style (solid, dashed with the transfer); cubie's default controller is not drawn; julia_cpu is on the error_vs_dt and error_vs_tol figures only. A series is one (key, package, controller kind, transfers) and is drawn when it has two or more x values. A figure with one package family (the cubie backends are one) or no series past three points goes under <kind>/limited_data/. runtime_vs_n, error_vs_runtime and states also get <problem>_algorithms.png (a subplot per algorithm) and <algorithm>_problems.png (a subplot per problem). With --set, what the store lacks of the sets goes to plots/<key>/incomplete.csv and the exit code is 1.
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
GRID_KINDS = ("runtime_vs_n", "error_vs_runtime", "states")
ERROR_LABEL = "RMS error (final state)"


class Kind:
    """One figure kind: its x and y columns, the spec fields the x axis varies, the packages shown, whether transfers splits the series, the rows it takes (timed, with an error, or with a build), the axis labels and the name a grid title uses."""

    def __init__(self, name, x, y, varying, packages, by_transfers, needs, x_label, y_label, title,
                 invert_x=False):
        self.name, self.x, self.y, self.varying = name, x, y, varying
        self.packages, self.by_transfers, self.needs = packages, by_transfers, needs
        self.x_label, self.y_label, self.title, self.invert_x = x_label, y_label, title, invert_x


KINDS = (
    Kind("runtime_vs_n", "n", "min_ms", ("n",), GPU_PACKAGES, True, "timed", "Trajectories", "Time (s)",
         "Batch time"),
    Kind("error_vs_runtime", "min_ms", "error", STEPPING_FIELDS, GPU_PACKAGES, True, "timed error",
         "Time (s)", ERROR_LABEL, "Work-precision"),
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
               "min_ms", "build_s", "error", "errored_pct", "key") + \
    tuple(f for f in store_mod.TRIAL_FIELDS if f not in ("package", "algorithm", "controller")) + \
    ("run_id", "trial_id", "group_id", "states", "reason", "finals")


# ------------------------------------------------------------------ rows

def with_errors(rows, errs):
    """The usable rows, each with its `error` against the golden (NaN without finals or golden) and `controller_kind`."""
    cache = {}
    out = []
    for row in shared.usable(rows):
        out.append(dict(row, error=errs.error(row), controller_kind=shared.controller_kind(row, cache)))
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
    """True when a row belongs on a kind: its package is shown, it is not cubie's default controller, and it has what the axes need."""
    if row["package"] not in kind.packages or row["controller_kind"] is None:
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
        if kind.x == "min_ms":
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
    """'Cubie (MLIR), Fixed-step dt=0.000977 + transfer' from a series key and its first row; ' + transfer' for the both transfers, no dt on the dt axis."""
    _, package, controller, transfers = key
    parts = [shared.package_name(package), shared.controller_text(controller, None if kind.x == "dt" else points[0][2])]
    return ", ".join(parts) + (" + transfer" if transfers == "both" else "")


def family(package):
    """The package family a comparison counts: the two cubie backends are one."""
    return "cubie" if package in ("cubie", "cubie_mlir") else package


def limited(series, builds=None):
    """True when a figure compares nothing: one package family, or no series past LIMITED_POINTS points."""
    everything = dict(series)
    everything.update(builds or {})
    if len({family(key[1]) for key in everything}) <= 1:
        return True
    return max((len(points) for points in everything.values()), default=0) <= LIMITED_POINTS


# --------------------------------------------------------------- figures

def cards_of(series):
    return sorted({key[0] for key in series})


def draw(panel, kind, series, cards):
    """Plot every series on a panel; returns [(card, label, handle)] with a heading per card, for a legend."""
    entries = []
    by_card = {}
    for key, points in series.items():
        by_card.setdefault(key[0], []).append((key, points))
    for card in cards:
        if card not in by_card:
            continue
        entries.append((card, shared.key_label(card), panel.plot([], [], linestyle="none")[0]))
        for key, points in by_card[card]:
            _, package, controller, transfers = key
            line = panel.plot([p[0] for p in points], [p[1] for p in points], label=series_label(kind, key, points),
                              color=shared.colour(package), marker=shared.marker(controller, cards.index(card)),
                              linestyle=shared.line(transfers) if transfers else "-", linewidth=1.5, markersize=6,
                              markeredgecolor="black", markeredgewidth=0.5)[0]
            entries.append((card, series_label(kind, key, points), line))
    panel.set_xscale("log")
    panel.set_yscale("log")
    if kind.invert_x:
        panel.invert_xaxis()
    panel.set_xlabel(kind.x_label)
    panel.set_ylabel(kind.y_label)
    panel.grid(True, which="both", alpha=0.3)
    return entries


def legend(target, entries, headings, **kwargs):
    """A legend of the entries with the heading labels in bold."""
    if not entries:
        return
    box = target.legend([h for _, _, h in entries], [text for _, text, _ in entries], fontsize=7, **kwargs)
    for text in box.get_texts():
        if text.get_text() in headings:
            text.set_fontweight("bold")


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
    plt = shared.pyplot()
    panels = 2 if builds is not None else 1
    fig, axes = plt.subplots(1, panels, figsize=(7.5 * panels, 5.0), squeeze=False)
    cards = cards_of(merged([series, builds or {}]))
    headings = {shared.key_label(c) for c in cards}
    legend(axes[0][0], draw(axes[0][0], kind, series, cards), headings)
    if builds is not None:
        legend(axes[0][1], draw(axes[0][1], BUILDS, builds, cards), headings)
    fig.suptitle(title_of(series or builds, algorithm), fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def render_grid(path, kind, panels, title):
    """One figure with a subplot per (name, series), one legend for the whole figure."""
    plt = shared.pyplot()
    count = len(panels)
    columns = min(4, math.ceil(math.sqrt(count)))
    rows = math.ceil(count / columns)
    width = 5.0 * columns
    fig, axes = plt.subplots(rows, columns, figsize=(width + 2.6, 3.8 * rows), squeeze=False)
    cards = cards_of(merged(series for _, series in panels))
    headings = {shared.key_label(c) for c in cards}
    entries = {}
    for index, (name, series) in enumerate(panels):
        panel = axes[index // columns][index % columns]
        for card, label, handle in draw(panel, kind, series, cards):
            entries.setdefault((card, label), handle)
        panel.set_title(name, fontsize=9)
    for index in range(count, rows * columns):
        axes[index // columns][index % columns].set_axis_off()
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, width / (width + 2.6), 0.96))
    listed = sorted(entries.items(), key=lambda item: (cards.index(item[0][0]), 0 if item[0][1] in headings else 1))
    legend(fig, [(card, label, handle) for (card, label), handle in listed], headings, loc="center left",
           bbox_to_anchor=(width / (width + 2.6), 0.5), ncol=1)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def csv_rows(kind, algorithm, series):
    out = []
    for key, points in series.items():
        for x, y, row in points:
            out.append(dict(row, kind=kind.name, algorithm=algorithm, series=series_label(kind, key, points), x=x, y=y,
                            controller=key[2], transfers=key[3] or row["transfers"]))
    return out


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
        for algorithm in sorted({a for _, a in panels_by}):
            panels = [(problem_title(s), s) for (p, a), s in sorted(panels_by.items()) if a == algorithm]
            written.append(render_grid(os.path.join(directory, shared.slug(algorithm) + "_problems.png"), kind,
                                       panels, "{0} algorithm: {1}, all problems".format(
                                           shared.algorithm_name(algorithm), kind.title)))
    return written


def run(store, set_names=(), where="", out=shared.PLOTS_DIR):
    """Write every figure and CSV of the selected rows under <out>/<key>/<kind>/ and, with several keys, <out>/all_cards/<kind>/; returns the paths written."""
    rows = with_errors(shared.select_rows(store, set_names, where), errors_mod.Errors(store))
    by_figure = {}
    for row in rows:
        by_figure.setdefault((row["key"], row["problem"], row["algorithm"]), []).append(row)
    figures, tables = {}, {}
    for (key, problem, algorithm), members in sorted(by_figure.items()):
        per_trial = one_per_trial(members)
        for kind in KINDS:
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
