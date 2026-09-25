"""paper_figures.py [--key KEY]* [--cache rows.pkl] [--root data] [--out plots/paper] [--no-sync]

Paper figures in plots.py's encoding, each with a CSV:

1_transfers: kernel and transfer time against trajectories, smallest and largest system, per card.
2_batch_size_<key>: kernel time, every algorithm, a pane per problem.
3_fixed_vs_adaptive_<key>: work-precision, algorithms by row, problems by column.
4_work_precision_<key>: each package's best algorithm per problem (see best_series).
5_cards_batch: Cubie kernel time on both cards.
6_fabbri_wp, 6_fabbri_batch: Fabbri-Linder, Cubie against Myokit.

`--cache` pickles rows with errors to skip the error pass.
"""

import math
import os
import pickle
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import shared  # noqa: E402

shared.under_suite_python()

import errors as errors_mod  # noqa: E402
import plots  # noqa: E402

KEYS = ("windows_RTX-4070-SUPER", "linux_RTX-2060-SUPER")
PROBLEMS = ("lorenz", "lorenz96", "lorenz96_20", "pleiades", "pollu", "ring_modulator",
            "ring_modulator_index2", "nand_gate")
FABBRI = "fabbri_linder"
GRID_ALGORITHMS = ("tsit5", "rosenbrock23_sciml", "kvaerno3")
TRANSFER_PROBLEMS = ("lorenz", "lorenz96")
BATCH = plots.kind_named("runtime_vs_n")
WP = plots.kind_named("error_vs_runtime")
TRANSFER = plots.Kind("transfer_vs_n", "n", "transfer_s", ("n",), plots.GPU_PACKAGES, False, "timed",
                      "Trajectories", "Transfer time (s)", "Transfer time")


# ------------------------------------------------------------------ rows

def load_rows(store, cache):
    """Every row with its error against the golden and its controller kind; read from and written to `cache` when given."""
    if cache and os.path.isfile(cache):
        with open(cache, "rb") as handle:
            rows = pickle.load(handle)
    else:
        errs = errors_mod.Errors(store)
        rows = []
        for row in store.rows():
            row = dict(row)
            row.pop("samples_ms", None)
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


def series(kind, rows, **match):
    """plots.series_of over the rows equal to every column given."""
    return plots.series_of(kind, [r for r in rows if all(r.get(k) == v for k, v in match.items())])


def by_algorithm(kind, rows, **match):
    """{algorithm: series} over every algorithm with a series among the rows matching."""
    out = {}
    for algorithm in sorted({r["algorithm"] for r in rows if all(r.get(k) == v for k, v in match.items())}):
        found = series(kind, rows, algorithm=algorithm, **match)
        if found:
            out[algorithm] = found
    return out


def tagged(by):
    """One series dict from {algorithm: series}, the algorithm appended to each key so they stay apart."""
    return plots.ordered({key + (algorithm,): points for algorithm, s in by.items() for key, points in s.items()})


def transfer_series(by):
    """{algorithm: series} of transfer time: with transfers minus without, at each trajectory count both have."""
    out = {}
    for algorithm, s in by.items():
        found = {}
        for key, points in s.items():
            if key[3] != "both":
                continue
            without = dict((x, y) for x, y, _ in s.get(key[:3] + ("none",), []))
            diff = [(x, y - without[x], row) for x, y, row in points if x in without and y > without[x]]
            if len(diff) >= 2:
                found[key[:3] + ("",)] = diff
        if found:
            out[algorithm] = found
    return out


def algorithm_label(kind, key, points):
    """'Cubie, Tsit5, adaptive'."""
    return "{0}, {1}, {2}".format(shared.package_name(key[1]), shared.algorithm_name(points[0][2]["algorithm"]),
                                  key[2])


def problem_label(problem):
    """'Lorenz 96 (20 states)', 'Ring modulator, index 2 (15 states)' from the catalogue."""
    from problems import load_problems
    entry = next(e for e in load_problems() if e["problem"] == problem)
    name = re.sub(r"\s*\((\d+)\)$", "", entry["display"])
    name = re.sub(r"\s*\((.+)\)$", r", \1", name)
    return "{0} ({1} states)".format(name, entry["states"])


# ----------------------------------------------------------- selection

def levels(low, high, count):
    return [low * (high / low) ** (i / (count - 1)) for i in range(count)]


def cheapest(points, level):
    """The least time among the points at or below an error level, errored rows left out."""
    return min((y for x, y, row in points if x <= level and shared.within_errored_limit(row)), default=math.nan)


def best_series(rows, key, problem):
    """{series key: points}, one per package: over nine error levels from the least error its curves reach to 100 times that, the curve cheapest at the most levels; ties to the lowest summed log time."""
    out = {}
    for package in plots.GPU_PACKAGES:
        curves = {}
        for algorithm, s in by_algorithm(WP, rows, key=key, package=package, problem=problem,
                                         transfers="none").items():
            for skey, points in s.items():
                if sum(shared.within_errored_limit(row) for _, _, row in points) >= 3:
                    curves[(algorithm, skey)] = points
        errors = [x for pts in curves.values() for x, _, row in pts if shared.within_errored_limit(row)]
        if not errors:
            continue
        wins, cost = dict.fromkeys(curves, 0), dict.fromkeys(curves, 0.0)
        for level in levels(min(errors), 100.0 * min(errors), 9):
            times = {c: cheapest(pts, level) for c, pts in curves.items()}
            reached = {c: t for c, t in times.items() if math.isfinite(t)}
            if reached:
                wins[min(reached, key=reached.get)] += 1
            for c, t in times.items():
                cost[c] += math.log(t) if math.isfinite(t) else 50.0
        chosen = max(curves, key=lambda c: (wins[c], -cost[c]))
        out[chosen[1]] = curves[chosen]
    return plots.ordered(out)


# ----------------------------------------------------------- figures

def table(out, stem, kind, series_dicts):
    shared.write_csv(os.path.join(out, stem + ".csv"), plots.CSV_COLUMNS,
                     [r for s in series_dicts if s for r in plots.csv_rows(kind, "", s)])


def grid(out, stem, kind, panels, title, columns, label=plots.series_label, panel_legends=False):
    path = os.path.join(out, stem + ".png")
    plots.render_grid(path, kind, panels, title, columns=columns, label=label, panel_legends=panel_legends)
    table(out, stem, kind, [s for _, s in panels])
    return path


def combined(out, stem, kind, panels, title, columns=None):
    path = os.path.join(out, stem + ".png")
    plots.render_combined_grid(path, kind, panels, title, columns=columns)
    table(out, stem, kind, [s for _, by in panels for s in by.values()])
    return path


def fig_transfers(rows, keys, out):
    """A row per (card, problem): kernel time on the left, transfer time on the right, every algorithm."""
    panels = []
    for key in keys:
        for problem in TRANSFER_PROBLEMS:
            by = by_algorithm(BATCH, rows, key=key, problem=problem)
            name = "{0}, {1}".format(shared.key_label(key), problem_label(problem))
            kernel = {a: {k: p for k, p in s.items() if k[3] == "none"} for a, s in by.items()}
            panels.append((name + ": kernel", tagged(kernel)))
            panels.append((name + ": transfers", tagged(transfer_series(by)), TRANSFER))
    path = os.path.join(out, "1_transfers.png")
    plots.render_grid(path, BATCH, panels, "Kernel time and transfer time", columns=2)
    table(out, "1_transfers", BATCH, [p[1] for p in panels])
    return path


def algorithm_grid(rows, key, kind, stem, title, out, **match):
    cells = {(a, p): series(kind, rows, key=key, algorithm=a, problem=p, **match)
             for a in GRID_ALGORITHMS for p in PROBLEMS}
    problems = [p for p in PROBLEMS if any(cells[(a, p)] for a in GRID_ALGORITHMS)]
    panels = [("{0}, {1}".format(shared.algorithm_name(a), problem_label(p)), cells[(a, p)])
              for a in GRID_ALGORITHMS for p in problems]
    return grid(out, stem, kind, panels, title, len(problems))


def fig_best_wp(rows, key, out):
    panels = [(problem_label(p), best_series(rows, key, p)) for p in PROBLEMS]
    return grid(out, "4_work_precision_" + shared.slug(key), WP, [(n, s) for n, s in panels if s],
                "{0}: each package's best algorithm".format(shared.key_label(key)), 3, algorithm_label,
                panel_legends=True)


def fig_batch(rows, key, out):
    panels = [(problem_label(p), by_algorithm(BATCH, rows, key=key, problem=p, transfers="none")) for p in PROBLEMS]
    return combined(out, "2_batch_size_" + shared.slug(key), BATCH, [(n, b) for n, b in panels if b],
                    "{0}: kernel time, every algorithm".format(shared.key_label(key)))


def fig_cards(rows, keys, out):
    panels = []
    for p in PROBLEMS:
        by = {}
        for key in keys:
            for algorithm, s in by_algorithm(BATCH, rows, key=key, package="cubie", problem=p,
                                             transfers="none").items():
                by.setdefault(algorithm, {}).update(s)
        if by:
            panels.append((problem_label(p), by))
    return combined(out, "5_cards_batch", BATCH, panels, "Cubie on both cards: kernel time, every algorithm")


def fig_fabbri(rows, key, out):
    paths = []
    for kind, stem, title in ((WP, "6_fabbri_wp", "work-precision"), (BATCH, "6_fabbri_batch", "kernel time")):
        by = {}
        for package in ("cubie", "myokit_cuda"):
            for algorithm, s in by_algorithm(kind, rows, key=key, package=package, problem=FABBRI,
                                             transfers="none").items():
                by.setdefault(algorithm, {}).update(s)
        if by:
            paths.append(combined(out, stem, kind, [(problem_label(FABBRI), by)],
                                  "{0}: {1}, Cubie against Myokit".format(shared.key_label(key), title), 1))
    return paths


def run(rows, keys, out):
    os.makedirs(out, exist_ok=True)
    paths = [fig_transfers(rows, keys, out)]
    for key in keys:
        tag, card = shared.slug(key), shared.key_label(key)
        paths.append(fig_batch(rows, key, out))
        paths.append(algorithm_grid(rows, key, WP, "3_fixed_vs_adaptive_" + tag, card + ": fixed and adaptive steps",
                                    out, transfers="none"))
        paths.append(fig_best_wp(rows, key, out))
    paths.append(fig_cards(rows, keys, out))
    paths += fig_fabbri(rows, keys[0], out)
    return paths


def main(argv=None):
    p = shared.parser(__doc__)
    p.add_argument("--key", action="append", default=[], help="a machine key; both keys without it")
    p.add_argument("--cache", default="", help="a pickle of the rows with their errors, read when present")
    p.set_defaults(out=os.path.join(shared.PLOTS_DIR, "paper"))
    args = p.parse_args(argv)
    shared.pull_store(args)
    store = shared.AnalysisStore(args.root)
    for path in run(load_rows(store, args.cache), args.key or KEYS, args.out):
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
