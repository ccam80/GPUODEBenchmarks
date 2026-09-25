"""paper_figures.py [--key KEY]* [--cache rows.pkl] [--root data] [--out plots/paper] [--no-sync]

Paper figures in plots.py's encoding, each with a CSV:

1_transfers: per card, each package's cheapest and costliest algorithm on Lorenz, with and without transfers.
2_batch_size_<key>: kernel time against trajectories; rows are algorithms, columns problems.
3_fixed_vs_adaptive_<key>: work-precision on the same grid.
4_work_precision_<key>: each package's best algorithm per problem (see best_series).
5_cards_batch, 5_cards_wp: Cubie Tsit5 on both cards.

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
            "ring_modulator_index2", "nand_gate", "fabbri_linder")
GRID_ALGORITHMS = ("tsit5", "rosenbrock23_sciml", "kvaerno3")
TRANSFER_PROBLEM = "lorenz"
CARD_PACKAGE, CARD_ALGORITHM = "cubie", "tsit5"
BATCH = plots.kind_named("runtime_vs_n")
WP = plots.kind_named("error_vs_runtime")


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


def algorithm_label(kind, key, points):
    """'Cubie, Tsit5, adaptive + transfer'."""
    _, package, controller, transfers = key
    return "{0}, {1}, {2}{3}".format(shared.package_name(package), shared.algorithm_name(points[0][2]["algorithm"]),
                                     controller, " + transfer" if transfers == "both" else "")


def problem_label(rows, problem):
    """'Lorenz 96 (20 states)', 'Ring modulator, index 2 (15 states)' from the catalogue."""
    from problems import load_problems
    entry = next(e for e in load_problems() if e["problem"] == problem)
    name = re.sub(r"\s*\((\d+)\)$", "", entry["display"])
    name = re.sub(r"\s*\((.+)\)$", r", \1", name)
    return "{0} ({1} states)".format(name, entry["states"])


# ----------------------------------------------------------- selection

def extreme_algorithms(rows, key, package, problem):
    """(least, most) costly (algorithm, controller kind) of a package's kernel-only batch series, compared at the largest trajectory count they share."""
    candidates = {}
    for algorithm in sorted({r["algorithm"] for r in rows if r["package"] == package and r["problem"] == problem}):
        for skey, points in series(BATCH, rows, key=key, package=package, problem=problem, algorithm=algorithm,
                                   transfers="none").items():
            candidates[(algorithm, skey[2])] = points
    if len(candidates) < 2:
        return None
    shared_n = set.intersection(*({x for x, _, _ in pts} for pts in candidates.values()))
    if not shared_n:
        return None
    n = max(shared_n)
    ranked = sorted(candidates, key=lambda c: next(y for x, y, _ in candidates[c] if x == n))
    return ranked[0], ranked[-1]


def levels(low, high, count):
    return [low * (high / low) ** (i / (count - 1)) for i in range(count)]


def cheapest(points, level):
    """The least time among the points at or below an error level, errored rows left out."""
    return min((x for x, y, row in points if y <= level and shared.within_errored_limit(row)), default=math.nan)


def best_series(rows, key, problem):
    """{series key: points}, one per package: over nine error levels from the least error its curves reach to 100 times that, the curve cheapest at the most levels; ties to the lowest summed log time."""
    out = {}
    for package in plots.GPU_PACKAGES:
        curves = {}
        for algorithm in sorted({r["algorithm"] for r in rows if r["package"] == package and r["problem"] == problem}):
            for skey, points in series(WP, rows, key=key, package=package, problem=problem, algorithm=algorithm,
                                       transfers="none").items():
                if sum(shared.within_errored_limit(row) for _, _, row in points) >= 3:
                    curves[(algorithm, skey)] = points
        errors = [y for pts in curves.values() for x, y, row in pts if shared.within_errored_limit(row)]
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

def grid(out, stem, kind, panels, title, columns, label=plots.series_label, panel_legends=False):
    path = os.path.join(out, stem + ".png")
    plots.render_grid(path, kind, panels, title, columns=columns, label=label, panel_legends=panel_legends)
    shared.write_csv(os.path.join(out, stem + ".csv"), plots.CSV_COLUMNS,
                     [r for _, s in panels if s for r in plots.csv_rows(kind, "", s)])
    return path


def fig_transfers(rows, keys, out):
    panels = []
    for key in keys:
        chosen = {}
        for package in plots.GPU_PACKAGES:
            ends = extreme_algorithms(rows, key, package, TRANSFER_PROBLEM)
            if ends:
                chosen[package] = ends
        for end, name in ((0, "least costly algorithm"), (1, "most costly algorithm")):
            pane = {}
            for package, ends in chosen.items():
                algorithm, controller = ends[end]
                pane.update(series(BATCH, rows, key=key, package=package, problem=TRANSFER_PROBLEM,
                                   algorithm=algorithm, controller_kind=controller))
            panels.append(("{0}: {1}".format(shared.key_label(key), name), plots.ordered(pane)))
    return grid(out, "1_transfers", BATCH, panels, "{0}: batch time with and without transfers".format(
        problem_label(rows, TRANSFER_PROBLEM)), 2, algorithm_label, panel_legends=True)


def algorithm_grid(rows, key, kind, stem, title, out, **match):
    cells = {(a, p): series(kind, rows, key=key, algorithm=a, problem=p, **match)
             for a in GRID_ALGORITHMS for p in PROBLEMS}
    problems = [p for p in PROBLEMS if any(cells[(a, p)] for a in GRID_ALGORITHMS)]
    panels = [("{0}, {1}".format(shared.algorithm_name(a), problem_label(rows, p)), cells[(a, p)])
              for a in GRID_ALGORITHMS for p in problems]
    return grid(out, stem, kind, panels, title, len(problems))


def fig_best_wp(rows, key, out):
    panels = [(problem_label(rows, p), best_series(rows, key, p)) for p in PROBLEMS]
    return grid(out, "4_work_precision_" + shared.slug(key), WP, [(n, s) for n, s in panels if s],
                "{0}: each package's best algorithm".format(shared.key_label(key)), 3, algorithm_label,
                panel_legends=True)


def fig_cards(rows, keys, kind, stem, title, out, **match):
    panels = []
    for p in PROBLEMS:
        pane = {}
        for key in keys:
            pane.update(series(kind, rows, key=key, package=CARD_PACKAGE, algorithm=CARD_ALGORITHM, problem=p, **match))
        if pane:
            panels.append((problem_label(rows, p), plots.ordered(pane)))
    return grid(out, stem, kind, panels, title, 4)


def run(rows, keys, out):
    os.makedirs(out, exist_ok=True)
    paths = [fig_transfers(rows, keys, out)]
    for key in keys:
        tag, card = shared.slug(key), shared.key_label(key)
        paths.append(algorithm_grid(rows, key, BATCH, "2_batch_size_" + tag, card + ": kernel time, adaptive steps",
                                    out, transfers="none", controller_kind="adaptive"))
        paths.append(algorithm_grid(rows, key, WP, "3_fixed_vs_adaptive_" + tag, card + ": fixed and adaptive steps",
                                    out, transfers="none"))
        paths.append(fig_best_wp(rows, key, out))
    name = "{0} {1}".format(shared.package_name(CARD_PACKAGE), shared.algorithm_name(CARD_ALGORITHM))
    paths.append(fig_cards(rows, keys, BATCH, "5_cards_batch", name + " on both cards: kernel time, adaptive steps",
                           out, transfers="none", controller_kind="adaptive"))
    paths.append(fig_cards(rows, keys, WP, "5_cards_wp", name + " on both cards: adaptive steps", out,
                           transfers="none", controller_kind="adaptive"))
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
