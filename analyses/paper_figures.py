"""paper_figures.py [--key KEY]* [--cache rows.pkl] [--root data] [--out plots/paper] [--no-sync]

Four figures per key, each with a CSV of its points:

1. overhead_transfers: batch time against trajectories with and without transfers, and the transfer share.
2. batch_size: kernel time against trajectories, one adaptive algorithm per problem pane.
3. fixed_vs_adaptive: fixed-step over adaptive time at equal error, median of five error levels.
4. work_precision: per problem pane, each package's best algorithm (see best_curves).

Rows over 10% errored are dropped; duplicates keep the latest. `--cache` pickles rows with errors.
"""

import math
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import shared  # noqa: E402

shared.under_suite_python()

import errors as errors_mod  # noqa: E402

KEYS = ("windows_RTX-4070-SUPER", "linux_RTX-2060-SUPER")
GPU_PACKAGES = ("cubie", "julia_gpu", "cpp", "jax", "pytorch", "myokit_cuda")
# Validated light-mode categorical slots, fixed per package; the marker is the secondary encoding.
PACKAGE_COLOURS = {"cubie": "#2a78d6", "julia_gpu": "#eb6834", "cpp": "#1baf7a", "jax": "#eda100",
                   "pytorch": "#e87ba4", "myokit_cuda": "#4a3aa7"}
PACKAGE_MARKERS = {"cubie": "o", "julia_gpu": "s", "cpp": "D", "jax": "^", "pytorch": "v", "myokit_cuda": "P"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

PERF_DT_POW = -10
PERF_TOL = 1.0e-5
WP_N = 131072

OVERHEAD = {"problem": "lorenz", "algorithm": "tsit5", "stepping": "fixed"}
BATCH_PANES = (("lorenz", "tsit5"), ("lorenz96", "tsit5"), ("pleiades", "tsit5"),
               ("pollu", "rosenbrock23_sciml"))
WP_PROBLEMS = ("lorenz", "lorenz96", "lorenz96_20", "pleiades", "pollu", "ring_modulator",
               "ring_modulator_index2", "nand_gate", "fabbri_linder")
RATIO_LEVELS = 5


# ------------------------------------------------------------------ rows

def load_rows(store, cache):
    """Every row with its `error` against the golden (NaN without finals, traces or golden), samples dropped; read from and written to `cache` when given."""
    if cache and os.path.isfile(cache):
        with open(cache, "rb") as handle:
            return pickle.load(handle)
    errs = errors_mod.Errors(store)
    out = []
    for row in store.rows():
        row = dict(row)
        row.pop("samples_ms", None)
        try:
            row["error"] = errs.error(row)
        except ValueError:
            row["error"] = math.nan
        errs._finals.clear()
        out.append(row)
    if cache:
        with open(cache, "wb") as handle:
            pickle.dump(out, handle)
    return out


def stepping(row):
    return "fixed" if row["controller"] == "fixed" else "adaptive"


def seconds(row):
    return shared.number(row.get("min_ms")) / 1000.0


def latest(rows, ident):
    """One row per ident(row), the most recently recorded."""
    held = {}
    for row in rows:
        k = ident(row)
        if k not in held or str(row["recorded_utc"]) > str(held[k]["recorded_utc"]):
            held[k] = row
    return list(held.values())


def usable(row):
    return shared.timed(row) and shared.within_errored_limit(row)


def perf_rows(rows, key):
    """The timing sweep over trajectory counts under a key: the default system, fixed step duration x 2^-10 or tolerance 1e-5."""
    def perf(row):
        if row["key"] != key or row["package"] not in GPU_PACKAGES:
            return False
        if row["precision"] != "float32" or row["system_params"] not in ("{}", '{"states":32}'):
            return False
        if stepping(row) == "fixed":
            return math.isclose(row["dt"], row["duration"] * 2.0 ** PERF_DT_POW, rel_tol=1e-9)
        return math.isclose(row["atol"], PERF_TOL, rel_tol=1e-9)
    kept = [r for r in rows if perf(r) and usable(r)]
    return latest(kept, lambda r: (r["package"], r["problem"], r["algorithm"], stepping(r), r["transfers"], r["n"]))


def wp_rows(rows, key):
    """The work-precision sweep under a key: kernel-only rows with a finite error against the golden."""
    def wp(row):
        return (row["key"] == key and row["package"] in GPU_PACKAGES and row["transfers"] == "none"
                and row["precision"] == "float32" and errors_mod.is_finite_positive(row["error"]))
    kept = [r for r in rows if wp(r) and usable(r)]
    return latest(kept, lambda r: (r["package"], r["problem"], r["algorithm"], stepping(r), r["n"], r["dt"],
                                   r["atol"], r["system_params"]))


def curve(rows, **match):
    return [r for r in rows if all((stepping(r) if k == "stepping" else r[k]) == v for k, v in match.items())]


# ----------------------------------------------------------- selection

def levels(low, high, count):
    return [low * (high / low) ** (i / (count - 1)) for i in range(count)]


def best_curves(rows, problem):
    """{package: (algorithm, stepping, rows in sweep order)}: per package, over nine error levels from the least error its curves reach to a hundred times that, the curve that is cheapest at the most levels (a level a curve never reaches counts against it); ties to the lowest summed log time."""
    out = {}
    for package in GPU_PACKAGES:
        curves = {}
        for r in rows:
            if r["problem"] == problem and r["package"] == package:
                curves.setdefault((r["algorithm"], stepping(r)), []).append(r)
        curves = {k: v for k, v in curves.items() if len(v) >= 3}
        if not curves:
            continue
        points = {k: [(seconds(r), r["error"]) for r in v] for k, v in curves.items()}
        errors = [e for pts in points.values() for _, e in pts]
        wins = dict.fromkeys(points, 0)
        cost = dict.fromkeys(points, 0.0)
        for level in levels(min(errors), 100.0 * min(errors), 9):
            times = {k: cheapest(pts, level) for k, pts in points.items()}
            reached = {k: t for k, t in times.items() if math.isfinite(t)}
            wins[min(reached, key=reached.get)] += 1
            for k, t in times.items():
                cost[k] += math.log(t) if math.isfinite(t) else 50.0
        chosen = max(points, key=lambda k: (wins[k], -cost[k]))
        out[package] = (chosen[0], chosen[1], sorted(curves[chosen], key=loose_first))
    return out


def loose_first(row):
    """Sweep order, loosest first: step size descending, then tolerance descending."""
    dt, tol = shared.number(row["dt"]), shared.number(row["atol"])
    return (-(dt if math.isfinite(dt) else 0.0), -(tol if math.isfinite(tol) else 0.0))


def problem_label(problem, rows):
    states = {r["states"] for r in rows if r["problem"] == problem}
    return shared.problem_name(problem, states.pop() if len(states) == 1 else None)


def cheapest(points, level):
    """The least time among the (time, error) points at or below an error level."""
    return min((t for t, e in points if e <= level), default=math.nan)


def fixed_adaptive_ratios(rows):
    """[(package, problem, algorithm, median ratio of fixed to adaptive time)] over the error levels both reach."""
    groups = {}
    for r in rows:
        groups.setdefault((r["package"], r["problem"], r["algorithm"]), {}).setdefault(stepping(r), []).append(
            (seconds(r), r["error"]))
    out = []
    for (package, problem, algorithm), by in sorted(groups.items()):
        fixed, adaptive = by.get("fixed", []), by.get("adaptive", [])
        if len(fixed) < 3 or len(adaptive) < 3:
            continue
        low = max(min(e for _, e in fixed), min(e for _, e in adaptive))
        high = min(max(e for _, e in fixed), max(e for _, e in adaptive))
        if not low < high:
            continue
        ratios = sorted(cheapest(fixed, lv) / cheapest(adaptive, lv) for lv in levels(low, high, RATIO_LEVELS))
        out.append((package, problem, algorithm, ratios[len(ratios) // 2]))
    return out


# --------------------------------------------------------------- drawing

def style(axis, xlabel, ylabel, logy=True):
    axis.set_xscale("log")
    if logy:
        axis.set_yscale("log")
    axis.set_xlabel(xlabel, color=INK)
    axis.set_ylabel(ylabel, color=INK)
    axis.grid(True, which="major", color=GRID, linewidth=0.6)
    axis.tick_params(colors=MUTED, labelsize=8)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(MUTED)


def line(axis, xs, ys, package, dashed=False, hollow=False, label=None):
    colour = PACKAGE_COLOURS[package]
    return axis.plot(xs, ys, color=colour, linestyle="--" if dashed else "-", linewidth=1.6,
                     marker=PACKAGE_MARKERS[package], markersize=5,
                     markerfacecolor="white" if hollow else colour, markeredgecolor=colour, label=label)[0]


def sweep(rows, **match):
    points = sorted(curve(rows, **match), key=lambda r: r["n"])
    return [r["n"] for r in points], [seconds(r) for r in points], points


def fig_overhead(perf, out, stem):
    plt = shared.pyplot()
    fig, (left, right) = plt.subplots(1, 2, figsize=(8.0, 3.4))
    table = []
    for package in GPU_PACKAGES:
        xs_k, ys_k, pk = sweep(perf, package=package, transfers="none", **OVERHEAD)
        xs_b, ys_b, pb = sweep(perf, package=package, transfers="both", **OVERHEAD)
        if len(xs_k) < 2 or len(xs_b) < 2:
            continue
        line(left, xs_k, ys_k, package, label=shared.package_name(package))
        line(left, xs_b, ys_b, package, dashed=True, hollow=True)
        both = dict(zip(xs_b, ys_b))
        shared_n = [n for n in xs_k if n in both]
        share = [100.0 * (both[n] - k) / both[n] for n, k in zip(xs_k, ys_k) if n in both]
        line(right, shared_n, share, package)
        table += [dict(r, figure="overhead_transfers") for r in pk + pb]
    style(left, "Trajectories", "Batch time (s)")
    style(right, "Trajectories", "Transfer share of batch time (%)", logy=False)
    right.set_ylim(min(-5.0, right.get_ylim()[0]), 100)
    right.axhline(0.0, color=MUTED, linewidth=0.8)
    kernel = plt.Line2D([], [], color=MUTED, linestyle="-", label="Kernel only")
    total = plt.Line2D([], [], color=MUTED, linestyle="--", label="With transfers")
    handles, _ = left.get_legend_handles_labels()
    left.legend(handles=handles + [kernel, total], fontsize=7, frameon=False)
    left.set_title("(a)", loc="left", fontsize=9)
    right.set_title("(b)", loc="left", fontsize=9)
    fig.suptitle("{0}, {1}, fixed step {2}".format(
        shared.problem_name(OVERHEAD["problem"]), shared.algorithm_name(OVERHEAD["algorithm"]),
        step_text(table[0]) if table else ""), fontsize=10, color=INK)
    return save(fig, out, stem, table)


def fig_batch(perf, out, stem):
    plt = shared.pyplot()
    fig, axes = plt.subplots(1, len(BATCH_PANES), figsize=(3.0 * len(BATCH_PANES), 3.2), sharey=False)
    table, seen = [], {}
    for axis, (problem, algorithm) in zip(axes, BATCH_PANES):
        for package in GPU_PACKAGES:
            xs, ys, pts = sweep(perf, package=package, problem=problem, algorithm=algorithm,
                                stepping="adaptive", transfers="none")
            if len(xs) < 2:
                continue
            seen[package] = line(axis, xs, ys, package)
            table += [dict(r, figure="batch_size") for r in pts]
        style(axis, "Trajectories", "Kernel time (s)" if axis is axes[0] else "")
        axis.set_title("{0}\n{1}, adaptive, tol 1e-5".format(shared.problem_name(problem),
                                                             shared.algorithm_name(algorithm)), fontsize=8)
    fig.legend(list(seen.values()), [shared.package_name(p) for p in seen], loc="lower center",
               ncol=len(seen), fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return save(fig, out, stem, table, tight=False)


def fig_fixed_adaptive(wp, out, stem):
    plt = shared.pyplot()
    ratios = fixed_adaptive_ratios(wp)
    problems = [p for p in WP_PROBLEMS if any(r[1] == p for r in ratios)]
    fig, axis = plt.subplots(1, 1, figsize=(6.0, 0.45 * len(problems) + 1.4))
    seen = {}
    offsets = {p: (i - (len(GPU_PACKAGES) - 1) / 2) * 0.09 for i, p in enumerate(GPU_PACKAGES)}
    for package, problem, algorithm, ratio in ratios:
        y = problems.index(problem) + offsets[package]
        seen[package] = axis.plot([ratio], [y], linestyle="none", marker=PACKAGE_MARKERS[package], markersize=6,
                                  color=PACKAGE_COLOURS[package], markeredgecolor="white", markeredgewidth=0.5)[0]
    axis.axvline(1.0, color=MUTED, linewidth=0.8)
    axis.set_yticks(range(len(problems)))
    axis.set_yticklabels([problem_label(p, wp) for p in problems], fontsize=8)
    axis.invert_yaxis()
    style(axis, "Fixed-step time / adaptive time at equal error", "", logy=False)
    wins = sum(r[3] > 1.0 for r in ratios)
    axis.set_title("Adaptive faster for {0} of {1} algorithms (right of the line)".format(wins, len(ratios)),
                   fontsize=9, color=INK)
    axis.legend(list(seen.values()), [shared.package_name(p) for p in seen], fontsize=7, frameon=False,
                loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=len(seen))
    table = [dict(package=p, problem=q, algorithm=a, ratio=r, figure="fixed_vs_adaptive") for p, q, a, r in ratios]
    return save(fig, out, stem, table, columns=("figure", "package", "problem", "algorithm", "ratio"))


def fig_wp(wp, out, stem):
    plt = shared.pyplot()
    problems = [p for p in WP_PROBLEMS if best_curves(wp, p)]
    columns = 3
    rows_n = math.ceil(len(problems) / columns)
    fig, axes = plt.subplots(rows_n, columns, figsize=(3.3 * columns, 2.9 * rows_n), squeeze=False)
    table = []
    for index, problem in enumerate(problems):
        axis = axes[index // columns][index % columns]
        medians = []
        for package, (algorithm, kind, pts) in best_curves(wp, problem).items():
            medians.append(sorted(r["error"] for r in pts)[len(pts) // 2])
            line(axis, [seconds(r) for r in pts], [r["error"] for r in pts], package, hollow=kind == "fixed",
                 label="{0}: {1}{2}".format(shared.package_name(package), shared.algorithm_name(algorithm),
                                            ", fixed" if kind == "fixed" else ""))
            table += [dict(r, figure="work_precision") for r in pts]
        style(axis, "Kernel time (s)" if index // columns == rows_n - 1 else "",
              "RMS error" if index % columns == 0 else "")
        # Clip diverged loose steps off the top.
        axis.set_ylim(top=min(axis.get_ylim()[1], 10.0 * max(medians)))
        axis.set_title(problem_label(problem, wp), fontsize=9, color=INK)
        axis.legend(fontsize=6, frameon=False)
    for index in range(len(problems), rows_n * columns):
        axes[index // columns][index % columns].set_axis_off()
    return save(fig, out, stem, table)


def step_text(row):
    dt = shared.number(row["dt"])
    for unit, scale in (("s", 1.0), ("ms", 1e-3), ("us", 1e-6), ("ns", 1e-9)):
        if dt >= scale:
            return "{0:.3g} {1}".format(dt / scale, unit)
    return "{0:g} s".format(dt)


CSV_COLUMNS = ("figure", "package", "problem", "algorithm", "controller", "transfers", "n", "dt", "atol", "min_ms",
               "error", "errored_pct", "run_id", "recorded_utc")


def save(fig, out, stem, table, columns=CSV_COLUMNS, tight=True):
    os.makedirs(out, exist_ok=True)
    if tight:
        fig.tight_layout()
    path = os.path.join(out, stem + ".png")
    fig.savefig(path, dpi=200)
    fig.savefig(os.path.join(out, stem + ".pdf"))
    shared.pyplot().close(fig)
    shared.write_csv(os.path.join(out, stem + ".csv"), columns, table)
    return path


def run(rows, keys, out):
    written = []
    for key in keys:
        perf, wp = perf_rows(rows, key), wp_rows(rows, key)
        tag = shared.slug(key)
        written.append(fig_overhead(perf, out, "1_overhead_transfers_" + tag))
        written.append(fig_batch(perf, out, "2_batch_size_" + tag))
        written.append(fig_fixed_adaptive(wp, out, "3_fixed_vs_adaptive_" + tag))
        written.append(fig_wp(wp, out, "4_work_precision_" + tag))
    return written


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
