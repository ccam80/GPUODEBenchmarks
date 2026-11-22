"""paper_figures.py [--key KEY]* [--cache rows.pkl] [--root data] [--out plots/paper] [--no-sync]

Paper figures as inclusion lists for plots.draw_figure; each writes a PNG and a CSV.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import shared  # noqa: E402

shared.under_suite_python()

import plots  # noqa: E402
from plots import Figure, Pane, problem_panes  # noqa: E402

KEYS = ("windows_RTX-4070-SUPER", "linux_RTX-2060-SUPER")
PROBLEMS = ("lorenz", "lorenz96", "lorenz96_20", "pleiades", "pollu", "ring_modulator",
            "ring_modulator_index2", "nand_gate")
TRANSFER_PROBLEMS = ("lorenz", "lorenz96")
BATCH = plots.kind_named("runtime_vs_n")
WP = plots.kind_named("error_vs_runtime")
INTERVAL_WP = plots.kind_named("interval_error_vs_runtime")
KERNEL = {"transfers": "none"}
NEW_PACKAGES_KEY = "linux_RTX-2060-SUPER"
NEW_PACKAGES = dict(key=NEW_PACKAGES_KEY, package=("cubie", "gradsolve", "modax"), **KERNEL)
NEW_PACKAGES_PROBLEMS = ("lorenz", "lorenz96", "pleiades", "pollu")
# We leave off the points whose error exceeds this, where a fixed-step solve diverged.
NEW_PACKAGES_ERROR_MAX = 1.0e3
# The algorithms gradsolve or modax ran, with modax's float64-LU Rodas5P drawn beside Rodas5P.
TSIT5 = (("tsit5",), "Tsit5")
RODAS5P = (("rodas5p", "rodas5p_lu64"), "Rodas5P")
NEW_PACKAGES_ALGORITHMS = (TSIT5, (("classical-rk4",), "RK4"), (("rosenbrock23_sciml",), "Rosenbrock23"), RODAS5P)
EVERY_NEW_PACKAGES_ALGORITHM = tuple(a for algorithms, _ in NEW_PACKAGES_ALGORITHMS for a in algorithms)
NEW_PACKAGES_TITLE = shared.key_label(NEW_PACKAGES_KEY) + ": "
EVERY_PACKAGE = dict(key=NEW_PACKAGES_KEY, **KERNEL)


def batch_figures(stem, algorithms, where, problems, title):
    """Return a kernel-time figure per (algorithm names, display name), a pane per problem, keeping errored rows."""
    return [Figure("{0}_batch_{1}".format(stem, shared.slug(names[0])), BATCH, problem_panes(problems),
                   where=dict(where, algorithm=names), columns=2, per_key=False, keep_errored=True,
                   title=NEW_PACKAGES_TITLE + title + ", " + name + " kernel time")
            for names, name in algorithms]


def wp_figures(stem, algorithms, where, problems, title):
    """Return a work-precision figure per (algorithm names, display name), a pane per problem."""
    return [Figure("{0}_wp_{1}".format(stem, shared.slug(names[0])), WP,
                   problem_panes(problems, error_max=NEW_PACKAGES_ERROR_MAX),
                   where=dict(where, algorithm=names), columns=2, per_key=False,
                   title=NEW_PACKAGES_TITLE + title + ", " + name + " work-precision")
            for names, name in algorithms]


FIGURES = (
    Figure("1_transfers", BATCH, [pane for key in KEYS for problem in TRANSFER_PROBLEMS for pane in (
        Pane(dict(key=key, problem=problem, **KERNEL), note="kernel time"),
        Pane(dict(key=key, problem=problem), note="transfer time", transform=plots.transfer_time,
             kind=plots.TRANSFER))], columns=2, per_key=False,
           title="Kernel time and transfer time, every algorithm"),
    Figure("2_batch_size", BATCH, problem_panes(PROBLEMS), where=KERNEL, title="kernel time, every algorithm"),
    Figure("3_fixed_vs_adaptive", WP, problem_panes(PROBLEMS), where=KERNEL,
           title="work-precision, every algorithm"),
    Figure("4_work_precision", WP, problem_panes(PROBLEMS, transform=plots.best_per_package), where=KERNEL,
           columns=3, title="each package's best algorithm"),
    Figure("5_cards_batch", BATCH, problem_panes(PROBLEMS), where=dict(package="cubie", **KERNEL), per_key=False,
           title="Cubie on both cards: kernel time, every algorithm"),
    Figure("6_fabbri_wp", INTERVAL_WP, problem_panes(("fabbri_linder",)), columns=1,
           where=dict(package=("cubie", "myokit_cuda"), **KERNEL), title="Cubie against Myokit"),
    Figure("6_fabbri_batch", BATCH, problem_panes(("fabbri_linder",)), columns=1,
           where=dict(package=("cubie", "myokit_cuda"), **KERNEL), title="Cubie against Myokit"),
    Figure("7_new_packages_wp", WP, problem_panes(NEW_PACKAGES_PROBLEMS, error_max=NEW_PACKAGES_ERROR_MAX),
           where=dict(NEW_PACKAGES, algorithm=EVERY_NEW_PACKAGES_ALGORITHM), columns=2, per_key=False,
           title=NEW_PACKAGES_TITLE + "Cubie, gradsolve and modax work-precision, every algorithm gradsolve or modax ran"),
    *batch_figures("7_new_packages", NEW_PACKAGES_ALGORITHMS, NEW_PACKAGES, NEW_PACKAGES_PROBLEMS,
                   "Cubie, gradsolve and modax"),
    *batch_figures("8_every_package", (TSIT5, RODAS5P), EVERY_PACKAGE, PROBLEMS, "every package"),
    *wp_figures("8_every_package", (TSIT5, RODAS5P), EVERY_PACKAGE, PROBLEMS, "every package"),
)


def main(argv=None):
    p = shared.parser(__doc__)
    p.add_argument("--key", action="append", default=[], help="a machine key; both keys without it")
    p.add_argument("--cache", default="", help="a pickle of the rows with their errors, read when present")
    p.set_defaults(out=os.path.join(shared.PLOTS_DIR, "paper"))
    args = p.parse_args(argv)
    shared.pull_store(args)
    rows = plots.load_rows(shared.AnalysisStore(args.root), args.cache)
    os.makedirs(args.out, exist_ok=True)
    for figure in FIGURES:
        for path in plots.draw_figure(figure, rows, list(args.key or KEYS), args.out):
            print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
