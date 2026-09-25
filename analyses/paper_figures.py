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
KERNEL = {"transfers": "none"}

FIGURES = (
    Figure("1_transfers", BATCH, [pane for key in KEYS for problem in TRANSFER_PROBLEMS for pane in (
        Pane(dict(key=key, problem=problem, **KERNEL), note="kernel time"),
        Pane(dict(key=key, problem=problem), note="transfer time", transform=plots.transfer_time,
             kind=plots.TRANSFER))], colour_by="package", columns=2, per_key=False,
           title="Kernel time and transfer time, every algorithm"),
    Figure("2_batch_size", BATCH, problem_panes(PROBLEMS), where=KERNEL, title="kernel time, every algorithm"),
    Figure("3_fixed_vs_adaptive", WP, problem_panes(PROBLEMS), where=KERNEL,
           title="work-precision, every algorithm"),
    Figure("4_work_precision", WP, problem_panes(PROBLEMS, transform=plots.best_per_package), where=KERNEL,
           columns=3, title="each package's best algorithm"),
    Figure("5_cards_batch", BATCH, problem_panes(PROBLEMS), where=dict(package="cubie", **KERNEL), per_key=False,
           title="Cubie on both cards: kernel time, every algorithm"),
    Figure("6_fabbri_wp", WP, problem_panes(("fabbri_linder",)), columns=1,
           where=dict(package=("cubie", "myokit_cuda"), **KERNEL), title="Cubie against Myokit"),
    Figure("6_fabbri_batch", BATCH, problem_panes(("fabbri_linder",)), columns=1,
           where=dict(package=("cubie", "myokit_cuda"), **KERNEL), title="Cubie against Myokit"),
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
