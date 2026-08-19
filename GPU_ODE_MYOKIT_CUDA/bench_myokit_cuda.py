#!/usr/bin/env python3

"""Myokit CUDA Euler ensemble benchmarks, one CellML model per problem with the swept scalar bound to diffusion_current."""

import os
import sys
import timeit
from pathlib import Path

import numpy as np

from myokit_cuda import MyokitCudaModel


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "runner_scripts"))

from bench_key import data_dir, dataset_key  # noqa: E402
from wp_common import (  # noqa: E402
    WATCHDOG_SECONDS,
    dts_for,
    ensemble_error,
    load_golden,
    parse_bench_args,
    times_outfile,
    wp_outfile,
)


MODELS_DIR = Path(__file__).resolve().parent / "models"
DATASET_KEY = dataset_key()
# The N-sweep steps duration * 2^-10, so 1024 steps keep the span exact.
STANDARD_STEPS = 1024
# Timed repeats per point; min is reported.
REPEATS = 20
# Myokit's generated CUDA kernel is forward Euler only.
ALGORITHM = "euler"

# problem -> (CellML component, ordered state variable names)
MODELS = {
    "lorenz": ("lorenz", ("lorenz.x", "lorenz.y", "lorenz.z")),
    "lorenz96": ("lorenz96", tuple(
        "lorenz96.x{0}".format(i) for i in range(1, 41))),
    "pleiades": ("pleiades", tuple(
        "pleiades.{0}{1}".format(prefix, i)
        for prefix in ("x", "y", "u", "v") for i in range(1, 8))),
}


def _capped_min_ms(run, repeats, setup=None):
    """Best-of-repeats after one warm-up; (ms, first_result), ms None on breach."""
    best = None
    first = None
    for attempt in range(repeats + 1):
        if setup is not None and attempt:
            setup()
        started = timeit.default_timer()
        result = run()
        elapsed = timeit.default_timer() - started
        if attempt == 0:
            first = result
        if elapsed > WATCHDOG_SECONDS:
            return None, first
        if attempt and (best is None or elapsed < best):
            best = elapsed
    return best * 1000.0, first


def timed_solve(model, cell_count, rho, dt, step_count, repeats):
    """(with_transfers_ms, device_only_ms, finals); NaN times on a breach."""
    initial_states = model.initial_states(cell_count)

    def run():
        return model.solve(
            dt=dt,
            step_count=step_count,
            initial_states=initial_states,
            diffusion_values=rho,
        )

    elapsed_ms, finals = _capped_min_ms(run, repeats)
    if elapsed_ms is None:
        return float("nan"), float("nan"), finals

    device_states, device_diffusion = model.to_device(initial_states, rho)
    pristine = device_states.copy()

    def run_on_device():
        return model.solve_on_device(
            dt, step_count, device_states, device_diffusion
        )

    def restore():
        # Untimed: reset the integrated-in-place state between timed runs.
        device_states[...] = pristine

    elapsed_dev_ms, _ = _capped_min_ms(run_on_device, repeats, setup=restore)
    if elapsed_dev_ms is None:
        return elapsed_ms, float("nan"), finals
    return elapsed_ms, elapsed_dev_ms, finals


def run_work_precision(model, problem, cell_count):
    """Write the fixed-step Myokit-CUDA work-precision sweep."""
    golden = load_golden(problem)
    sweep = problem.sweep(cell_count, dtype=np.float32)
    output = wp_outfile(
        "MYOKIT_CUDA",
        "Myokit_cuda",
        "fixed",
        ALGORITHM,
        DATASET_KEY,
        problem,
    )
    with open(output, "w", encoding="utf-8") as handle:
        # Later settings are slower, so a breach abandons the leg.
        breached = False
        for dt in dts_for(ALGORITHM, problem):
            if breached:
                handle.write("{0:.10g} nan nan\n".format(dt))
                continue
            step_count = int(round(problem["duration"] / dt))
            elapsed_ms, _, finals = timed_solve(
                model,
                cell_count,
                sweep,
                dt,
                step_count,
                repeats=20,
            )
            if np.isnan(elapsed_ms):
                print("WATCHDOG wp fixed dt={0:g}: run exceeded the cap"
                      .format(dt))
                breached = True
                error = float("nan")
            else:
                error = ensemble_error(finals, golden)
            print(
                "wp fixed dt={0:g}: {1:.2f} ms, err={2:.3e}"
                .format(dt, elapsed_ms, error)
            )
            handle.write(
                "{0:.10g} {1} {2:.10e}\n".format(
                    dt, elapsed_ms, error
                )
            )
            handle.flush()


def load_model(problem):
    """Build the CUDA model for a problem and check its state order."""
    name = problem["problem"]
    if name not in MODELS:
        raise SystemExit(
            "no Myokit CellML model for problem '{0}'".format(name)
        )
    component, states = MODELS[name]
    model = MyokitCudaModel(
        MODELS_DIR / "{0}.cellml".format(name),
        diffusion_variable="{0}.{1}".format(
            component, problem["sweep_parameter"]
        ),
    )
    if model.state_names != states:
        raise RuntimeError(
            "unexpected {0} state order: {1}".format(name, model.state_names)
        )
    return model


def run_warm(problems):
    """Compile each requested model once so CuPy's kernel cache is hot."""
    import timeit

    for problem in problems:
        if not problem.runs("myokit_cuda", ALGORITHM):
            continue
        started = timeit.default_timer()
        try:
            model = load_model(problem)
            model.solve(
                dt=problem.timing_dt,
                step_count=1,
                initial_states=model.initial_states(64),
                diffusion_values=problem.sweep(64, dtype=np.float32),
            )
            print("warmed {0} in {1:.1f}s".format(
                problem.name, timeit.default_timer() - started))
        except Exception as exc:
            print("FAILED warm {0}: {1}".format(problem.name, exc))


def run_problem(problem, cell_counts, wp_mode):
    """The ascending N sweep or the work-precision sweep, on one compiled model."""
    model = load_model(problem)
    if wp_mode:
        run_work_precision(model, problem, cell_counts[0])
        return

    timing_file = Path(times_outfile(
        "MYOKIT_CUDA", "Myokit_cuda", "fixed", ALGORITHM, DATASET_KEY, problem
    ))
    with timing_file.open("a", encoding="utf-8") as handle:
        for index, cell_count in enumerate(cell_counts):
            sweep = problem.sweep(cell_count, dtype=np.float32)
            elapsed_ms, elapsed_dev_ms, finals = timed_solve(
                model,
                cell_count,
                sweep,
                problem.timing_dt,
                STANDARD_STEPS,
                repeats=REPEATS,
            )
            print(
                "{0} {1} solves with Myokit-CUDA Euler completed in "
                "{2:.1f} ms ({3:.1f} ms without transfers)"
                .format(cell_count, problem.name, elapsed_ms, elapsed_dev_ms)
            )
            handle.write("{0} {1} {2}\n".format(
                cell_count, elapsed_ms, elapsed_dev_ms))
            handle.flush()

            # The pairwise numerical cross-check reads this fixed CSV name.
            if cell_count == 32768 and np.isfinite(elapsed_ms):
                numerical_file = (
                    Path(data_dir("numerical", DATASET_KEY, REPO_ROOT,
                                  problem))
                    / "myokit_cuda.csv"
                )
                np.savetxt(numerical_file, finals, delimiter=",")

            if not np.isfinite(elapsed_ms):
                # Larger sizes are slower, so the sweep is abandoned.
                print("WATCHDOG {0} fixed {1} N={2}: run exceeded the cap"
                      .format(problem.name, ALGORITHM, cell_count))
                for rest in cell_counts[index + 1:]:
                    handle.write("{0} nan nan\n".format(rest))
                handle.flush()
                break


def _lorenz96_cellml(n):
    """Write and return the path of a cyclic n-state lorenz96 CellML model."""
    outdir = MODELS_DIR / "generated"
    outdir.mkdir(exist_ok=True)
    path = outdir / "lorenz96_{0}.cellml".format(n)
    variables = "\n".join(
        '    <variable name="x{0}" units="dimensionless" '
        'initial_value="{1}"/>'.format(i, 9 if i == 1 else 8)
        for i in range(1, n + 1))
    rows = []
    for i in range(1, n + 1):
        ip1 = i % n + 1
        im1 = (i - 2) % n + 1
        im2 = (i - 3) % n + 1
        rows.append(
            "      <apply><eq/><apply><diff/><bvar><ci>time</ci></bvar>"
            "<ci>x{0}</ci></apply><apply><plus/><apply><minus/><apply>"
            "<times/><apply><minus/><ci>x{1}</ci><ci>x{2}</ci></apply>"
            "<ci>x{3}</ci></apply><ci>x{0}</ci></apply><ci>F</ci></apply>"
            "</apply>".format(i, ip1, im2, im1))
    path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<model\n    name="lorenz96"\n'
        '    xmlns="http://www.cellml.org/cellml/1.1#"\n'
        '    xmlns:cellml="http://www.cellml.org/cellml/1.1#">\n'
        '  <component name="environment">\n'
        '    <variable\n        name="time"\n'
        '        units="dimensionless"\n'
        '        public_interface="out"/>\n'
        '  </component>\n\n'
        '  <component name="lorenz96">\n'
        '    <variable\n        name="time"\n'
        '        units="dimensionless"\n'
        '        public_interface="in"/>\n'
        + variables + '\n'
        '    <variable name="F" units="dimensionless" initial_value="8"/>\n\n'
        '    <math xmlns="http://www.w3.org/1998/Math/MathML">\n'
        + "\n".join(rows) + '\n'
        '    </math>\n'
        '  </component>\n\n'
        '  <connection>\n'
        '    <map_components component_1="environment" '
        'component_2="lorenz96"/>\n'
        '    <map_variables variable_1="time" variable_2="time"/>\n'
        '  </connection>\n'
        '</model>\n',
        encoding="utf-8")
    return path


def run_states(cell_count):
    """Runtime-by-states sweep: lorenz96 resized along STATES_GRID at one
    fixed ensemble size."""
    import timeit

    from problems import states_row
    from wp_common import STATES_GRID, states_outfile

    outfile = Path(states_outfile(
        "MYOKIT_CUDA", "Myokit_cuda", "fixed", ALGORITHM, DATASET_KEY))
    with outfile.open("w", encoding="utf-8") as handle:
        for index, nstates in enumerate(STATES_GRID):
            row = states_row(nstates)
            sweep = row.sweep(cell_count, dtype=np.float32)
            elapsed_ms = elapsed_dev_ms = build_s = float("nan")
            try:
                started = timeit.default_timer()
                model = MyokitCudaModel(
                    _lorenz96_cellml(nstates),
                    diffusion_variable="lorenz96.F",
                )
                model.solve(
                    dt=row.timing_dt,
                    step_count=1,
                    initial_states=model.initial_states(cell_count),
                    diffusion_values=sweep,
                )
                build_s = timeit.default_timer() - started
                elapsed_ms, elapsed_dev_ms, _ = timed_solve(
                    model,
                    cell_count,
                    sweep,
                    row.timing_dt,
                    STANDARD_STEPS,
                    repeats=REPEATS,
                )
                print(
                    "{0} lorenz96 states={1} solves with Myokit-CUDA Euler "
                    "completed in {2:.1f} ms ({3:.1f} ms without transfers)"
                    .format(cell_count, nstates, elapsed_ms, elapsed_dev_ms)
                )
            except Exception as exc:
                print("FAILED lorenz96 states={0} fixed {1} N={2}: {3}"
                      .format(nstates, ALGORITHM, cell_count, exc))
            handle.write("{0} {1} {2} {3}\n".format(
                nstates, elapsed_ms, elapsed_dev_ms, build_s))
            handle.flush()
            if not np.isfinite(elapsed_ms) and np.isfinite(build_s):
                # Larger systems are slower, so the sweep is abandoned.
                print("WATCHDOG lorenz96 states={0} fixed {1} N={2}: run "
                      "exceeded the cap".format(nstates, ALGORITHM,
                                                cell_count))
                for rest in STATES_GRID[index + 1:]:
                    handle.write("{0} nan nan nan\n".format(rest))
                handle.flush()
                break


def main(argv=None):
    """Run a standard timing point or the fixed work-precision sweep."""
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        raise SystemExit(
            "usage: bench_myokit_cuda.py <N|N,N,...>|wp "
            "[algorithm|all] [--problem <name|all>]"
        )
    cell_counts, analysis, algorithms, problems = parse_bench_args(
        argv, "myokit_cuda"
    )
    if not algorithms:
        print("Myokit CUDA supports forward Euler only; skipping.")
        return 0
    if not problems:
        print("Myokit CUDA runs none of the requested problems; skipping.")
        return 0

    os.chdir(REPO_ROOT)
    if analysis == "warm":
        run_warm(problems)
        return 0
    if analysis == "states":
        from problems import STATES_PROBLEM
        if not any(p.name == STATES_PROBLEM for p in problems):
            print("Myokit CUDA does not run {0}; skipping the states sweep."
                  .format(STATES_PROBLEM))
            return 0
        run_states(cell_counts[0])
        return 0
    for problem in problems:
        if not problem.runs("myokit_cuda", ALGORITHM):
            continue
        run_problem(problem, cell_counts, analysis == "wp")
    return 0


if __name__ == "__main__":
    sys.exit(main())
