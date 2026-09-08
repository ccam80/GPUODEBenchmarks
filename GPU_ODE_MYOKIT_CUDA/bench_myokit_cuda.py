#!/usr/bin/env python3

"""Myokit CUDA Euler ensemble benchmarks, one CellML model per problem with the swept scalar bound to diffusion_current."""

import os
import sys
from pathlib import Path

import numpy as np

from myokit_cuda import MyokitCudaModel


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "runner_scripts"))

from bench_key import data_dir, dataset_key  # noqa: E402
from protocol import TIMING_DT_K  # noqa: E402
from results import Leg  # noqa: E402
from resume import skip_point, skip_wp_leg  # noqa: E402
from wp_common import (  # noqa: E402
    REPEAT_CAP,
    ensemble_error,
    errored_pct,
    load_golden,
    parse_bench_args,
    timed_min_ms,
)


MODELS_DIR = Path(__file__).resolve().parent / "models"
DATASET_KEY = dataset_key()
# The N-sweep steps duration * 2^-timing_k, so 2^timing_k steps keep the span exact.
STANDARD_STEPS = 2 ** TIMING_DT_K
REPEATS = REPEAT_CAP
# Myokit's generated CUDA kernel is forward Euler only.
ALGORITHM = "euler"

# problem -> (CellML component, ordered state variable names)
MODELS = {
    "lorenz": ("lorenz", ("lorenz.x", "lorenz.y", "lorenz.z")),
    "lorenz96": ("lorenz96", tuple(
        "lorenz96.x{0}".format(i) for i in range(1, 33))),
    "pleiades": ("pleiades", tuple(
        "pleiades.{0}{1}".format(prefix, i)
        for prefix in ("x", "y", "u", "v") for i in range(1, 8))),
}


def device_leg(model, initial_states, rho, dt, step_count, repeats,
               on_breach=None):
    """(device_only_ms, samples) on resident inputs with the result left on the device; ms is None on a breach."""
    device_states, device_diffusion = model.to_device(initial_states, rho)
    pristine = device_states.copy()

    def run_on_device():
        return model.solve_on_device(
            dt, step_count, device_states, device_diffusion
        )

    def restore():
        # Untimed: reset the integrated-in-place state between timed runs.
        device_states[...] = pristine

    elapsed_dev_ms, _, samples = timed_min_ms(run_on_device, repeats,
                                              on_breach, setup=restore)
    return elapsed_dev_ms, samples


def timed_solve(model, cell_count, rho, dt, step_count, repeats,
                on_breach=None):
    """(with_transfers_ms, device_only_ms, finals, samples_both, samples_none); NaN times on a breach. on_breach fills the leg before a watchdog hard-exit."""
    initial_states = model.initial_states(cell_count)

    def run():
        return model.solve(
            dt=dt,
            step_count=step_count,
            initial_states=initial_states,
            diffusion_values=rho,
        )

    elapsed_ms, finals, samples_both = timed_min_ms(run, repeats, on_breach)
    if elapsed_ms is None:
        return float("nan"), float("nan"), finals, samples_both, None
    elapsed_dev_ms, samples_none = device_leg(model, initial_states, rho, dt,
                                              step_count, repeats, on_breach)
    if elapsed_dev_ms is None:
        return elapsed_ms, float("nan"), finals, samples_both, samples_none
    return elapsed_ms, elapsed_dev_ms, finals, samples_both, samples_none


def run_work_precision(model, problem, cell_count, leg):
    """Record the fixed-step Myokit-CUDA work-precision sweep, timed on the resident inputs."""
    golden = load_golden(problem)
    sweep = problem.sweep(cell_count, dtype=np.float32)
    initial_states = model.initial_states(cell_count)
    # Later settings are slower, so a breach abandons the leg.
    dts = list(problem.dts(ALGORITHM))
    for index, dt in enumerate(dts):
        step_count = int(round(problem["duration"] / dt))

        def on_breach(rest=dts[index:], at=dt):
            # The hard exit skips the abandon path, so fill it here.
            leg.nan_wp(rest)
            print("WATCHDOG wp fixed dt={0:g}: run never returned"
                  .format(at))

        # One untimed host solve for the finals; the timed leg is device only.
        finals = model.solve(dt=dt, step_count=step_count,
                             initial_states=initial_states,
                             diffusion_values=sweep)
        elapsed_ms, samples = device_leg(model, initial_states, sweep, dt,
                                         step_count, REPEATS, on_breach)
        elapsed_ms = float("nan") if elapsed_ms is None else elapsed_ms
        breached = np.isnan(elapsed_ms)
        if breached:
            print("WATCHDOG wp fixed dt={0:g}: run exceeded the cap"
                  .format(dt))
            error = float("nan")
        else:
            error = ensemble_error(finals, golden)
        pct = errored_pct(finals)
        print(
            "wp fixed dt={0:g}: {1:.2f} ms, err={2:.3e}, errored={3:.1f}%"
            .format(dt, elapsed_ms, error, pct)
        )
        leg.record_wp(dt, elapsed_ms, error, pct, samples=samples)
        if breached:
            leg.nan_wp(dts[index + 1:])
            break


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
    """Compile each problem's model."""
    import timeit

    def warm_one(build_model, row, label):
        started = timeit.default_timer()
        try:
            model = build_model()
            model.solve(
                dt=row.timing_dt,
                step_count=1,
                initial_states=model.initial_states(64),
                diffusion_values=row.sweep(64, dtype=np.float32),
            )
            print("warmed {0} in {1:.1f}s".format(
                label, timeit.default_timer() - started))
        except Exception as exc:
            print("FAILED warm {0}: {1}".format(label, exc))

    for problem in problems:
        if not problem.supports("myokit_cuda"):
            continue
        warm_one(lambda: load_model(problem), problem, problem.name)


def run_problem(problem, cell_counts, wp_mode):
    """The ascending N sweep or the work-precision sweep, on one compiled model."""
    if wp_mode:
        leg = Leg("myokit_cuda", DATASET_KEY, "wp", problem, ALGORITHM,
                  "fixed")
        if skip_wp_leg(leg, problem.dts(ALGORITHM)):
            print("-- resume: skipping wp {0} fixed {1} (already covered)"
                  .format(problem.name, ALGORITHM))
            return
        model = load_model(problem)
        run_work_precision(model, problem, cell_counts[0], leg)
        return

    leg = Leg("myokit_cuda", DATASET_KEY, "times", problem, ALGORITHM,
              "fixed")
    run_counts = [n for n in cell_counts if not skip_point(leg, n)]
    if not run_counts:
        print("-- resume: skipping {0} fixed {1} (already covered)"
              .format(problem.name, ALGORITHM))
        return
    if len(run_counts) < len(cell_counts):
        print("-- resume: {0} fixed {1} runs N={2}".format(
            problem.name, ALGORITHM,
            ",".join(str(n) for n in run_counts)))
    model = load_model(problem)
    for index, cell_count in enumerate(run_counts):
        sweep = problem.sweep(cell_count, dtype=np.float32)
        elapsed_ms, elapsed_dev_ms, finals, samples_both, samples_none = (
            timed_solve(
                model,
                cell_count,
                sweep,
                problem.timing_dt,
                STANDARD_STEPS,
                repeats=REPEATS,
            ))
        print(
            "{0} {1} solves with Myokit-CUDA Euler completed in "
            "{2:.1f} ms ({3:.1f} ms without transfers)"
            .format(cell_count, problem.name, elapsed_ms, elapsed_dev_ms)
        )
        pct = 100.0 if finals is None else errored_pct(finals)
        leg.record_times(cell_count, elapsed_ms, elapsed_dev_ms, pct,
                         samples_both, samples_none)

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
            leg.nan_times(run_counts[index + 1:])
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


def run_states(grid):
    """Runtime-by-states sweep: lorenz96 resized along the requested grid at
    one fixed ensemble size."""
    import tempfile
    import timeit

    from problems import STATES_PROBLEM, states_row
    from wp_common import STATES_N

    # Throwaway CuPy cache dir, set before cupy's first import: compiles run cold.
    os.environ["CUPY_CACHE_DIR"] = tempfile.mkdtemp(prefix="myokit_states_")

    cell_count = STATES_N
    leg = Leg("myokit_cuda", DATASET_KEY, "states", STATES_PROBLEM, ALGORITHM,
              "fixed")
    run_grid = [s for s in grid if not skip_point(leg, cell_count, s)]
    if not run_grid:
        print("-- resume: skipping states fixed {0} (already covered)"
              .format(ALGORITHM))
        return
    for index, nstates in enumerate(run_grid):
        row = states_row(nstates)
        sweep = row.sweep(cell_count, dtype=np.float32)
        elapsed_ms = elapsed_dev_ms = build_s = float("nan")
        finals = samples_both = samples_none = None
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
            elapsed_ms, elapsed_dev_ms, finals, samples_both, samples_none = (
                timed_solve(
                    model,
                    cell_count,
                    sweep,
                    row.timing_dt,
                    STANDARD_STEPS,
                    repeats=REPEATS,
                ))
            print(
                "{0} lorenz96 states={1} solves with Myokit-CUDA Euler "
                "completed in {2:.1f} ms ({3:.1f} ms without transfers)"
                .format(cell_count, nstates, elapsed_ms, elapsed_dev_ms)
            )
        except Exception as exc:
            print("FAILED lorenz96 states={0} fixed {1} N={2}: {3}"
                  .format(nstates, ALGORITHM, cell_count, exc))
        pct = 100.0 if finals is None else errored_pct(finals)
        leg.record_times(cell_count, elapsed_ms, elapsed_dev_ms, pct,
                         samples_both, samples_none, build_s=build_s,
                         states=nstates)
        if not np.isfinite(elapsed_ms) and np.isfinite(build_s):
            # Larger systems are slower, so the sweep is abandoned.
            print("WATCHDOG lorenz96 states={0} fixed {1} N={2}: run "
                  "exceeded the cap".format(nstates, ALGORITHM,
                                            cell_count))
            leg.nan_states(run_grid[index + 1:])
            break


def main(argv=None):
    """Run a standard timing point or the fixed work-precision sweep."""
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        raise SystemExit(
            "usage: bench_myokit_cuda.py <N|N,N,...>|wp "
            "[algorithm|all] [--problem <name|all>] [--mode <fixed|adaptive|all>]"
        )
    cell_counts, analysis, algorithms, problems, modes = parse_bench_args(
        argv, "myokit_cuda"
    )
    if not algorithms or "fixed" not in modes:
        print("Myokit CUDA supports fixed-step forward Euler only; skipping.")
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
        run_states(cell_counts)
        return 0
    for problem in problems:
        if not problem.supports("myokit_cuda"):
            continue
        run_problem(problem, cell_counts, analysis == "wp")
    return 0


if __name__ == "__main__":
    sys.exit(main())
