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
}


def timed_solve(model, cell_count, rho, dt, step_count, repeats):
    """Warm up, then return (with_transfers_ms, device_only_ms, finals)."""
    initial_states = model.initial_states(cell_count)

    def run():
        return model.solve(
            dt=dt,
            step_count=step_count,
            initial_states=initial_states,
            diffusion_values=rho,
        )

    finals = run()
    elapsed = timeit.repeat(
        run,
        setup="gc.enable()",
        repeat=repeats,
        number=1,
    )

    device_states, device_diffusion = model.to_device(initial_states, rho)
    pristine = device_states.copy()

    def run_on_device():
        return model.solve_on_device(
            dt, step_count, device_states, device_diffusion
        )

    def restore():
        # Untimed: reset the integrated-in-place state between timed runs.
        device_states[...] = pristine

    run_on_device()
    restore()
    elapsed_dev = timeit.repeat(
        run_on_device,
        setup=restore,
        repeat=repeats,
        number=1,
    )
    return min(elapsed) * 1000.0, min(elapsed_dev) * 1000.0, finals


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
        for dt in dts_for(ALGORITHM, problem):
            step_count = int(round(problem["duration"] / dt))
            elapsed_ms, _, finals = timed_solve(
                model,
                cell_count,
                sweep,
                dt,
                step_count,
                repeats=20,
            )
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


def run_problem(problem, cell_count, wp_mode):
    """One timing point or the fixed work-precision sweep for a problem."""
    model = load_model(problem)
    if wp_mode:
        run_work_precision(model, problem, cell_count)
        return

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
        "{0} {1} solves with Myokit-CUDA Euler completed in {2:.1f} ms "
        "({3:.1f} ms without transfers)"
        .format(cell_count, problem.name, elapsed_ms, elapsed_dev_ms)
    )

    timing_file = Path(times_outfile(
        "MYOKIT_CUDA", "Myokit_cuda", "fixed", ALGORITHM, DATASET_KEY, problem
    ))
    with timing_file.open("a", encoding="utf-8") as handle:
        handle.write("{0} {1} {2}\n".format(cell_count, elapsed_ms, elapsed_dev_ms))

    # The pairwise numerical cross-check reads this fixed CSV name.
    if cell_count == 32768:
        numerical_file = (
            Path(data_dir("numerical", DATASET_KEY, REPO_ROOT, problem))
            / "myokit_cuda.csv"
        )
        np.savetxt(numerical_file, finals, delimiter=",")


def main(argv=None):
    """Run a standard timing point or the fixed work-precision sweep."""
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        raise SystemExit(
            "usage: bench_myokit_cuda.py <N>|wp "
            "[algorithm|all] [--problem <name|all>]"
        )
    cell_count, wp_mode, algorithms, problems = parse_bench_args(
        argv, "myokit_cuda"
    )
    if not algorithms:
        print("Myokit CUDA supports forward Euler only; skipping.")
        return 0
    if not problems:
        print("Myokit CUDA runs none of the requested problems; skipping.")
        return 0

    os.chdir(REPO_ROOT)
    for problem in problems:
        run_problem(problem, cell_count, wp_mode)
    return 0


if __name__ == "__main__":
    sys.exit(main())
