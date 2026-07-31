#!/usr/bin/env python3
"""Benchmark Myokit's generated CUDA Euler kernel on the Lorenz ensemble."""

import os
import sys
import timeit
from pathlib import Path

import numpy as np

from myokit_cuda import MyokitCudaModel


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "runner_scripts"))

from bench_key import dataset_key, data_dir  # noqa: E402
from wp_common import (  # noqa: E402
    DTS,
    N_WP,
    ensemble_error,
    load_golden,
    wp_outfile,
)


MODEL_PATH = Path(__file__).resolve().parent / "models" / "lorenz.cellml"
DATASET_KEY = dataset_key()
STANDARD_DT = 0.001
STANDARD_STEPS = 1000
# Timed repeats per point; min is reported.
REPEATS = 20


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


def run_work_precision(model, cell_count):
    """Write the fixed-step Myokit-CUDA work-precision sweep."""
    if cell_count != N_WP:
        raise SystemExit("wp mode must be run with N = {0}".format(N_WP))
    golden = load_golden()
    rho = np.linspace(
        0.0, 21.0, cell_count, dtype=np.float32
    )
    output = wp_outfile(
        "MYOKIT_CUDA",
        "Myokit_cuda",
        "fixed",
        DATASET_KEY,
    )
    with open(output, "w", encoding="utf-8") as handle:
        for dt in DTS:
            step_count = int(round(1.0 / dt))
            elapsed_ms, _, finals = timed_solve(
                model,
                cell_count,
                rho,
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


def main(argv=None):
    """Run a standard timing point or the fixed work-precision sweep."""
    argv = sys.argv[1:] if argv is None else argv
    if not argv or len(argv) > 2:
        raise SystemExit(
            "usage: bench_myokit_cuda.py <trajectory-count> [wp]"
        )
    cell_count = int(argv[0])
    mode = argv[1].lower() if len(argv) == 2 else "timing"
    if mode not in ("timing", "wp"):
        raise SystemExit("second argument must be wp when supplied")

    os.chdir(REPO_ROOT)
    model = MyokitCudaModel(
        MODEL_PATH,
        diffusion_variable="lorenz.rho",
    )
    if model.state_names != (
        "lorenz.x",
        "lorenz.y",
        "lorenz.z",
    ):
        raise RuntimeError(
            "unexpected Lorenz state order: {0}".format(model.state_names)
        )

    if mode == "wp":
        run_work_precision(model, cell_count)
        return 0

    rho = np.linspace(
        0.0, 21.0, cell_count, dtype=np.float32
    )
    elapsed_ms, elapsed_dev_ms, finals = timed_solve(
        model,
        cell_count,
        rho,
        STANDARD_DT,
        STANDARD_STEPS,
        repeats=REPEATS,
    )
    print(
        "{0} ODE solves with Myokit-CUDA Euler completed in {1:.1f} ms "
        "({2:.1f} ms without transfers)"
        .format(cell_count, elapsed_ms, elapsed_dev_ms)
    )

    timing_file = (
        Path(data_dir("MYOKIT_CUDA", DATASET_KEY, REPO_ROOT))
        / "Myokit_cuda_times_unadaptive.txt"
    )
    with timing_file.open("a", encoding="utf-8") as handle:
        handle.write("{0} {1} {2}\n".format(cell_count, elapsed_ms, elapsed_dev_ms))

    if cell_count == N_WP:
        numerical_file = (
            Path(data_dir("numerical", DATASET_KEY, REPO_ROOT))
            / "myokit_cuda.csv"
        )
        np.savetxt(numerical_file, finals, delimiter=",")
    return 0


if __name__ == "__main__":
    sys.exit(main())
