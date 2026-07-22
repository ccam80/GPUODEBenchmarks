#!/usr/bin/env python
# coding: utf-8
"""Regenerate ONLY the radau_iia_5 adaptive numerical-equivalence CSVs.

Focused counterpart to numerical_equivalence.py: reruns radau_iia_5's
adaptive sweep (default + matched controller tiers) over the tolerance grid
and overwrites just that algorithm's cubie adaptive outputs, so the radau
values can be swapped into the dataset without rerunning the whole suite.

radau_iia_5 maps Julia's PredictiveController to cubie's Gustafsson
controller. cubie's Gustafsson safety factor is set via `safety` (the
former `gamma` key is now an overloaded method coefficient — setting it
corrupts the solve), so the matched tier mirrors Julia's gamma onto
`safety`, identical to numerical_equivalence.py.

Run from the repo root inside the GPU_ODE_CUBIE venv (numba-cuda backend,
the canonical committed dataset), or the MLIR venv with
CUBIE_CUDA_BACKEND=mlir:
    python GPU_ODE_CUBIE/numerical_equivalence_radau.py
"""

import os
import sys

import numpy as np
import cubie as qb

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# ne_common resolves data/ paths relative to the working directory.
os.chdir(_REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "runner_scripts"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "runner_scripts",
                                "numerical_equivalence"))
from bench_key import dataset_key
from ne_common import (TOLS_NE, DT0_NE, DT_MIN_NE, DT_MAX_NE, N_NE,
                       load_golden_ne, ensemble_error, load_controller_constants,
                       cubie_ne_adaptive_file, write_ne_adaptive_csv)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ALIAS = "radau_iia_5"
ORDER = 5  # classical order, from algorithms.csv
precision = np.float32
DATASET_KEY = dataset_key()

lorenz_system = qb.create_ODE_system(
    """
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    """,
    states={'x': 1.0, 'y': 0.0, 'z': 0.0},
    parameters={'rho': 21.0},
    constants={'sigma': 10.0, 'beta': 8.0 / 3.0},
    name="Lorenz",
    precision=precision,
)

golden_rho, golden_states = load_golden_ne()
initial_conditions = {'x': 1.0, 'y': 0.0, 'z': 0.0}
parameters = {'rho': golden_rho}


def solve_finals(solver, initials_array, parameter_array):
    """One solve; returns a copied (N_NE, 3) float32 finals array."""
    solution = solver.solve(
        initial_values=initials_array,
        parameters=parameter_array,
        blocksize=64,
        duration=1.0,
    )
    finals = np.array(solution.state[-1, :, :].T, copy=True)
    if finals.shape != (N_NE, 3):
        raise ValueError("expected ({0}, 3) finals, got {1}"
                         .format(N_NE, finals.shape))
    return finals


def matched_controller_settings():
    """Cubie controller kwargs mirroring Julia's resolved radau defaults.

    Julia's PredictiveController (Radau) maps to cubie's Gustafsson
    controller; the safety factor is Julia's gamma, passed via `safety`.
    """
    c = load_controller_constants().get(ALIAS)
    if c is None:
        raise SystemExit("no julia controller constants for {0}".format(ALIAS))
    if c["controller"] != "PredictiveController":
        raise SystemExit("{0} controller is {1}, expected PredictiveController"
                         .format(ALIAS, c["controller"]))
    return {"step_controller": "gustafsson", "safety": c["gamma"]}


tiers = [
    ("default", {"step_controller": "pi"}),
    ("matched", matched_controller_settings()),
]

for tier, controller_settings in tiers:
    print("=== adaptive {0} [{1}] (order {2}) ==="
          .format(ALIAS, tier, ORDER))
    solver = qb.Solver(
        lorenz_system,
        algorithm=ALIAS,
        dt=DT0_NE,
        dt_min=DT_MIN_NE,
        dt_max=DT_MAX_NE,
        atol=TOLS_NE[0],
        rtol=TOLS_NE[0],
        save_every=1.0,
        step_controller=controller_settings["step_controller"],
        output_types=['state'],
        time_logging_level=None,
    )
    extra = {k: v for k, v in controller_settings.items()
             if k != "step_controller"}
    if extra:
        recognised = solver.update(extra, silent=True)
        ignored = set(extra) - set(recognised)
        print("  controller settings applied: {0}{1}".format(
            sorted(recognised),
            " (ignored: {0})".format(sorted(ignored)) if ignored else ""))
        if ignored:
            raise SystemExit("controller settings ignored: {0}".format(
                sorted(ignored)))
    initials_array, parameter_array = solver.build_grid(
        initial_values=initial_conditions, parameters=parameters)

    per_tol = []
    for tol in TOLS_NE:
        solver.update(atol=tol, rtol=tol, dt=DT0_NE)
        finals = solve_finals(solver, initials_array, parameter_array)
        err = ensemble_error(finals, golden_states)
        print("  tol={0:<8g} err={1:.6e}".format(tol, err))
        per_tol.append((tol, finals, None, None))

    outfile = cubie_ne_adaptive_file(ALIAS, tier, DATASET_KEY)
    write_ne_adaptive_csv(outfile, per_tol)
    print("  wrote {0}".format(outfile))
