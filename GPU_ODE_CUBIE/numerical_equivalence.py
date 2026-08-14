#!/usr/bin/env python
# coding: utf-8
"""Numerical-equivalence (ne) sweeps for cubie.

Two sweeps over every algorithm mutually supported by cubie and
DifferentialEquations.jl (protocol and paths in
runner_scripts/numerical_equivalence/ne_common.py):

* fixed:    error-vs-dt convergence study (fixed step controller) — isolates
  the tableau from the controller. erk-family rows are excluded.
* adaptive: error-vs-tolerance study at atol = rtol over the mutual
  adaptive set (the ``adaptive`` column of algorithms.csv), run up to
  twice per algorithm:
    - "default" tier: cubie's own PI controller defaults — cubie's real
      controller dynamics.
    - "matched" tier: controller constants mirrored from the Julia run's
      resolved defaults (data/numerical_equivalence/julia/
      controller_constants.csv, written by ne_diffeq.jl), so both stacks
      run identical controller type, gains and tolerances. Divergence
      between the two stacks in this tier is the CI-gate signal. When the
      matched settings equal the default tier's, the default results are
      written for the matched file.

Run from the repo root (inside the GPU_ODE_CUBIE venv):
    python GPU_ODE_CUBIE/numerical_equivalence.py [fixed|adaptive|all]
"""

import os
import argparse
import sys

import numpy as np
import cubie as qb
from cubie.integrators.algorithms import resolve_alias
from cubie.time_logger import default_timelogger

default_timelogger.set_verbosity(None)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "runner_scripts"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "runner_scripts",
                                "numerical_equivalence"))
from bench_key import dataset_key
from ne_common import (DTS_NE, TOLS_NE, DT0_NE, DT_MIN_NE, DT_MAX_NE, N_NE,
                       algorithm_names, load_algorithms, load_golden_ne, ensemble_error,
                       load_controller_constants, cubie_ne_file,
                       cubie_ne_adaptive_file, write_ne_csv,
                       write_ne_adaptive_csv, runs_fixed,
                       cubie_default_controller, controllers_equal)

_parser = argparse.ArgumentParser(description="cubie Float32 equivalence sweeps.")
_parser.add_argument("--controller", choices=("fixed", "adaptive", "all"), default="all")
_parser.add_argument("--algorithm", choices=algorithm_names(), default="all")
_args = _parser.parse_args()
MODE = _args.controller
ALGORITHM = _args.algorithm

DATASET_KEY = dataset_key()

precision = np.float32

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

# The golden file's rho column is the float32-rounded grid every consumer
# integrates; cubie's cast to float32 is exact on these values.
golden_rho, golden_states = load_golden_ne()

initial_conditions = {'x': 1.0, 'y': 0.0, 'z': 0.0}
parameters = {'rho': golden_rho}

failures = []


def cubie_is_adaptive(alias):
    """Whether cubie's implementation carries an embedded error estimate."""
    _, tableau = resolve_alias(alias)
    if tableau is not None:
        return tableau.has_error_estimate
    # Bespoke (non-tableau) steps: crank_nicolson derives an embedded
    # estimate; explicit/backwards euler do not.
    return {"crank_nicolson": True}.get(alias, False)


def solve_finals(solver, initials_array, parameter_array):
    """One solve; returns a copied (N_NE, 3) float32 finals array."""
    solution = solver.solve(
        initial_values=initials_array,
        parameters=parameter_array,
        blocksize=64,
        duration=1.0,
    )
    # Copy: the returned array views cubie's output buffer, which the next
    # solve overwrites in place.
    finals = np.array(solution.state[-1, :, :].T, copy=True)
    if finals.dtype != precision:
        raise TypeError("expected float32 output, got {0}"
                        .format(finals.dtype))
    if finals.shape != (N_NE, 3):
        raise ValueError("expected ({0}, 3) finals, got {1}"
                         .format(N_NE, finals.shape))
    return finals


# ---------------------------------------------------------------------------
# Fixed-step error-vs-dt sweep
# ---------------------------------------------------------------------------
if MODE in ("fixed", "all"):
    for row in load_algorithms(ALGORITHM):
        alias = row["cubie_alias"]
        if not runs_fixed(row):
            print("=== fixed {0}: skipped (no fixed sweep for erk) ==="
                  .format(alias))
            continue
        print("=== fixed {0} (order {1}) ===".format(alias, row["order"]))
        solver = qb.Solver(
            lorenz_system,
            algorithm=alias,
            dt=DTS_NE[0],
            save_every=1.0,
            step_controller='fixed',
            output_types=['state'],
            time_logging_level=None,
        )
        initials_array, parameter_array = solver.build_grid(
            initial_values=initial_conditions, parameters=parameters)

        per_dt_finals = []
        for dt in DTS_NE:
            try:
                solver.update(dt=dt)
                finals = solve_finals(solver, initials_array,
                                      parameter_array)
                err = ensemble_error(finals, golden_states)
                print("  dt={0:<12g} err={1:.6e}".format(dt, err))
                per_dt_finals.append((dt, finals))
            except Exception as exc:  # record and continue: broken dt points
                # are findings for the report, not reasons to abort.
                print("  dt={0:<12g} FAILED: {1}: {2}"
                      .format(dt, type(exc).__name__, exc))
                failures.append((alias, dt, "{0}: {1}".format(
                    type(exc).__name__, exc)))

        if per_dt_finals:
            outfile = cubie_ne_file(alias, DATASET_KEY)
            write_ne_csv(outfile, per_dt_finals)
            print("  wrote {0}".format(outfile))
        else:
            print("  no successful dt points; nothing written")

# ---------------------------------------------------------------------------
# Adaptive error-vs-tolerance sweeps (default + matched controller tiers)
# ---------------------------------------------------------------------------
if MODE in ("adaptive", "all"):
    constants = load_controller_constants()

    def matched_controller_settings(alias, order):
        """Cubie controller kwargs mirroring Julia's resolved defaults.

        Julia's PI updates dt*gamma*EEst^(-beta1)*errold^(+beta2); cubie's PI
        gain is safety*EEst^(-kp/(order+1))*errold^(-ki/(order+1)) with order
        the classical order it feeds the exponent, so kp = beta1*(order+1)
        and ki = -beta2*(order+1). qmin/qmax bound the same gain quantity as
        cubie's min_gain/max_gain. Julia's PredictiveController (Radau) maps
        to cubie's gustafsson controller.
        """
        c = constants.get(alias)
        if c is None:
            return None, "no julia controller constants"
        if c["controller"] == "PIController":
            return {
                "step_controller": "pi",
                "kp": c["beta1"] * (order + 1),
                "ki": -c["beta2"] * (order + 1),
                "safety": c["gamma"],
                "min_gain": c["qmin"],
                "max_gain": c["qmax"],
            }, None
        if c["controller"] == "PredictiveController":
            return {
                "step_controller": "gustafsson",
                "safety": c["gamma"],
            }, None
        return None, "unmapped julia controller {0}".format(c["controller"])

    for row in load_algorithms(ALGORITHM):
        alias = row["cubie_alias"]
        if not row["adaptive"]:
            print("=== adaptive {0}: skipped (not in the mutual adaptive "
                  "set) ===".format(alias))
            continue
        if not cubie_is_adaptive(alias):
            raise SystemExit(
                "algorithms.csv marks {0} adaptive but cubie reports no "
                "embedded error estimate; fix the csv".format(alias))
        if alias not in constants:
            print("=== adaptive {0}: skipped (not adaptive in "
                  "OrdinaryDiffEq) ===".format(alias))
            continue

        matched, why_not = matched_controller_settings(alias, row["order"])
        tiers = [("default", {"step_controller": "pi"})]
        matched_reuses_default = False
        if matched is None:
            print("=== adaptive {0}: no matched tier ({1}) ==="
                  .format(alias, why_not))
        elif controllers_equal(matched, cubie_default_controller(
                alias, row["family"], row["order"])):
            # Matched equals default; reuse the default results.
            matched_reuses_default = True
            print("=== adaptive {0}: matched tier equals cubie's defaults; "
                  "reusing the default results ===".format(alias))
        else:
            tiers.append(("matched", matched))

        for tier, controller_settings in tiers:
            print("=== adaptive {0} [{1}] (order {2}) ==="
                  .format(alias, tier, row["order"]))
            try:
                solver = qb.Solver(
                    lorenz_system,
                    algorithm=alias,
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
                    ignored = set(extra) - recognised
                    print("  controller settings applied: {0}{1}".format(
                        sorted(recognised),
                        " (ignored: {0})".format(sorted(ignored))
                        if ignored else ""))
                initials_array, parameter_array = solver.build_grid(
                    initial_values=initial_conditions, parameters=parameters)
            except Exception as exc:
                print("  solver construction FAILED: {0}: {1}"
                      .format(type(exc).__name__, exc))
                failures.append((alias, tier, "{0}: {1}".format(
                    type(exc).__name__, exc)))
                continue

            per_tol = []
            for tol in TOLS_NE:
                try:
                    solver.update(atol=tol, rtol=tol, dt=DT0_NE)
                    finals = solve_finals(solver, initials_array,
                                          parameter_array)
                    err = ensemble_error(finals, golden_states)
                    print("  tol={0:<8g} err={1:.6e}".format(tol, err))
                    per_tol.append((tol, finals, None, None))
                except Exception as exc:
                    print("  tol={0:<8g} FAILED: {1}: {2}"
                          .format(tol, type(exc).__name__, exc))
                    failures.append((alias, tol, "{0} [{1}]: {2}".format(
                        type(exc).__name__, tier, exc)))

            if per_tol:
                outfile = cubie_ne_adaptive_file(alias, tier, DATASET_KEY)
                write_ne_adaptive_csv(outfile, per_tol)
                print("  wrote {0}".format(outfile))
                if tier == "default" and matched_reuses_default:
                    outfile = cubie_ne_adaptive_file(alias, "matched",
                                                     DATASET_KEY)
                    write_ne_adaptive_csv(outfile, per_tol)
                    print("  wrote {0} (copy of default)".format(outfile))
            else:
                print("  no successful tolerance points; nothing written")

if failures:
    print("\n{0} failed points:".format(len(failures)))
    for alias, setting, msg in failures:
        print("  {0} @ {1}: {2}".format(alias, setting, msg))
