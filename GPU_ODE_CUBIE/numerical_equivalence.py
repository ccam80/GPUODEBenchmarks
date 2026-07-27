#!/usr/bin/env python
# coding: utf-8
"""Numerical-equivalence (ne) sweeps for cubie.

Two sweeps over every algorithm mutually supported by cubie and
DifferentialEquations.jl (protocol and paths in
runner_scripts/numerical_equivalence/ne_common.py):

* fixed:    error-vs-dt convergence study (fixed step controller) — isolates
  the tableau from the controller.
* adaptive: error-vs-tolerance study at atol = rtol, run twice per
  algorithm:
    - "default" tier: cubie's own PI controller defaults — cubie's real
      controller dynamics.
    - "matched" tier: controller constants mirrored from the Julia run's
      resolved defaults (data/numerical_equivalence/julia/
      controller_constants.csv, written by ne_diffeq.jl), so both stacks
      run identical controller type, gains and tolerances. Divergence
      between the two stacks in this tier is the CI-gate signal.

Run from the repo root (inside the GPU_ODE_CUBIE venv):
    python GPU_ODE_CUBIE/numerical_equivalence.py [fixed|adaptive|all]
"""

import os
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
                       load_algorithms, load_golden_ne, ensemble_error,
                       load_controller_constants, cubie_ne_file,
                       cubie_ne_adaptive_file, write_ne_csv,
                       write_ne_adaptive_csv)

MODE = sys.argv[1].lower() if len(sys.argv) > 1 else "all"
if MODE not in ("fixed", "adaptive", "all"):
    sys.exit("usage: numerical_equivalence.py [fixed|adaptive|all]")

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

# With a fixed step controller cubie does not derive the inner Newton/Krylov
# tolerances from atol/rtol (that path only runs for adaptive controllers),
# so pin them to match what OrdinaryDiffEq enforces in the paired Julia run.
# Both stacks accept the Newton solve when eta*||dz|| < kappa with
# kappa = 1/100, where ||dz|| scales the Newton update by
# atol + rtol*max(|u_prev|, |u_stage|). The Julia run pins
# abstol_j=1e-6, reltol_j=1e-3 (the OrdinaryDiffEq defaults), and kappa
# is internal to both implementations, so identical inner enforcement is
# simply Julia's own tolerances:
#   newton_atol = abstol_j = 1e-6
#   newton_rtol = reltol_j = 1e-3
# Julia solves its 3x3 linear systems with a dense LU (exact), so cubie's
# matrix-free Krylov tolerances sit 10x below the Newton tolerances to
# make the linear-solve error negligible. Keys not used by an algorithm
# family (e.g. newton_* for explicit steps) are ignored. (In the adaptive
# sweeps cubie derives these tolerances itself as atol/10, rtol/10, so no
# pin is applied there.)
INNER_SOLVER_SETTINGS = {
    "newton_atol": 1e-6,
    "newton_rtol": 1e-3,
    "krylov_atol": 1e-7,
    "krylov_rtol": 1e-4,
}

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
    for row in load_algorithms():
        alias = row["cubie_alias"]
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
        recognised = solver.update(dict(INNER_SOLVER_SETTINGS), silent=True)
        if recognised:
            print("  inner solver settings applied: {0}"
                  .format(sorted(recognised)))
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
        cubie's min_gain/max_gain, and Julia's qsteady deadband acts on
        q = 1/gain, hence the inverted bounds. Julia's PredictiveController
        (Radau) maps to cubie's gustafsson controller — same Gustafsson
        family, matched safety only (documented approximate match).
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
                "deadband_min": 1.0 / c["qsteady_max"],
                "deadband_max": 1.0 / c["qsteady_min"],
            }, None
        if c["controller"] == "PredictiveController":
            # cubie's Gustafsson controller takes the step-size safety factor
            # as `safety` (same as the PI path above); `gamma` is now an
            # overloaded method/tableau coefficient and setting it corrupts
            # the solve. Julia's PredictiveController gamma IS the safety
            # factor, so map it to `safety`.
            return {
                "step_controller": "gustafsson",
                "safety": c["gamma"],
            }, None
        return None, "unmapped julia controller {0}".format(c["controller"])

    for row in load_algorithms():
        alias = row["cubie_alias"]
        if not cubie_is_adaptive(alias):
            print("=== adaptive {0}: skipped (no embedded error estimate "
                  "in cubie) ===".format(alias))
            continue
        if alias not in constants:
            print("=== adaptive {0}: skipped (not adaptive in "
                  "OrdinaryDiffEq) ===".format(alias))
            continue

        matched, why_not = matched_controller_settings(alias, row["order"])
        tiers = [("default", {"step_controller": "pi"})]
        if matched is not None:
            tiers.append(("matched", matched))
        else:
            print("=== adaptive {0}: no matched tier ({1}) ==="
                  .format(alias, why_not))

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
            else:
                print("  no successful tolerance points; nothing written")

if failures:
    print("\n{0} failed points:".format(len(failures)))
    for alias, setting, msg in failures:
        print("  {0} @ {1}: {2}".format(alias, setting, msg))
