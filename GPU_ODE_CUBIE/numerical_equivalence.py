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
    - "default" tier: cubie's shipped controller defaults — cubie's real
      controller dynamics.
    - "matched" tier: controller constants mirrored from the Julia run's
      resolved defaults (data/numerical_equivalence/julia/
      controller_constants.csv, written by ne_diffeq.jl), so both stacks
      run identical controller type, gains and tolerances, isolating
      controller-caused error. When the matched settings equal the default
      tier's, the default results are written for the matched file.

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
from cubie_systems import build_system, final_states, output_types
from problems import problem_names, resolve_problems
from ne_common import (TOLS_NE, N_NE, algorithm_names, dts_ne, dt_pins_ne,
                       load_algorithms, load_golden_ne, ensemble_error,
                       load_controller_constants, cubie_ne_file,
                       cubie_ne_adaptive_file, write_ne_csv,
                       write_ne_adaptive_csv, runs_fixed,
                       cubie_default_controller, controllers_equal)

_parser = argparse.ArgumentParser(description="cubie Float32 equivalence sweeps.")
_parser.add_argument("--controller", choices=("fixed", "adaptive", "all"), default="all")
_parser.add_argument("--algorithm", choices=algorithm_names(), default="all")
_parser.add_argument("--problem", default="all",
                     help="all | comma list of " + ", ".join(problem_names()))
_args = _parser.parse_args()
MODE = _args.controller
ALGORITHM = _args.algorithm
PROBLEMS = resolve_problems(_args.problem, "cubie")

DATASET_KEY = dataset_key()

precision = np.float32

failures = []


def cubie_is_adaptive(alias):
    """Whether cubie's implementation carries an embedded error estimate."""
    _, tableau = resolve_alias(alias)
    if tableau is not None:
        return tableau.has_error_estimate
    # Bespoke (non-tableau) steps: crank_nicolson derives an embedded
    # estimate; explicit/backwards euler do not.
    return {"crank_nicolson": True}.get(alias, False)


def solve_finals(solver, initials_array, parameter_array, ctx):
    """One solve; returns a copied (N_NE, states) float32 finals array."""
    solution = solver.solve(
        initial_values=initials_array,
        parameters=parameter_array,
        blocksize=64,
        duration=ctx["duration"],
    )
    # Copy: the returned array views cubie's output buffer, which the next
    # solve overwrites in place.
    finals = np.array(final_states(ctx["system"], solution,
                                   ctx["problem"]), copy=True)
    if finals.dtype != precision:
        raise TypeError("expected float32 output, got {0}"
                        .format(finals.dtype))
    if finals.shape != (N_NE, ctx["nstates"]):
        raise ValueError("expected ({0}, {1}) finals, got {2}"
                         .format(N_NE, ctx["nstates"], finals.shape))
    return finals


def problem_context(problem):
    """System, ensemble grid and golden states for one problem."""
    system, initial_conditions = build_system(problem, precision)
    # The golden file's parameter column is the float32-rounded grid every
    # consumer integrates; cubie's cast to float32 is exact on these values.
    golden_sweep, golden_states = load_golden_ne(problem)
    dt0, dt_min, dt_max = dt_pins_ne(problem)
    return {
        "problem": problem,
        "system": system,
        "initial_conditions": initial_conditions,
        "parameters": {problem["sweep_parameter"]: golden_sweep},
        "golden_states": golden_states,
        "duration": problem["duration"],
        "nstates": problem["states"],
        "dts": dts_ne(problem),
        "dt0": dt0,
        "dt_min": dt_min,
        "dt_max": dt_max,
    }


# ---------------------------------------------------------------------------
# Fixed-step error-vs-dt sweep
# ---------------------------------------------------------------------------
def run_fixed(ctx):
    for row in load_algorithms(ALGORITHM):
        alias = row["cubie_alias"]
        if not runs_fixed(row):
            print("=== fixed {0}: skipped (no fixed sweep for erk) ==="
                  .format(alias))
            continue
        print("=== {0} fixed {1} (order {2}) ==="
              .format(ctx["problem"].name, alias, row["order"]))
        solver = qb.Solver(
            ctx["system"],
            algorithm=alias,
            dt=ctx["dts"][0],
            save_every=ctx["duration"],
            step_controller='fixed',
            output_types=output_types(ctx["system"]),
            time_logging_level=None,
        )
        initials_array, parameter_array = solver.build_grid(
            initial_values=ctx["initial_conditions"],
            parameters=ctx["parameters"])

        per_dt_finals = []
        for dt in ctx["dts"]:
            try:
                solver.update(dt=dt)
                finals = solve_finals(solver, initials_array,
                                      parameter_array, ctx)
                err = ensemble_error(finals, ctx["golden_states"])
                print("  dt={0:<12g} err={1:.6e}".format(dt, err))
                per_dt_finals.append((dt, finals))
            except Exception as exc:  # record and continue: broken dt points
                # are findings for the report, not reasons to abort.
                print("  dt={0:<12g} FAILED: {1}: {2}"
                      .format(dt, type(exc).__name__, exc))
                failures.append((alias, dt, "{0}: {1}".format(
                    type(exc).__name__, exc)))

        if per_dt_finals:
            outfile = cubie_ne_file(alias, DATASET_KEY, ctx["problem"])
            write_ne_csv(outfile, per_dt_finals)
            print("  wrote {0}".format(outfile))
        else:
            print("  no successful dt points; nothing written")

# ---------------------------------------------------------------------------
# Adaptive error-vs-tolerance sweeps (default + matched controller tiers)
# ---------------------------------------------------------------------------
def run_adaptive(ctx):
    constants = load_controller_constants(ctx["problem"])

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
        tiers = [("default", {})]
        matched_reuses_default = False
        shipped = cubie_default_controller(alias, row["family"], row["order"])
        if matched is None:
            print("=== adaptive {0}: no matched tier ({1}) ==="
                  .format(alias, why_not))
        elif (shipped is not None
              and matched["step_controller"] == shipped["step_controller"]
              and controllers_equal(dict(shipped, **matched), shipped)):
            # Matched resolves to the shipped defaults; reuse the results.
            matched_reuses_default = True
            print("=== adaptive {0}: matched tier equals cubie's defaults; "
                  "reusing the default results ===".format(alias))
        else:
            tiers.append(("matched", matched))

        for tier, controller_settings in tiers:
            print("=== {0} adaptive {1} [{2}] (order {3}) ==="
                  .format(ctx["problem"].name, alias, tier, row["order"]))
            try:
                # The default tier passes no controller: cubie as shipped.
                controller_kwargs = {}
                if "step_controller" in controller_settings:
                    controller_kwargs["step_controller"] = (
                        controller_settings["step_controller"])
                solver = qb.Solver(
                    ctx["system"],
                    algorithm=alias,
                    dt=ctx["dt0"],
                    dt_min=ctx["dt_min"],
                    dt_max=ctx["dt_max"],
                    atol=TOLS_NE[0],
                    rtol=TOLS_NE[0],
                    save_every=ctx["duration"],
                    output_types=output_types(ctx["system"]),
                    time_logging_level=None,
                    **controller_kwargs,
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
                    initial_values=ctx["initial_conditions"],
                    parameters=ctx["parameters"])
            except Exception as exc:
                print("  solver construction FAILED: {0}: {1}"
                      .format(type(exc).__name__, exc))
                failures.append((alias, tier, "{0}: {1}".format(
                    type(exc).__name__, exc)))
                continue

            per_tol = []
            for tol in TOLS_NE:
                try:
                    solver.update(atol=tol, rtol=tol, dt=ctx["dt0"])
                    finals = solve_finals(solver, initials_array,
                                          parameter_array, ctx)
                    err = ensemble_error(finals, ctx["golden_states"])
                    print("  tol={0:<8g} err={1:.6e}".format(tol, err))
                    per_tol.append((tol, finals, None, None))
                except Exception as exc:
                    print("  tol={0:<8g} FAILED: {1}: {2}"
                          .format(tol, type(exc).__name__, exc))
                    failures.append((alias, tol, "{0} [{1}]: {2}".format(
                        type(exc).__name__, tier, exc)))

            if per_tol:
                outfile = cubie_ne_adaptive_file(alias, tier, DATASET_KEY,
                                                 ctx["problem"])
                write_ne_adaptive_csv(outfile, per_tol)
                print("  wrote {0}".format(outfile))
                if tier == "default" and matched_reuses_default:
                    outfile = cubie_ne_adaptive_file(alias, "matched",
                                                     DATASET_KEY,
                                                     ctx["problem"])
                    write_ne_adaptive_csv(outfile, per_tol)
                    print("  wrote {0} (copy of default)".format(outfile))
            else:
                print("  no successful tolerance points; nothing written")

if not PROBLEMS:
    print("cubie runs none of the requested problems; skipping.")
    sys.exit(0)

for _problem in PROBLEMS:
    _ctx = problem_context(_problem)
    if MODE in ("fixed", "all"):
        run_fixed(_ctx)
    if MODE in ("adaptive", "all"):
        run_adaptive(_ctx)

if failures:
    print("\n{0} failed points:".format(len(failures)))
    for alias, setting, msg in failures:
        print("  {0} @ {1}: {2}".format(alias, setting, msg))
