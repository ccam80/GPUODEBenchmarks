#!/usr/bin/env python

# Diffrax ensemble benchmarks via vmap: bench_diffrax.py <N>|wp [algorithm|all] [--problem <name|all>]

from collections.abc import Callable
from typing import ClassVar

import diffrax
import jax
import optimistix as optx
import jax.numpy as jnp
import numpy as np
import os
import sys

from diffrax import AbstractERK, ButcherTableau
from diffrax._local_interpolation import ThirdOrderHermitePolynomialInterpolation

# Dataset key ("<os>_<gpu>") so output files are keyed per machine and can be
# additively populated across machines without clobbering each other.
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runner_scripts"))
from algorithms import supported_for
from bench_key import dataset_key, data_dir
from jax_systems import build_problem
from wp_common import (TIMING_TOL, parse_bench_args, timed_min_ms,
                       times_outfile)

DATASET_KEY = dataset_key()

FIXED_ALGORITHMS = supported_for("jax", "fixed")
ADAPTIVE_ALGORITHMS = supported_for("jax", "adaptive")

NS, ANALYSIS, ALGORITHMS, PROBLEMS = parse_bench_args(
    sys.argv[1:], "jax")
# Timed repeats per point; min is reported.
REPEATS = 20

# Persistent XLA compilation cache under the shared generated/ cache root.
# The states sweep stays off it: its build_s is the compile.
if ANALYSIS != "states":
    jax.config.update("jax_compilation_cache_dir", os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "generated", "jax_cache"))
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

# %%


print("Working on :", jax.default_backend())

# This is a GPU benchmark: refuse to silently record CPU timings (e.g. on
# native Windows, where no CUDA jaxlib wheels exist — use Linux or WSL2).
if jax.default_backend() == "cpu":
    print("ERROR: JAX is running on the CPU backend; aborting so CPU "
          "timings are not recorded as GPU results.")
    sys.exit(1)


# %%
# Classical RK4 from the standard tableau; b_error is zeroed, so fixed-step only.
_rk4_tableau = ButcherTableau(
    a_lower=(
        np.array([1 / 2]),
        np.array([0.0, 1 / 2]),
        np.array([0.0, 0.0, 1.0]),
    ),
    b_sol=np.array([1 / 6, 1 / 3, 1 / 3, 1 / 6]),
    b_error=np.zeros(4),
    c=np.array([1 / 2, 1 / 2, 1.0]),
)


class ClassicalRK4(AbstractERK):
    """The classical fourth-order Runge--Kutta method, fixed-step only."""

    tableau: ClassVar[ButcherTableau] = _rk4_tableau
    interpolation_cls: ClassVar[
        Callable[..., ThirdOrderHermitePolynomialInterpolation]
    ] = ThirdOrderHermitePolynomialInterpolation.from_k

    def order(self, terms):
        del terms
        return 4


def make_solver(algorithm, fixed_tol=None):
    """Diffrax solver; a fixed step size leaves nothing for an implicit solver to take its root-finder tolerances from, so fixed_tol supplies them."""
    if algorithm == "euler":
        return diffrax.Euler()
    if algorithm == "classical-rk4":
        return ClassicalRK4()
    if algorithm == "tsit5":
        return diffrax.Tsit5()
    if algorithm == "kvaerno3":
        if fixed_tol is None:
            return diffrax.Kvaerno3()
        return diffrax.Kvaerno3(root_finder=diffrax.VeryChord(
            rtol=fixed_tol, atol=fixed_tol, norm=optx.rms_norm))
    raise ValueError("no diffrax solver for {0}".format(algorithm))


def best_times_ms(solve, args, label, n):
    """Best of REPEATS timed runs in ms as (with_transfers, device_only, abandon); abandon means every larger N is hopeless too."""
    try:
        compiled = solve.lower(args).compile()
    except Exception as err:
        print("FAILED {0} at N={1} ({2}: {3})".format(
            label, n, type(err).__name__, err))
        return float("nan"), float("nan"), True
    usage = compiled.memory_analysis()
    limit = jax.local_devices()[0].memory_stats()["bytes_limit"]
    if usage is not None:
        needed = (usage.temp_size_in_bytes + usage.argument_size_in_bytes
                  + usage.output_size_in_bytes - usage.alias_size_in_bytes)
        print("{0} at N={1}: needs {2:.2f} GiB, device limit {3:.2f} GiB".format(
            label, n, needed / 2**30, limit / 2**30))
        if needed > limit:
            print("ERROR: the {0} solve does not fit in device memory at "
                  "N={1}; no timing recorded.".format(label, n))
            return float("nan"), float("nan"), True

    host_args = np.asarray(jax.device_get(args))

    def with_transfers():
        # jnp.asarray is the h2d, device_get the d2h.
        return jax.device_get(jax.block_until_ready(solve(jnp.asarray(host_args))))

    def device_only():
        # Args already resident, results left on device; block_until_ready only.
        return jax.block_until_ready(solve(args))

    try:
        both, _ = timed_min_ms(with_transfers, REPEATS)
        none = None
        if both is not None:
            none, _ = timed_min_ms(device_only, REPEATS)
        if both is None or none is None:
            print("WATCHDOG {0} at N={1}: run exceeded the cap".format(
                label, n))
            return (float("nan") if both is None else both, float("nan"),
                    True)
    except Exception as err:
        print("FAILED {0} at N={1} ({2}: {3})".format(
            label, n, type(err).__name__, err))
        return float("nan"), float("nan"), False
    return both, none, False


# %%
# JIT-compiled ensemble solves; fixed uses the default ConstantStepSize.
def make_fixed(problem, algorithm, dt0=None, max_steps=4096):
    solver = make_solver(algorithm, fixed_tol=TIMING_TOL)
    vector_field, y0 = build_problem(problem)
    duration = problem["duration"]
    dt0 = problem.timing_dt if dt0 is None else dt0

    @jax.jit
    @jax.vmap
    def main(p):
        terms = diffrax.ODETerm(vector_field(p))
        return diffrax.diffeqsolve(
            terms, solver, 0.0, duration, dt0, y0, max_steps=max_steps)
    return main


def make_adaptive(problem, algorithm, tol=TIMING_TOL, max_steps=65536):
    solver = make_solver(algorithm)
    vector_field, y0 = build_problem(problem)
    duration = problem["duration"]
    dt0 = problem.timing_dt

    @jax.jit
    @jax.vmap
    def main(p):
        terms = diffrax.ODETerm(vector_field(p))
        return diffrax.diffeqsolve(
            terms, solver, 0.0, duration, dt0, y0, max_steps=max_steps,
            stepsize_controller=diffrax.PIDController(rtol=tol, atol=tol))
    return main


# vmap/JIT ordering makes no measurable difference: https://colab.research.google.com/drive/1d7G-O5JX31lHbg7jTzzozbo5-Gp7DBEv


def run_wp(problem, parameterList):
    """dt / tolerance sweep at N = N_WP; see runner_scripts/wp_common.py."""
    from wp_common import (dts_for, TOLS, load_golden, ensemble_error,
                           timed_min_ms, wp_outfile)

    golden = load_golden(problem)

    def bench(m, setting, outfh):
        """Write one row; False when a run breached the watchdog."""
        breached = False
        try:
            t_ms, sol = timed_min_ms(
                lambda: jax.block_until_ready(m(parameterList)), 20)
            if t_ms is None:
                breached = True
                t_ms, err = float("nan"), float("nan")
            else:
                err = ensemble_error(np.array(sol.ys[:, -1, :]), golden)
        except Exception as err_exc:
            print("FAILED wp {0} setting={1:g}: {2}".format(
                problem.name, setting, err_exc))
            t_ms, err = float("nan"), float("nan")
        print("wp {0} setting={1:g}: {2:.2f} ms, err={3:.3e}".format(
            problem.name, setting, t_ms, err))
        outfh.write("{0:.10g} {1} {2:.10e}\n".format(setting, t_ms, err))
        outfh.flush()
        return not breached

    def sweep(outfile, settings, make):
        # Later settings are slower, so a breach abandons the leg as NaN rows.
        with open(outfile, "w") as f:
            breached = False
            for setting in settings:
                if breached:
                    f.write("{0:.10g} nan nan\n".format(setting))
                    continue
                if not bench(make(setting), setting, f):
                    print("WATCHDOG wp {0} setting={1:g}: run exceeded "
                          "the cap".format(problem.name, setting))
                    breached = True

    for algorithm in ALGORITHMS:
        if not problem.runs("jax", algorithm):
            continue
        if algorithm in FIXED_ALGORITHMS:
            # max_steps covers the finest euler dt (2^17 steps).
            sweep(wp_outfile("JAX", "Jax", "fixed", algorithm, DATASET_KEY,
                             problem),
                  dts_for(algorithm, problem),
                  lambda dt: make_fixed(problem, algorithm, dt,
                                        max_steps=262144))
        if algorithm in ADAPTIVE_ALGORITHMS:
            sweep(wp_outfile("JAX", "Jax", "adaptive", algorithm, DATASET_KEY,
                             problem),
                  TOLS,
                  lambda tol: make_adaptive(problem, algorithm, tol))


def run_times(problem):
    """N-sweep: each (algorithm, mode) leg walks the sizes ascending on one jitted ensemble."""
    for algorithm in ALGORITHMS:
        if not problem.runs("jax", algorithm):
            continue
        for mode in ("fixed", "adaptive"):
            supported = (FIXED_ALGORITHMS if mode == "fixed"
                         else ADAPTIVE_ALGORITHMS)
            if algorithm not in supported:
                continue
            main = (make_fixed(problem, algorithm) if mode == "fixed"
                    else make_adaptive(problem, algorithm))
            outfile = times_outfile("JAX", "Jax", mode, algorithm,
                                    DATASET_KEY, problem)
            with open(outfile, "a+") as file:
                for index, n in enumerate(NS):
                    parameterList = jnp.asarray(problem.sweep(n))
                    best_time, best_time_dev, abandon = best_times_ms(
                        main, parameterList,
                        "{0} {1}".format(mode, algorithm), n)
                    print("{:} ODE solves ({}, {}, {}) completed in {:.1f} "
                          "ms ({:.1f} ms without transfers)".format(
                              n, problem.name, algorithm, mode, best_time,
                              best_time_dev))
                    file.write('{0} {1} {2}\n'.format(
                        n, best_time, best_time_dev))
                    file.flush()
                    # The pairwise numerical cross-check reads this fixed CSV name.
                    if (mode == "fixed" and n == 32768
                            and algorithm == "tsit5"
                            and np.isfinite(best_time)):
                        sol = main(parameterList)
                        final_states = np.array(sol.ys[:, -1, :])
                        np.savetxt(os.path.join(
                            data_dir("numerical", DATASET_KEY,
                                     problem=problem),
                            "jax.csv"), final_states, delimiter=',')
                    if abandon:
                        # Larger sizes are slower or bigger, so the leg ends.
                        for rest in NS[index + 1:]:
                            file.write('{0} nan nan\n'.format(rest))
                        file.flush()
                        break


def run_states():
    """Runtime-by-states sweep: lorenz96 resized along STATES_GRID at one
    fixed ensemble size."""
    import timeit

    from problems import states_row
    from wp_common import STATES_N, states_outfile

    n = STATES_N
    grid = NS
    for algorithm in ALGORITHMS:
        for mode in ("fixed", "adaptive"):
            supported = (FIXED_ALGORITHMS if mode == "fixed"
                         else ADAPTIVE_ALGORITHMS)
            if algorithm not in supported:
                continue
            outfile = states_outfile("JAX", "Jax", mode, algorithm,
                                     DATASET_KEY)
            with open(outfile, "w") as file:
                for index, nstates in enumerate(grid):
                    row = states_row(nstates)
                    main = (make_fixed(row, algorithm) if mode == "fixed"
                            else make_adaptive(row, algorithm))
                    parameterList = jnp.asarray(row.sweep(n))
                    label = "states={0} {1} {2}".format(nstates, mode,
                                                        algorithm)
                    try:
                        started = timeit.default_timer()
                        jax.block_until_ready(main(parameterList))
                        build_s = timeit.default_timer() - started
                    except Exception as err:
                        print("FAILED {0} at N={1} ({2}: {3})".format(
                            label, n, type(err).__name__, err))
                        file.write('{0} nan nan nan\n'.format(nstates))
                        file.flush()
                        continue
                    best_time, best_time_dev, abandon = best_times_ms(
                        main, parameterList, label, n)
                    print("{:} ODE solves ({}) completed in {:.1f} ms "
                          "({:.1f} ms without transfers)".format(
                              n, label, best_time, best_time_dev))
                    file.write('{0} {1} {2} {3}\n'.format(
                        nstates, best_time, best_time_dev, build_s))
                    file.flush()
                    if abandon:
                        # Larger systems are slower, so the leg ends.
                        for rest in grid[index + 1:]:
                            file.write('{0} nan nan nan\n'.format(rest))
                        file.flush()
                        break


def run_warm():
    """Compile every timing and wp-setting kernel without running them.
    States-sweep kernels are never warmed: the sweep measures the compile."""
    import timeit

    from wp_common import N_WP, TOLS, dts_for

    counts = NS or [8]

    def warm_one(build, args, label):
        started = timeit.default_timer()
        try:
            build().lower(args).compile()
            print("warmed {0} in {1:.1f}s".format(
                label, timeit.default_timer() - started))
        except Exception as err:
            print("FAILED warm {0} ({1}: {2})".format(
                label, type(err).__name__, err))

    for problem in PROBLEMS:
        for algorithm in ALGORITHMS:
            if not problem.runs("jax", algorithm):
                continue
            if algorithm in FIXED_ALGORITHMS:
                for n in counts:
                    warm_one(lambda: make_fixed(problem, algorithm),
                             jnp.asarray(problem.sweep(n)),
                             f"{problem.name} fixed {algorithm} N={n}")
                wp_args = jnp.asarray(problem.sweep(N_WP))
                for dt in dts_for(algorithm, problem):
                    warm_one(lambda: make_fixed(problem, algorithm, dt,
                                                max_steps=262144),
                             wp_args,
                             f"{problem.name} fixed {algorithm} dt={dt:g}")
            if algorithm in ADAPTIVE_ALGORITHMS:
                for n in counts:
                    warm_one(lambda: make_adaptive(problem, algorithm),
                             jnp.asarray(problem.sweep(n)),
                             f"{problem.name} adaptive {algorithm} N={n}")
                wp_args = jnp.asarray(problem.sweep(N_WP))
                for tol in TOLS:
                    warm_one(lambda: make_adaptive(problem, algorithm, tol),
                             wp_args,
                             f"{problem.name} adaptive {algorithm} "
                             f"tol={tol:g}")


# %%
if ANALYSIS == "warm":
    run_warm()
    sys.exit(0)

if ANALYSIS == "states":
    from problems import STATES_PROBLEM
    if not any(p.name == STATES_PROBLEM for p in PROBLEMS):
        print("diffrax does not run {0}; skipping the states sweep."
              .format(STATES_PROBLEM))
        sys.exit(0)
    run_states()
    sys.exit(0)

if not PROBLEMS:
    print("diffrax runs none of the requested problems; skipping.")
    sys.exit(0)

for _problem in PROBLEMS:
    if ANALYSIS == "wp":
        # Setting up parameters for parallel simulation
        run_wp(_problem, jnp.asarray(_problem.sweep(NS[0])))
    else:
        run_times(_problem)
