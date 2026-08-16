#!/usr/bin/env python

# Diffrax ensemble benchmarks via vmap: bench_diffrax.py <N> [wp] [algorithm|all] [--problem <name|all>]

from collections.abc import Callable
from typing import ClassVar

import diffrax
import jax
import optimistix as optx
import jax.numpy as jnp
import numpy as np
import os
import timeit
import sys

from diffrax import AbstractERK, ButcherTableau
from diffrax._local_interpolation import ThirdOrderHermitePolynomialInterpolation

# Dataset key ("<os>_<gpu>") so output files are keyed per machine and can be
# additively populated across machines without clobbering each other.
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runner_scripts"))
from algorithms import get_algorithm, supported_for
from bench_key import dataset_key, data_dir
from jax_systems import build_problem
from wp_common import (ADAPTIVE, FIXED_TOL, NEWTON_TOL_FACTOR, TIMING_TOL,
                       parse_bench_args, times_outfile)

DATASET_KEY = dataset_key()

FIXED_ALGORITHMS = supported_for("jax", "fixed")
ADAPTIVE_ALGORITHMS = supported_for("jax", "adaptive")

numberOfParameters, WP_MODE, ALGORITHMS, PROBLEMS = parse_bench_args(
    sys.argv[1:], "jax")
# Timed repeats per point; min is reported.
REPEATS = 20

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


def make_solver(algorithm, tol):
    """Diffrax solver; implicit stages converge to the protocol's Newton tolerance."""
    if algorithm == "euler":
        return diffrax.Euler()
    if algorithm == "classical-rk4":
        return ClassicalRK4()
    if algorithm == "tsit5":
        return diffrax.Tsit5()
    if algorithm == "kvaerno3":
        # A relative tolerance below the working epsilon is unreachable.
        newton_tol = max(tol * NEWTON_TOL_FACTOR,
                         float(np.finfo(np.float32).eps))
        # Newton, matching cubie's stage solver rather than diffrax's chord.
        return diffrax.Kvaerno3(
            root_finder=optx.Newton(rtol=newton_tol, atol=newton_tol))
    raise ValueError("no diffrax solver for {0}".format(algorithm))


def best_times_ms(solve, args, label):
    """Best of REPEATS timed runs in ms as (with_transfers, device_only); exits 1 without recording when the solve does not fit in device memory."""
    try:
        compiled = solve.lower(args).compile()
    except Exception as err:
        print("FAILED {0} at N={1} ({2}: {3})".format(
            label, numberOfParameters, type(err).__name__, err))
        return float("nan"), float("nan")
    usage = compiled.memory_analysis()
    limit = jax.local_devices()[0].memory_stats()["bytes_limit"]
    if usage is not None:
        needed = (usage.temp_size_in_bytes + usage.argument_size_in_bytes
                  + usage.output_size_in_bytes - usage.alias_size_in_bytes)
        print("{0} at N={1}: needs {2:.2f} GiB, device limit {3:.2f} GiB".format(
            label, numberOfParameters, needed / 2**30, limit / 2**30))
        if needed > limit:
            print("ERROR: the {0} solve does not fit in device memory at "
                  "N={1}; no timing recorded.".format(label, numberOfParameters))
            sys.exit(1)

    host_args = np.asarray(jax.device_get(args))

    def with_transfers():
        # jnp.asarray is the h2d, device_get the d2h.
        return jax.device_get(jax.block_until_ready(solve(jnp.asarray(host_args))))

    def device_only():
        # Args already resident, results left on device; block_until_ready only.
        return jax.block_until_ready(solve(args))

    try:
        both = min(timeit.repeat(with_transfers, repeat=REPEATS, number=1)) * 1000
        none = min(timeit.repeat(device_only, repeat=REPEATS, number=1)) * 1000
    except Exception as err:
        print("FAILED {0} at N={1} ({2}: {3})".format(
            label, numberOfParameters, type(err).__name__, err))
        return float("nan"), float("nan")
    return both, none


# %%
# JIT-compiled ensemble solves; fixed uses the default ConstantStepSize.
def make_fixed(problem, algorithm, dt0=None, max_steps=4096):
    solver = make_solver(algorithm, FIXED_TOL)
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


def make_controller(algorithm, tol):
    """Diffrax controller matching the shared adaptive protocol.

    Diffrax scales the gain exponent by ``error_order``, so passing the
    algorithm order plus one reproduces cubie's ``kp / (order + 1)``."""
    return diffrax.PIDController(
        rtol=tol,
        atol=tol,
        icoeff=ADAPTIVE["kp"],
        pcoeff=ADAPTIVE["ki"],
        dcoeff=ADAPTIVE["kd"],
        error_order=get_algorithm(algorithm)["order"] + 1,
        safety=ADAPTIVE["safety"],
        factormin=ADAPTIVE["min_gain"],
        factormax=ADAPTIVE["max_gain"],
        dtmin=ADAPTIVE["dt_min"],
        dtmax=ADAPTIVE["dt_max"],
        norm=optx.rms_norm,
    )


def make_adaptive(problem, algorithm, tol=TIMING_TOL, max_steps=65536):
    solver = make_solver(algorithm, tol)
    vector_field, y0 = build_problem(problem)
    duration = problem["duration"]
    dt0 = problem.timing_dt
    controller = make_controller(algorithm, tol)

    @jax.jit
    @jax.vmap
    def main(p):
        terms = diffrax.ODETerm(vector_field(p))
        return diffrax.diffeqsolve(
            terms, solver, 0.0, duration, dt0, y0, max_steps=max_steps,
            stepsize_controller=controller)
    return main


# vmap/JIT ordering makes no measurable difference: https://colab.research.google.com/drive/1d7G-O5JX31lHbg7jTzzozbo5-Gp7DBEv


def run_wp(problem, parameterList):
    """dt / tolerance sweep at N = N_WP; see runner_scripts/wp_common.py."""
    from wp_common import (dts_for, TOLS, N_WP, load_golden, ensemble_error,
                           wp_outfile)

    if numberOfParameters != N_WP:
        sys.exit("wp mode must be run with N = {0}".format(N_WP))
    golden = load_golden(problem)

    def bench(m, setting, outfh):
        try:
            sol = m(parameterList)
            jax.block_until_ready(sol.ys)  # warm-up (JIT) + numerical result
            err = ensemble_error(np.array(sol.ys[:, -1, :]), golden)
            res = timeit.repeat(
                lambda: jax.block_until_ready(m(parameterList).ys),
                repeat=20, number=1)
            t_ms = min(res) * 1000
        except Exception as err_exc:
            print("FAILED wp {0} setting={1:g}: {2}".format(
                problem.name, setting, err_exc))
            t_ms, err = float("nan"), float("nan")
        print("wp {0} setting={1:g}: {2:.2f} ms, err={3:.3e}".format(
            problem.name, setting, t_ms, err))
        outfh.write("{0:.10g} {1} {2:.10e}\n".format(setting, t_ms, err))

    for algorithm in ALGORITHMS:
        if algorithm in FIXED_ALGORITHMS:
            outfile = wp_outfile("JAX", "Jax", "fixed", algorithm, DATASET_KEY,
                                 problem)
            with open(outfile, "w") as f:
                for dt in dts_for(algorithm, problem):
                    # max_steps covers the finest euler dt (2^17 steps).
                    bench(make_fixed(problem, algorithm, dt,
                                     max_steps=262144), dt, f)
        if algorithm in ADAPTIVE_ALGORITHMS:
            outfile = wp_outfile("JAX", "Jax", "adaptive", algorithm,
                                 DATASET_KEY, problem)
            with open(outfile, "w") as f:
                for tol in TOLS:
                    bench(make_adaptive(problem, algorithm, tol), tol, f)


def run_times(problem, parameterList):
    """N-sweep: use jax.vmap to compute parallel solutions of the ODE."""
    for algorithm in ALGORITHMS:
        if algorithm in FIXED_ALGORITHMS:
            main = make_fixed(problem, algorithm)
            best_time, best_time_dev = best_times_ms(
                main, parameterList, "fixed {0}".format(algorithm))
            print("{:} ODE solves ({}, {}, fixed) completed in {:.1f} ms "
                  "({:.1f} ms without transfers)".format(
                      numberOfParameters, problem.name, algorithm, best_time,
                      best_time_dev))
            outfile = times_outfile("JAX", "Jax", "fixed", algorithm,
                                    DATASET_KEY, problem)
            with open(outfile, "a+") as file:
                file.write('{0} {1} {2}\n'.format(
                    numberOfParameters, best_time, best_time_dev))
            # The pairwise numerical cross-check reads this fixed CSV name.
            if (numberOfParameters == 32768 and algorithm == "tsit5"
                    and np.isfinite(best_time)):
                sol = main(parameterList)
                final_states = np.array(sol.ys[:, -1, :])
                np.savetxt(os.path.join(
                    data_dir("numerical", DATASET_KEY, problem=problem),
                    "jax.csv"), final_states, delimiter=',')

        if algorithm in ADAPTIVE_ALGORITHMS:
            main = make_adaptive(problem, algorithm)
            best_time, best_time_dev = best_times_ms(
                main, parameterList, "adaptive {0}".format(algorithm))
            print("{:} ODE solves ({}, {}, adaptive) completed in {:.1f} ms "
                  "({:.1f} ms without transfers)".format(
                      numberOfParameters, problem.name, algorithm, best_time,
                      best_time_dev))
            outfile = times_outfile("JAX", "Jax", "adaptive", algorithm,
                                    DATASET_KEY, problem)
            with open(outfile, "a+") as file:
                file.write('{0} {1} {2}\n'.format(
                    numberOfParameters, best_time, best_time_dev))


# %%
if not PROBLEMS:
    print("diffrax runs none of the requested problems; skipping.")
    sys.exit(0)

for _problem in PROBLEMS:
    # Setting up parameters for parallel simulation
    _parameters = jnp.asarray(_problem.sweep(numberOfParameters))
    if WP_MODE:
        run_wp(_problem, _parameters)
    else:
        run_times(_problem, _parameters)
