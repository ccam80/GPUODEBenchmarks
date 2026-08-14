#!/usr/bin/env python
# coding: utf-8
# %%
# Benchmarking Diffrax ODE solvers for ensemble problems via vmap, once per
# algorithm: euler/classical-rk4/tsit5 fixed, tsit5 adaptive (PIDController).
# Usage: bench_diffrax.py <N> [wp] [algorithm|all]

# Created By: Utkarsh
# Last Updated: 19 April 2023


# %%
from collections.abc import Callable
from typing import ClassVar

import diffrax
import equinox as eqx
import jax
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
from bench_key import dataset_key, data_dir
from wp_common import parse_bench_args, times_outfile

DATASET_KEY = dataset_key()

FIXED_ALGORITHMS = ("euler", "classical-rk4", "tsit5")
ADAPTIVE_ALGORITHMS = ("tsit5",)
SUPPORTED = ("euler", "classical-rk4", "tsit5")

numberOfParameters, WP_MODE, ALGORITHMS = parse_bench_args(sys.argv[1:], SUPPORTED)
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


def make_solver(algorithm):
    if algorithm == "euler":
        return diffrax.Euler()
    if algorithm == "classical-rk4":
        return ClassicalRK4()
    if algorithm == "tsit5":
        return diffrax.Tsit5()
    raise ValueError("no diffrax solver for {0}".format(algorithm))


def best_times_ms(solve, args, label):
    """Best of REPEATS timed runs in ms as (with_transfers, device_only). Exits
    1 without recording if the compiled solve does not fit in device memory."""
    compiled = solve.lower(args).compile()
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
        print("ERROR: the {0} solve failed at N={1} ({2}: {3}); no timing "
              "recorded.".format(label, numberOfParameters,
                                 type(err).__name__, err))
        sys.exit(1)
    return both, none


# %%
# Defining the Lorenz Problem
class Lorenz(eqx.Module):
    k1: float

    def __call__(self, t, y, args):
        f0 = 10.0*(y[1] - y[0])
        f1 = self.k1 * y[0] - y[1] - y[0] * y[2]
        f2 = y[0] * y[1] - (8/3)*y[2]
        return jnp.stack([f0, f1, f2])


# %%
# JIT-compiled ensemble solves; fixed uses the default ConstantStepSize.
def make_fixed(algorithm, dt0=2.0 ** -10, max_steps=4096):
    solver = make_solver(algorithm)

    @jax.jit
    @jax.vmap
    def main(k1):
        lorenz = Lorenz(k1)
        terms = diffrax.ODETerm(lorenz)
        return diffrax.diffeqsolve(
            terms, solver, 0.0, 1.0, dt0,
            jnp.array([1.0, 0.0, 0.0]), max_steps=max_steps)
    return main


def make_adaptive(algorithm, tol=1e-8, max_steps=65536):
    solver = make_solver(algorithm)

    @jax.jit
    @jax.vmap
    def main(k1):
        lorenz = Lorenz(k1)
        terms = diffrax.ODETerm(lorenz)
        return diffrax.diffeqsolve(
            terms, solver, 0.0, 1.0, 0.001,
            jnp.array([1.0, 0.0, 0.0]), max_steps=max_steps,
            stepsize_controller=diffrax.PIDController(rtol=tol, atol=tol))
    return main


# %%
# Setting up parameters for parallel simulation
parameterList = jnp.linspace(0.0,21.0,numberOfParameters)

# Test that vmap and JIT ordering does not make a noticeable difference:
# https://colab.research.google.com/drive/1d7G-O5JX31lHbg7jTzzozbo5-Gp7DBEv?usp=sharing

# ========================================
# WORK-PRECISION (wp) MODE
# ========================================
# Sweeps dt / tolerance per algorithm at N=131072; see runner_scripts/wp_common.py.
# wp timings block_until_ready so the full solve is measured.
if WP_MODE:
    from wp_common import (dts_for, TOLS, N_WP, load_golden, ensemble_error,
                           wp_outfile)

    if numberOfParameters != N_WP:
        sys.exit("wp mode must be run with N = {0}".format(N_WP))
    golden = load_golden()

    def bench(m, setting, outfh):
        sol = m(parameterList)
        jax.block_until_ready(sol.ys)  # warm-up (JIT) + numerical result
        err = ensemble_error(np.array(sol.ys[:, -1, :]), golden)
        res = timeit.repeat(
            lambda: jax.block_until_ready(m(parameterList).ys),
            repeat=20, number=1)
        t_ms = min(res) * 1000
        print("wp setting={0:g}: {1:.2f} ms, err={2:.3e}".format(
            setting, t_ms, err))
        outfh.write("{0:.10g} {1} {2:.10e}\n".format(setting, t_ms, err))

    for algorithm in ALGORITHMS:
        if algorithm in FIXED_ALGORITHMS:
            outfile = wp_outfile("JAX", "Jax", "fixed", algorithm, DATASET_KEY)
            with open(outfile, "w") as f:
                for dt in dts_for(algorithm):
                    # max_steps covers the finest euler dt (2^17 steps).
                    bench(make_fixed(algorithm, dt, max_steps=262144), dt, f)
        if algorithm in ADAPTIVE_ALGORITHMS:
            outfile = wp_outfile("JAX", "Jax", "adaptive", algorithm,
                                 DATASET_KEY)
            with open(outfile, "w") as f:
                for tol in TOLS:
                    bench(make_adaptive(algorithm, tol), tol, f)

    sys.exit(0)

# %%
# N-sweep: use jax.vmap to compute parallel solutions of the ODE.
for algorithm in ALGORITHMS:
    if algorithm in FIXED_ALGORITHMS:
        main = make_fixed(algorithm)
        best_time, best_time_dev = best_times_ms(
            main, parameterList, "fixed {0}".format(algorithm))
        print("{:} ODE solves ({}, fixed) completed in {:.1f} ms "
              "({:.1f} ms without transfers)".format(
                  numberOfParameters, algorithm, best_time, best_time_dev))
        outfile = times_outfile("JAX", "Jax", "fixed", algorithm, DATASET_KEY)
        with open(outfile, "a+") as file:
            file.write('{0} {1} {2}\n'.format(
                numberOfParameters, best_time, best_time_dev))
        # The pairwise numerical cross-check reads this fixed CSV name.
        if numberOfParameters == 32768 and algorithm == "tsit5":
            sol = main(parameterList)
            final_states = np.array(sol.ys[:, -1, :])
            np.savetxt(os.path.join(data_dir("numerical", DATASET_KEY), "jax.csv"),
                       final_states, delimiter=',')

    if algorithm in ADAPTIVE_ALGORITHMS:
        main = make_adaptive(algorithm)
        best_time, best_time_dev = best_times_ms(
            main, parameterList, "adaptive {0}".format(algorithm))
        print("{:} ODE solves ({}, adaptive) completed in {:.1f} ms "
              "({:.1f} ms without transfers)".format(
                  numberOfParameters, algorithm, best_time, best_time_dev))
        outfile = times_outfile("JAX", "Jax", "adaptive", algorithm,
                                DATASET_KEY)
        with open(outfile, "a+") as file:
            file.write('{0} {1} {2}\n'.format(
                numberOfParameters, best_time, best_time_dev))
