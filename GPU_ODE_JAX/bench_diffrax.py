#!/usr/bin/env python
# coding: utf-8
# %%
# Benchmarking Diffrax ODE solvers for ensemble problems, via vmap. The Lorenz ODE is integrated by Tsit5.

# Created By: Utkarsh
# Last Updated: 19 April 2023


# %%
import time

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import os
import timeit
import sys

numberOfParameters = int(sys.argv[1])

# Dataset key ("<os>_<gpu>") keys output files per machine.
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runner_scripts"))
from bench_key import dataset_key, data_dir
# Timed repeats per point (min is reported) and the shared solver settings.
from protocol import PERF_ADAPTIVE_TOL, PERF_FIXED_DT, REPEATS
DATASET_KEY = dataset_key()

# %%


print("Working on :", jax.default_backend())

# This is a GPU benchmark: refuse to silently record CPU timings (e.g. on
# native Windows, where no CUDA jaxlib wheels exist — use Linux or WSL2).
if jax.default_backend() == "cpu":
    print("ERROR: JAX is running on the CPU backend; aborting so CPU "
          "timings are not recorded as GPU results.")
    sys.exit(1)


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
# JIT compilation of ODE solver
@jax.jit
@jax.vmap
def main(k1):
    lorenz = Lorenz(k1)
    terms = diffrax.ODETerm(lorenz)
    t0 = 0.0
    t1 = 1.0
    y0 = jnp.array([1.0, 0.0, 0.0])
    dt0 = PERF_FIXED_DT
    solver = diffrax.Tsit5()
    saveat = diffrax.SaveAt(ts = jnp.array([t0,t1]))
    sol = diffrax.diffeqsolve(
        terms,
        solver,
        t0,
        t1,
        dt0,
        y0,
    )
    return sol

# %%
# Setting up parameters for parallel simulation
parameterList = jnp.linspace(0.0,21.0,numberOfParameters)

# Test that vmap and JIT ordering does not make a noticeable difference:
# https://colab.research.google.com/drive/1d7G-O5JX31lHbg7jTzzozbo5-Gp7DBEv?usp=sharing

# ========================================
# WORK-PRECISION (wp) MODE
# ========================================
# `bench_diffrax.py 32768 wp` sweeps fixed dt / adaptive tolerance at N=32768
# and records "<setting> <time_ms> <error-vs-golden>" per point. Protocol and
# sweep grids live in runner_scripts/wp_common.py. Note: wp timings call
# block_until_ready so the full solve (not just the async dispatch) is
# measured.
if len(sys.argv) > 2 and sys.argv[2] == "wp":
    from wp_common import (DTS, TOLS, N_WP, load_golden, ensemble_error,
                           wp_outfile)

    if numberOfParameters != N_WP:
        sys.exit("wp mode must be run with N = {0}".format(N_WP))
    golden = load_golden()

    def make_fixed(dt0):
        @jax.jit
        @jax.vmap
        def m(k1):
            lorenz = Lorenz(k1)
            terms = diffrax.ODETerm(lorenz)
            return diffrax.diffeqsolve(
                terms, diffrax.Tsit5(), 0.0, 1.0, dt0,
                jnp.array([1.0, 0.0, 0.0]), max_steps=65536)
        return m

    def make_adaptive(tol):
        @jax.jit
        @jax.vmap
        def m(k1):
            lorenz = Lorenz(k1)
            terms = diffrax.ODETerm(lorenz)
            return diffrax.diffeqsolve(
                terms, diffrax.Tsit5(), 0.0, 1.0, PERF_FIXED_DT,
                jnp.array([1.0, 0.0, 0.0]), max_steps=65536,
                stepsize_controller=diffrax.PIDController(rtol=tol, atol=tol))
        return m

    def bench(make, setting, outfh):
        m = make(setting)
        sol = m(parameterList)
        jax.block_until_ready(sol.ys)  # warm-up (JIT) + numerical result
        err = ensemble_error(np.array(sol.ys[:, -1, :]), golden)
        res = timeit.repeat(
            lambda: jax.block_until_ready(m(parameterList).ys),
            repeat=REPEATS, number=1)
        t_ms = min(res) * 1000
        print("wp setting={0:g}: {1:.2f} ms, err={2:.3e}".format(
            setting, t_ms, err))
        outfh.write("{0:.10g} {1} {2:.10e}\n".format(setting, t_ms, err))

    with open(wp_outfile("JAX", "Jax", "fixed", DATASET_KEY), "w") as f:
        for dt in DTS:
            bench(make_fixed, dt, f)
    with open(wp_outfile("JAX", "Jax", "adaptive", DATASET_KEY), "w") as f:
        for tol in TOLS:
            bench(make_adaptive, tol, f)

    sys.exit(0)

# %%
# Use jax.vmap to compute parallel solutions of the ODE
best_time, best_time_dev = best_times_ms(main, parameterList, "fixed time-stepping")
print("{:} ODE solves with fixed time-stepping completed in {:.1f} ms "
      "({:.1f} ms without transfers)".format(numberOfParameters, best_time, best_time_dev))


# %%
# Save the minimum time 
file = open(os.path.join(data_dir("JAX", DATASET_KEY), "Jax_times_unadaptive.txt"), "a+")
file.write('{0} {1} {2}\n'.format(numberOfParameters, best_time, best_time_dev))
file.close()

# Save numerical output for 32768-trajectory run
if numberOfParameters == 32768:
    sol = main(parameterList)
    # Extract final state values (last time point for each trajectory)
    final_states = np.array(sol.ys[:, -1, :])  # shape: (trajectories, states)
    np.savetxt(os.path.join(data_dir("numerical", DATASET_KEY), "jax.csv"), final_states, delimiter=',')


# %%
# Repeat the same for adaptive time-stepping
@jax.jit
@jax.vmap
def main(k1):
    lorenz = Lorenz(k1)
    terms = diffrax.ODETerm(lorenz)
    t0 = 0.0
    t1 = 1.0
    y0 = jnp.array([1.0, 0.0, 0.0])
    dt0 = PERF_FIXED_DT
    solver = diffrax.Tsit5()
    saveat = diffrax.SaveAt(ts = jnp.array([t0,t1]))
    stepsize_controller = diffrax.PIDController(
        rtol=PERF_ADAPTIVE_TOL, atol=PERF_ADAPTIVE_TOL)
    sol = diffrax.diffeqsolve(
        terms,
        solver,
        t0,
        t1,
        dt0,
        y0,
#         saveat=saveat,
        stepsize_controller=stepsize_controller,
    )
    return sol


# %%


import timeit


# %%


best_time, best_time_dev = best_times_ms(main, parameterList, "adaptive time-stepping")


# %%

print("{:} ODE solves with adaptive time-stepping completed in {:.1f} ms "
      "({:.1f} ms without transfers)".format(numberOfParameters, best_time, best_time_dev))


# %%


file = open(os.path.join(data_dir("JAX", DATASET_KEY), "Jax_times_adaptive.txt"), "a+")
file.write('{0} {1} {2}\n'.format(numberOfParameters, best_time, best_time_dev))
file.close()

