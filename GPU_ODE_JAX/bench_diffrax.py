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

# Dataset key ("<os>_<gpu>") so output files are keyed per machine and can be
# additively populated across machines without clobbering each other.
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runner_scripts"))
from bench_key import dataset_key
DATASET_KEY = dataset_key()

# %%


print("Working on :", jax.default_backend())

# This is a GPU benchmark: refuse to silently record CPU timings (e.g. on
# native Windows, where no CUDA jaxlib wheels exist — use Linux or WSL2).
if jax.default_backend() == "cpu":
    print("ERROR: JAX is running on the CPU backend; aborting so CPU "
          "timings are not recorded as GPU results.")
    sys.exit(1)


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
    dt0 = 0.001
    solver = diffrax.Tsit5()
    saveat = diffrax.SaveAt(ts = jnp.array([t0,t1]))
    stepsize_controller = diffrax.PIDController(rtol=1e-6, atol=1e-3)
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
                terms, diffrax.Tsit5(), 0.0, 1.0, 0.001,
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
            repeat=20, number=1)
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
res = timeit.repeat(lambda: main(parameterList),repeat = 100,number = 1)

best_time  = min(res)*1000
print("{:} ODE solves with fixed time-stepping completed in {:.1f} ms".format(numberOfParameters, best_time))


# %%
# Save the minimum time 
os.makedirs("./data/JAX", exist_ok=True)
file = open("./data/JAX/Jax_times_unadaptive_{0}.txt".format(DATASET_KEY),"a+")
file.write('{0} {1}\n'.format(numberOfParameters, best_time))
file.close()

# Save numerical output for 32768-trajectory run
if numberOfParameters == 32768:
    os.makedirs("./data/numerical", exist_ok=True)
    sol = main(parameterList)
    # Extract final state values (last time point for each trajectory)
    final_states = np.array(sol.ys[:, -1, :])  # shape: (trajectories, states)
    np.savetxt("./data/numerical/jax_{0}.csv".format(DATASET_KEY), final_states, delimiter=',')


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
    dt0 = 0.001
    solver = diffrax.Tsit5()
    saveat = diffrax.SaveAt(ts = jnp.array([t0,t1]))
    stepsize_controller = diffrax.PIDController(rtol=1e-8, atol=1e-8)
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


res = timeit.repeat(lambda: main(parameterList),repeat = 100,number = 1)


# %%

best_time  = min(res)*1000
print("{:} ODE solves with adaptive time-stepping completed in {:.1f} ms".format(numberOfParameters, best_time))


# %%


file = open("./data/JAX/Jax_times_adaptive_{0}.txt".format(DATASET_KEY),"a+")
file.write('{0} {1}\n'.format(numberOfParameters, best_time))
file.close()

