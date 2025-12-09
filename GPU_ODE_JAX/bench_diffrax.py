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

# Add parent directory to path for model_definitions
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_definitions import get_jax_model_class, get_model_params, get_parameter_list

numberOfParameters = int(sys.argv[1])
model_name = sys.argv[2] if len(sys.argv) > 2 else "lorenz"

# %%


print("Working on :", jax.default_backend())

# Get model definition
ModelClass = get_jax_model_class(model_name)
model_params = get_model_params(model_name, np.float32)
initial_conditions = model_params['initial_conditions']
tspan = model_params['tspan']
param_name = model_params['parameter_name']

# Convert initial conditions to array
y0_list = list(initial_conditions.values())
y0 = jnp.array(y0_list)


# %%
# JIT compilation of ODE solver
@jax.jit
@jax.vmap
def main(param_value):
    model = ModelClass(param_value)
    terms = diffrax.ODETerm(model)
    t0 = tspan[0]
    t1 = tspan[1]
    dt0 = 0.001
    solver = diffrax.Tsit5()
    saveat = diffrax.SaveAt(ts = jnp.array([t0, t1]))
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
parameterList = jnp.array(get_parameter_list(model_name, numberOfParameters, np.float32))

# Test that vmap and JIT ordering does not make a noticeable difference:
# https://colab.research.google.com/drive/1d7G-O5JX31lHbg7jTzzozbo5-Gp7DBEv?usp=sharing

# %%
# Use jax.vmap to compute parallel solutions of the ODE
res = timeit.repeat(lambda: main(parameterList),repeat = 100,number = 1)

best_time  = min(res)*1000
print("{:} ODE solves with fixed time-stepping completed in {:.1f} ms".format(numberOfParameters, best_time))


# %%
# Save the minimum time 
file = open("./data/JAX/Jax_times_unadaptive.txt","a+")
file.write('{0} {1}\n'.format(numberOfParameters, best_time))
file.close()

# Save numerical output for 32768-trajectory run
if numberOfParameters == 32768:
    os.makedirs("./data/numerical", exist_ok=True)
    sol = main(parameterList)
    # Extract final state values (last time point for each trajectory)
    final_states = np.array(sol.ys[:, -1, :])  # shape: (trajectories, states)
    np.savetxt("./data/numerical/jax.csv", final_states, delimiter=',')


# %%
# Repeat the same for adaptive time-stepping
@jax.jit
@jax.vmap
def main_adaptive(param_value):
    model = ModelClass(param_value)
    terms = diffrax.ODETerm(model)
    t0 = tspan[0]
    t1 = tspan[1]
    dt0 = 0.001
    solver = diffrax.Tsit5()
    saveat = diffrax.SaveAt(ts = jnp.array([t0, t1]))
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


res = timeit.repeat(lambda: main_adaptive(parameterList),repeat = 100,number = 1)


# %%

best_time  = min(res)*1000
print("{:} ODE solves with adaptive time-stepping completed in {:.1f} ms".format(numberOfParameters, best_time))


# %%


file = open("./data/JAX/Jax_times_adaptive.txt","a+")
file.write('{0} {1}\n'.format(numberOfParameters, best_time))
file.close()

