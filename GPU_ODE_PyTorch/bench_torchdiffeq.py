#!/usr/bin/env python
# coding: utf-8
# %%
# Benchmarking torchdiffeq ODE solvers for ensemble problems, via vmap. The Lorenz ODE is integrated by Tsit5.

# Created By: Utkarsh
# Last Updated: 19 April 2023

# %%

import torch
import sys
import os
import timeit
import sys
import numpy as np

# Add parent directory to path for model_definitions
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_definitions import get_pytorch_model_class, get_model_params, get_parameter_list

numberOfParameters = int(sys.argv[1])
model_name = sys.argv[2] if len(sys.argv) > 2 else "lorenz"

# %%


import torchdiffeq
import math
import torch.nn as nn
import timeit
from torchdiffeq import odeint

# Get model definition
ModelClass = get_pytorch_model_class(model_name)
model_params = get_model_params(model_name, np.float32)
initial_conditions = model_params['initial_conditions']
tspan = model_params['tspan']
param_name = model_params['parameter_name']

# Convert initial conditions to tensor
u0_list = list(initial_conditions.values())
u0 = torch.tensor(u0_list).cuda()

# Time points
t = torch.linspace(tspan[0], tspan[1], 2).cuda()

# %%

## Checking if torch installation has cuda enabled
print("CUDA enabled: ", torch.backends.cuda.is_built())


# %%
# Define the solve without gradient calculations
# Note: I was't able to JIT compile the code with this application, torchdiffeq + vmap
def solve(p):
    with torch.no_grad():
        model = ModelClass(**{param_name: p})
        traj = odeint(model, u0, t, method='rk4', options=dict(step_size=0.001))
        return traj

# %%
# Generate parameter list
param_list = get_parameter_list(model_name, numberOfParameters, np.float32)
parameters = torch.tensor(param_list).cuda()


# %%

import timeit
res = timeit.repeat(lambda: torch.vmap(solve)(parameters), repeat = 10, number = 1)


# %%
# Print the best result

best_time  = min(res)*1000
print("{:} ODE solves with fixed time-stepping completed in {:.1f} ms".format(numberOfParameters, best_time))


# %%
# Save the result

file = open("./data/PYTORCH/Torch_times_unadaptive.txt","a+")
file.write('{0} {1}\n'.format(numberOfParameters, best_time))
file.close()

# Save numerical output for 32768-trajectory run
if numberOfParameters == 32768:
    os.makedirs("./data/numerical", exist_ok=True)
    traj = torch.vmap(solve)(parameters)
    # Extract final state values (last time point for each trajectory)
    final_states = traj[:, -1, :].cpu().numpy()  # shape: (trajectories, states)
    np.savetxt("./data/numerical/pytorch.csv", final_states, delimiter=',')


# %%
