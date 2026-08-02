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

numberOfParameters = int(sys.argv[1])

# Dataset key ("<os>_<gpu>") keys output files per machine.
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runner_scripts"))
from bench_key import dataset_key, data_dir
# Timed repeats per point (min is reported) and the shared fixed step size.
from protocol import PERF_FIXED_DT, REPEATS
DATASET_KEY = dataset_key()

# %%


import torchdiffeq
import math
import torch.nn as nn
import timeit
from torchdiffeq import odeint


# %%

## Checking if torch installation has cuda enabled
print("CUDA enabled: ", torch.backends.cuda.is_built())


# %%
# Defining the Lorenz ODE problem
class LorenzODE(torch.nn.Module):

    def __init__(self, rho = torch.tensor(21.0)):
        super(LorenzODE, self).__init__()
        self.sigma = nn.Parameter(torch.as_tensor([10.0]))
        self.rho = nn.Parameter(rho)
        self.beta = nn.Parameter(torch.as_tensor([8/3]))

    def forward(self, t, u):
        x, y, z = u[0],u[1],u[2]
        du1 = self.sigma[0] * (y - x)
        du2 = x * (self.rho - z) - y
        du3 = x * y - self.beta[0] * z
        return torch.stack([du1, du2, du3])


# %%
# Uncomment for smoke test

# u0 = torch.tensor([1.0,0.0,0.0]).cuda()
# t = torch.linspace(0, 1.0, 1001).cuda()
# y = odeint(LorenzODE(), u0, t, method='rk4',options=dict(step_size=0.001))


# %%
# Define the solve without gradient calculations
# Note: I was't able to JIT compile the code with this application, torchdiffeq + vmap
def solve(p):
    with torch.no_grad():
        traj = odeint(LorenzODE(rho = p), u0, t, method='rk4', options=dict(step_size=PERF_FIXED_DT))
        return traj

# Define the initial conditions and timepoints to save
u0 = torch.tensor([1.0,0.0,0.0]).cuda()
t = torch.linspace(0, 1.0, 2).cuda()


# %%
# Generate parameter list
parameters = torch.linspace(0.0,21.0,numberOfParameters).cuda()


# ========================================
# WORK-PRECISION (wp) MODE
# ========================================
# `bench_torchdiffeq.py 32768 wp` sweeps the fixed step size at N=32768 and
# records "<dt> <time_ms> <error-vs-golden>" per point. Protocol and sweep
# grids live in runner_scripts/wp_common.py. There is no adaptive sweep:
# torchdiffeq's adaptive solvers have data-dependent control flow that
# torch.vmap cannot trace (the reason this benchmark is fixed-step only).
# Note: unlike the N-sweep above, wp timings synchronize the device so the
# full solve (not just the async dispatch) is measured.
if len(sys.argv) > 2 and sys.argv[2] == "wp":
    from wp_common import DTS, N_WP, load_golden, ensemble_error, wp_outfile

    if numberOfParameters != N_WP:
        sys.exit("wp mode must be run with N = {0}".format(N_WP))
    golden = load_golden()

    with open(wp_outfile("PYTORCH", "Torch", "fixed", DATASET_KEY), "w") as f:
        for dt in DTS:
            def solve_dt(p, dt=dt):
                with torch.no_grad():
                    return odeint(LorenzODE(rho=p), u0, t, method='rk4',
                                  options=dict(step_size=dt))

            def run():
                traj = torch.vmap(solve_dt)(parameters)
                torch.cuda.synchronize()
                return traj

            traj = run()  # warm-up + numerical result
            err = ensemble_error(traj[:, -1, :].cpu().numpy(), golden)
            res = timeit.repeat(run, repeat=REPEATS, number=1)
            t_ms = min(res) * 1000
            print("wp fixed dt={0:g}: {1:.2f} ms, err={2:.3e}".format(dt, t_ms, err))
            f.write("{0:.10g} {1} {2:.10e}\n".format(dt, t_ms, err))

    sys.exit(0)


# %%

import timeit

parameters_host = parameters.cpu().numpy()


def with_transfers():
    # .cuda() is the h2d, .cpu() the d2h.
    p = torch.from_numpy(parameters_host).cuda()
    out = torch.vmap(solve)(p).cpu()
    torch.cuda.synchronize()
    return out


def device_only():
    # Params already resident, results left on device.
    out = torch.vmap(solve)(parameters)
    torch.cuda.synchronize()
    return out


torch.vmap(solve)(parameters); torch.cuda.synchronize()  # warmup
best_time = min(timeit.repeat(with_transfers, repeat=REPEATS, number=1)) * 1000
best_time_dev = min(timeit.repeat(device_only, repeat=REPEATS, number=1)) * 1000
print("{:} ODE solves with fixed time-stepping completed in {:.1f} ms "
      "({:.1f} ms without transfers)".format(numberOfParameters, best_time, best_time_dev))


# %%
# Save the result

file = open(os.path.join(data_dir("PYTORCH", DATASET_KEY), "Torch_times_unadaptive.txt"), "a+")
file.write('{0} {1} {2}\n'.format(numberOfParameters, best_time, best_time_dev))
file.close()

# Save numerical output for 32768-trajectory run
if numberOfParameters == 32768:
    traj = torch.vmap(solve)(parameters)
    # Extract final state values (last time point for each trajectory)
    final_states = traj[:, -1, :].cpu().numpy()  # shape: (trajectories, states)
    np.savetxt(os.path.join(data_dir("numerical", DATASET_KEY), "pytorch.csv"), final_states, delimiter=',')


# %%
