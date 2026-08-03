#!/usr/bin/env python
# coding: utf-8
# %%
# Benchmarking torchdiffeq ODE solvers for ensemble problems, via vmap. The
# Lorenz ODE is integrated once per supported algorithm so every timing file
# compares like-for-like against the other frameworks (issue #29):
#
#     fixed: euler, classical-rk4, tsit5 (custom fixed-grid solver below)
#
# There is no adaptive sweep: torchdiffeq's adaptive solvers have
# data-dependent control flow that torch.vmap cannot trace.
#
# Usage: bench_torchdiffeq.py <N> [wp] [algorithm|all]

# Created By: Utkarsh
# Last Updated: 19 April 2023

# %%

import torch
import sys
import os
import timeit
import numpy as np

# Dataset key ("<os>_<gpu>") so output files are keyed per machine and can be
# additively populated across machines without clobbering each other.
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runner_scripts"))
from bench_key import dataset_key
from wp_common import parse_bench_args, times_outfile

DATASET_KEY = dataset_key()

FIXED_ALGORITHMS = ("euler", "classical-rk4", "tsit5")
SUPPORTED = ("euler", "classical-rk4", "tsit5")

numberOfParameters, WP_MODE, ALGORITHMS = parse_bench_args(sys.argv[1:], SUPPORTED)
# Timed repeats per point; min is reported.
REPEATS = 20

# %%


import torch.nn as nn
from torchdiffeq import odeint
from torchdiffeq._impl.odeint import SOLVERS
from torchdiffeq._impl.solvers import FixedGridODESolver
from torchdiffeq._impl.misc import Perturb


# torchdiffeq has no Tsit5; register a fixed-grid one built from the
# Tsitouras 5(4) coefficients (same values as cubie/diffrax/DiffEqGPU).
# Fixed-grid only: the embedded error weights are not carried.
_TSIT5_C = (0.161, 0.327, 0.9, 0.9800255409045097, 1.0, 1.0)
_TSIT5_A = (
    (0.161,),
    (-0.008480655492356989, 0.335480655492357),
    (2.8971530571054935, -6.359448489975075, 4.3622954328695815),
    (5.325864828439257, -11.748883564062828, 7.4955393428898365,
     -0.09249506636175525),
    (5.86145544294642, -12.92096931784711, 8.159367898576159,
     -0.071584973281401, -0.028269050394068383),
    (0.09646076681806523, 0.01, 0.4798896504144996, 1.379008574103742,
     -3.290069515436081, 2.324710524099774),
)
_TSIT5_B = (0.09646076681806523, 0.01, 0.4798896504144996, 1.379008574103742,
            -3.290069515436081, 2.324710524099774, 0.0)


class Tsit5Fixed(FixedGridODESolver):
    order = 5

    def _step_func(self, func, t0, dt, t1, y0):
        f0 = func(t0, y0, perturb=Perturb.NEXT if self.perturb else Perturb.NONE)
        k = [f0]
        for ci, ai in zip(_TSIT5_C, _TSIT5_A):
            yi = y0
            for aij, kj in zip(ai, k):
                yi = yi + dt * aij * kj
            k.append(func(t0 + ci * dt, yi))
        dy = None
        for bi, ki in zip(_TSIT5_B, k):
            term = dt * bi * ki
            dy = term if dy is None else dy + term
        return dy, f0


SOLVERS["tsit5"] = Tsit5Fixed

# canonical algorithm name -> torchdiffeq method string
METHODS = {"euler": "euler", "classical-rk4": "rk4", "tsit5": "tsit5"}


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
# Define the solve without gradient calculations
# Note: I was't able to JIT compile the code with this application, torchdiffeq + vmap
def make_solve(algorithm, dt=0.001):
    method = METHODS[algorithm]

    def solve(p):
        with torch.no_grad():
            return odeint(LorenzODE(rho=p), u0, t, method=method,
                          options=dict(step_size=dt))
    return solve

# Define the initial conditions and timepoints to save
u0 = torch.tensor([1.0,0.0,0.0]).cuda()
t = torch.linspace(0, 1.0, 2).cuda()


# %%
# Generate parameter list
parameters = torch.linspace(0.0,21.0,numberOfParameters).cuda()


# ========================================
# WORK-PRECISION (wp) MODE
# ========================================
# `bench_torchdiffeq.py 32768 wp [algorithm]` sweeps the fixed step size at
# N=32768 per algorithm and records "<dt> <time_ms> <error-vs-golden>" per
# point. Protocol and sweep grids live in runner_scripts/wp_common.py.
# Note: unlike the N-sweep below, wp timings synchronize the device so the
# full solve (not just the async dispatch) is measured.
if WP_MODE:
    from wp_common import dts_for, N_WP, load_golden, ensemble_error, wp_outfile

    if numberOfParameters != N_WP:
        sys.exit("wp mode must be run with N = {0}".format(N_WP))
    golden = load_golden()

    for algorithm in ALGORITHMS:
        outfile = wp_outfile("PYTORCH", "Torch", "fixed", algorithm,
                             DATASET_KEY)
        with open(outfile, "w") as f:
            for dt in dts_for(algorithm):
                solve_dt = make_solve(algorithm, dt)

                def run():
                    traj = torch.vmap(solve_dt)(parameters)
                    torch.cuda.synchronize()
                    return traj

                traj = run()  # warm-up + numerical result
                err = ensemble_error(traj[:, -1, :].cpu().numpy(), golden)
                res = timeit.repeat(run, repeat=5, number=1)
                t_ms = min(res) * 1000
                print("wp fixed {0} dt={1:g}: {2:.2f} ms, err={3:.3e}".format(
                    algorithm, dt, t_ms, err))
                f.write("{0:.10g} {1} {2:.10e}\n".format(dt, t_ms, err))

    sys.exit(0)


# %%
# N-sweep timing benchmark.

parameters_host = parameters.cpu().numpy()

os.makedirs("./data/PYTORCH", exist_ok=True)

for algorithm in ALGORITHMS:
    solve = make_solve(algorithm)

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
    print("{:} ODE solves ({}, fixed) completed in {:.1f} ms "
          "({:.1f} ms without transfers)".format(
              numberOfParameters, algorithm, best_time, best_time_dev))

    outfile = times_outfile("PYTORCH", "Torch", "fixed", algorithm, DATASET_KEY)
    with open(outfile, "a+") as file:
        file.write('{0} {1} {2}\n'.format(
            numberOfParameters, best_time, best_time_dev))

    # The pairwise numerical cross-check keys on the pre-#29 run mode:
    # classical-rk4 was the benchmarked algorithm, so it keeps the CSV name.
    if numberOfParameters == 32768 and algorithm == "classical-rk4":
        os.makedirs("./data/numerical", exist_ok=True)
        traj = torch.vmap(solve)(parameters)
        # Extract final state values (last time point for each trajectory)
        final_states = traj[:, -1, :].cpu().numpy()  # shape: (trajectories, states)
        np.savetxt("./data/numerical/pytorch_{0}.csv".format(DATASET_KEY),
                   final_states, delimiter=',')


# %%
