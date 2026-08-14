#!/usr/bin/env python

# torchdiffeq ensemble benchmarks via vmap, fixed-step only: bench_torchdiffeq.py <N> [wp] [algorithm|all] [--problem <name|all>]


import torch
import sys
import os
import timeit
import numpy as np

# Dataset key ("<os>_<gpu>") so output files are keyed per machine and can be
# additively populated across machines without clobbering each other.
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runner_scripts"))
from bench_key import dataset_key, data_dir
from torch_systems import build_problem
from wp_common import parse_bench_args, times_outfile

DATASET_KEY = dataset_key()

FIXED_ALGORITHMS = ("euler", "classical-rk4", "tsit5")
SUPPORTED = ("euler", "classical-rk4", "tsit5")

numberOfParameters, WP_MODE, ALGORITHMS, PROBLEMS = parse_bench_args(
    sys.argv[1:], SUPPORTED, framework="pytorch")
# Timed repeats per point; min is reported.
REPEATS = 20

# %%


from torchdiffeq import odeint
from torchdiffeq._impl.odeint import SOLVERS
from torchdiffeq._impl.solvers import FixedGridODESolver
from torchdiffeq._impl.misc import Perturb


# Fixed-grid Tsit5 from the Tsitouras 5(4) coefficients, registered below.
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
# torchdiffeq under vmap does not JIT compile, so the solve stays interpreted.
def make_solve(problem, algorithm, dt=None):
    method = METHODS[algorithm]
    module_factory, u0 = build_problem(problem)
    dt = problem.timing_dt if dt is None else dt
    # Endpoints only: the benchmark scores the final state.
    t = torch.linspace(0, problem["duration"], 2).cuda()

    def solve(p):
        with torch.no_grad():
            return odeint(module_factory(p), u0, t, method=method,
                          options=dict(step_size=dt))
    return solve


def run_wp(problem, parameters):
    """dt sweep at N = N_WP; see runner_scripts/wp_common.py."""
    from wp_common import dts_for, N_WP, load_golden, ensemble_error, wp_outfile

    if numberOfParameters != N_WP:
        sys.exit("wp mode must be run with N = {0}".format(N_WP))
    golden = load_golden(problem)

    for algorithm in ALGORITHMS:
        outfile = wp_outfile("PYTORCH", "Torch", "fixed", algorithm,
                             DATASET_KEY, problem)
        with open(outfile, "w") as f:
            for dt in dts_for(algorithm, problem):
                solve_dt = make_solve(problem, algorithm, dt)

                def run():
                    traj = torch.vmap(solve_dt)(parameters)
                    torch.cuda.synchronize()
                    return traj

                traj = run()  # warm-up + numerical result
                err = ensemble_error(traj[:, -1, :].cpu().numpy(), golden)
                res = timeit.repeat(run, repeat=5, number=1)
                t_ms = min(res) * 1000
                print("wp {0} fixed {1} dt={2:g}: {3:.2f} ms, err={4:.3e}"
                      .format(problem.name, algorithm, dt, t_ms, err))
                f.write("{0:.10g} {1} {2:.10e}\n".format(dt, t_ms, err))


def run_times(problem, parameters):
    """N-sweep timing benchmark."""
    parameters_host = parameters.cpu().numpy()

    for algorithm in ALGORITHMS:
        solve = make_solve(problem, algorithm)

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
        print("{:} ODE solves ({}, {}, fixed) completed in {:.1f} ms "
              "({:.1f} ms without transfers)".format(
                  numberOfParameters, problem.name, algorithm, best_time,
                  best_time_dev))

        outfile = times_outfile("PYTORCH", "Torch", "fixed", algorithm,
                                DATASET_KEY, problem)
        with open(outfile, "a+") as file:
            file.write('{0} {1} {2}\n'.format(
                numberOfParameters, best_time, best_time_dev))

        # The pairwise numerical cross-check reads this fixed CSV name.
        if numberOfParameters == 32768 and algorithm == "classical-rk4":
            traj = torch.vmap(solve)(parameters)
            # Extract final state values (last time point for each trajectory)
            final_states = traj[:, -1, :].cpu().numpy()  # (trajectories, states)
            np.savetxt(os.path.join(
                data_dir("numerical", DATASET_KEY, problem=problem),
                "pytorch.csv"), final_states, delimiter=',')


# %%
if not PROBLEMS:
    print("torchdiffeq runs none of the requested problems; skipping.")
    sys.exit(0)

for _problem in PROBLEMS:
    # Generate parameter list
    _parameters = torch.from_numpy(
        _problem.sweep(numberOfParameters, dtype=np.float32)).cuda()
    if WP_MODE:
        run_wp(_problem, _parameters)
    else:
        run_times(_problem, _parameters)

# %%
