#!/usr/bin/env python

# torchdiffeq ensemble benchmarks via vmap, fixed-step only: bench_torchdiffeq.py <N>|wp [algorithm|all] [--problem <name|all>]


import torch
import sys
import os
import numpy as np

# Dataset key ("<os>_<gpu>") so output files are keyed per machine and can be
# additively populated across machines without clobbering each other.
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runner_scripts"))
from algorithms import supported_for
from bench_key import dataset_key, data_dir
from torch_systems import build_problem
from wp_common import parse_bench_args, times_outfile

DATASET_KEY = dataset_key()

FIXED_ALGORITHMS = supported_for("pytorch", "fixed")

NS, ANALYSIS, ALGORITHMS, PROBLEMS = parse_bench_args(
    sys.argv[1:], "pytorch")
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
    from wp_common import (dts_for, load_golden, ensemble_error, timed_min_ms,
                           wp_outfile)

    golden = load_golden(problem)

    for algorithm in ALGORITHMS:
        if not problem.runs("pytorch", algorithm):
            continue
        outfile = wp_outfile("PYTORCH", "Torch", "fixed", algorithm,
                             DATASET_KEY, problem)
        with open(outfile, "w") as f:
            # Later settings are slower, so a breach abandons the leg.
            breached = False
            for dt in dts_for(algorithm, problem):
                if breached:
                    f.write("{0:.10g} nan nan\n".format(dt))
                    continue
                solve_dt = make_solve(problem, algorithm, dt)

                def run():
                    traj = torch.vmap(solve_dt)(parameters)
                    torch.cuda.synchronize()
                    return traj

                t_ms, traj = timed_min_ms(run, 5)
                if t_ms is None:
                    print("WATCHDOG wp {0} fixed {1} dt={2:g}: run exceeded "
                          "the cap".format(problem.name, algorithm, dt))
                    breached = True
                    t_ms, err = float("nan"), float("nan")
                else:
                    err = ensemble_error(traj[:, -1, :].cpu().numpy(), golden)
                print("wp {0} fixed {1} dt={2:g}: {3:.2f} ms, err={4:.3e}"
                      .format(problem.name, algorithm, dt, t_ms, err))
                f.write("{0:.10g} {1} {2:.10e}\n".format(dt, t_ms, err))
                f.flush()


def run_times(problem):
    """N-sweep timing: one leg per algorithm, sizes ascending on one solve."""
    from wp_common import timed_min_ms

    for algorithm in ALGORITHMS:
        if not problem.runs("pytorch", algorithm):
            continue
        solve = make_solve(problem, algorithm)
        outfile = times_outfile("PYTORCH", "Torch", "fixed", algorithm,
                                DATASET_KEY, problem)
        with open(outfile, "a+") as file:
            for index, n in enumerate(NS):
                parameters_host = problem.sweep(n, dtype=np.float32)
                parameters = torch.from_numpy(parameters_host).cuda()

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

                best_time, _ = timed_min_ms(with_transfers, REPEATS)
                best_time_dev = None
                if best_time is not None:
                    best_time_dev, _ = timed_min_ms(device_only, REPEATS)
                breached = best_time is None or best_time_dev is None
                if breached:
                    print("WATCHDOG {0} fixed {1} N={2}: run exceeded the "
                          "cap".format(problem.name, algorithm, n))
                    best_time = (float("nan") if best_time is None
                                 else best_time)
                    best_time_dev = float("nan")
                else:
                    print("{:} ODE solves ({}, {}, fixed) completed in "
                          "{:.1f} ms ({:.1f} ms without transfers)".format(
                              n, problem.name, algorithm, best_time,
                              best_time_dev))

                file.write('{0} {1} {2}\n'.format(
                    n, best_time, best_time_dev))
                file.flush()

                # The pairwise numerical cross-check reads this fixed CSV name.
                if (n == 32768 and algorithm == "classical-rk4"
                        and np.isfinite(best_time)):
                    traj = torch.vmap(solve)(parameters)
                    # Extract final state values (last time point for each trajectory)
                    final_states = traj[:, -1, :].cpu().numpy()  # (trajectories, states)
                    np.savetxt(os.path.join(
                        data_dir("numerical", DATASET_KEY, problem=problem),
                        "pytorch.csv"), final_states, delimiter=',')

                if breached:
                    # Larger sizes are slower, so the leg is abandoned.
                    for rest in NS[index + 1:]:
                        file.write('{0} nan nan\n'.format(rest))
                    file.flush()
                    break


def run_states():
    """Runtime-by-states sweep: lorenz96 resized along the requested grid at
    one fixed ensemble size; torchdiffeq is fixed-step only."""
    import timeit

    from problems import states_row
    from wp_common import STATES_N, states_outfile, timed_min_ms

    n = STATES_N
    grid = NS
    for algorithm in ALGORITHMS:
        outfile = states_outfile("PYTORCH", "Torch", "fixed", algorithm,
                                 DATASET_KEY)
        with open(outfile, "w") as file:
            for index, nstates in enumerate(grid):
                row = states_row(nstates)
                solve = make_solve(row, algorithm)
                parameters_host = row.sweep(n, dtype=np.float32)
                parameters = torch.from_numpy(parameters_host).cuda()

                def with_transfers():
                    p = torch.from_numpy(parameters_host).cuda()
                    out = torch.vmap(solve)(p).cpu()
                    torch.cuda.synchronize()
                    return out

                def device_only():
                    out = torch.vmap(solve)(parameters)
                    torch.cuda.synchronize()
                    return out

                t_ms = t_dev = build_s = float("nan")
                breached = False
                try:
                    started = timeit.default_timer()
                    device_only()
                    build_s = timeit.default_timer() - started
                    best, _ = timed_min_ms(with_transfers, REPEATS)
                    best_dev = None
                    if best is not None:
                        best_dev, _ = timed_min_ms(device_only, REPEATS)
                    breached = best is None or best_dev is None
                    if not breached:
                        t_ms, t_dev = best, best_dev
                        print("{:} ODE solves (lorenz96 states={}, {}, "
                              "fixed) completed in {:.1f} ms ({:.1f} ms "
                              "without transfers)".format(
                                  n, nstates, algorithm, t_ms, t_dev))
                except Exception as exc:
                    print("FAILED lorenz96 states={0} fixed {1} N={2}: {3}"
                          .format(nstates, algorithm, n, exc))
                file.write('{0} {1} {2} {3}\n'.format(
                    nstates, t_ms, t_dev, build_s))
                file.flush()
                if breached:
                    # Larger systems are slower, so the leg is abandoned.
                    print("WATCHDOG lorenz96 states={0} fixed {1} N={2}: "
                          "run exceeded the cap".format(nstates, algorithm,
                                                        n))
                    for rest in grid[index + 1:]:
                        file.write('{0} nan nan nan\n'.format(rest))
                    file.flush()
                    break


# %%
if ANALYSIS == "warm":
    print("torchdiffeq runs eagerly; there is nothing to warm.")
    sys.exit(0)

if ANALYSIS == "states":
    from problems import STATES_PROBLEM
    if not any(p.name == STATES_PROBLEM for p in PROBLEMS):
        print("torchdiffeq does not run {0}; skipping the states sweep."
              .format(STATES_PROBLEM))
        sys.exit(0)
    run_states()
    sys.exit(0)

if not PROBLEMS:
    print("torchdiffeq runs none of the requested problems; skipping.")
    sys.exit(0)

for _problem in PROBLEMS:
    if ANALYSIS == "wp":
        # Generate parameter list
        run_wp(_problem, torch.from_numpy(
            _problem.sweep(NS[0], dtype=np.float32)).cuda())
    else:
        run_times(_problem)

# %%
