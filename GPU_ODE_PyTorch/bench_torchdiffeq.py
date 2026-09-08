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
from results import Leg
from resume import skip_point, skip_wp_leg
from wp_common import REPEAT_CAP, errored_pct, parse_bench_args

DATASET_KEY = dataset_key()

NS, ANALYSIS, ALGORITHMS, PROBLEMS, MODES = parse_bench_args(
    sys.argv[1:], "pytorch")
# torchdiffeq is fixed-step only, so an adaptive-only run has no legs.
FIXED_ALGORITHMS = supported_for("pytorch", "fixed") if "fixed" in MODES else ()
ALGORITHMS = [name for name in ALGORITHMS if name in FIXED_ALGORITHMS]
REPEATS = REPEAT_CAP

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
    from wp_common import load_golden, ensemble_error, timed_min_ms

    golden = load_golden(problem)

    for algorithm in ALGORITHMS:
        if not problem.supports("pytorch"):
            continue
        dts = list(problem.dts(algorithm))
        leg = Leg("pytorch", DATASET_KEY, "wp", problem, algorithm, "fixed")
        if skip_wp_leg(leg, dts):
            print("-- resume: skipping wp {0} fixed {1} (already covered)"
                  .format(problem.name, algorithm))
            continue
        # Later settings are slower, so a breach abandons the leg.
        for index, dt in enumerate(dts):
            solve_dt = make_solve(problem, algorithm, dt)

            def run():
                traj = torch.vmap(solve_dt)(parameters)
                torch.cuda.synchronize()
                return traj

            def on_breach(rest=dts[index:], at=dt):
                # The hard exit skips the abandon path, so fill it here.
                leg.nan_wp(rest)
                print("WATCHDOG wp {0} fixed {1} dt={2:g}: run never "
                      "returned".format(problem.name, algorithm, at))

            # Parameters are already resident and results stay on device.
            t_ms, traj, samples = timed_min_ms(run, REPEATS, on_breach)
            finals = traj[:, -1, :].cpu().numpy()
            pct = errored_pct(finals)
            breached = t_ms is None
            if breached:
                print("WATCHDOG wp {0} fixed {1} dt={2:g}: run exceeded "
                      "the cap".format(problem.name, algorithm, dt))
                t_ms, err = float("nan"), float("nan")
            else:
                err = ensemble_error(finals, golden)
            print("wp {0} fixed {1} dt={2:g}: {3:.2f} ms, err={4:.3e}, "
                  "errored={5:.1f}%".format(problem.name, algorithm, dt,
                                            t_ms, err, pct))
            leg.record_wp(dt, t_ms, err, pct, samples=samples)
            if breached:
                leg.nan_wp(dts[index + 1:])
                break


def run_times(problem):
    """N-sweep timing: one leg per algorithm, sizes ascending on one solve."""
    from wp_common import timed_min_ms

    for algorithm in ALGORITHMS:
        if not problem.supports("pytorch"):
            continue
        leg = Leg("pytorch", DATASET_KEY, "times", problem, algorithm,
                  "fixed")
        run_ns = [n for n in NS if not skip_point(leg, n)]
        if not run_ns:
            print("-- resume: skipping {0} fixed {1} (already covered)"
                  .format(problem.name, algorithm))
            continue
        if len(run_ns) < len(NS):
            print("-- resume: {0} fixed {1} runs N={2}".format(
                problem.name, algorithm,
                ",".join(str(n) for n in run_ns)))
        solve = make_solve(problem, algorithm)
        for index, n in enumerate(run_ns):
            parameters_host = problem.sweep(n, dtype=np.float32)
            parameters = None

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

            # An exhausted card ends the leg the way a breach does.
            exhausted = False
            best_time = None
            best_time_dev = None
            out = None
            samples_both = samples_none = None
            try:
                parameters = torch.from_numpy(parameters_host).cuda()
                best_time, out, samples_both = timed_min_ms(with_transfers,
                                                            REPEATS)
                if best_time is not None:
                    best_time_dev, _, samples_none = timed_min_ms(
                        device_only, REPEATS)
            except torch.OutOfMemoryError as err:
                exhausted = True
                parameters = None
                torch.cuda.empty_cache()
                print("OOM {0} fixed {1} N={2}: {3}".format(
                    problem.name, algorithm, n,
                    str(err).splitlines()[0]))
            breached = (exhausted or best_time is None
                        or best_time_dev is None)
            if breached:
                if not exhausted:
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

            pct = (100.0 if out is None
                   else errored_pct(np.asarray(out[:, -1, :])))
            leg.record_times(n, best_time, best_time_dev, pct, samples_both,
                             samples_none)

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
                leg.nan_times(run_ns[index + 1:])
                break


def run_states():
    """Runtime-by-states sweep: lorenz96 resized along the requested grid at
    one fixed ensemble size; torchdiffeq is fixed-step only."""
    import timeit

    from problems import STATES_PROBLEM, states_row
    from wp_common import STATES_N, timed_min_ms

    n = STATES_N
    grid = NS
    for algorithm in ALGORITHMS:
        leg = Leg("pytorch", DATASET_KEY, "states", STATES_PROBLEM, algorithm,
                  "fixed")
        run_grid = [s for s in grid if not skip_point(leg, n, s)]
        if not run_grid:
            print("-- resume: skipping states fixed {0} (already covered)"
                  .format(algorithm))
            continue
        for index, nstates in enumerate(run_grid):
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
            out = None
            breached = False
            samples_both = samples_none = None
            try:
                started = timeit.default_timer()
                device_only()
                build_s = timeit.default_timer() - started

                best, out, samples_both = timed_min_ms(with_transfers,
                                                       REPEATS)
                best_dev = None
                if best is not None:
                    best_dev, _, samples_none = timed_min_ms(device_only,
                                                             REPEATS)
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
            pct = (100.0 if out is None
                   else errored_pct(np.asarray(out[:, -1, :])))
            leg.record_times(n, t_ms, t_dev, pct, samples_both, samples_none,
                             build_s=build_s, states=nstates)
            if breached:
                # Larger systems are slower, so the leg is abandoned.
                print("WATCHDOG lorenz96 states={0} fixed {1} N={2}: "
                      "run exceeded the cap".format(nstates, algorithm,
                                                    n))
                leg.nan_states(run_grid[index + 1:])
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
