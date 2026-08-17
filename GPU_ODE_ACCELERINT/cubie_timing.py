"""Time cubie's radau_iia_5 on the same ensemble the accelerInt CPU reference runs."""

import argparse
import gc
import os
import sys
import timeit

import numpy as np
from numba import cuda

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "runner_scripts"))

from cubie_systems import build_system, final_states  # noqa: E402
from problems import get_problem  # noqa: E402

PRECISION = np.float32
ALGORITHM = "radau_iia_5"  # overridden by --algorithm


def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("-n", type=int, default=512)
    p.add_argument("--problem", default="ring_modulator")
    p.add_argument("--algorithm", default=ALGORITHM)
    p.add_argument("--precision", default="single", choices=["single", "double"])
    p.add_argument("--sweep-min", type=float, default=2.5e-5)
    p.add_argument("--sweep-max", type=float, default=1.0e-1)
    p.add_argument("--rtol", type=float, default=1.0e-5)
    p.add_argument("--atol", type=float, default=1.0e-6)
    # Duration fractions, as in runner_scripts/numerical_equivalence/ne_common.py.
    p.add_argument("--dt0-fraction", type=float, default=1.0e-2)
    p.add_argument("--dt-min-fraction", type=float, default=1.0e-6)
    p.add_argument("--dt-max-fraction", type=float, default=0.5)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--blocksize", type=int, default=64)
    p.add_argument("--newton-max-iters", type=int, default=None)
    p.add_argument("--krylov-max-iters", type=int, default=None)
    p.add_argument("--finals", default=None, help="write final states here as float32")
    return p.parse_args(argv)


def main(argv):
    args = parse_args(argv)
    import cubie as qb

    precision = np.float32 if args.precision == "single" else np.float64
    problem = get_problem(args.problem)
    duration = problem["duration"]
    system, initial_conditions = build_system(problem, precision,
                                              name_suffix="_accelerint_" + args.precision)
    sweep = np.logspace(np.log10(args.sweep_min), np.log10(args.sweep_max),
                        args.n).astype(precision)

    extra = {}
    if args.newton_max_iters is not None:
        extra["newton_max_iters"] = args.newton_max_iters
    if args.krylov_max_iters is not None:
        extra["krylov_max_iters"] = args.krylov_max_iters

    solver = qb.Solver(
        system,
        algorithm=args.algorithm,
        atol=args.atol,
        rtol=args.rtol,
        dt=duration * args.dt0_fraction,
        dt_min=duration * args.dt_min_fraction,
        dt_max=duration * args.dt_max_fraction,
        save_every=duration,
        output_types=["state"],
        time_logging_level=None,
        **extra,
    )
    initials_array, parameter_array = solver.build_grid(
        initial_values=initial_conditions,
        parameters={problem["sweep_parameter"]: sweep})
    d_initials = cuda.to_device(initials_array)
    d_parameters = cuda.to_device(parameter_array)

    def with_transfers():
        return solver.solve(initial_values=initials_array,
                            parameters=parameter_array,
                            blocksize=args.blocksize, duration=duration)

    def device_only():
        return solver.solve(initial_values=d_initials,
                            parameters=d_parameters,
                            blocksize=args.blocksize, duration=duration)

    solution = with_transfers()
    finals = final_states(system, solution, problem)
    converged = int(np.isfinite(finals).all(axis=1).sum())

    gc.collect()
    t_full = min(timeit.repeat(with_transfers, setup="gc.enable()",
                               repeat=args.repeats, number=1))
    device_only()
    t_dev = min(timeit.repeat(device_only, setup="gc.enable()",
                              repeat=args.repeats, number=1))

    print("n {0}  Cs [{1:g}, {2:g}]  rtol {3:g}  atol {4:g}".format(
        args.n, args.sweep_min, args.sweep_max, args.rtol, args.atol))
    print("finite {0}/{1}".format(converged, args.n))
    print("with_transfers_s {0:.6f}".format(t_full))
    print("device_only_s    {0:.6f}".format(t_dev))

    if args.finals:
        finals.astype(np.float32).tofile(args.finals)
        print("finals -> {0}".format(args.finals))
    solver.close()


if __name__ == "__main__":
    main(sys.argv[1:])
