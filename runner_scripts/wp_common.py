"""Work-precision sweep protocol: setting, time and error rows under data/<package>/<key>/<problem>/, mirrored by the Julia and MPGOS writers."""

import os

import numpy as np

from algorithms import resolve_algorithms
from bench_key import data_dir
from problems import DEFAULT_PROBLEM, get_problem, resolve_problems

TOLS = [10.0 ** -k for k in range(2, 9)]       # 1e-2 .. 1e-8, 7 points

N_WP = 131072

# Adaptive N-sweep tolerance; mirrored in the Julia and MPGOS writers.
TIMING_TOL = 1.0e-8

# Per-run wall-clock ceiling in seconds; mirrored by the Julia and MPGOS writers.
WATCHDOG_SECONDS = float(os.environ.get("BENCH_WATCHDOG_SECONDS", "120"))


def timed_min_ms(run, repeats):
    """Best-of-repeats wall time in ms after one warm-up; None on a breach."""
    import timeit
    best = None
    for attempt in range(repeats + 1):
        elapsed = timeit.default_timer()
        result = run()
        elapsed = timeit.default_timer() - elapsed
        if elapsed > WATCHDOG_SECONDS:
            return None, result
        if attempt and (best is None or elapsed < best):
            best = elapsed
    return best * 1000.0, result


def _row(problem):
    """Accept a problem row or a problem name."""
    return problem if isinstance(problem, dict) else get_problem(problem)


def dts_for(algorithm, problem=DEFAULT_PROBLEM):
    """The fixed-step dt grid appropriate to the algorithm and problem."""
    return _row(problem).dts(algorithm)


def golden_path(problem=DEFAULT_PROBLEM):
    """Path of the Float64 reference final states for a problem."""
    return os.path.join(
        "data", "numerical",
        "golden_{0}_{1}.csv".format(_row(problem)["problem"], N_WP))


def load_golden(problem=DEFAULT_PROBLEM):
    """Load the Float64 golden final states, shape (N_WP, states)."""
    row = _row(problem)
    path = golden_path(row)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "{0} not found - generate it first with "
            "`julia -t auto --project=. runner_scripts/golden/generate_golden.jl "
            "--problem {1}`".format(path, row["problem"]))
    golden = np.loadtxt(path, delimiter=",")
    if golden.shape != (N_WP, row["states"]):
        raise ValueError("golden reference has shape {0}, expected ({1}, {2})"
                         .format(golden.shape, N_WP, row["states"]))
    return golden


def ensemble_error(final_states, golden):
    """l2-at-final error over the ensemble, computed in float64."""
    diff = np.asarray(final_states, dtype=np.float64) - golden
    return float(np.sqrt(np.mean(diff ** 2)))


def wp_outfile(framework_dir, prefix, mode, algorithm, dataset_key,
               problem=DEFAULT_PROBLEM):
    """Path of the wp output file under data/<package>/<key>/<problem>."""
    return os.path.join(data_dir(framework_dir, dataset_key, problem=problem),
                        "{0}_wp_{1}_{2}.txt".format(prefix, mode, algorithm))


def times_outfile(framework_dir, prefix, mode, algorithm, dataset_key,
                  problem=DEFAULT_PROBLEM):
    """Path of the N-sweep timing file under data/<package>/<key>/<problem>."""
    return os.path.join(data_dir(framework_dir, dataset_key, problem=problem),
                        "{0}_times_{1}_{2}.txt".format(prefix, mode, algorithm))


def parse_bench_args(argv, framework):
    """Parse <N>|wp [algorithm|all] [--problem <name|all>] into (n, wp, algorithms, problems); wp mode always runs at N_WP."""
    if not argv:
        raise SystemExit(
            "usage: <N>|wp [algorithm|all] [--problem <name|all>]")
    wp = argv[0] == "wp"
    n = N_WP if wp else int(argv[0])
    request = "all"
    problem_request = "all"
    rest = list(argv[1:])
    while rest:
        tok = rest.pop(0)
        if tok in ("--problem", "-s"):
            if not rest:
                raise SystemExit("--problem requires a value")
            problem_request = rest.pop(0)
        elif tok.startswith("--problem="):
            problem_request = tok.split("=", 1)[1]
        else:
            request = tok
    algorithms = resolve_algorithms(request, framework)
    problems = resolve_problems(problem_request, framework)
    return n, wp, algorithms, problems
