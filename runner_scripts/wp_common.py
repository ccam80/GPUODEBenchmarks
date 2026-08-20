"""Work-precision sweep protocol: setting, time and error rows under data/<package>/<key>/<problem>/, mirrored by the Julia and MPGOS writers."""

import os

import numpy as np

from algorithms import resolve_algorithms
from bench_key import data_dir
from problems import DEFAULT_PROBLEM, get_problem, resolve_problems

TOLS = [10.0 ** -k for k in range(2, 9)]       # 1e-2 .. 1e-8, 7 points

N_WP = 131072

# States sweep sizes; BENCH_STATES_GRID (comma list) overrides the default.
_STATES_ENV = os.environ.get("BENCH_STATES_GRID", "")
STATES_GRID = (tuple(sorted(int(tok) for tok in _STATES_ENV.split(",")))
               if _STATES_ENV else (4, 8, 16, 32, 64, 128))
STATES_N = 131072

# Adaptive N-sweep tolerance; mirrored in the Julia and MPGOS writers.
TIMING_TOL = 1.0e-5

# Per-run wall-clock ceiling in seconds; mirrored by the Julia and MPGOS writers.
WATCHDOG_SECONDS = float(os.environ.get("BENCH_WATCHDOG_SECONDS", "120"))


# Columns of the per-repeat timing log; mirrored by the Julia and MPGOS writers.
SAMPLE_FIELDS = ("analysis", "problem", "algorithm", "mode", "transfers",
                 "setting_kind", "setting", "n", "states", "repeat", "ms")


def timed_min_ms(run, repeats, sink=None):
    """Best-of-repeats wall time in ms after one warm-up; None on a breach. ``sink(attempt, ms)`` sees every attempt, warm-up (0) and breaching run included."""
    import timeit
    best = None
    for attempt in range(repeats + 1):
        elapsed = timeit.default_timer()
        result = run()
        elapsed = timeit.default_timer() - elapsed
        if sink is not None:
            sink(attempt, elapsed * 1000.0)
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


def states_outfile(framework_dir, prefix, mode, algorithm, dataset_key):
    """Path of the states-sweep timing file under the lorenz96 problem dir."""
    from problems import STATES_PROBLEM
    return os.path.join(
        data_dir(framework_dir, dataset_key, problem=STATES_PROBLEM),
        "{0}_states_{1}_{2}.txt".format(prefix, mode, algorithm))


def samples_outfile(framework_dir, prefix, analysis, mode, algorithm,
                    dataset_key, problem=DEFAULT_PROBLEM):
    """Path of the per-repeat timing log beside its reduced output file."""
    return os.path.join(data_dir(framework_dir, dataset_key, problem=problem),
                        "{0}_samples_{1}_{2}_{3}.csv".format(
                            prefix, analysis, mode, algorithm))


class SampleLog(object):
    """Per-repeat timing rows for one (analysis, mode, algorithm) leg; ``truncate`` matches the sibling output file's open mode."""

    def __init__(self, path, truncate=False):
        header = ",".join(SAMPLE_FIELDS) + "\n"
        if truncate:
            self._handle = open(path, "w")
            self._handle.write(header)
            self._handle.flush()
            return
        # Exclusive create: one header even when sibling processes open the log.
        try:
            with open(path, "x") as fresh:
                fresh.write(header)
        except FileExistsError:
            pass
        self._handle = open(path, "a")

    def sink(self, analysis, problem, algorithm, mode, transfers, n, states,
             setting_kind="none", setting=float("nan")):
        """A ``sink(repeat, ms)`` callable for one timed point."""
        head = "{0},{1},{2},{3},{4},{5},{6:.10g},{7},{8}".format(
            analysis, problem, algorithm, mode, transfers, setting_kind,
            setting, n, states)

        def record(repeat, ms):
            self._handle.write("{0},{1},{2:.6f}\n".format(head, repeat, ms))
            self._handle.flush()

        return record

    def close(self):
        self._handle.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def parse_bench_args(argv, framework):
    """Parse <N|N,N,...>|wp|states|warm[:N,N,...] [algorithm|all] [--problem <name|all>] into (ns, analysis, algorithms, problems)."""
    if not argv:
        raise SystemExit("usage: <N|N,N,...>|wp|states|warm[:N,N,...] "
                         "[algorithm|all] [--problem <name|all>]")
    if argv[0] == "wp":
        analysis, ns = "wp", [N_WP]
    elif argv[0] == "states":
        # In states mode ns is the state-count grid; the ensemble is STATES_N.
        analysis, ns = "states", list(STATES_GRID)
    elif argv[0] == "warm" or argv[0].startswith("warm:"):
        _, _, counts = argv[0].partition(":")
        analysis = "warm"
        ns = sorted(int(tok) for tok in counts.split(",")) if counts else []
    else:
        # Ascending, so each leg walks its sweep on kernels compiled once.
        analysis = "times"
        ns = sorted(int(tok) for tok in argv[0].split(","))
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
    return ns, analysis, algorithms, problems
