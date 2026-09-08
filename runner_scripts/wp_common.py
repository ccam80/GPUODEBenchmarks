"""Work-precision sweep helpers: setting, time and error rows under data/<package>/<key>/<problem>/; constants come from protocol.toml."""

import os
import sys
import threading

import numpy as np

from algorithms import resolve_algorithms, resolve_modes
from bench_key import data_dir
from problems import DEFAULT_PROBLEM, get_problem, resolve_problems
from protocol import (N_WP, REPEAT_CAP, REPEAT_SCHEDULE,  # noqa: F401
                      REPEAT_SPREAD, STATES_GRID, STATES_N, TIMING_TOL, TOLS,
                      WATCHDOG_EXIT_CODE, WATCHDOG_SECONDS)


def run_watchdogged(run, on_breach):
    """Run run(); when it never returns, run on_breach() and hard-exit.

    The soft cap in timed_min_ms only sees a run that comes back, so a solve
    that never returns needs this. Mirrors run_watchdogged in
    runner_scripts/watchdog.jl.
    """
    finished = threading.Event()

    def fire():
        if finished.is_set():
            return
        try:
            on_breach()
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            # A hung kernel blocks every exit path except a hard exit.
            os._exit(WATCHDOG_EXIT_CODE)

    # Margin over the soft cap: only never-returning runs reach the hard exit.
    timer = threading.Timer(WATCHDOG_SECONDS * 2.0 + 30.0, fire)
    timer.daemon = True
    timer.start()
    try:
        return run()
    finally:
        finished.set()
        timer.cancel()


def errored_pct(finals):
    """Percent of trajectories (rows) with a non-finite final state."""
    a = np.asarray(finals)
    if a.size == 0:
        return 0.0
    bad = ~np.isfinite(a)
    if bad.ndim > 1:
        bad = bad.any(axis=tuple(range(1, bad.ndim)))
    return 100.0 * float(bad.sum()) / float(bad.size)


# Columns of the per-repeat timing log; mirrored by the Julia and MPGOS writers.
SAMPLE_FIELDS = ("analysis", "problem", "algorithm", "mode", "transfers",
                 "setting_kind", "setting", "n", "states", "repeat", "ms")


def repeat_bounds(first_s, cap):
    """(floor, ceiling) repeats for a leg whose first timed run took first_s seconds, both capped at cap."""
    for limit, floor, ceiling in REPEAT_SCHEDULE:
        if first_s < limit:
            return min(floor, cap), min(ceiling, cap)


def repeats_done(timed_s, floor, ceiling):
    """True when the timed runs so far settle the leg's minimum: the ceiling is reached, or the floor is and median/min - 1 is within REPEAT_SPREAD."""
    if len(timed_s) >= ceiling:
        return True
    if len(timed_s) < floor:
        return False
    import statistics
    return statistics.median(timed_s) / min(timed_s) - 1.0 <= REPEAT_SPREAD


def timed_min_ms(run, repeats, on_breach=None):
    """(best_ms, result, samples) after one warm-up; best_ms None on a breach. samples holds every attempt in ms, warm-up first. The repeat count follows the first timed run's duration, capped at `repeats`. With on_breach, a run that never returns hard-exits through run_watchdogged."""
    import timeit
    samples = []
    timed = []
    floor = ceiling = None
    while True:
        elapsed = timeit.default_timer()
        result = (run() if on_breach is None
                  else run_watchdogged(run, on_breach))
        elapsed = timeit.default_timer() - elapsed
        samples.append(elapsed * 1000.0)
        if elapsed > WATCHDOG_SECONDS:
            return None, result, samples
        if len(samples) == 1:
            continue                     # the warm-up carries the compile
        timed.append(elapsed)
        if floor is None:
            floor, ceiling = repeat_bounds(timed[0], repeats)
        if repeats_done(timed, floor, ceiling):
            return min(timed) * 1000.0, result, samples


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


def samples_outfile(framework_dir, prefix, analysis, mode, algorithm,
                    dataset_key, problem=DEFAULT_PROBLEM):
    """Path of the per-repeat timing log under data/<package>/<key>/<problem>."""
    return os.path.join(data_dir(framework_dir, dataset_key, problem=problem),
                        "{0}_samples_{1}_{2}_{3}.csv".format(
                            prefix, analysis, mode, algorithm))


def sample_point(analysis, problem, algorithm, mode, n, states,
                 setting_kind="none", setting=float("nan")):
    """The identity of one timed point, shared by its timed legs."""
    return {"analysis": analysis, "problem": problem, "algorithm": algorithm,
            "mode": mode, "setting_kind": setting_kind, "setting": setting,
            "n": n, "states": states}


def reset_samples(path):
    """Drop a leg's log, for the sweeps whose reduced file is rewritten."""
    if os.path.exists(path):
        os.remove(path)


def append_samples(path, point, transfers, samples):
    """Append one row per attempt of one timed leg, warm-up as repeat 0."""
    header = not os.path.exists(path)
    with open(path, "a") as handle:
        if header:
            handle.write(",".join(SAMPLE_FIELDS) + "\n")
        head = "{analysis},{problem},{algorithm},{mode}".format(**point)
        tail = "{setting_kind},{setting:.10g},{n},{states}".format(**point)
        for repeat, ms in enumerate(samples):
            handle.write("{0},{1},{2},{3},{4:.6f}\n".format(
                head, transfers, tail, repeat, ms))


def parse_bench_args(argv, framework):
    """Parse <N|N,N,...>|wp|states|warm[:N,N,...]|optimize [algorithm|all] [--problem <name|all>] [--mode <fixed|adaptive|all>] into (ns, analysis, algorithms, problems, modes)."""
    if not argv:
        raise SystemExit("usage: <N|N,N,...>|wp|states|warm[:N,N,...]|optimize "
                         "[algorithm|all] [--problem <name|all>] "
                         "[--mode <fixed|adaptive|all>]")
    if argv[0] == "wp":
        analysis, ns = "wp", [N_WP]
    elif argv[0] == "optimize":
        analysis, ns = "optimize", [N_WP]
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
    mode_request = "all"
    rest = list(argv[1:])
    while rest:
        tok = rest.pop(0)
        if tok in ("--problem", "-s", "--mode"):
            if not rest:
                raise SystemExit("{0} requires a value".format(tok))
            if tok == "--mode":
                mode_request = rest.pop(0)
            else:
                problem_request = rest.pop(0)
        elif tok.startswith("--problem="):
            problem_request = tok.split("=", 1)[1]
        elif tok.startswith("--mode="):
            mode_request = tok.split("=", 1)[1]
        else:
            request = tok
    algorithms = resolve_algorithms(request, framework)
    problems = resolve_problems(problem_request, framework)
    return ns, analysis, algorithms, problems, resolve_modes(mode_request)
