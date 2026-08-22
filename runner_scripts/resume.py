"""Continuation of partial runs: skip benchmark points already on disk and
points before a run-order cursor; mirrored by resume.jl for the Julia bench.

Environment contract (set by run_benchmark.sh / run_benchmark.bat):

- BENCH_RESUME=1
    Skip every point whose row is already in its output file. A row counts
    even when its timings are NaN: a recorded failure is a result.

- BENCH_RESUME_FROM=problem[:algorithm][:fixed|adaptive][:N]
    A cursor into the deterministic run order (problems.csv order, then
    algorithms.csv order, then fixed before adaptive, then ascending N).
    Points strictly before the cursor are skipped:
      lorenz                          - start at lorenz; earlier problems skip
      lorenz:tsit5                    - ... and lorenz legs before tsit5 skip
      lorenz:tsit5:adaptive:131072    - ... that leg starts at N=131072
      lorenz:131072                   - every lorenz leg starts at N=131072
    The N component of the problem[:N] form floors every leg of that problem;
    with an algorithm named it floors only the named leg, and later legs run
    in full. In the states sweep N is the state count.

Work-precision legs have no N axis: the cursor applies down to (problem,
algorithm, mode), and BENCH_RESUME skips a wp leg only when its file already
holds a row per setting (a partial wp file is rewritten whole).
"""

import os

from algorithms import algorithm_names, get_algorithm
from problems import get_problem, problem_names

MODES = ("fixed", "adaptive")

_CURSOR_CACHE = []          # [] = unparsed, [None] or [dict] once parsed


def resume_enabled():
    """True when BENCH_RESUME asks for skip-what-is-on-disk continuation."""
    return os.environ.get("BENCH_RESUME", "") not in ("", "0")


def parse_cursor(spec):
    """BENCH_RESUME_FROM spec -> {problem, algorithm, mode, n} indices/value.

    algorithm/mode/n are None when the spec omits them; a bad spec exits.
    """
    parts = [tok for tok in spec.split(":")]
    if not parts or not parts[0]:
        raise SystemExit("BENCH_RESUME_FROM requires a problem name, got "
                         "'{0}'".format(spec))
    get_problem(parts[0])
    cursor = {"problem": problem_names().index(parts[0]),
              "algorithm": None, "mode": None, "n": None}
    for tok in parts[1:]:
        if tok.isdigit():
            if cursor["n"] is not None:
                raise SystemExit("BENCH_RESUME_FROM '{0}': more than one N"
                                 .format(spec))
            cursor["n"] = int(tok)
        elif tok in MODES:
            if cursor["algorithm"] is None or cursor["mode"] is not None \
                    or cursor["n"] is not None:
                raise SystemExit(
                    "BENCH_RESUME_FROM '{0}': the mode goes after the "
                    "algorithm and before N".format(spec))
            cursor["mode"] = MODES.index(tok)
        else:
            if cursor["algorithm"] is not None or cursor["n"] is not None:
                raise SystemExit(
                    "BENCH_RESUME_FROM '{0}': expected problem[:algorithm]"
                    "[:fixed|adaptive][:N]".format(spec))
            get_algorithm(tok)
            cursor["algorithm"] = algorithm_names().index(tok)
    if cursor["algorithm"] is not None and cursor["mode"] is None:
        cursor["mode"] = 0
    return cursor


def cursor():
    """The parsed BENCH_RESUME_FROM cursor, or None; parsed once."""
    if not _CURSOR_CACHE:
        spec = os.environ.get("BENCH_RESUME_FROM", "")
        _CURSOR_CACHE.append(parse_cursor(spec) if spec else None)
    return _CURSOR_CACHE[0]


def _reset_cache():
    """Forget the parsed cursor (tests change the environment)."""
    del _CURSOR_CACHE[:]


def active():
    """True when either continuation mechanism is switched on."""
    return resume_enabled() or cursor() is not None


def cursor_skips(problem, algorithm, mode, n=None):
    """True when (problem, algorithm, mode[, n]) is before the cursor."""
    cur = cursor()
    if cur is None:
        return False
    pi = problem_names().index(problem)
    if pi != cur["problem"]:
        return pi < cur["problem"]
    if cur["algorithm"] is None:
        return (cur["n"] is not None and n is not None and n < cur["n"])
    ai = algorithm_names().index(algorithm)
    mi = MODES.index(mode)
    if (ai, mi) != (cur["algorithm"], cur["mode"]):
        return (ai, mi) < (cur["algorithm"], cur["mode"])
    return cur["n"] is not None and n is not None and n < cur["n"]


def recorded_values(path):
    """First-column integers of the rows already in an output file."""
    values = set()
    try:
        with open(path) as handle:
            for line in handle:
                fields = line.split()
                if len(fields) < 2:
                    continue
                try:
                    values.add(int(float(fields[0])))
                except ValueError:
                    continue
    except OSError:
        pass
    return values


def _row_count(path):
    try:
        with open(path) as handle:
            return sum(1 for line in handle if len(line.split()) >= 2)
    except OSError:
        return 0


def skip_point(problem, algorithm, mode, n, outfile):
    """True when one (problem, algorithm, mode, n) sweep point is covered."""
    if cursor_skips(problem, algorithm, mode, n):
        return True
    return resume_enabled() and n in recorded_values(outfile)


def wp_settings_count(problem, algorithm, mode):
    """Rows a complete wp file holds; grids mirrored across the writers."""
    if mode == "fixed":
        return len(get_problem(problem).dts(algorithm))
    from wp_common import TOLS
    return len(TOLS)


def skip_wp_leg(problem, algorithm, mode, outfile):
    """True when a whole work-precision leg is covered."""
    if cursor_skips(problem, algorithm, mode):
        return True
    return (resume_enabled() and _row_count(outfile)
            >= wp_settings_count(problem, algorithm, mode))


def _cli(argv):
    """point <problem> <alg> <mode> <N> <outfile> | leg <problem> <alg>
    <mode> <outfile>: prints "skip" or "run" for the shell runners."""
    usage = ("usage: resume.py point <problem> <algorithm> <mode> <N> "
             "<outfile> | leg <problem> <algorithm> <mode> <outfile>")
    if len(argv) >= 1 and argv[0] == "point" and len(argv) == 6:
        skip = skip_point(argv[1], argv[2], argv[3], int(argv[4]), argv[5])
    elif len(argv) >= 1 and argv[0] == "leg" and len(argv) == 5:
        skip = skip_wp_leg(argv[1], argv[2], argv[3], argv[4])
    else:
        raise SystemExit(usage)
    print("skip" if skip else "run")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_cli(sys.argv[1:]))
