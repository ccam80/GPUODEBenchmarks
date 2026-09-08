"""Continuation of partial runs against the result store; mirrored by resume.jl.

BENCH_RESUME=1 skips every point with a row in the store (NaN rows count).
BENCH_NO_OVERWRITE=1 skips only points whose row holds a finite time.
BENCH_RESUME_FROM is a cursor problem[:algorithm][:fixed|adaptive][:N] into the
run order (problems.csv, then algorithms.csv, fixed before adaptive, N
ascending); points strictly before it are skipped. The problem[:N] form floors
every leg of that problem at N; in the states sweep N is the state count. A wp
leg is skipped only when every setting is covered. BENCH_FLOOR=1 skips nothing
and each result merges into the store keeping the lower time.
"""

import os

from algorithms import MODES, algorithm_names, get_algorithm
from problems import get_problem, problem_names
from protocol import N_WP, STATES_N, TOLS
from results import floor_enabled  # noqa: F401

_CURSOR_CACHE = []          # [] = unparsed, [None] or [dict] once parsed


def resume_enabled():
    """True when BENCH_RESUME asks for skip-what-is-on-disk continuation."""
    return os.environ.get("BENCH_RESUME", "") not in ("", "0")


def no_overwrite_enabled():
    """True when BENCH_NO_OVERWRITE asks to keep only finite recorded rows."""
    return os.environ.get("BENCH_NO_OVERWRITE", "") not in ("", "0")


def parse_cursor(spec):
    """BENCH_RESUME_FROM spec -> cursor dict; omitted parts are None."""
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
    """True when any continuation mechanism is switched on."""
    return resume_enabled() or no_overwrite_enabled() or cursor() is not None


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


def _status_skips(status):
    """Whether a recorded status is covered under the active flags."""
    if resume_enabled() and status != "absent":
        return True
    return no_overwrite_enabled() and status == "finite"


def skip_point(leg, n, states=None):
    """True when one N or states point of a results.Leg is covered."""
    key = states if leg.analysis == "states" else n
    if cursor_skips(leg.problem.name, leg.algorithm, leg.mode, key):
        return True
    return _status_skips(leg.status(n, states))


def skip_wp_leg(leg, settings):
    """True when every setting of a work-precision results.Leg is covered."""
    if cursor_skips(leg.problem.name, leg.algorithm, leg.mode):
        return True
    if not (resume_enabled() or no_overwrite_enabled()):
        return False
    return all(_status_skips(leg.status(N_WP, setting=setting))
               for setting in settings)


def _cli(argv):
    """Shell entry: "skip"/"run" for a point or a wp leg of a package."""
    from results import Leg
    usage = ("usage: resume.py point <package> <key> <analysis> <problem> "
             "<algorithm> <mode> <N|states> | leg <package> <key> <problem> "
             "<algorithm> <mode>")
    if len(argv) == 8 and argv[0] == "point":
        package, key, analysis, problem, algorithm, mode, value = argv[1:]
        leg = Leg(package, key, analysis, problem, algorithm, mode)
        if analysis == "states":
            skip = skip_point(leg, STATES_N, int(value))
        else:
            skip = skip_point(leg, int(value))
    elif len(argv) == 6 and argv[0] == "leg":
        package, key, problem, algorithm, mode = argv[1:]
        leg = Leg(package, key, "wp", problem, algorithm, mode)
        settings = (leg.problem.dts(algorithm) if mode == "fixed" else TOLS)
        skip = skip_wp_leg(leg, settings)
    else:
        raise SystemExit(usage)
    print("skip" if skip else "run")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_cli(sys.argv[1:]))
