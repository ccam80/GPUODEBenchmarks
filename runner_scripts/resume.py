"""Continuation of partial runs; mirrored by resume.jl.

BENCH_RESUME=1 skips every point whose row is already in its output file
(NaN rows count as recorded). BENCH_NO_OVERWRITE=1 skips only points whose
row holds a finite time; NaN and absent rows rerun. BENCH_RESUME_FROM is a
cursor problem[:algorithm][:fixed|adaptive][:N] into the run order
(problems.csv, then algorithms.csv, fixed before adaptive, N ascending);
points strictly before it are skipped. The problem[:N] form floors every
leg of that problem at N; in the states sweep N is the state count. A wp
leg is skipped only when its file holds a row per setting. BENCH_FLOOR=1
skips and prunes nothing; each result merges into the recorded file,
keeping the lower time per column (per (time, error) pair for wp rows).
"""

import math
import os

from algorithms import algorithm_names, get_algorithm
from problems import get_problem, problem_names

MODES = ("fixed", "adaptive")

_CURSOR_CACHE = []          # [] = unparsed, [None] or [dict] once parsed


def resume_enabled():
    """True when BENCH_RESUME asks for skip-what-is-on-disk continuation."""
    return os.environ.get("BENCH_RESUME", "") not in ("", "0")


def no_overwrite_enabled():
    """True when BENCH_NO_OVERWRITE asks to keep only finite recorded rows."""
    return os.environ.get("BENCH_NO_OVERWRITE", "") not in ("", "0")


def floor_enabled():
    """True when BENCH_FLOOR asks to merge re-runs by keeping the lower time."""
    return os.environ.get("BENCH_FLOOR", "") not in ("", "0")


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


def _finite(token):
    """The token as a finite float, or None."""
    try:
        value = float(token)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def numeric_values(path):
    """First-column integers of the rows whose time field is finite."""
    values = set()
    try:
        with open(path) as handle:
            for line in handle:
                fields = line.split()
                if len(fields) < 2 or _finite(fields[1]) is None:
                    continue
                try:
                    values.add(int(float(fields[0])))
                except ValueError:
                    continue
    except OSError:
        pass
    return values


def prune_reruns(outfile, ns):
    """Drop the rows for the points about to rerun."""
    # --floor merges at write time; the recorded rows must survive.
    if floor_enabled() or not active() or not ns:
        return
    rerun = set(ns)

    def stale(line):
        fields = line.split()
        if len(fields) < 2:
            return False
        try:
            return int(float(fields[0])) in rerun
        except ValueError:
            return False

    try:
        with open(outfile) as handle:
            lines = handle.readlines()
    except OSError:
        return
    kept = [line for line in lines if not stale(line)]
    if len(kept) < len(lines):
        with open(outfile, "w") as handle:
            handle.writelines(kept)


def _row_count(path, finite_only=False):
    try:
        with open(path) as handle:
            return sum(1 for line in handle
                       if len(line.split()) >= 2
                       and (not finite_only
                            or _finite(line.split()[1]) is not None))
    except OSError:
        return 0


def skip_point(problem, algorithm, mode, n, outfile):
    """True when one (problem, algorithm, mode, n) sweep point is covered."""
    if cursor_skips(problem, algorithm, mode, n):
        return True
    if resume_enabled() and n in recorded_values(outfile):
        return True
    return no_overwrite_enabled() and n in numeric_values(outfile)


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
    expected = wp_settings_count(problem, algorithm, mode)
    if resume_enabled() and _row_count(outfile) >= expected:
        return True
    return (no_overwrite_enabled()
            and _row_count(outfile, finite_only=True) >= expected)


def _lower(recorded, new):
    """The lower of two times; NaN loses to any finite value."""
    if math.isnan(recorded):
        return new
    if math.isnan(new):
        return recorded
    return min(recorded, new)


def _read_lines(path):
    try:
        with open(path) as handle:
            return handle.readlines()
    except OSError:
        return []


def merge_min_row(path, key, values, sep=" "):
    """--floor: merge `key values...` into path, keeping the lower value per column; recorded fields beyond the merged columns survive."""
    values = [float(v) for v in values]
    out = []
    merged = False
    for line in _read_lines(path):
        fields = line.split()
        row_key = None
        if fields:
            try:
                row_key = int(float(fields[0]))
            except ValueError:
                pass
        if merged or row_key != key:
            out.append(line)
            continue
        recorded = []
        for i in range(len(values)):
            try:
                recorded.append(float(fields[1 + i]))
            except (IndexError, ValueError):
                recorded.append(float("nan"))
        combined = [_lower(r, v) for r, v in zip(recorded, values)]
        tail = fields[1 + len(values):]
        out.append(sep.join([str(key)] + [str(v) for v in combined] + tail)
                   + "\n")
        merged = True
    if not merged:
        out.append(sep.join([str(key)] + [str(v) for v in values]) + "\n")
    with open(path, "w") as handle:
        handle.writelines(out)


def merge_wp_row(path, setting, t_ms, err):
    """--floor: merge one wp row, keeping the (time, error) pair with the lower time."""
    new_row = "{0:.10g} {1} {2:.10e}\n".format(setting, t_ms, err)
    out = []
    merged = False
    for line in _read_lines(path):
        fields = line.split()
        row_key = None
        if fields:
            try:
                row_key = float(fields[0])
            except ValueError:
                pass
        if (merged or row_key is None
                or not math.isclose(row_key, setting, rel_tol=1e-8)):
            out.append(line)
            continue
        merged = True
        try:
            recorded_t = float(fields[1])
        except (IndexError, ValueError):
            recorded_t = float("nan")
        if math.isnan(recorded_t) or (not math.isnan(t_ms)
                                      and t_ms < recorded_t):
            out.append(new_row)
        else:
            out.append(line)
    if not merged:
        out.append(new_row)
    with open(path, "w") as handle:
        handle.writelines(out)


def write_times_row(handle, path, key, values):
    """One N-sweep or states row through the open handle, or a --floor merge into path keeping the lower time per column."""
    if floor_enabled():
        merge_min_row(path, key, values)
        return
    handle.write(" ".join([str(key)] + [str(float(v)) for v in values])
                 + "\n")
    handle.flush()


def write_wp_row(handle, path, setting, t_ms, err):
    """One wp row through the open handle, or a --floor merge into path keeping the pair with the lower time."""
    if floor_enabled():
        merge_wp_row(path, setting, t_ms, err)
        return
    handle.write("{0:.10g} {1} {2:.10e}\n".format(setting, t_ms, err))
    handle.flush()


def _cli(argv):
    """Shell entry: "skip"/"run" for a point or wp leg; "prune" drops one
    point's rows; "merge" floor-merges one times/states row."""
    usage = ("usage: resume.py point <problem> <algorithm> <mode> <N> "
             "<outfile> | leg <problem> <algorithm> <mode> <outfile> | "
             "prune <N> <outfile> | merge <outfile> <space|tab> <N> "
             "<value>...")
    if len(argv) >= 1 and argv[0] == "point" and len(argv) == 6:
        skip = skip_point(argv[1], argv[2], argv[3], int(argv[4]), argv[5])
    elif len(argv) >= 1 and argv[0] == "leg" and len(argv) == 5:
        skip = skip_wp_leg(argv[1], argv[2], argv[3], argv[4])
    elif len(argv) >= 1 and argv[0] == "prune" and len(argv) == 3:
        prune_reruns(argv[2], [int(argv[1])])
        return 0
    elif len(argv) >= 5 and argv[0] == "merge" and argv[2] in ("space",
                                                               "tab"):
        merge_min_row(argv[1], int(argv[3]), [float(v) for v in argv[4:]],
                      sep=" " if argv[2] == "space" else "\t")
        return 0
    else:
        raise SystemExit(usage)
    print("skip" if skip else "run")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_cli(sys.argv[1:]))
