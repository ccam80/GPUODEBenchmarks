"""Master per-repeat timing table: every per-repeat log under data/, tagged with
the package, machine key and writer prefix its path carries.

The benchmark writers leave one `<Prefix>_samples_<analysis>_<mode>_<algorithm>.csv`
beside each reduced output file, holding one row per attempt behind that point's
minimum (see `runner_scripts/wp_common.py`). This walks them all and writes one
master file, replacing it whole each run::

    $ python3 runner_scripts/collect_samples.py

Each row keeps its log's columns and gains the four the path and the file's shape
carry: the `package` and machine `key` directories, the writer `prefix`, and a
`series` index separating repeat runs of the same leg (see `SERIES_DOC`). No third
party imports, so it runs under a bare interpreter on any machine that holds data.
"""

import argparse
import csv
import os
import sys

# Columns of the per-repeat logs, as written by runner_scripts/wp_common.py,
# runner_scripts/samples.jl and GPU_ODE_MPGOS/Bench.cu.
SAMPLE_FIELDS = ("analysis", "problem", "algorithm", "mode", "transfers",
                 "setting_kind", "setting", "n", "states", "repeat", "ms")

# Columns the path and the file's shape add in front of a log's own.
TAG_FIELDS = ("package", "key", "prefix", "series")

MASTER_FIELDS = TAG_FIELDS + SAMPLE_FIELDS

# A leg's rows sit together, and its blocks then its attempts run in order.
SORT_FIELDS = ("package", "key", "prefix", "analysis", "problem", "algorithm",
               "mode", "transfers", "setting_kind", "setting", "n", "states",
               "series", "repeat")

SERIES_DOC = ("Each block of rows headed by repeat 0 is one timed leg, and "
              "gets its own series index counted from 0 within its file. The "
              "N sweep appends, so a re-run or a resumed run leaves a second "
              "block for a leg it repeats: same point and transfers, later "
              "series. Take a minimum within one series, never across two.")

DEFAULT_OUT = os.path.join("data", "master_run_times.csv")

_SAMPLES_SPLIT = "_samples_"

# Sort keys read as numbers where the column holds one; nan sorts last.
_NUMERIC_FIELDS = ("setting", "n", "states", "repeat", "series")


def repo_root():
    """The checkout holding this script."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def is_samples_file(name):
    """True for a per-repeat log's file name."""
    return name.endswith(".csv") and _SAMPLES_SPLIT in name


def prefix_of(name):
    """The writer prefix a log's file name starts with, e.g. Cubie_mlir."""
    return name.split(_SAMPLES_SPLIT, 1)[0]


def find_sample_files(data_root):
    """Every per-repeat log under a data root, in a stable order."""
    found = []
    for base, dirs, names in os.walk(data_root):
        dirs.sort()
        for name in sorted(names):
            if is_samples_file(name):
                found.append(os.path.join(base, name))
    return found


def tags_for(path, data_root):
    """(package, key, prefix) from a log's path; either directory is "" when the
    log sits above it."""
    parts = os.path.relpath(path, data_root).split(os.sep)
    package = parts[0] if len(parts) > 1 else ""
    key = parts[1] if len(parts) > 2 else ""
    return package, key, prefix_of(parts[-1])


def read_samples(path, data_root, on_skip=None):
    """Rows of one log as master dicts. Rows whose shape does not match the
    header are handed to on_skip and dropped: a log being appended to while this
    runs can end in a torn line."""
    package, key, prefix = tags_for(path, data_root)
    rows = []
    series = -1
    with open(path, "r", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            return rows
        if tuple(header) != SAMPLE_FIELDS:
            if on_skip is not None:
                on_skip(path, 1, "header is {0}".format(",".join(header)))
            return rows
        for number, record in enumerate(reader, start=2):
            if not record:
                continue
            if len(record) != len(SAMPLE_FIELDS):
                if on_skip is not None:
                    on_skip(path, number, "{0} of {1} columns".format(
                        len(record), len(SAMPLE_FIELDS)))
                continue
            row = dict(zip(SAMPLE_FIELDS, record))
            if row["repeat"] == "0":
                series += 1
            row.update(package=package, key=key, prefix=prefix,
                       series=max(series, 0))
            rows.append(row)
    return rows


def _sorts_last(value):
    """Read a sort key as a number; anything unreadable, nan included, sorts last."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("inf")
    return number if number == number else float("inf")


def _sort_key(row):
    """Group a row with its siblings, then order the group by series and repeat."""
    return tuple(_sorts_last(row[field]) if field in _NUMERIC_FIELDS
                 else str(row[field]) for field in SORT_FIELDS)


def collect(data_root, paths=None, on_skip=None):
    """Every per-repeat row under a data root, sorted for a readable diff. Pass
    paths to collect a walk already made."""
    rows = []
    for path in (find_sample_files(data_root) if paths is None else paths):
        rows.extend(read_samples(path, data_root, on_skip=on_skip))
    rows.sort(key=_sort_key)
    return rows


def write_master(rows, out_path):
    """Replace the master file with these rows, atomically."""
    parent = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(parent, exist_ok=True)
    scratch = out_path + ".partial"
    with open(scratch, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MASTER_FIELDS,
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(scratch, out_path)
    return out_path


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Collect every per-repeat timing log under a data root "
                    "into one master file, replacing it.",
        epilog=SERIES_DOC)
    parser.add_argument("--data-root", default=None,
                        help="directory holding the package directories "
                             "(default: <repo>/data)")
    parser.add_argument("--out", default=None,
                        help="master file to replace (default: "
                             "<repo>/{0})".format(DEFAULT_OUT))
    parser.add_argument("--quiet", action="store_true",
                        help="only report skipped rows")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    root = repo_root()
    data_root = args.data_root or os.path.join(root, "data")
    out_path = args.out or os.path.join(root, DEFAULT_OUT)
    if not os.path.isdir(data_root):
        sys.stderr.write("no data root at {0}\n".format(data_root))
        return 1

    skipped = []

    def on_skip(path, number, why):
        skipped.append(path)
        sys.stderr.write("skipped {0}:{1} ({2})\n".format(
            os.path.relpath(path, data_root), number, why))

    files = find_sample_files(data_root)
    rows = collect(data_root, files, on_skip=on_skip)
    write_master(rows, out_path)
    if not args.quiet:
        print("{0} rows from {1} logs{2} -> {3}".format(
            len(rows), len(files),
            ", {0} skipped".format(len(skipped)) if skipped else "",
            os.path.relpath(out_path, root)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
