"""Box-side housekeeping of the store, run on the box beside its data/ tree (or in-process against a local tree): under the key's lock, delete the clock logs clocks/<key>_<stamp>.csv that no result row under key=<key> names and that are at least a day old. `serve` holds the key's lock while a client uploads: it prints `held`, waits for one stdin line, prunes on `prune` (or `prune --dry-run`) and releases on anything else or EOF, so an upload and the pruning after it are one serialized step and a failed upload prunes nothing. Needs pyarrow alone. CLI: box_prune.py --root DIR --key KEY [--min-age-days D] [--dry-run] probe | prune | serve."""

import argparse
import glob
import os
import re
import sys
import time

MIN_AGE_S = 86400.0
LOCK_TIMEOUT_S = 3600.0
STAMP = r"\d{8}T\d{6}Z"


class KeyLock:
    """An exclusive lock on <root>/.sync_<key>.lock held by an open file: flock where it exists, else msvcrt; it dies with the process."""

    def __init__(self, root, key, timeout=LOCK_TIMEOUT_S):
        self.path = os.path.join(root, ".sync_{0}.lock".format(key))
        self.timeout = timeout
        self.handle = None

    def __enter__(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.handle = open(self.path, "a+")
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                _lock(self.handle)
                return self
            except OSError:
                if time.monotonic() > deadline:
                    self.handle.close()
                    self.handle = None
                    raise TimeoutError("store locked for another upload: " + self.path)
                time.sleep(0.2)

    def __exit__(self, *_):
        if self.handle is not None:
            try:
                _unlock(self.handle)
            finally:
                self.handle.close()
                self.handle = None


def _lock(handle):
    try:
        import fcntl
    except ImportError:
        import msvcrt
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        return
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock(handle):
    try:
        import fcntl
    except ImportError:
        import msvcrt
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return
    fcntl.flock(handle, fcntl.LOCK_UN)


def runs_named(root, key):
    """The run names the result rows under key=<key> carry, read from the `run` column of every results file; a file that cannot be read raises, since a log it might name must not be pruned."""
    import pyarrow.parquet as pq
    named = set()
    pattern = os.path.join(root, "key=" + key, "package=*", "results", "*.parquet")
    for path in sorted(glob.glob(pattern)):
        schema = pq.read_schema(path)
        if "run" not in schema.names:
            continue
        for run in pq.read_table(path, columns=["run"]).column("run").to_pylist():
            if run:
                named.add(run)
    return named


def orphan_logs(root, key, min_age_s=MIN_AGE_S, now=None):
    """The clocks/<key>_<stamp>.csv files at least min_age_s old that no row under the key names, sorted."""
    now = time.time() if now is None else now
    clocks = os.path.join(root, "clocks")
    if not os.path.isdir(clocks):
        return []
    named = runs_named(root, key)
    log = re.compile("^" + re.escape(key) + "_" + STAMP + r"\.csv$")
    found = []
    for name in sorted(os.listdir(clocks)):
        path = os.path.join(clocks, name)
        if not log.match(name) or not os.path.isfile(path) or name[:-4] in named:
            continue
        if now - os.stat(path).st_mtime < min_age_s:
            continue
        found.append(name)
    return found


def prune(root, key, min_age_s=MIN_AGE_S, dry_run=False):
    """Delete the key's orphan logs (list them alone under dry_run); returns their file names."""
    names = orphan_logs(root, key, min_age_s)
    if not dry_run:
        for name in names:
            os.remove(os.path.join(root, "clocks", name))
    return names


def serve(root, key, min_age_s=MIN_AGE_S, stdin=sys.stdin, stdout=sys.stdout):
    """Hold the key's lock: print `held`, read one line, prune on `prune [--dry-run]` printing each name, release; returns 0."""
    with KeyLock(root, key):
        stdout.write("held\n")
        stdout.flush()
        line = stdin.readline().strip()
        if line.split() and line.split()[0] == "prune":
            for name in prune(root, key, min_age_s, dry_run="--dry-run" in line.split()):
                stdout.write(name + "\n")
        stdout.write("released\n")
        stdout.flush()
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="box_prune.py", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("probe", "prune", "serve"))
    parser.add_argument("--root", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--min-age-days", type=float, default=MIN_AGE_S / 86400.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not os.path.isdir(args.root):
        sys.stderr.write("box_prune: no store at {0}\n".format(args.root))
        return 2
    if args.command == "probe":
        import pyarrow  # noqa: F401 - the probe is that it imports
        print("ready")
        return 0
    min_age_s = args.min_age_days * 86400.0
    if args.command == "prune":
        with KeyLock(args.root, args.key):
            for name in prune(args.root, args.key, min_age_s, args.dry_run):
                print(name)
        return 0
    return serve(args.root, args.key, min_age_s)


if __name__ == "__main__":
    sys.exit(main())
