#!/usr/bin/env python3
"""Rename legacy data files into the algorithm-tagged layout:
<Prefix>_{times,wp}_<unadaptive|fixed|adaptive>_<os>_<gpu>.txt becomes
<Prefix>_{times,wp}_<fixed|adaptive>_<algorithm>_<os>_<gpu>.txt, using the
per-framework algorithm each legacy mode ran (FRAMEWORKS table below).

Usage: python runner_scripts/migrate_data_layout.py [--dry-run] [data-dir]
"""

import os
import re
import sys

# directory -> (prefix, fixed algorithm or None, adaptive algorithm or None)
FRAMEWORKS = {
    "CUBIE": ("Cubie", "classical-rk4", "tsit5"),
    "CUBIE_MLIR": ("Cubie_mlir", "classical-rk4", "tsit5"),
    "Julia": ("Julia", "tsit5", "tsit5"),
    "JAX": ("Jax", "tsit5", "tsit5"),
    "PYTORCH": ("Torch", "classical-rk4", None),
    "MYOKIT_CUDA": ("Myokit_cuda", "euler", None),
    "CPP": ("MPGOS", "classical-rk4", "cash-karp-54"),
}


def planned_renames(data_dir):
    """Yield (old_path, new_path) for every legacy-named file found."""
    for subdir, (prefix, fixed_alg, adaptive_alg) in FRAMEWORKS.items():
        dpath = os.path.join(data_dir, subdir)
        if not os.path.isdir(dpath):
            continue
        # Legacy names have exactly two underscore fields after the mode.
        pat = re.compile(
            "^" + re.escape(prefix)
            + r"_(times|wp)_(unadaptive|fixed|adaptive)_([^_]+_[^_]+)\.txt$")
        for fname in sorted(os.listdir(dpath)):
            m = pat.match(fname)
            if m is None:
                continue
            kind, mode, key = m.groups()
            new_mode = "fixed" if mode in ("unadaptive", "fixed") else "adaptive"
            algorithm = fixed_alg if new_mode == "fixed" else adaptive_alg
            if algorithm is None:
                print("WARNING: {0} has no known {1} algorithm; leaving {2}"
                      .format(subdir, new_mode, fname))
                continue
            new_name = "{0}_{1}_{2}_{3}_{4}.txt".format(
                prefix, kind, new_mode, algorithm, key)
            yield os.path.join(dpath, fname), os.path.join(dpath, new_name)


def main(argv):
    dry_run = "--dry-run" in argv
    argv = [a for a in argv if a != "--dry-run"]
    data_dir = argv[0] if argv else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

    renamed = 0
    for old, new in planned_renames(data_dir):
        if os.path.exists(new):
            print("SKIP (target exists): {0} -> {1}".format(old, new))
            continue
        print("{0}{1} -> {2}".format("DRY RUN: " if dry_run else "", old, new))
        if not dry_run:
            os.rename(old, new)
        renamed += 1
    print("{0} file(s) {1}".format(
        renamed, "would be renamed" if dry_run else "renamed"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
