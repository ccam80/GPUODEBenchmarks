#!/usr/bin/env python3
"""Move keyed files into data/<package>/<key>/ and plots/<group>/; --dry-run lists the moves."""

from __future__ import annotations

import argparse
import os
import re
import shutil
from pathlib import Path

PACKAGE_DIRS = {"Julia": "Julia", "CPP": "MPGOS", "JAX": "Jax", "PYTORCH": "Torch",
                "CUBIE": "Cubie", "CUBIE_MLIR": "Cubie_mlir",
                "MYOKIT_CUDA": "Myokit_cuda"}
NUMERICAL_PACKAGES = ["cubie_unadaptive", "cubie_adaptive", "cubie_mlir_unadaptive",
                      "cubie_mlir_adaptive", "myokit_cuda", "jax", "pytorch",
                      "julia_adaptive", "julia_fixed", "mpgos_internalsave", "mpgos"]
KEY = re.compile(r"^(?P<stem>.+)_(?P<os>linux|macos|windows|unknown)_(?P<gpu>[^_]+)$")


def moves(root):
    """Pairs of (source, destination) for every file that should move."""
    out = []

    for package, prefix in PACKAGE_DIRS.items():
        d = root / "data" / package
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.txt")):
            m = KEY.match(f.stem)
            if not m or not m.group("stem").startswith(prefix + "_"):
                continue
            key = "{0}_{1}".format(m.group("os"), m.group("gpu"))
            out.append((f, d / key / (m.group("stem") + ".txt")))

    d = root / "data" / "numerical"
    if d.is_dir():
        for f in sorted(d.glob("*.csv")):
            if f.stem.startswith("golden"):
                continue
            m = KEY.match(f.stem)
            if not m or m.group("stem") not in NUMERICAL_PACKAGES:
                continue
            key = "{0}_{1}".format(m.group("os"), m.group("gpu"))
            out.append((f, d / key / (m.group("stem") + ".csv")))

    d = root / "data" / "numerical_equivalence" / "cubie"
    if d.is_dir():
        for f in sorted(d.glob("*.csv")):
            m = KEY.match(f.stem)
            if not m:
                continue
            key = "{0}_{1}".format(m.group("os"), m.group("gpu"))
            out.append((f, d / key / (m.group("stem") + ".csv")))

    d = root / "data" / "clocks"
    if d.is_dir():
        for f in sorted(d.glob("calibration_*.csv")):
            key = f.stem[len("calibration_"):]
            out.append((f, d / key / "calibration.csv"))

    plots = root / "plots"
    if plots.is_dir():
        for f in sorted(plots.glob("*.png")):
            stem = f.stem
            # Equivalence plots carry the full key; Lorenz plots a single field.
            for prefix, name in (("numerical_equivalence_adaptive_", "numerical_equivalence_adaptive"),
                                 ("numerical_equivalence_", "numerical_equivalence")):
                if stem.startswith(prefix):
                    out.append((f, plots / stem[len(prefix):] / (name + ".png")))
                    break
            else:
                for prefix, name in (("Lorenz_wp_", "Lorenz_wp"), ("Lorenz_", "Lorenz")):
                    if stem.startswith(prefix):
                        middle, _, group = stem[len(prefix):].rpartition("_")
                        target = "_".join(q for q in (name, middle) if q) + ".png"
                        out.append((f, plots / group / target))
                        break

    for f in sorted(root.glob("pairwise_comparisons_*.md")):
        out.append((f, plots / f.stem[len("pairwise_comparisons_"):] / "pairwise_comparisons.md"))
    for f in sorted(root.glob("numerical_equivalence_*.md")):
        out.append((f, plots / f.stem[len("numerical_equivalence_"):] / "numerical_equivalence.md"))

    return [(s, t) for s, t in out if s.resolve() != t.resolve()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    pairs = moves(args.root.resolve())
    if not pairs:
        print("Nothing to move.")
        return 0
    for src, dst in pairs:
        rel_src = src.relative_to(args.root.resolve())
        rel_dst = dst.relative_to(args.root.resolve())
        print("{0}  ->  {1}".format(rel_src, rel_dst))
        if not args.dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
    print("{0} file(s){1}".format(len(pairs), " (dry run)" if args.dry_run else " moved"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
