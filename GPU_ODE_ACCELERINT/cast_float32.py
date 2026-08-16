#!/usr/bin/env python
"""Usage: cast_float32.py <accelerInt-checkout> <output-dir> [single|double]"""

import os
import re
import sys

# accelerInt files the radau2a CPU build needs.
SOURCES = (
    "generic/solver_main.c",
    "generic/solver_generic.c",
    "generic/solver_interface.c",
    "generic/solver_interface.h",
    "generic/solver.h",
    "generic/solver_props.h",
    "generic/complexInverse.c",
    "generic/complexInverse.h",
    "generic/fd_jacob.c",
    "generic/lapack_dfns.h",
    "generic/solver_init.h",
    "generic/read_initial_conditions.h",
    "generic/timer.h",
    "radau2a/radau2a.c",
    "radau2a/radau2a_init.c",
    "radau2a/radau2a_props.c",
    "radau2a/radau2a_props.h",
)

# Real and complex LAPACK kernels have single-precision twins.
_LAPACK = {
    "dgetrf_": "sgetrf_",
    "dgetrs_": "sgetrs_",
    "dgetri_": "sgetri_",
    "zgetrf_": "cgetrf_",
    "zgetrs_": "cgetrs_",
    "zgetri_": "cgetri_",
}

# libm calls that have an f-suffixed single-precision form.
_LIBM = (
    "fabs", "sqrt", "pow", "exp", "log", "log10", "fmin", "fmax",
    "sin", "cos", "tan", "ceil", "floor", "fmod",
    "cabs", "creal", "cimag", "conj", "csqrt", "cexp", "cpow",
)

_LIMITS = {
    "DBL_EPSILON": "FLT_EPSILON",
    "DBL_MIN": "FLT_MIN",
    "DBL_MAX": "FLT_MAX",
}


def _rules():
    """(name, compiled pattern, replacement) in application order."""
    rules = [("double -> float", re.compile(r"\bdouble\b"), "float")]
    for old, new in _LIMITS.items():
        rules.append((old, re.compile(r"\b%s\b" % old), new))
    for old, new in _LAPACK.items():
        rules.append((old, re.compile(r"\b%s" % old), new))
    for name in _LIBM:
        # Match the call site only, so identifiers like exp4 are left alone.
        rules.append((name, re.compile(r"\b%s\s*\(" % name), "%sf(" % name))
    return rules


def cast_file(text, rules, counts):
    for name, pattern, replacement in rules:
        text, n = pattern.subn(replacement, text)
        counts[name] = counts.get(name, 0) + n
    return text


def main(argv):
    if len(argv) not in (3, 4):
        raise SystemExit(__doc__)
    src_root, out_root = argv[1], argv[2]
    precision = argv[3] if len(argv) == 4 else "single"
    if precision not in ("single", "double"):
        raise SystemExit("precision must be single or double")
    rules = _rules() if precision == "single" else []
    counts = {}
    for relative in SOURCES:
        src = os.path.join(src_root, relative)
        if not os.path.isfile(src):
            raise SystemExit("missing accelerInt source: {0}".format(src))
        with open(src, "r", encoding="utf-8") as handle:
            text = handle.read()
        out = os.path.join(out_root, relative)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(cast_file(text, rules, counts))
    print("wrote {0} files at {1} precision".format(len(SOURCES), precision))
    for name, n in sorted(counts.items()):
        if n:
            print("  {0:<16} {1}".format(name, n))


if __name__ == "__main__":
    main(sys.argv)
