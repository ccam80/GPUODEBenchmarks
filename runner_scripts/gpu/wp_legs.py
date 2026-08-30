#!/usr/bin/env python
"""Work-precision legs a python framework runs, one "<problem> <algorithm>" per
line, so a runner can give each leg its own process."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))

from algorithms import resolve_algorithms, supported_for
from problems import resolve_problems


def main(argv):
    if len(argv) < 1:
        raise SystemExit("usage: wp_legs.py <framework> [algorithm] [problem]")
    framework = argv[0]
    algorithm = argv[1] if len(argv) > 1 else "all"
    problem = argv[2] if len(argv) > 2 else "all"
    algorithms = resolve_algorithms(algorithm, framework)
    modes = ("fixed", "adaptive")
    for row in resolve_problems(problem, framework):
        for alg in algorithms:
            # Only legs the framework actually runs in some step mode.
            if any(alg in supported_for(framework, mode) for mode in modes):
                print("{0} {1}".format(row.name, alg))


if __name__ == "__main__":
    main(sys.argv[1:])
