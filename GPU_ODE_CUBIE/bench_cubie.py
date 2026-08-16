#!/usr/bin/env python

"""Cubie benchmarks on the stock numba-cuda backend: bench_cubie.py <N>|wp [algorithm|all] [--problem <name|all>]"""

import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "runner_scripts"))
from cubie_bench import run

if __name__ == "__main__":
    sys.exit(run(sys.argv[1:], framework="cubie", framework_dir="CUBIE",
                 prefix="Cubie", numerical_tag="cubie"))
