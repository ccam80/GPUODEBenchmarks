#!/usr/bin/env python

"""Cubie benchmarks on the numba-cuda-mlir backend: bench_cubie_mlir.py <N> [wp] [algorithm|all] [--problem <name|all>]"""

import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "runner_scripts"))
from cubie_bench import run

if __name__ == "__main__":
    sys.exit(run(sys.argv[1:], framework="cubie_mlir",
                 framework_dir="CUBIE_MLIR", prefix="Cubie_mlir",
                 numerical_tag="cubie_mlir", name_suffix="_mlir"))
