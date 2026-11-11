#!/usr/bin/env python

"""gradsolve runner: bench_gradsolve.py --trials <path> [--floor]"""

import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "runner_scripts"))
from gradsolve_bench import run

if __name__ == "__main__":
    sys.exit(run(sys.argv[1:]))
