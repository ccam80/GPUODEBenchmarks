#!/usr/bin/env python

"""modax runner: bench_modax.py --trials <path> [--floor]"""

import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "runner_scripts"))
from modax_bench import run

if __name__ == "__main__":
    sys.exit(run(sys.argv[1:]))
