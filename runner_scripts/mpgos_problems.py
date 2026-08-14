"""Print the requested problems MPGOS has a definition for, one per line."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from problems import resolve_problems

HEADERS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "GPU_ODE_MPGOS", "problems")

for row in resolve_problems(sys.argv[1] if len(sys.argv) > 1 else "all", "cpp"):
    if os.path.isfile(os.path.join(HEADERS, row["problem"] + ".cuh")):
        print(row["problem"])
