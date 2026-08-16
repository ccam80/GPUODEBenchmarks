"""Shared dataset-key helper for the Python benchmark writers.

Produces a key "<os>_<gpu>" identifying the current machine so benchmark output
files can be additively populated across machines. The GPU name comes from
nvidia-smi (the single source of truth shared by every framework) and is
sanitised identically everywhere: tokenise on non-alphanumeric characters, drop
the "NVIDIA"/"GeForce" vendor words, and join the rest with '-'.
e.g. "NVIDIA GeForce RTX 2060 SUPER" -> "RTX-2060-SUPER".

Import from a benchmark script with::

    import os, sys
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runner_scripts"))
    from bench_key import dataset_key
"""

import os
import platform
import re
import subprocess


def _gpu_name_raw():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
        )
        if out.returncode == 0:
            for line in out.stdout.splitlines():
                line = line.strip()
                if line:
                    return line
    except Exception:
        pass
    return ""


def _sanitize_gpu(raw):
    tokens = [t for t in re.split(r"[^A-Za-z0-9]+", raw)
              if t and t not in ("NVIDIA", "GeForce")]
    return "-".join(tokens) if tokens else "unknown-gpu"


def _os_key():
    return {"Linux": "linux", "Darwin": "macos", "Windows": "windows"}.get(
        platform.system(), "unknown")


def dataset_key():
    """Return "<os>_<gpu>" for this machine."""
    return "{0}_{1}".format(_os_key(), _sanitize_gpu(_gpu_name_raw()))


def data_dir(package, key=None, root="", problem=None):
    """Directory holding one machine's files for a package and problem; creates it."""
    d = os.path.join(root, "data", package, key or dataset_key())
    if problem is not None:
        d = os.path.join(d, problem["problem"] if isinstance(problem, dict)
                         else problem)
    os.makedirs(d, exist_ok=True)
    return d


def group_dir(group, problem=None):
    """Directory holding one group's plots and reports; creates it."""
    d = os.path.join("plots", group)
    if problem is not None:
        d = os.path.join(d, problem["problem"] if isinstance(problem, dict)
                         else problem)
    os.makedirs(d, exist_ok=True)
    return d


if __name__ == "__main__":
    print(dataset_key())
