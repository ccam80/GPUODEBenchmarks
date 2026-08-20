#!/usr/bin/env python3

"""Check that the CUDA toolchain on PATH is the one the wheels target."""

import re
import subprocess

REQUIRED_MAJOR = 13

_PROBES = (
    (["nvcc", "--version"], r"release\s+(\d+)(?:\.\d+)?"),
    (["nvidia-smi"], r"CUDA Version:\s*(\d+)(?:\.\d+)?"),
)


def detect_cuda_major():
    """Return the CUDA major from nvcc or nvidia-smi, or raise RuntimeError."""
    for command, pattern in _PROBES:
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=20,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
        match = re.search(
            pattern,
            result.stdout + "\n" + result.stderr,
            flags=re.IGNORECASE,
        )
        if match is not None:
            return int(match.group(1))
    raise RuntimeError(
        "Could not detect CUDA from nvcc or nvidia-smi on PATH."
    )


def require_cuda13():
    """Raise RuntimeError unless CUDA 13 is the toolchain on PATH."""
    major = detect_cuda_major()
    if major != REQUIRED_MAJOR:
        raise RuntimeError(
            "CUDA {0} found on PATH; this suite needs CUDA {1}."
            .format(major, REQUIRED_MAJOR)
        )
