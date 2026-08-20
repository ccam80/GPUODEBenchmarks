#!/usr/bin/env python3

"""Detect the installed CUDA major version for the setup scripts."""

import re
import subprocess

SUPPORTED_MAJORS = (12, 13)

_PROBES = (
    (["nvcc", "--version"], r"release\s+(\d+)(?:\.\d+)?"),
    (["nvidia-smi"], r"CUDA Version:\s*(\d+)(?:\.\d+)?"),
)


def detect_cuda_major():
    """Return 12 or 13 from the toolchain on PATH, or raise RuntimeError."""
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
            major = int(match.group(1))
            if major in SUPPORTED_MAJORS:
                return major
            raise RuntimeError(
                "CUDA {0} is unsupported; expected CUDA {1}."
                .format(major, " or ".join(str(m) for m in SUPPORTED_MAJORS))
            )
    raise RuntimeError(
        "Could not detect CUDA from nvcc or nvidia-smi on PATH."
    )
