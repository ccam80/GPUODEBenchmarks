#!/usr/bin/env python3
"""Create the Myokit-CUDA benchmark environment."""

import platform
import re
import subprocess
import sys
from pathlib import Path


def run(command):
    """Run a setup command and fail on a non-zero exit."""
    print("+ {0}".format(" ".join(str(part) for part in command)))
    subprocess.run(command, check=True)


def cuda_major():
    """Detect CUDA 12 or 13 from the toolchain already on PATH."""
    probes = (
        (["nvcc", "--version"], r"release\s+(\d+)(?:\.\d+)?"),
        (["nvidia-smi"], r"CUDA Version:\s*(\d+)(?:\.\d+)?"),
    )
    for command, pattern in probes:
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
            if major in (12, 13):
                return major
            raise RuntimeError(
                "CUDA {0} is unsupported; expected CUDA 12 or 13."
                .format(major)
            )
    raise RuntimeError(
        "Could not detect CUDA from nvcc or nvidia-smi on PATH."
    )


def main():
    """Create a venv and install pinned Myokit plus matched CuPy."""
    script_dir = Path(__file__).resolve().parent
    environment = script_dir / "venv"
    if not environment.exists():
        run([sys.executable, "-m", "venv", str(environment)])

    if platform.system() == "Windows":
        python = environment / "Scripts" / "python.exe"
    else:
        python = environment / "bin" / "python"

    major = cuda_major()
    cupy_package = "cupy-cuda{0}x".format(major)
    run([str(python), "-m", "pip", "install", "--upgrade", "pip"])
    run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "-r",
            str(script_dir / "requirements.txt"),
            cupy_package,
        ]
    )
    run(
        [
            str(python),
            "-c",
            (
                "import cupy, myokit; "
                "assert myokit.__version__ == '1.39.2'; "
                "print('Myokit', myokit.__version__); "
                "print('CuPy', cupy.__version__); "
                "print('CUDA devices', cupy.cuda.runtime.getDeviceCount())"
            ),
        ]
    )
    print("Myokit-CUDA environment is ready at {0}".format(environment))
    return 0


if __name__ == "__main__":
    sys.exit(main())
