#!/usr/bin/env python3
"""Create the Myokit-CUDA benchmark environment."""

import platform
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "runner_scripts"))
from cuda_toolkit import require_cuda13

CUPY_PACKAGE = "cupy-cuda13x"
CUPY_VERSION = "14.2.0"


def run(command):
    """Run a setup command and fail on a non-zero exit."""
    print("+ {0}".format(" ".join(str(part) for part in command)))
    subprocess.run(command, check=True)


def myokit_version(requirements):
    """Return the pinned myokit version from a requirements file."""
    match = re.search(
        r"^myokit==(\S+)$",
        requirements.read_text(),
        flags=re.MULTILINE,
    )
    if match is None:
        raise RuntimeError(
            "No pinned myokit== line in {0}".format(requirements)
        )
    return match.group(1)


def main():
    """Create a venv and install pinned Myokit plus matched CuPy."""
    require_cuda13()
    script_dir = Path(__file__).resolve().parent
    environment = script_dir / "venv"
    if not environment.exists():
        run([sys.executable, "-m", "venv", str(environment)])

    if platform.system() == "Windows":
        python = environment / "Scripts" / "python.exe"
    else:
        python = environment / "bin" / "python"

    requirements = script_dir / "requirements.txt"
    run([str(python), "-m", "pip", "install", "--upgrade", "pip"])
    run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "-r",
            str(requirements),
            "{0}=={1}".format(CUPY_PACKAGE, CUPY_VERSION),
        ]
    )
    run(
        [
            str(python),
            "-c",
            (
                "import cupy, myokit; "
                "assert myokit.__version__ == '{0}'; "
                "assert cupy.__version__ == '{1}'; "
                "print('Myokit', myokit.__version__); "
                "print('CuPy', cupy.__version__); "
                "print('CUDA devices', cupy.cuda.runtime.getDeviceCount())"
            ).format(myokit_version(requirements), CUPY_VERSION),
        ]
    )
    print("Myokit-CUDA environment is ready at {0}".format(environment))
    return 0


if __name__ == "__main__":
    sys.exit(main())
