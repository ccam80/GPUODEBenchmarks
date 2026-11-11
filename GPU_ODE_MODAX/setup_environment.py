#!/usr/bin/env python3
"""Build GPU_ODE_MODAX/venv on Linux with modax-solvers and its CUDA 13 dependencies."""
import os
import platform
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "runner_scripts"))
from cuda_toolkit import require_cuda13

MODAX_VERSION = "0.0.7"
JAX_VERSION = "0.11.1"

# We set up on Linux only, because jax's CUDA plugins and numba-enzyme-cuda exist for Linux only.
CUDA_PLATFORM = "Linux"


def run(cmd, env=None):
    """Run a command, streaming its output, and return True when it exits with status 0."""
    return subprocess.run(cmd, text=True, encoding="utf-8", errors="replace", env=env).returncode == 0


def distributions(venv_python):
    """Return the lower-cased names of every distribution installed in the venv."""
    result = subprocess.run(
        [str(venv_python), "-c",
         "import importlib.metadata as m;"
         "print('\\n'.join(sorted((d.metadata['Name'] or '').lower() for d in m.distributions())))"],
        capture_output=True, text=True)
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def main():
    script_dir = Path(__file__).parent.resolve()
    print("Setting up the modax environment...")
    if platform.system() != CUDA_PLATFORM:
        print(f"Skipping: modax's jax CUDA plugins and Enzyme wheels are Linux-only, not "
              f"{platform.system()}. Set this suite up on Linux.")
        return 0
    try:
        require_cuda13()
    except RuntimeError as error:
        print(f"Error: {error}")
        return 1

    venv_path = script_dir / "venv"
    if venv_path.exists():
        print("Virtual environment already exists, using existing one...")
    elif not run([sys.executable, "-m", "venv", str(venv_path)]):
        print("Failed to create the virtual environment")
        return 1
    venv_python = venv_path / "bin" / "python"
    if not run([str(venv_python), "-m", "pip", "install", "--upgrade", "pip", "uv"]):
        print("Failed to install uv")
        return 1
    venv_uv = venv_path / "bin" / "uv"

    # Resolve the whole stack at once; the runner needs pyarrow and tzdata to write results.
    print(f"Installing modax-solvers[cuda13] {MODAX_VERSION} and jax[cuda13] {JAX_VERSION}...")
    if not run([str(venv_uv), "pip", "install", "-p", str(venv_python),
                f"modax-solvers[cuda13]=={MODAX_VERSION}", f"jax[cuda13]=={JAX_VERSION}",
                "pyarrow", "tzdata"]):
        print("Failed to install the modax stack")
        return 1

    names = distributions(venv_python)
    # Refuse upstream numba-enzyme, because it installs the same numba_enzyme package as numba-enzyme-cuda.
    if "numba-enzyme" in names:
        print("Error: upstream numba-enzyme is installed beside numba-enzyme-cuda. Remove it with:")
        print(f"  {venv_uv} pip uninstall -p {venv_python} numba-enzyme")
        return 1
    generations = sorted({name.split("-")[1] for name in names if name.startswith("jax-cuda")})
    if len(generations) > 1:
        print("Error: jax CUDA plugin packages from several CUDA generations are installed side "
              "by side. Delete the venv and re-run this script.")
        return 1

    # Import on jax's CPU backend so that setup never needs a free GPU.
    print("Verifying the installation...")
    check = ("import importlib.metadata as m, jax, modax.tsit5, modax.rodas5P;"
             "print('modax', m.version('modax-solvers'), 'jax', jax.__version__)")
    if not run([str(venv_python), "-c", check], env=dict(os.environ, JAX_PLATFORMS="cpu")):
        print("Error: the modax stack does not import")
        return 1
    print("modax environment setup complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
