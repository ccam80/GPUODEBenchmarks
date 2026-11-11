#!/usr/bin/env python3
"""Build GPU_ODE_GRADSOLVE/venv on Linux with gradsolve, NVIDIA Warp and a CUDA 13 jax."""
import os
import platform
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "runner_scripts"))
from cuda_toolkit import require_cuda13

GRADSOLVE_VERSION = "0.2.1"
WARP_VERSION = "1.19.0"
JAX_VERSION = "0.11.1"

# We set up on Linux only, because jax publishes its CUDA plugins for Linux only.
CUDA_PLATFORM = "Linux"


def run(cmd, env=None):
    """Run a command, streaming its output, and return True when it exits with status 0."""
    return subprocess.run(cmd, text=True, encoding="utf-8", errors="replace", env=env).returncode == 0


def jax_cuda_generations(venv_python):
    """Return the venv's jax-cuda plugin names and the CUDA generations they belong to."""
    result = subprocess.run(
        [str(venv_python), "-c",
         "import importlib.metadata as m;"
         "print('\\n'.join(sorted(d.metadata['Name'] for d in m.distributions()"
         " if (d.metadata['Name'] or '').startswith('jax-cuda'))))"],
        capture_output=True, text=True)
    plugins = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return plugins, sorted({name.split("-")[1] for name in plugins})


def main():
    script_dir = Path(__file__).parent.resolve()
    print("Setting up the gradsolve environment...")
    if platform.system() != CUDA_PLATFORM:
        print(f"Skipping: jax publishes no CUDA wheels for {platform.system()}. "
              "Set this suite up on Linux.")
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
    print(f"Installing gradsolve {GRADSOLVE_VERSION}, warp-lang {WARP_VERSION} and "
          f"jax[cuda13] {JAX_VERSION}...")
    if not run([str(venv_uv), "pip", "install", "-p", str(venv_python),
                f"gradsolve=={GRADSOLVE_VERSION}", f"warp-lang=={WARP_VERSION}",
                f"jax[cuda13]=={JAX_VERSION}", "numpy", "scipy", "pyarrow", "tzdata"]):
        print("Failed to install the gradsolve stack")
        return 1

    plugins, generations = jax_cuda_generations(venv_python)
    if len(generations) > 1:
        print("Error: jax CUDA plugin packages from several CUDA generations are installed "
              f"side by side: {', '.join(plugins)}. Delete the venv and re-run this script.")
        return 1

    # Import on jax's CPU backend so that setup never needs a free GPU.
    print("Verifying the installation...")
    check = ("import gradsolve, jax, warp; print('gradsolve', gradsolve.__version__, "
             "'jax', jax.__version__, 'warp', warp.__version__)")
    if not run([str(venv_python), "-c", check], env=dict(os.environ, JAX_PLATFORMS="cpu")):
        print("Error: the gradsolve stack does not import")
        return 1
    print("gradsolve environment setup complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
