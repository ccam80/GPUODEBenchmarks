#!/usr/bin/env python3
"""Linux-only setup script for the JAX (Diffrax) ODE benchmarking environment."""
import os
import sys
import subprocess
import platform
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "runner_scripts"))
from cuda_toolkit import require_cuda13

JAX_VERSION = "0.11.1"
DIFFRAX_VERSION = "0.7.2"
EQUINOX_VERSION = "0.13.8"

# jax's CUDA plugins are manylinux-only; a CPU jax cannot run this suite.
CUDA_PLATFORM = "Linux"


def run_command(cmd, shell=False, check=True, cwd=None):
    """Run a command and handle errors, streaming output in real-time."""
    try:
        # Stream output directly to terminal for real-time feedback
        result = subprocess.run(
            cmd,
            shell=shell,
            check=check,
            cwd=cwd,
            text=True,
            encoding='utf-8',
            errors='replace'  # Replace encoding errors instead of failing
        )
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error: Command failed with exit code {e.returncode}")
        return False


def main():
    script_dir = Path(__file__).parent.resolve()
    os.chdir(script_dir)

    print("Setting up JAX/Diffrax environment...")

    if platform.system() != CUDA_PLATFORM:
        print(f"Skipping: jax publishes no CUDA wheels for "
              f"{platform.system()}. Set this suite up on Linux or WSL2.")
        return 0

    # Check if Python is available
    try:
        result = subprocess.run([sys.executable, "--version"], capture_output=True, text=True)
        print(f"Using Python: {result.stdout.strip()}")
    except Exception as e:
        print(f"Error: python3 is not installed: {e}")
        return 1

    try:
        require_cuda13()
    except RuntimeError as error:
        print(f"Error: {error}")
        return 1
    jax_spec = f"jax[cuda13]=={JAX_VERSION}"

    # Create or use existing venv
    venv_path = script_dir / "venv"
    if venv_path.exists():
        print("Virtual environment already exists, using existing one...")
    else:
        print("Creating virtual environment...")
        if not run_command([sys.executable, "-m", "venv", str(venv_path)]):
            print("Failed to create virtual environment")
            return 1

    venv_python = venv_path / "bin" / "python"
    venv_pip = venv_path / "bin" / "pip"

    # Upgrade pip using python -m pip (required for proper upgrade)
    print("Upgrading pip...")
    if not run_command([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"]):
        print("Failed to upgrade pip")
        return 1

    # Install uv package manager
    print("Installing uv package manager...")
    if not run_command([str(venv_pip), "install", "uv"]):
        print("Failed to install uv")
        return 1

    venv_uv = venv_path / "bin" / "uv"

    # One resolve for the whole stack, so diffrax cannot pull a different jax.
    print(f"Installing {jax_spec}, diffrax {DIFFRAX_VERSION}, "
          f"equinox {EQUINOX_VERSION}...")
    if not run_command([str(venv_uv), "pip", "install", "-p", str(venv_python),
                        jax_spec,
                        f"diffrax=={DIFFRAX_VERSION}",
                        f"equinox=={EQUINOX_VERSION}",
                        "numpy", "scipy"]):
        print("Failed to install the JAX stack")
        return 1

    # Fail when jax-cuda plugin wheels from two CUDA generations coexist.
    print("Checking for stale jax-cuda plugin generations...")
    result = subprocess.run(
        [str(venv_python), "-c",
         "import importlib.metadata as m;"
         "names = sorted(d.metadata['Name'] for d in m.distributions()"
         " if (d.metadata['Name'] or '').startswith('jax-cuda'));"
         "print('\\n'.join(names))"],
        capture_output=True, text=True)
    plugins = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    generations = sorted({name.split("-")[1] for name in plugins})
    if len(generations) > 1:
        print("Error: jax CUDA plugin packages from multiple CUDA generations "
              f"are installed side by side: {', '.join(plugins)}")
        print("The stale generation's PJRT plugin raises ALREADY_EXISTS on "
              "every run. Delete the venv and re-run this script, or remove "
              "the stale packages with:")
        stale = [name for name in plugins if name.split("-")[1] != generations[-1]]
        print(f"  {venv_uv} pip uninstall -p {venv_python} {' '.join(stale)}")
        return 1

    # Verify installation
    print("Verifying installation...")
    if not run_command([str(venv_python), "-c", "import jax; print('JAX version:', jax.__version__); print('JAX backend:', jax.default_backend())"]):
        print("Warning: JAX verification failed")

    if not run_command([str(venv_python), "-c", "import diffrax; print('Diffrax', diffrax.__version__)"]):
        print("Warning: Diffrax verification failed")

    if not run_command([str(venv_python), "-c", "import equinox; print('Equinox', equinox.__version__)"]):
        print("Warning: Equinox verification failed")

    print("\nJAX/Diffrax environment setup complete!")
    print(f"To activate: source {venv_path / 'bin' / 'activate'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
