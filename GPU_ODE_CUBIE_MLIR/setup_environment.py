#!/usr/bin/env python3
"""
Cross-platform setup for the CUBIE (MLIR backend) ODE benchmarking suite.

There is no separate MLIR environment any more. Since ccam80/cubie#617 the MLIR
backend ships alongside the default numba-cuda backend in a single cubie
install, and the active backend is chosen at *import time* by the
``CUBIE_CUDA_BACKEND`` environment variable (``numba-cuda`` | ``mlir``).

GPU_ODE_CUBIE/setup_environment.py therefore builds one shared venv holding
cubie from PyPI plus *both* backend wheels (``numba-cuda`` and
``cubie-numba-cuda-mlir``). This script just points ``GPU_ODE_CUBIE_MLIR/venv``
at that shared venv, so the existing launchers -- which activate
``./GPU_ODE_CUBIE_MLIR/venv`` and export ``CUBIE_CUDA_BACKEND=mlir`` -- keep
working unchanged.

Works on Linux, Windows, and macOS.
"""
import os
import subprocess
import sys
import platform
from pathlib import Path

SHARED_SUITE = "GPU_ODE_CUBIE"


def link_dir(link_path, target):
    """Point `link_path` at `target`, cross-platform.

    Uses a symlink where available. On Windows, where symlinks need developer
    mode or elevation, falls back to a directory junction (`mklink /J`), which
    needs neither.
    """
    try:
        os.symlink(target, link_path, target_is_directory=True)
        return True
    except (OSError, NotImplementedError, AttributeError) as exc:
        if platform.system() != "Windows":
            print(f"Error: could not create symlink {link_path} -> {target}: {exc}")
            return False
        print(f"Symlink unavailable ({exc}); falling back to a directory junction...")

    try:
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link_path), str(target)],
                       check=True, text=True)
        return True
    except (subprocess.CalledProcessError, OSError) as exc:
        print(f"Error: could not create junction {link_path} -> {target}: {exc}")
        return False


def main():
    script_dir = Path(__file__).parent.resolve()
    shared_venv = (script_dir.parent / SHARED_SUITE / "venv").resolve()
    venv_path = script_dir / "venv"

    print("Setting up CUBIE-MLIR environment...")
    print(f"Shared cubie venv: {shared_venv}")

    # The shared venv is built by the CUBIE suite (step 1/6 of
    # setup_all_environments.py). Build it on demand if run out of order.
    if not shared_venv.exists():
        print(f"Shared venv not found; running {SHARED_SUITE}/setup_environment.py first...")
        shared_setup = script_dir.parent / SHARED_SUITE / "setup_environment.py"
        if not shared_setup.exists():
            print(f"Error: {shared_setup} not found")
            return 1
        if subprocess.run([sys.executable, str(shared_setup)]).returncode != 0:
            print("Error: shared cubie environment setup failed")
            return 1

    # Resolve through any existing link before comparing, so an already-correct
    # link is left alone. `resolve()` follows both symlinks and Windows
    # junctions, so this recognises either form.
    if venv_path.exists() or venv_path.is_symlink():
        if venv_path.resolve() == shared_venv:
            print("venv already points at the shared cubie environment; nothing to do.")
            return 0
        if venv_path.is_symlink():
            print("Replacing stale venv link...")
            venv_path.unlink()
        else:
            print(f"Error: {venv_path} exists and is not a link to the shared venv.")
            print("This suite now shares the CUBIE venv. Remove it and re-run:")
            print(f"  rm -rf {venv_path}")
            return 1

    if not link_dir(venv_path, shared_venv):
        return 1

    # Verify cubie imports through the link and resolves to the MLIR backend.
    is_windows = platform.system() == "Windows"
    venv_python = (venv_path / ("Scripts/python.exe" if is_windows else "bin/python"))
    print("Verifying installation (MLIR backend)...")
    verify_env = os.environ.copy()
    verify_env["CUBIE_CUDA_BACKEND"] = "mlir"
    verify_code = (
        "import cubie; "
        "from cubie.cuda_backend import CUDA_BACKEND, IS_MLIR; "
        "assert IS_MLIR, 'resolved backend: ' + CUDA_BACKEND; "
        "print('Cubie', cubie.__version__, 'installed; backend =', CUDA_BACKEND)"
    )
    if subprocess.run([str(venv_python), "-c", verify_code], env=verify_env).returncode != 0:
        print("Failed to import cubie or the MLIR backend did not activate")
        return 1

    print("\nCUBIE-MLIR environment setup complete!")
    print(f"{venv_path} -> {shared_venv}")
    print("Backend is selected at runtime via CUBIE_CUDA_BACKEND=mlir "
          "(the benchmark launchers set this automatically).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
