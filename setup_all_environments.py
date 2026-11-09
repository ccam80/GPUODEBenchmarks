#!/usr/bin/env python3
"""
Master script to setup all GPU ODE benchmark environments.
Cross-platform: Works on Linux, Windows, and macOS.
"""
import os
import sys
import subprocess
import platform
from pathlib import Path


def run_setup_script(script_path, name):
    """Run a setup script and return success status, streaming output in real-time."""
    print("=" * 50)
    print(f"Setting up {name} environment...")
    print("=" * 50)
    
    if not script_path.exists():
        print(f"✗ {name} setup script not found: {script_path}")
        return False
    
    try:
        # Stream output directly to terminal for real-time feedback
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            text=True,
            encoding='utf-8',
            errors='replace'  # Replace encoding errors instead of failing
        )
        print(f"✓ {name} setup completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {name} setup failed")
        return False
    except Exception as e:
        print(f"✗ {name} setup failed with error: {e}")
        return False


def main():
    script_dir = Path(__file__).parent.resolve()
    os.chdir(script_dir)
    
    print("=" * 50)
    print("Setting up all GPU ODE benchmark environments")
    print("=" * 50)
    print()
    
    # Track failures
    failed_setups = []
    
    # Setup CUBIE
    print("=" * 50)
    print("1/7: Setting up CUBIE environment...")
    print("=" * 50)
    cubie_setup = script_dir / "GPU_ODE_CUBIE" / "setup_environment.py"
    if not run_setup_script(cubie_setup, "CUBIE"):
        failed_setups.append("CUBIE")
    print()

    # Setup JAX/Diffrax
    print("=" * 50)
    print("2/7: Setting up JAX/Diffrax environment...")
    print("=" * 50)
    jax_setup = script_dir / "GPU_ODE_JAX" / "setup_environment.py"
    if not run_setup_script(jax_setup, "JAX"):
        failed_setups.append("JAX")
    print()

    # Setup PyTorch/torchdiffeq
    print("=" * 50)
    print("3/7: Setting up PyTorch/torchdiffeq environment...")
    print("=" * 50)
    pytorch_setup = script_dir / "GPU_ODE_PyTorch" / "setup_environment.py"
    if not run_setup_script(pytorch_setup, "PyTorch"):
        failed_setups.append("PyTorch")
    print()

    # Setup Myokit-CUDA
    print("=" * 50)
    print("4/7: Setting up Myokit-CUDA environment...")
    print("=" * 50)
    myokit_cuda_setup = (
        script_dir / "GPU_ODE_MYOKIT_CUDA" / "setup_environment.py"
    )
    if not run_setup_script(myokit_cuda_setup, "Myokit-CUDA"):
        failed_setups.append("Myokit-CUDA")
    print()

    # Setup gradsolve
    print("=" * 50)
    print("5/7: Setting up gradsolve environment...")
    print("=" * 50)
    gradsolve_setup = script_dir / "GPU_ODE_GRADSOLVE" / "setup_environment.py"
    if not run_setup_script(gradsolve_setup, "gradsolve"):
        failed_setups.append("gradsolve")
    print()

    # Setup modax
    print("=" * 50)
    print("6/7: Setting up modax environment...")
    print("=" * 50)
    modax_setup = script_dir / "GPU_ODE_MODAX" / "setup_environment.py"
    if not run_setup_script(modax_setup, "modax"):
        failed_setups.append("modax")
    print()

    # Setup Julia
    print("=" * 50)
    print("7/7: Setting up Julia environment...")
    print("=" * 50)
    julia_setup = script_dir / "setup_julia.py"
    if not run_setup_script(julia_setup, "Julia"):
        failed_setups.append("Julia")
    print()
    
    # Summary
    print("=" * 50)
    print("Setup Summary")
    print("=" * 50)
    
    is_windows = platform.system() == "Windows"
    
    if not failed_setups:
        print("✓ All environments setup successfully!")
        print()
        print("To use the environments:")
        if is_windows:
            print("  CUBIE:      GPU_ODE_CUBIE\\venv\\Scripts\\activate.bat")
            print("  JAX:        GPU_ODE_JAX\\venv\\Scripts\\activate.bat")
            print("  PyTorch:    GPU_ODE_PyTorch\\venv\\Scripts\\activate.bat")
            print(
                "  Myokit-CUDA: "
                "GPU_ODE_MYOKIT_CUDA\\venv\\Scripts\\activate.bat"
            )
        else:
            print("  CUBIE:      source GPU_ODE_CUBIE/venv/bin/activate")
            print("  JAX:        source GPU_ODE_JAX/venv/bin/activate")
            print("  PyTorch:    source GPU_ODE_PyTorch/venv/bin/activate")
            print(
                "  Myokit-CUDA: "
                "source GPU_ODE_MYOKIT_CUDA/venv/bin/activate"
            )
            print("  gradsolve:  source GPU_ODE_GRADSOLVE/venv/bin/activate")
            print("  modax:      source GPU_ODE_MODAX/venv/bin/activate")
        print("  Julia:      julia --project=.")
        return 0
    else:
        print("✗ Some environments failed to setup:")
        for env in failed_setups:
            print(f"  - {env}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
