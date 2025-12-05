#!/usr/bin/env python3
"""
Cross-platform setup script for Julia environment for GPU ODE benchmarking.
Works on Linux, Windows, and macOS.
"""
import os
import sys
import subprocess
import shutil
from pathlib import Path


def run_command(cmd, shell=False, check=True, cwd=None):
    """Run a command and handle errors."""
    try:
        result = subprocess.run(
            cmd,
            shell=shell,
            check=check,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.stdout:
            print(result.stdout, end='')
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error: Command failed with exit code {e.returncode}")
        if e.stderr:
            print(e.stderr)
        return False


def main():
    script_dir = Path(__file__).parent.resolve()
    os.chdir(script_dir)
    
    print("Setting up Julia environment...")
    
    # Check if Julia is available
    if not shutil.which("julia"):
        print("Error: julia is not installed")
        print("Please install Julia from https://julialang.org/downloads/")
        return 1
    
    print("Julia version:")
    if not run_command(["julia", "--version"]):
        return 1
    
    # Instantiate and precompile the main project
    print("Installing Julia packages for main project...")
    julia_cmd = [
        "julia", "--project=.",
        "-e",
        "using Pkg; Pkg.instantiate(); Pkg.resolve(); Pkg.precompile()"
    ]
    if not run_command(julia_cmd):
        print("Failed to install Julia packages")
        return 1
    
    # Add CUDA for GPU support (NVIDIA)
    print("Adding CUDA package for GPU support...")
    julia_cmd = [
        "julia", "--project=.",
        "-e",
        'using Pkg; Pkg.add("CUDA")'
    ]
    if not run_command(julia_cmd):
        print("Failed to add CUDA package")
        return 1
    
    print("\nJulia environment setup complete!")
    print("To test the installation, run:")
    print("  julia --project=. -e 'using DiffEqGPU, CUDA'")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
