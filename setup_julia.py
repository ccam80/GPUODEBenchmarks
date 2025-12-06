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
    
    print("Setting up Julia environment...")
    
    # Check if Julia is available
    if not shutil.which("julia"):
        print("Error: julia is not installed")
        print("Please install Julia from https://julialang.org/downloads/")
        return 1
    
    print("Julia version:")
    if not run_command(["julia", "--version"]):
        return 1
    
    # Add CUDA first (before other dependencies to avoid compatibility issues)
    print("Adding CUDA package for GPU support...")
    julia_cmd = [
        "julia", "--project=.",
        "-e",
        'using Pkg; Pkg.add("CUDA")'
    ]
    if not run_command(julia_cmd):
        print("Failed to add CUDA package")
        return 1
    
    # Manually add all dependencies without using pinned versions
    print("Adding Julia packages manually (without pinned versions)...")
    packages = [
        "BenchmarkTools",
        "CSV",
        "Catalyst",
        "DataFrames",
        "DiffEqBase",
        "DiffEqDevTools",
        "DiffEqGPU",
        "ModelingToolkit",
        "OrdinaryDiffEq",
        "ReactionNetworkImporters",
        "RecursiveArrayTools",
        "SciMLBase",
        "SimpleDiffEq",
        "StaticArrays"
    ]
    
    for package in packages:
        print(f"Adding {package}...")
        julia_cmd = [
            "julia", "--project=.",
            "-e",
            f'using Pkg; Pkg.add("{package}")'
        ]
        if not run_command(julia_cmd):
            print(f"Warning: Failed to add {package}")
            # Continue with other packages even if one fails
    
    # Precompile all packages
    print("Precompiling packages...")
    julia_cmd = [
        "julia", "--project=.",
        "-e",
        "using Pkg; Pkg.precompile()"
    ]
    if not run_command(julia_cmd):
        print("Warning: Precompilation had issues, but continuing...")
    
    print("\nJulia environment setup complete!")
    print("To test the installation, run:")
    print("  julia --project=. -e 'using DiffEqGPU, CUDA'")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
