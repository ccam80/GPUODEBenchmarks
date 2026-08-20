#!/usr/bin/env python3

"""Instantiate the committed Julia environment, or re-resolve it with --update."""

import argparse
import os
import sys
import subprocess
import shutil
from pathlib import Path

# Plots is required by runner_scripts/plot/*.jl.
CORE_PACKAGES = ["BenchmarkTools", "CSV", "DataFrames", "StaticArrays", "Plots"]

# Solver sub-libraries the numerical-equivalence suite needs beyond the umbrella.
DIFFEQ_PACKAGES = [
    "DiffEqBase",
    "DiffEqDevTools",
    "DiffEqGPU",
    "OrdinaryDiffEq",
    "OrdinaryDiffEqCore",
    "OrdinaryDiffEqExplicitRK",
    "OrdinaryDiffEqFIRK",
    "OrdinaryDiffEqHighOrderRK",
    "OrdinaryDiffEqLowOrderRK",
    "OrdinaryDiffEqRosenbrock",
    "OrdinaryDiffEqSDIRK",
    "OrdinaryDiffEqVerner",
    "RecursiveArrayTools",
    "SciMLBase",
    "SimpleDiffEq",
]

MODELING_PACKAGES = ["Catalyst", "ModelingToolkit", "ReactionNetworkImporters"]


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


def julia(code):
    """Run one snippet in the repo's Julia project."""
    return run_command(["julia", "--project=.", "-e", code])


def add_packages(names):
    """Add packages as a group, falling back to one at a time."""
    pkg_list = ", ".join(f'"{name}"' for name in names)
    if julia(f"using Pkg; Pkg.add([{pkg_list}])"):
        return True
    print("Warning: group add failed, trying individually...")
    ok = True
    for name in names:
        print(f"Adding {name}...")
        ok = julia(f'using Pkg; Pkg.add("{name}")') and ok
    return ok


def resolve_latest():
    """Re-resolve the whole environment and rewrite Project/Manifest."""
    # CUDA resolves first, against an empty project.
    print("Adding CUDA package for GPU support...")
    if not julia('using Pkg; Pkg.add("CUDA")'):
        print("Failed to add CUDA package")
        return False
    print("Adding core utility packages...")
    add_packages(CORE_PACKAGES)
    print("Adding DiffEq packages...")
    add_packages(DIFFEQ_PACKAGES)
    print("Adding modeling packages...")
    add_packages(MODELING_PACKAGES)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--update",
        action="store_true",
        help="re-resolve to the newest compatible versions and rewrite the "
             "committed Project.toml/Manifest.toml",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent.resolve()
    os.chdir(script_dir)

    print("Setting up Julia environment...")

    # Check if Julia is available
    if not shutil.which("julia"):
        print("Error: julia is not installed")
        print("Please install Julia from https://julialang.org/downloads/")
        return 1

    # The Manifest targets the juliaup "1.13" channel; pin this directory.
    if shutil.which("juliaup"):
        if not run_command(["juliaup", "override", "set", "1.13"]):
            print("Failed to pin the juliaup 1.13 channel override; the "
                  "Manifest will not resolve on another julia version")
            return 1

    print("Julia version:")
    if not run_command(["julia", "--version"]):
        return 1

    manifest = script_dir / "Manifest.toml"
    if args.update or not manifest.is_file():
        if not manifest.is_file():
            print("No Manifest.toml found; resolving from scratch.")
        if not resolve_latest():
            return 1
    else:
        print("Instantiating the pinned environment from Manifest.toml...")
        if not julia("using Pkg; Pkg.instantiate()"):
            print("Failed to instantiate the pinned environment")
            return 1

    print("Precompiling packages...")
    if not julia("using Pkg; Pkg.precompile()"):
        print("Precompilation failed")
        return 1

    print("\nJulia environment setup complete!")
    print("To test the installation, run:")
    print("  julia --project=. -e 'using DiffEqGPU, CUDA'")

    return 0


if __name__ == "__main__":
    sys.exit(main())
