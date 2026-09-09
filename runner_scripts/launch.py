"""The runner registry for bench.py: the argv and environment that execute a package's trials file, the suite interpreter and the julia launcher."""

import os
import shlex
import sys

from cubie_adapter import PACKAGES as CUBIE_PACKAGES
from store import PACKAGES

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VENV = {"cubie": "GPU_ODE_CUBIE/venv", "cubie_mlir": "GPU_ODE_CUBIE_MLIR/venv",
        "pytorch": "GPU_ODE_PyTorch/venv", "jax": "GPU_ODE_JAX/venv",
        "myokit_cuda": "GPU_ODE_MYOKIT_CUDA/venv"}
ENV = {"cubie": {"CUBIE_MAX_CACHE_ENTRIES": "0"},
       "cubie_mlir": {"CUBIE_MAX_CACHE_ENTRIES": "0"},
       "jax": {"XLA_PYTHON_CLIENT_PREALLOCATE": "false"}}

# package -> callable returning the runner argv; bench.py appends `--trials <path> [--floor]`.
RUNNERS = {}


class UnportedPackage(Exception):
    """The package has no runner on the trials contract yet."""


def runner_argv(package):
    """The argv that runs a package's trials file, without the --trials tail."""
    if package not in PACKAGES:
        raise ValueError("unknown package '{0}'".format(package))
    if package not in RUNNERS:
        raise UnportedPackage("{0} has no runner registered in launch.RUNNERS".format(package))
    return list(RUNNERS[package]())


def runner_env(package):
    """Extra environment for a package's runner."""
    return dict(ENV.get(package, {}))


def ordered(packages):
    """cubie, then cubie_mlir, then the rest in the given order."""
    return ([p for p in packages if p == "cubie"]
            + [p for p in packages if p == "cubie_mlir"]
            + [p for p in packages if p not in CUBIE_PACKAGES])


def venv_python(package):
    """The package venv's interpreter, or this one when the venv is absent."""
    root = os.path.join(REPO_ROOT, VENV[package])
    for candidate in (os.path.join(root, "Scripts", "python.exe"),
                      os.path.join(root, "bin", "python3"),
                      os.path.join(root, "bin", "python")):
        if os.path.isfile(candidate):
            return candidate
    return sys.executable


def cubie_python():
    """The shared cubie interpreter, for the NE, overlap and comparison scripts."""
    return venv_python("cubie")


def suite_python():
    """The suite interpreter: GPU_ODE_CUBIE/venv with pyarrow and duckdb; bench.py, the analyses and the shell wrappers run under it."""
    return venv_python("cubie")


def julia_command():
    """The julia launcher as argv: `julia +1.13`, or JULIA when set."""
    return shlex.split(os.environ.get("JULIA", "julia +1.13"))
