"""The runner registry for bench.py: per package, the interpreter, script and environment that consume a trial file (`<runner argv> --trials <path> [--floor]`)."""

import os
import platform
import shlex
import sys

from store import PACKAGES

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CUBIE_PACKAGES = ("cubie", "cubie_mlir")

VENV = {"cubie": "GPU_ODE_CUBIE/venv", "cubie_mlir": "GPU_ODE_CUBIE_MLIR/venv",
        "pytorch": "GPU_ODE_PyTorch/venv", "jax": "GPU_ODE_JAX/venv",
        "myokit_cuda": "GPU_ODE_MYOKIT_CUDA/venv"}
ENV = {"cubie": {"CUBIE_MAX_CACHE_ENTRIES": "0"},
       "cubie_mlir": {"CUBIE_MAX_CACHE_ENTRIES": "0"},
       "jax": {"XLA_PYTHON_CLIENT_PREALLOCATE": "false"}}


class Command:
    """One runner invocation: argv, extra environment, and the exit codes that are not failures."""

    def __init__(self, label, argv, env=None, ok=(0,)):
        self.label, self.argv, self.env, self.ok = label, list(argv), dict(env or {}), tuple(ok)


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


def suite_python():
    """The suite interpreter: GPU_ODE_CUBIE/venv with pyarrow and duckdb; bench.py, the store CLI and the analyses run under it."""
    return venv_python("cubie")


def julia_command():
    """The julia launcher as argv: `julia +1.13`, or JULIA when set."""
    return shlex.split(os.environ.get("JULIA", "julia +1.13"))


def _script(path):
    return os.path.join(REPO_ROOT, *path.split("/"))


def _python_runner(package, script):
    return [venv_python(package), _script(script)]


def _cpp_runner():
    if platform.system() == "Windows":
        return ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                _script("runner_scripts/gpu/run_ode_cpp.ps1")]
    return ["bash", _script("runner_scripts/gpu/run_ode_cpp.sh")]


# package -> callable giving the argv a trial file is appended to.
RUNNERS = {
    "cubie": lambda: _python_runner("cubie", "GPU_ODE_CUBIE/bench_cubie.py"),
    "cubie_mlir": lambda: _python_runner("cubie_mlir", "GPU_ODE_CUBIE_MLIR/bench_cubie_mlir.py"),
    "jax": lambda: _python_runner("jax", "GPU_ODE_JAX/bench_diffrax.py"),
    "pytorch": lambda: _python_runner("pytorch", "GPU_ODE_PyTorch/bench_torchdiffeq.py"),
    "myokit_cuda": lambda: _python_runner("myokit_cuda", "GPU_ODE_MYOKIT_CUDA/bench_myokit_cuda.py"),
    "cpp": _cpp_runner,
    "julia_gpu": lambda: [suite_python(), _script("runner_scripts/gpu/julia_driver.py")],
    "julia_cpu": lambda: julia_command() + ["-t", "auto", "--project=.",
                                            _script("GPU_ODE_Julia/bench_ode_cpu.jl")],
}


def runner_command(package, trials_path, floor=False):
    """The Command that runs a package's trial file."""
    if package not in RUNNERS:
        raise ValueError("no runner registered for '{0}' (known: {1})".format(
            package, ", ".join(PACKAGES)))
    argv = list(RUNNERS[package]()) + ["--trials", trials_path]
    if floor:
        argv.append("--floor")
    return Command(package, argv, ENV.get(package, {}))
