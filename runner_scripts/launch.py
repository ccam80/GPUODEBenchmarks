"""Per-package command sequences for bench.py: which interpreter, script, environment and process split each package's analyses use."""

import os
import platform
import shlex
import sys

from algorithms import MODES
from cubie_adapter import PACKAGES as CUBIE_PACKAGES
from protocol import WATCHDOG_EXIT_CODE

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PACKAGES = CUBIE_PACKAGES + ("julia", "cpp", "pytorch", "jax", "myokit_cuda")
ANALYSES = ("optimize", "warm", "performance", "states", "work-precision")

VENV = {"cubie": "GPU_ODE_CUBIE/venv", "cubie_mlir": "GPU_ODE_CUBIE_MLIR/venv",
        "pytorch": "GPU_ODE_PyTorch/venv", "jax": "GPU_ODE_JAX/venv",
        "myokit_cuda": "GPU_ODE_MYOKIT_CUDA/venv"}
SCRIPT = {"cubie": "GPU_ODE_CUBIE/bench_cubie.py",
          "cubie_mlir": "GPU_ODE_CUBIE_MLIR/bench_cubie_mlir.py",
          "pytorch": "GPU_ODE_PyTorch/bench_torchdiffeq.py",
          "jax": "GPU_ODE_JAX/bench_diffrax.py",
          "myokit_cuda": "GPU_ODE_MYOKIT_CUDA/bench_myokit_cuda.py"}
ENV = {"cubie": {"CUBIE_MAX_CACHE_ENTRIES": "0"},
       "cubie_mlir": {"CUBIE_MAX_CACHE_ENTRIES": "0"},
       "jax": {"XLA_PYTHON_CLIENT_PREALLOCATE": "false"}}
# Work-precision legs run one process each so a watchdog hard-exit abandons one leg.
WP_PER_LEG = ("pytorch", "jax", "myokit_cuda")
# Packages whose performance sweep fills its cache before timing.
PERF_WARM = ("cubie", "cubie_mlir", "jax", "myokit_cuda")


class Command:
    """One subprocess of a stage: argv, extra environment, and the exit codes that are not failures."""

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


def cubie_python():
    """The shared cubie interpreter, for the NE, overlap and comparison scripts."""
    return venv_python("cubie")


def julia_command():
    """The julia launcher as argv; JULIA may name a channel such as `julia +1.13`."""
    return shlex.split(os.environ.get("JULIA", "julia"))


def mode_args(mode):
    """The --mode tail a bench script or driver takes; nothing when every mode runs."""
    return [] if mode == "all" else ["--mode", mode]


def wp_legs(package, algorithm, problem, mode="all"):
    """(problem, algorithm) work-precision legs the package runs in the requested modes."""
    from algorithms import resolve_algorithms, resolve_modes, supported_for
    from problems import resolve_problems
    algorithms = resolve_algorithms(algorithm, package)
    modes = resolve_modes(mode)
    legs = []
    for row in resolve_problems(problem, package):
        for name in algorithms:
            if any(name in supported_for(package, m) for m in modes):
                legs.append((row.name, name))
    return legs


def _python_commands(package, analysis, nlist, algorithm, problem, mode):
    python = venv_python(package)
    script = os.path.join(REPO_ROOT, SCRIPT[package])
    env = ENV.get(package, {})
    csv = ",".join(str(n) for n in nlist)
    tail = mode_args(mode)

    def bench(*args):
        return [python, script] + list(args) + tail

    if analysis == "optimize":
        if package not in CUBIE_PACKAGES:
            return []
        return [Command("optimize", bench("optimize", algorithm, "--problem", problem), env)]
    if analysis == "warm":
        return [Command("warm", bench("warm:" + csv, algorithm, "--problem", problem), env)]
    if analysis == "states":
        return [Command("states", bench("states", algorithm), env)]
    if analysis == "work-precision":
        commands = []
        if package in CUBIE_PACKAGES:
            commands.append(Command("optimize", bench("optimize", algorithm, "--problem", problem), env))
        if package in WP_PER_LEG:
            for leg_problem, leg_algorithm in wp_legs(package, algorithm, problem, mode):
                commands.append(Command(
                    "wp {0} {1}".format(leg_problem, leg_algorithm),
                    bench("wp", leg_algorithm, "--problem", leg_problem), env,
                    ok=(0, WATCHDOG_EXIT_CODE)))
        else:
            commands.append(Command("wp", bench("wp", algorithm, "--problem", problem), env))
        return commands
    commands = []
    if package in CUBIE_PACKAGES:
        commands.append(Command("optimize", bench("optimize", algorithm, "--problem", problem), env))
    if package in PERF_WARM:
        commands.append(Command("warm", bench("warm:" + csv, algorithm, "--problem", problem), env))
    commands.append(Command("performance", bench(csv, algorithm, "--problem", problem), env))
    return commands


def _julia_commands(analysis, nlist, algorithm, problem, mode):
    driver = [sys.executable, os.path.join(REPO_ROOT, "runner_scripts", "gpu", "julia_driver.py")]
    tail = mode_args(mode)
    if analysis == "optimize":
        return []
    if analysis == "warm":
        return [Command("warm", julia_command() + ["--project=.", "-e", "using Pkg; Pkg.precompile()"])]
    if analysis == "states":
        return [Command("states", driver + ["states", algorithm] + tail)]
    if analysis == "work-precision":
        return [Command("wp", driver + ["wp", algorithm, problem] + tail)]
    csv = ",".join(str(n) for n in nlist)
    return [Command("performance", driver + ["performance", csv, algorithm, problem] + tail)]


def _cpp_commands(analysis, nmax, algorithm, problem, mode):
    if analysis == "optimize":
        return []
    if platform.system() == "Windows":
        argv = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                os.path.join(REPO_ROOT, "runner_scripts", "gpu", "run_ode_cpp.ps1")]
    else:
        argv = ["bash", os.path.join(REPO_ROOT, "runner_scripts", "gpu", "run_ode_cpp.sh")]
    argv += ["-a", analysis, "-n", nmax, "-g", algorithm, "-s", problem, "-m", mode]
    return [Command(analysis, argv)]


def commands(package, analysis, nlist, nmax, algorithm, problem, mode="all"):
    """The commands one (package, analysis) stage runs."""
    if mode != "all" and mode not in MODES:
        raise ValueError("unknown mode '{0}'".format(mode))
    if package == "julia":
        return _julia_commands(analysis, nlist, algorithm, problem, mode)
    if package == "cpp":
        return _cpp_commands(analysis, nmax, algorithm, problem, mode)
    return _python_commands(package, analysis, nlist, algorithm, problem, mode)


def store_analysis(analysis):
    """The result-store analysis name a bench analysis writes."""
    return {"work-precision": "wp", "states": "states"}.get(analysis, "times")
