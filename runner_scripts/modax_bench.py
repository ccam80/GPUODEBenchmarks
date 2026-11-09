#!/usr/bin/env python

"""Run modax's Tsit5 and Rodas5P solvers for runner.py.

Run Rodas5P as `rodas5p` with modax's default float32 LU factorisation, and as
`rodas5p_lu64` with a float64 LU factorisation.

Compile each problem's scalar right-hand side as a modax callback and jit one
solve per set of solve settings, saving the state at t = 0 and t = duration.
Compiling the solve also compiles the CUDA kernel through numba-cuda-mlir, and
for Rodas5P its Enzyme Jacobian. modax integrates in float64 only and adaptively
only, under its PID controller, so a trial must name the default controller; it
may set the gains pcoeff, icoeff and dcoeff but not dt_min or dt_max. Compile a
cold build from a fresh callback in a fresh jax cache.
"""

import json
import math
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import runner  # noqa: E402
import scalar_systems  # noqa: E402
from jax_bench import ADAPTIVE_MAX_STEPS, GIB, memory_shortfall, problem_row, set_cache_dir  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(REPO_ROOT, "generated", "modax_jax_cache")

CONTROLLERS = ("default",)
PRECISION = "float64"
# Map each algorithm to the modax module whose solve runs it and the options that select the method's variant.
SOLVERS = {"tsit5": ("modax.tsit5", {}),
           "rodas5p": ("modax.rodas5P", {"lu_precision": "fp32"}),
           "rodas5p_lu64": ("modax.rodas5P", {"lu_precision": "fp64"})}
GAINS = ("pcoeff", "icoeff", "dcoeff")
MAX_STEPS_RETCODE = "max_steps: the step bound was reached before the duration"


def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def check_trial(trial):
    """Raise ValueError if the trial asks for a precision other than float64, a controller other than default, or an algorithm modax lacks."""
    if trial["precision"] != PRECISION:
        raise ValueError("modax integrates in {0} only, not {1}".format(PRECISION, trial["precision"]))
    if trial["controller"] not in CONTROLLERS:
        raise ValueError("unknown controller " + trial["controller"])
    if trial["algorithm"] not in SOLVERS:
        raise ValueError("modax has no {0}".format(trial["algorithm"]))


def solve_options(trial):
    """Return the keyword arguments of modax's solve for a trial.

    Pass the tolerances, the suite's step bound, any PID gains the trial sets,
    first_step (None to let modax start from 1e-6 of the span) and, for Rodas5P,
    the LU precision the algorithm names. Raise ValueError if the trial sets
    dt_min, dt_max or a gain modax does not take.
    """
    if _finite(trial["dt_min"]) or _finite(trial["dt_max"]):
        raise ValueError("modax takes no dt_min or dt_max")
    gains = json.loads(trial["gains"]) if trial["gains"] else {}
    unknown = sorted(set(gains) - set(GAINS))
    if unknown:
        raise ValueError("modax's controller takes {0} only, not {1}".format(", ".join(GAINS), ", ".join(unknown)))
    options = {"rtol": float(trial["rtol"]), "atol": float(trial["atol"]),
               "first_step": float(trial["dt"]) if _finite(trial["dt"]) else None,
               "max_steps": ADAPTIVE_MAX_STEPS}
    options.update({name: float(value) for name, value in gains.items()})
    options.update(SOLVERS[trial["algorithm"]][1])
    return options


def stepping_key(trial):
    """Return the trial fields that select a jitted solve."""
    return tuple(trial[field] for field in ("controller", "dt", "dt_min", "dt_max", "atol", "rtol", "gains"))


def finals_of(states, loop_steps, max_steps, duration):
    """Return (finals, t_final, retcode) from final states and loop step counts.

    Set NaN states and a failure message for every trajectory whose loop reached
    the step bound, which means it stopped short of the duration. Report the
    duration as t_final, because modax reports no end time.
    """
    finals = np.array(states, copy=True)
    stopped = np.asarray(loop_steps).reshape(-1) >= int(max_steps)
    finals[stopped] = np.nan
    retcode = [MAX_STEPS_RETCODE if flag else "" for flag in stopped]
    return finals, np.full(finals.shape[0], float(duration)), retcode


class Build:
    """Hold the jitted solves of one problem's callback and algorithm.

    Check the memory of each solve once per ensemble size, and keep the device
    parameters of the last upload for solves on resident inputs.
    """

    def __init__(self, trial, cold=False):
        import importlib

        import jax
        import jax.numpy as jnp
        check_trial(trial)
        self.row = problem_row(trial)
        self.duration = float(trial["duration"])
        self.algorithm = trial["algorithm"]
        self.module = importlib.import_module(SOLVERS[self.algorithm][0])
        self.system = scalar_systems.system(self.row)
        self.states = len(self.system.y0)
        self.cache_dir = None
        self.solves = {}
        self.checked = set()
        self.resident_n = None
        self.resident = None
        if cold:
            self.cache_dir = tempfile.mkdtemp(prefix="modax_cold_")
            set_cache_dir(self.cache_dir)
            jax.clear_caches()
        try:
            jax.config.update("jax_enable_x64", True)
            # modax caches kernels by callback object, so a fresh callback gives a cold build its own kernels.
            self.ode_fn = scalar_systems.right_hand_side(self.system, scalar_systems.MATH, "({0},)",
                                                         arguments="y, t, p")
            self.y0 = jnp.asarray(self.system.y0, dtype=jnp.float64)
            self.t_span = jnp.asarray([0.0, self.duration], dtype=jnp.float64)
        except BaseException:
            self.close()
            raise

    def solve_of(self, trial):
        """Return the jitted solve that maps grid values to (final states, loop step counts), building it on first use."""
        import jax
        import jax.numpy as jnp
        key = stepping_key(trial)
        if key not in self.solves:
            options = solve_options(trial)
            solve, ode_fn, t_span = self.module.solve, self.ode_fn, self.t_span
            y0 = self.y0

            @jax.jit
            def run(values):
                params = values.reshape(-1, 1)
                y0s = jnp.broadcast_to(y0, (params.shape[0], y0.shape[0]))
                history, stats = solve(ode_fn, y0s, t_span, params, return_stats=True, **options)
                return history[:, -1, :], stats["loop_steps"]

            self.solves[key] = run
        return self.solves[key]

    def compile(self, trial, values):
        """Lower and compile the trial's solve at its n.

        Raise MemoryError if the compiled solve needs more memory than the device has.
        """
        import jax
        import jax.numpy as jnp
        compiled = self.solve_of(trial).lower(jnp.asarray(values)).compile()
        limit = jax.local_devices()[0].memory_stats()["bytes_limit"]
        shortfall = memory_shortfall(compiled.memory_analysis(), limit)
        if shortfall is not None:
            raise MemoryError("RESOURCE_EXHAUSTED: the compiled solve needs {0:.2f} GiB; the device "
                              "limit is {1:.2f} GiB".format(shortfall[0] / GIB, shortfall[1] / GIB))
        self.checked.add((stepping_key(trial), int(values.shape[0])))

    def prepare(self, trial, values):
        if (stepping_key(trial), int(values.shape[0])) not in self.checked:
            self.compile(trial, values)
        return self.solve_of(trial)

    def host_solve(self, trial, values):
        """Solve from host arrays and copy the result back, timing both copies."""
        import jax
        import jax.numpy as jnp
        solve = self.prepare(trial, values)
        host = np.ascontiguousarray(values, dtype=np.float64)
        return jax.device_get(jax.block_until_ready(solve(jnp.asarray(host))))

    def device_solve(self, trial, values):
        """Solve on resident device parameters and leave the result on the device.

        Upload the parameters first when the resident ones belong to another grid.
        """
        import jax
        import jax.numpy as jnp
        solve = self.prepare(trial, values)
        n = int(values.shape[0])
        if self.resident_n != n:
            self.resident = None
            self.resident = jnp.asarray(np.ascontiguousarray(values, dtype=np.float64))
            self.resident_n = n
        return jax.block_until_ready(solve(self.resident))

    def close(self):
        self.solves = {}
        self.resident = None
        if self.cache_dir is not None:
            set_cache_dir(CACHE_DIR)
            shutil.rmtree(self.cache_dir, ignore_errors=True)
            self.cache_dir = None


class ModaxAdapter:
    """Run modax trials for runner.py on one machine."""

    controllers = CONTROLLERS

    def __init__(self, key, root):
        self.key, self.root = key, root

    def version(self):
        import importlib.metadata
        return importlib.metadata.version("modax-solvers")

    def states(self, trial):
        return int(problem_row(trial)["states"])

    def build(self, trial, cold=False):
        return Build(trial, cold)

    def compile(self, build, trial, values):
        build.compile(trial, values)

    def optimize(self, build, trial):
        raise NotImplementedError("modax has no launch geometry to optimize")

    def solve(self, build, trial, values, transfers):
        if transfers == "both":
            return build.host_solve(trial, values)
        return build.device_solve(trial, values)

    def finals(self, build, result):
        """Return (finals, t_final, retcode) for the result of a solve."""
        states, loop_steps = result
        return finals_of(np.asarray(states), np.asarray(loop_steps), ADAPTIVE_MAX_STEPS, build.duration)


def run(argv):
    """Run a trial file on the GPU with the persistent jax compilation cache.

    Exit if jax sees no GPU.
    """
    import jax
    import modax  # noqa: F401 - the import sets jax's x64 flag
    if jax.default_backend() == "cpu":
        raise SystemExit("jax is running on the CPU backend; this is a GPU benchmark, so nothing is recorded")
    print("jax backend: " + jax.default_backend(), flush=True)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    os.makedirs(CACHE_DIR, exist_ok=True)
    set_cache_dir(CACHE_DIR)
    return runner.main(argv, lambda key, root: ModaxAdapter(key, root))
