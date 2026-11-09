#!/usr/bin/env python

"""The modax adapter for runner.py: a build is one problem's scalar right-hand side as a modax callback and its Tsit5 or Rodas5P solve, jitted per stepping over the ensemble's parameters, saving the state at t = 0 and t = duration; compile lowers and compiles at the trial's n, which compiles the CUDA kernel (and for Rodas5P its Enzyme Jacobian) through numba-cuda-mlir; a solve runs through host arrays (`both`) or on the resident device parameters (`none`). modax integrates in float64 only and adaptively only, under its PID controller (`default`), whose gains a trial may set as pcoeff, icoeff and dcoeff. modax takes no step-size pins, so a trial that sets dt_min or dt_max is refused. A cold build compiles a fresh callback in a fresh jax cache."""

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
# algorithm -> the modax module whose solve runs it.
SOLVERS = {"tsit5": "modax.tsit5", "rodas5p": "modax.rodas5P"}
GAINS = ("pcoeff", "icoeff", "dcoeff")
MAX_STEPS_RETCODE = "max_steps: the step bound was reached before the duration"


def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def check_trial(trial):
    """Refuses what modax cannot run: a precision other than float64, a controller other than default, an algorithm it has no solver for."""
    if trial["precision"] != PRECISION:
        raise ValueError("modax integrates in {0} only, not {1}".format(PRECISION, trial["precision"]))
    if trial["controller"] not in CONTROLLERS:
        raise ValueError("unknown controller " + trial["controller"])
    if trial["algorithm"] not in SOLVERS:
        raise ValueError("modax has no {0}".format(trial["algorithm"]))


def solve_options(trial):
    """The keyword arguments of modax's solve for a trial: the tolerances, first_step (None leaves modax to start from 1e-6 of the span), the suite's step bound and any PID gains the trial sets. Raises ValueError on a step-size pin or a gain modax does not take."""
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
    return options


def stepping_key(trial):
    """The trial fields that select a jitted solve."""
    return tuple(trial[field] for field in ("controller", "dt", "dt_min", "dt_max", "atol", "rtol", "gains"))


def finals_of(states, loop_steps, max_steps, duration):
    """(finals, t_final, retcode): a trajectory whose loop ran to the step bound stopped short of the duration and carries NaN states; modax reports no end time, so t_final is the duration."""
    finals = np.array(states, copy=True)
    stopped = np.asarray(loop_steps).reshape(-1) >= int(max_steps)
    finals[stopped] = np.nan
    retcode = [MAX_STEPS_RETCODE if flag else "" for flag in stopped]
    return finals, np.full(finals.shape[0], float(duration)), retcode


class Build:
    """One problem's callback and algorithm: a jitted solve per stepping, one memory check per (stepping, n), the device parameters of the last upload."""

    def __init__(self, trial, cold=False):
        import importlib

        import jax
        import jax.numpy as jnp
        check_trial(trial)
        self.row = problem_row(trial)
        self.duration = float(trial["duration"])
        self.algorithm = trial["algorithm"]
        self.module = importlib.import_module(SOLVERS[self.algorithm])
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
            # modax caches its kernels on the callback object, so a fresh callback compiles afresh.
            self.ode_fn = scalar_systems.right_hand_side(self.system, scalar_systems.MATH, "({0},)",
                                                         arguments="y, t, p")
            self.y0 = jnp.asarray(self.system.y0, dtype=jnp.float64)
            self.t_span = jnp.asarray([0.0, self.duration], dtype=jnp.float64)
        except BaseException:
            self.close()
            raise

    def solve_of(self, trial):
        """The jitted solve of a trial's stepping, built on first use: grid values in, (final states, loop step counts) out."""
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
        """Lower and compile the trial's solve at its n; refuses, as an out-of-memory error, a solve whose compiled footprint exceeds the device."""
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
        """One solve through host arrays: the upload is the h2d, device_get the d2h."""
        import jax
        import jax.numpy as jnp
        solve = self.prepare(trial, values)
        host = np.ascontiguousarray(values, dtype=np.float64)
        return jax.device_get(jax.block_until_ready(solve(jnp.asarray(host))))

    def device_solve(self, trial, values):
        """One solve on the resident parameters, uploaded when they are not this grid's; the result stays on the device."""
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
    """The runner adapter of the modax package on one machine."""

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
        """(finals, t_final, retcode) of a solve: the state at the duration of every trajectory, the duration, and a step-bound message for a trajectory that stopped short."""
        states, loop_steps = result
        return finals_of(np.asarray(states), np.asarray(loop_steps), ADAPTIVE_MAX_STEPS, build.duration)


def run(argv):
    """Entry point of the modax suite: a GPU backend and the persistent compilation cache, then the trial file through runner.main."""
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
