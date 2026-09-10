#!/usr/bin/env python

"""The jax adapter for runner.py: a build is one Diffrax vector field and initial state whose vmapped, jitted solve is built per stepping; compile lowers and compiles at the trial's n (in a fresh compilation cache when cold); a solve runs through host arrays (`both`) or on the resident device parameters (`none`). Every algorithm here steps at a constant dt or under Diffrax's PID controller, so `fixed` and `default` are the controllers a trial may name."""

import json
import math
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import runner  # noqa: E402
from problems import as_problem  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(REPO_ROOT, "generated", "jax_cache")

CONTROLLERS = ("fixed", "default")
PRECISIONS = ("float32", "float64")
# The step bound of an adaptive solve: duration over a 1e-6 step floor, rounded up to a power of two.
ADAPTIVE_MAX_STEPS = 1048576
# The smallest step bound of a fixed solve; larger runs take twice their step count rounded up to a power of two.
FIXED_MIN_MAX_STEPS = 4096
GIB = float(2 ** 30)


def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def problem_row(trial):
    """The catalogue row of a trial, resized to its system_params states."""
    row = as_problem(trial["problem"])
    params = json.loads(trial["system_params"]) if trial["system_params"] else {}
    if "states" in params:
        row = row.resized(params["states"])
    return row


def fixed_max_steps(duration, dt):
    """The step bound of a constant-step solve: at least FIXED_MIN_MAX_STEPS, else twice the step count rounded up to a power of two."""
    steps = int(math.ceil(float(duration) / float(dt)))
    return max(FIXED_MIN_MAX_STEPS, 2 * (1 << (steps - 1).bit_length()))


def stepping(trial):
    """The solve settings a trial's stepping asks for: a `fixed` plan carries dt0 and its step bound; a `default` plan the tolerances, dt0 (None leaves Diffrax to pick), the finite step pins, the explicit gains and the adaptive step bound; both carry the Newton tolerances when the trial has them."""
    controller = trial["controller"]
    if controller not in CONTROLLERS:
        raise ValueError("unknown controller " + controller)
    newton = None
    if _finite(trial["newton_atol"]) and _finite(trial["newton_rtol"]):
        newton = (float(trial["newton_atol"]), float(trial["newton_rtol"]))
    if controller == "fixed":
        if not _finite(trial["dt"]) or trial["dt"] <= 0.0:
            raise ValueError("a fixed stepping needs a finite positive dt, got {0!r}".format(trial["dt"]))
        return {"kind": "fixed", "dt0": float(trial["dt"]),
                "max_steps": fixed_max_steps(trial["duration"], trial["dt"]), "newton": newton}
    return {"kind": "adaptive", "dt0": float(trial["dt"]) if _finite(trial["dt"]) else None,
            "atol": float(trial["atol"]), "rtol": float(trial["rtol"]),
            "dtmin": float(trial["dt_min"]) if _finite(trial["dt_min"]) else None,
            "dtmax": float(trial["dt_max"]) if _finite(trial["dt_max"]) else None,
            "gains": json.loads(trial["gains"]) if trial["gains"] else {},
            "max_steps": ADAPTIVE_MAX_STEPS, "newton": newton}


def stepping_key(trial):
    """The trial fields that select a jitted solve."""
    return tuple(trial[field] for field in ("controller", "dt", "dt_min", "dt_max", "atol", "rtol",
                                            "gains", "newton_atol", "newton_rtol"))


def memory_shortfall(usage, limit):
    """(needed, limit) in bytes when a compiled solve's temporaries, arguments and outputs exceed the device limit, else None; usage None (no analysis) never fails."""
    if usage is None:
        return None
    needed = (usage.temp_size_in_bytes + usage.argument_size_in_bytes
              + usage.output_size_in_bytes - usage.alias_size_in_bytes)
    if needed > limit:
        return needed, limit
    return None


def finals_of(ys, retcode, duration):
    """(finals, t_final, retcode) from the last saved states and the per-trajectory result messages: the duration where the message is empty and NaN otherwise."""
    finals = np.asarray(ys)
    retcode = [str(text) for text in retcode]
    ok = np.array([text == "" for text in retcode], dtype=bool)
    t_final = np.where(ok, float(duration), np.nan)
    return finals, t_final, retcode


# ------------------------------------------------------------------ diffrax

def make_solver(algorithm, newton=None):
    """The Diffrax solver of an algorithm; an implicit one takes a chord root finder at the given (atol, rtol) and Diffrax's own otherwise."""
    import diffrax
    if algorithm == "euler":
        return diffrax.Euler()
    if algorithm == "classical-rk4":
        return classical_rk4()
    if algorithm == "tsit5":
        return diffrax.Tsit5()
    if algorithm == "kvaerno3":
        if newton is None:
            return diffrax.Kvaerno3()
        import optimistix as optx
        atol, rtol = newton
        return diffrax.Kvaerno3(root_finder=diffrax.VeryChord(rtol=rtol, atol=atol, norm=optx.rms_norm))
    raise ValueError("no diffrax solver for {0}".format(algorithm))


def classical_rk4():
    """The classical fourth-order Runge-Kutta method from its tableau; b_error is zero, so it steps at a constant dt only."""
    from collections.abc import Callable
    from typing import ClassVar

    from diffrax import AbstractERK, ButcherTableau
    from diffrax._local_interpolation import ThirdOrderHermitePolynomialInterpolation

    rk4_tableau = ButcherTableau(
        a_lower=(np.array([1 / 2]), np.array([0.0, 1 / 2]), np.array([0.0, 0.0, 1.0])),
        b_sol=np.array([1 / 6, 1 / 3, 1 / 3, 1 / 6]),
        b_error=np.zeros(4),
        c=np.array([1 / 2, 1 / 2, 1.0]),
    )

    class ClassicalRK4(AbstractERK):
        tableau: ClassVar[ButcherTableau] = rk4_tableau
        interpolation_cls: ClassVar[Callable[..., ThirdOrderHermitePolynomialInterpolation]] = \
            ThirdOrderHermitePolynomialInterpolation.from_k

        def order(self, terms):
            del terms
            return 4

    return ClassicalRK4()


def make_controller(plan):
    """Diffrax's constant step size for a fixed plan, else its PID controller at the plan's tolerances, pins and gains."""
    import diffrax
    if plan["kind"] == "fixed":
        return diffrax.ConstantStepSize()
    return diffrax.PIDController(rtol=plan["rtol"], atol=plan["atol"], dtmin=plan["dtmin"],
                                 dtmax=plan["dtmax"], **plan["gains"])


def make_solve(vector_field, y0, duration, algorithm, plan):
    """The jitted, vmapped ensemble solve of one stepping: parameters in, the Diffrax Solution at t = duration out, failures reported through the result codes rather than raised."""
    import diffrax
    import jax
    solver = make_solver(algorithm, plan["newton"])
    controller = make_controller(plan)

    @jax.jit
    @jax.vmap
    def solve(p):
        terms = diffrax.ODETerm(vector_field(p))
        return diffrax.diffeqsolve(terms, solver, 0.0, duration, plan["dt0"], y0,
                                   max_steps=plan["max_steps"], stepsize_controller=controller,
                                   throw=False)

    return solve


def retcodes(result):
    """The Diffrax result message of every trajectory; empty on success. A vmapped RESULTS item holds one code per trajectory in `_value` and the enumeration keeps its messages by code in `_index_to_message`, the two tables its own `RESULTS[item]` text is printed from."""
    import diffrax
    messages = diffrax.RESULTS._index_to_message
    return [messages[int(code)] for code in np.asarray(result._value).reshape(-1)]


def set_cache_dir(path):
    """Point the persistent compilation cache at a directory, dropping what the process holds of the previous one."""
    from jax.experimental.compilation_cache import compilation_cache
    compilation_cache.reset_cache()
    compilation_cache.set_cache_dir(path)


# ---------------------------------------------------------------------- build

class Build:
    """One vector field and initial state; a jitted solve per stepping, one memory check per (stepping, n), the device parameters of the last upload."""

    def __init__(self, trial, cold=False):
        import jax
        import jax.numpy as jnp
        from jax_systems import build_problem
        self.row = problem_row(trial)
        if trial["precision"] not in PRECISIONS:
            raise ValueError("precision '{0}' is not float32 or float64".format(trial["precision"]))
        self.precision = trial["precision"]
        self.duration = float(trial["duration"])
        self.cache_dir = None
        self.solves = {}
        self.checked = set()
        self.resident_n = None
        self.resident = None
        if cold:
            self.cache_dir = tempfile.mkdtemp(prefix="jax_cold_")
            set_cache_dir(self.cache_dir)
            jax.clear_caches()
        try:
            jax.config.update("jax_enable_x64", self.precision == "float64")
            vector_field, y0 = build_problem(self.row)
            self.vector_field = vector_field
            self.y0 = jnp.asarray(y0, dtype=self.precision)
            self.states = int(self.y0.shape[0])
            self.algorithm = trial["algorithm"]
        except BaseException:
            self.close()
            raise

    def solve_of(self, trial):
        """The jitted solve of a trial's stepping, built on first use."""
        key = stepping_key(trial)
        if key not in self.solves:
            self.solves[key] = make_solve(self.vector_field, self.y0, self.duration, self.algorithm,
                                          stepping(trial))
        return self.solves[key]

    def compile(self, trial, values):
        """Lower and compile the trial's solve at its n; refuses, as an out-of-memory error, a solve whose compiled footprint exceeds the device."""
        import jax
        import jax.numpy as jnp
        solve = self.solve_of(trial)
        compiled = solve.lower(jnp.asarray(values)).compile()
        limit = jax.local_devices()[0].memory_stats()["bytes_limit"]
        shortfall = memory_shortfall(compiled.memory_analysis(), limit)
        if shortfall is not None:
            raise MemoryError("RESOURCE_EXHAUSTED: the compiled solve needs {0:.2f} GiB; the device "
                              "limit is {1:.2f} GiB".format(shortfall[0] / GIB, shortfall[1] / GIB))
        self.checked.add((stepping_key(trial), int(values.shape[0])))

    def prepare(self, trial, values):
        """The trial's solve, compiled and memory-checked once per (stepping, n)."""
        if (stepping_key(trial), int(values.shape[0])) not in self.checked:
            self.compile(trial, values)
        return self.solve_of(trial)

    def host_solve(self, trial, values):
        """One solve through host arrays: the upload is the h2d, device_get the d2h."""
        import jax
        import jax.numpy as jnp
        solve = self.prepare(trial, values)
        host = np.ascontiguousarray(values)
        return jax.device_get(jax.block_until_ready(solve(jnp.asarray(host))))

    def device_solve(self, trial, values):
        """One solve on the resident parameters, uploaded when they are not this grid's; the result stays on the device."""
        import jax
        import jax.numpy as jnp
        solve = self.prepare(trial, values)
        n = int(values.shape[0])
        if self.resident_n != n:
            self.resident = None
            self.resident = jnp.asarray(np.ascontiguousarray(values))
            self.resident_n = n
        return jax.block_until_ready(solve(self.resident))

    def close(self):
        self.solves = {}
        self.resident = None
        if self.cache_dir is not None:
            set_cache_dir(CACHE_DIR)
            shutil.rmtree(self.cache_dir, ignore_errors=True)
            self.cache_dir = None


class JaxAdapter:
    """The runner adapter of the jax package on one machine."""

    controllers = CONTROLLERS

    def __init__(self, key, root):
        self.key, self.root = key, root

    def version(self):
        import jax
        return jax.__version__

    def states(self, trial):
        return int(problem_row(trial)["states"])

    def build(self, trial, cold=False):
        return Build(trial, cold)

    def compile(self, build, trial, values):
        build.compile(trial, values)

    def optimize(self, build, trial):
        raise NotImplementedError("jax has no launch geometry to optimize")

    def solve(self, build, trial, values, transfers):
        if transfers == "both":
            return build.host_solve(trial, values)
        return build.device_solve(trial, values)

    def finals(self, build, result):
        """(finals, t_final, retcode) of a Solution: the last saved state of every trajectory, the duration where Diffrax reports success and NaN otherwise, and Diffrax's message on failure."""
        ys = np.asarray(result.ys)[:, -1, :]
        return finals_of(ys, retcodes(result.result), build.duration)


def run(argv):
    """Entry point of the jax suite: a GPU backend and the persistent compilation cache, then the trial file through runner.main."""
    import jax
    if jax.default_backend() == "cpu":
        raise SystemExit("jax is running on the CPU backend; this is a GPU benchmark, so nothing is recorded")
    print("jax backend: " + jax.default_backend(), flush=True)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    os.makedirs(CACHE_DIR, exist_ok=True)
    set_cache_dir(CACHE_DIR)
    return runner.main(argv, lambda key, root: JaxAdapter(key, root))
