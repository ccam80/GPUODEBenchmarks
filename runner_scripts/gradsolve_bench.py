#!/usr/bin/env python

"""Run gradsolve's solvers for runner.py.

Run adaptive Tsit5 and Rosenbrock23 on gradsolve's fused NVIDIA Warp kernels, one
thread per trajectory, with each problem registered through
gradsolve.register_jax_field. Run fixed-step Tsit5 and RK4 on gradsolve's lax.scan
solvers, jitted over the ensemble.

We launch the fused kernels through gradsolve's Warp launch layer rather than
gradsolve.solve, because solve fixes the first step at a hundredth of the span,
fixes the step bound at 4096 and returns host arrays. Launching directly gives a
trial the protocol's dt0, the suite's step bound and solves on resident device
inputs. Compile a cold build into fresh Warp and jax caches.
"""

import math
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import runner  # noqa: E402
import scalar_systems  # noqa: E402
from jax_bench import (ADAPTIVE_MAX_STEPS, GIB, memory_shortfall, problem_row,  # noqa: E402
                       set_cache_dir)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JAX_CACHE_DIR = os.path.join(REPO_ROOT, "generated", "gradsolve_jax_cache")
WARP_CACHE_DIR = os.path.join(REPO_ROOT, "generated", "gradsolve_warp_cache")

CONTROLLERS = ("fixed", "default")
PRECISIONS = ("float32", "float64")
# Map each adaptive algorithm to its fused kernel module and to whether its field needs a Jacobian.
FUSED = {"tsit5": ("warp_ode", False), "rosenbrock23_sciml": ("warp_rosenbrock", True)}
# Map each fixed-step algorithm to its lax.scan solver module.
SCANS = {"tsit5": "fixed_step_tsit5", "classical-rk4": "fixed_step_explicit"}
DEVICE = "cuda:0"
# Bind these names to jax.numpy for the scalar right-hand sides.
JAX_FUNCTIONS = ("exp", "sin", "sqrt", "stack")


def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def step_count(duration, dt):
    """Return the number of steps of size dt that span the duration.

    Raise ValueError if dt is not positive or does not divide the duration.
    """
    if not _finite(dt) or dt <= 0.0:
        raise ValueError("a fixed stepping needs a finite positive dt, got {0!r}".format(dt))
    steps = int(round(float(duration) / float(dt)))
    if steps < 1 or abs(steps * float(dt) - float(duration)) > 1e-9 * float(duration):
        raise ValueError("dt {0!r} does not divide the duration {1!r}".format(dt, duration))
    return steps


def plan(trial):
    """Return the solve settings for a trial.

    Return the scan module and step count for a fixed step size. For an adaptive
    controller, return the fused kernel module, the tolerances, dt0 and the
    suite's step bound; use gradsolve's own dt0 of a hundredth of the span when
    the trial sets none. Raise ValueError if gradsolve does not run the algorithm
    under the trial's controller, or if the trial sets dt_min, dt_max or gains,
    which the fused kernels do not take.
    """
    controller, algorithm = trial["controller"], trial["algorithm"]
    duration = float(trial["duration"])
    if controller not in CONTROLLERS:
        raise ValueError("unknown controller " + controller)
    if controller == "fixed":
        if algorithm not in SCANS:
            raise ValueError("gradsolve has no fixed-step {0}".format(algorithm))
        return {"kind": "fixed", "module": SCANS[algorithm], "steps": step_count(duration, trial["dt"])}
    if algorithm not in FUSED:
        raise ValueError("gradsolve has no adaptive {0}".format(algorithm))
    if _finite(trial["dt_min"]) or _finite(trial["dt_max"]):
        raise ValueError("gradsolve's fused kernels take no dt_min or dt_max")
    if trial["gains"] not in ("", "{}"):
        raise ValueError("gradsolve's fused kernels take no controller gains, got " + trial["gains"])
    module, stiff = FUSED[algorithm]
    dt0 = float(trial["dt"]) if _finite(trial["dt"]) else duration / 100.0
    return {"kind": "adaptive", "module": module, "stiff": stiff, "dt0": dt0,
            "atol": float(trial["atol"]), "rtol": float(trial["rtol"]),
            "max_steps": ADAPTIVE_MAX_STEPS}


def plan_key(trial):
    """Return the trial fields that select a compiled solve within a build."""
    return tuple(trial[field] for field in ("controller", "dt", "dt_min", "dt_max", "atol", "rtol", "gains"))


def finals_of(states, status, duration):
    """Return (finals, t_final, retcode) from final states and kernel statuses.

    Set NaN states and a failure message for every trajectory with a nonzero
    status, which means it did not reach the duration within the step bound.
    Report the duration as t_final, because gradsolve reports no end time.
    """
    finals = np.array(states, copy=True)
    codes = np.asarray(status).reshape(-1)
    finals[codes != 0] = np.nan
    retcode = ["" if code == 0 else "status {0}: the step bound was reached".format(int(code)) for code in codes]
    return finals, np.full(finals.shape[0], float(duration)), retcode


def to_host(array):
    """Return a host copy of a Warp, jax or numpy array."""
    numpy = getattr(array, "numpy", None)
    return np.asarray(numpy() if callable(numpy) else array)


def as_memory_error(exc):
    """Return a MemoryError for a Warp allocation failure, and any other exception unchanged."""
    if isinstance(exc, RuntimeError) and "allocate" in str(exc).lower():
        return MemoryError("OUT_OF_MEMORY: " + str(exc))
    return exc


class ScanProblem:
    """Present a scalar system to gradsolve's scan solvers through its Problem protocol."""

    is_stiff = False

    def __init__(self, name, dim, duration, f_jax):
        self.name, self.dim, self.t0, self.t1, self.f_jax = name, dim, 0.0, float(duration), f_jax


class Build:
    """Hold the solves of one problem, precision and algorithm.

    Build one solve per set of solve settings: a fused kernel on the problem's
    registered gradsolve field, or a jitted scan. Keep the device inputs of the
    last upload for solves on resident inputs.
    """

    serial = 0

    def __init__(self, trial, cold=False):
        import gradsolve  # noqa: F401 - the import sets jax's x64 flag, which the build then overrides
        import jax
        import jax.numpy as jnp
        import warp as wp
        if trial["precision"] not in PRECISIONS:
            raise ValueError("precision '{0}' is not float32 or float64".format(trial["precision"]))
        Build.serial += 1
        self.row = problem_row(trial)
        self.precision = trial["precision"]
        self.np_dtype = np.float32 if self.precision == "float32" else np.float64
        self.duration = float(trial["duration"])
        self.algorithm = trial["algorithm"]
        self.system = scalar_systems.system(self.row)
        self.states = len(self.system.y0)
        self.y0 = np.asarray(self.system.y0, dtype=self.np_dtype)
        self.cache_dirs = []
        self.solves = {}
        self.checked = set()
        self.resident_key = None
        self.resident = None
        if cold:
            self.cache_dirs = [tempfile.mkdtemp(prefix="gradsolve_jax_cold_"),
                               tempfile.mkdtemp(prefix="gradsolve_warp_cold_")]
            set_cache_dir(self.cache_dirs[0])
            jax.clear_caches()
            wp.config.kernel_cache_dir = self.cache_dirs[1]
        try:
            jax.config.update("jax_enable_x64", self.precision == "float64")
            namespace = {name: getattr(jnp, name) for name in JAX_FUNCTIONS}
            self.f_jax = scalar_systems.right_hand_side(self.system, namespace, "stack([{0}])")
            # Register a cold build under a name of its own so that it never reuses a kernel this process loaded.
            self.name = "suite_{0}_{1}".format(self.row["problem"], self.states)
            if cold:
                self.name += "_cold{0}".format(Build.serial)
        except BaseException:
            self.close()
            raise

    # ------------------------------------------------------------ fused
    def fused(self, plan):
        """Register the problem's field and return (module, kernel, field_key) for an adaptive plan.

        Raise RuntimeError if a Tsit5 plan's problem has more states than
        gradsolve's Tsit5 kernel runs on a GPU.
        """
        from gradsolve import register_jax_field
        from gradsolve.warp import warp_ode, warp_rosenbrock
        module = warp_rosenbrock if plan["stiff"] else warp_ode
        # We ignore gradsolve's 12-state Rosenbrock23 limit, so a run records where that kernel fails on the GPU.
        if not plan["stiff"] and self.states > warp_ode._GPU_VEC_DIM_CEILING:
            raise RuntimeError("gradsolve's fused warp_ode kernel runs at most {0} states on a GPU; {1} has {2}"
                               .format(warp_ode._GPU_VEC_DIM_CEILING, self.row["problem"], self.states))
        key = self.name + ("_stiff" if plan["stiff"] else "")
        register_jax_field(key, self.f_jax, self.states, 1, stiff=plan["stiff"])
        if plan["stiff"]:
            kernel = module._precision_spec(key, self.states, self.precision)[4]
        else:
            kernel = module._precision_spec(key, self.states, self.precision, vecp_P=1)[3]
        return module, kernel, key

    def fused_inputs(self, plan, module, key, values):
        """Upload the initial states and grid values as the kernel's (y0, params) device arrays."""
        import warp as wp
        n = int(values.shape[0])
        y0 = np.ascontiguousarray(np.broadcast_to(self.y0, (n, self.states)))
        params = np.asarray(values, dtype=self.np_dtype).reshape(n, 1)
        if plan["stiff"]:
            _, _, vecD, _, _ = module._precision_spec(key, self.states, self.precision)
            y0, params, _ = module._validate_batch(y0, params, self.np_dtype, self.states)
            return wp.array(y0, dtype=vecD, device=DEVICE), wp.array(params, dtype=vecD, device=DEVICE)
        _, wp_scalar, vecD, _ = module._precision_spec(key, self.states, self.precision, vecp_P=1)
        vecP = wp.types.vector(length=1, dtype=wp_scalar)
        return wp.array(y0, dtype=vecD, device=DEVICE), wp.array(params, dtype=vecP, device=DEVICE)

    def fused_launch(self, solve, y0_wp, p_wp):
        """Launch the kernel on device inputs, wait for it, and return the device (y_out, status) arrays."""
        import warp as wp
        plan, kernel = solve["plan"], solve["kernel"]
        n = y0_wp.shape[0]
        wp_scalar = wp.float32 if self.precision == "float32" else wp.float64
        conf = wp.array(np.array([0.0, self.duration, plan["rtol"], plan["atol"], plan["dt0"]],
                                 dtype=self.np_dtype), dtype=wp_scalar, device=DEVICE)
        y_out = wp.zeros(n, dtype=y0_wp.dtype, device=DEVICE)
        n_acc = wp.zeros(n, dtype=wp.int32, device=DEVICE)
        n_rej = wp.zeros(n, dtype=wp.int32, device=DEVICE)
        status = wp.zeros(n, dtype=wp.int32, device=DEVICE)
        dt_rec = wp.zeros((1, 1), dtype=wp_scalar, device=DEVICE)
        wp.launch(kernel, dim=n, inputs=[y0_wp, p_wp, conf, int(plan["max_steps"]), 0,
                                         y_out, n_acc, n_rej, status, dt_rec], device=DEVICE)
        wp.synchronize_device(DEVICE)
        return y_out, status

    # ------------------------------------------------------------- scans
    def scan(self, plan):
        """Return the jitted fixed-step scan that maps grid values to final states."""
        import importlib

        import jax
        import jax.numpy as jnp
        module = importlib.import_module("gradsolve.solvers." + plan["module"])
        problem = ScanProblem(self.name, self.states, self.duration, self.f_jax)
        y0 = jnp.asarray(self.y0)
        steps = plan["steps"]

        @jax.jit
        def solve(values):
            params = values.reshape(-1, 1)
            return module.solve_jax(problem, jnp.broadcast_to(y0, (params.shape[0], y0.shape[0])), params,
                                    n_steps=steps)

        return solve

    # ------------------------------------------------------------ solves
    def solve_of(self, trial):
        """Return the solve for a trial's settings, building it on first use."""
        key = plan_key(trial)
        if key not in self.solves:
            trial_plan = plan(trial)
            if trial_plan["kind"] == "fixed":
                self.solves[key] = {"plan": trial_plan, "scan": self.scan(trial_plan)}
            else:
                module, kernel, field = self.fused(trial_plan)
                self.solves[key] = {"plan": trial_plan, "module": module, "kernel": kernel, "key": field}
        return self.solves[key]

    def compile(self, trial, values):
        """Load a fused kernel's Warp module, or lower and compile the scan at the trial's n.

        Raise MemoryError if the compiled scan needs more memory than the device has.
        """
        import jax
        import jax.numpy as jnp
        import warp as wp
        solve = self.solve_of(trial)
        if "scan" in solve:
            compiled = solve["scan"].lower(jnp.asarray(values)).compile()
            limit = jax.local_devices()[0].memory_stats()["bytes_limit"]
            shortfall = memory_shortfall(compiled.memory_analysis(), limit)
            if shortfall is not None:
                raise MemoryError("RESOURCE_EXHAUSTED: the compiled solve needs {0:.2f} GiB; the device "
                                  "limit is {1:.2f} GiB".format(shortfall[0] / GIB, shortfall[1] / GIB))
        else:
            wp.load_module(solve["kernel"].module, device=DEVICE)
        self.checked.add((plan_key(trial), int(values.shape[0])))

    def prepare(self, trial, values):
        if (plan_key(trial), int(values.shape[0])) not in self.checked:
            self.compile(trial, values)
        return self.solve_of(trial)

    def host_solve(self, trial, values):
        """Solve from host arrays and return (states, status) on the host, timing both copies."""
        import jax
        import jax.numpy as jnp
        solve = self.prepare(trial, values)
        host = np.ascontiguousarray(values)
        if "scan" in solve:
            states = jax.device_get(jax.block_until_ready(solve["scan"](jnp.asarray(host))))
            return states, np.zeros(states.shape[0], dtype=np.int32)
        try:
            y0_wp, p_wp = self.fused_inputs(solve["plan"], solve["module"], solve["key"], host)
            y_out, status = self.fused_launch(solve, y0_wp, p_wp)
            return y_out.numpy().reshape(-1, self.states), status.numpy()
        except RuntimeError as exc:
            raise as_memory_error(exc)

    def device_solve(self, trial, values):
        """Solve on resident device inputs and leave the results on the device.

        Upload the inputs first when the resident ones belong to another grid.
        """
        import jax
        import jax.numpy as jnp
        solve = self.prepare(trial, values)
        n = int(values.shape[0])
        key = (n, "scan" in solve, solve["plan"].get("stiff"))
        if self.resident_key != key:
            self.resident = None
            host = np.ascontiguousarray(values)
            if "scan" in solve:
                self.resident = jnp.asarray(host)
            else:
                self.resident = self.fused_inputs(solve["plan"], solve["module"], solve["key"], host)
            self.resident_key = key
        if "scan" in solve:
            return jax.block_until_ready(solve["scan"](self.resident)), None
        try:
            return self.fused_launch(solve, *self.resident)
        except RuntimeError as exc:
            raise as_memory_error(exc)

    def close(self):
        self.solves = {}
        self.resident = None
        if self.cache_dirs:
            import warp as wp
            set_cache_dir(JAX_CACHE_DIR)
            wp.config.kernel_cache_dir = WARP_CACHE_DIR
            for path in self.cache_dirs:
                shutil.rmtree(path, ignore_errors=True)
            self.cache_dirs = []


class GradsolveAdapter:
    """Run gradsolve trials for runner.py on one machine."""

    controllers = CONTROLLERS

    def __init__(self, key, root):
        self.key, self.root = key, root

    def version(self):
        import gradsolve
        import warp
        return "{0}+warp{1}".format(gradsolve.__version__, warp.__version__)

    def states(self, trial):
        return int(problem_row(trial)["states"])

    def build(self, trial, cold=False):
        return Build(trial, cold)

    def compile(self, build, trial, values):
        build.compile(trial, values)

    def optimize(self, build, trial):
        raise NotImplementedError("gradsolve has no launch geometry to optimize")

    def solve(self, build, trial, values, transfers):
        if transfers == "both":
            return build.host_solve(trial, values)
        return build.device_solve(trial, values)

    def finals(self, build, result):
        """Return (finals, t_final, retcode) for the result of a solve."""
        states, status = result
        states = to_host(states).reshape(-1, build.states)
        status = np.zeros(states.shape[0], dtype=np.int32) if status is None else to_host(status)
        return finals_of(states, status, build.duration)


def run(argv):
    """Run a trial file on the GPU with the persistent jax and Warp caches.

    Exit if jax or Warp sees no GPU.
    """
    import gradsolve  # noqa: F401
    import jax
    import warp as wp
    if jax.default_backend() == "cpu":
        raise SystemExit("jax is running on the CPU backend; this is a GPU benchmark, so nothing is recorded")
    print("jax backend: " + jax.default_backend(), flush=True)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    os.makedirs(JAX_CACHE_DIR, exist_ok=True)
    os.makedirs(WARP_CACHE_DIR, exist_ok=True)
    set_cache_dir(JAX_CACHE_DIR)
    wp.config.kernel_cache_dir = WARP_CACHE_DIR
    wp.init()
    if not wp.is_cuda_available():
        raise SystemExit("Warp sees no CUDA device; this is a GPU benchmark, so nothing is recorded")
    return runner.main(argv, lambda key, root: GradsolveAdapter(key, root))
