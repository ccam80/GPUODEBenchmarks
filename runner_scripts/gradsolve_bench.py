#!/usr/bin/env python

"""The gradsolve adapter for runner.py. An adaptive trial runs one of gradsolve's fused NVIDIA Warp kernels, one thread per trajectory: Tsit5 (warp_ode) or Rosenbrock23 (warp_rosenbrock), each on the problem's right-hand side registered through gradsolve.register_jax_field. A fixed trial runs gradsolve's fixed-step lax.scan solver for Tsit5 or classical RK4 at the trial's step count, jitted and vmapped over the ensemble. The adapter drives the kernels through the packages' launch layer, because gradsolve.solve fixes the first step at a hundredth of the span and the step bound at 4096 and returns host arrays; that way a trial gets the protocol's dt0, the suite's step bound and a solve on resident device inputs (`none`) as well as through host arrays (`both`). The fused kernels take no step-size pins or controller gains, so a trial that sets them is refused. A cold build compiles in fresh Warp and jax caches."""

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
# algorithm -> the fused kernel module that runs it adaptively, and whether its field is the stiff (Jacobian) form.
FUSED = {"tsit5": ("warp_ode", False), "rosenbrock23_sciml": ("warp_rosenbrock", True)}
# algorithm -> the fixed-step scan module that runs it at a constant step.
SCANS = {"tsit5": "fixed_step_tsit5", "classical-rk4": "fixed_step_explicit"}
DEVICE = "cuda:0"
# Functions the scalar right-hand sides call, bound to jax.numpy when the build imports jax.
JAX_FUNCTIONS = ("exp", "sin", "sqrt", "stack")


def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def step_count(duration, dt):
    """The whole number of steps of size dt that span the duration; refuses a dt that does not divide it."""
    if not _finite(dt) or dt <= 0.0:
        raise ValueError("a fixed stepping needs a finite positive dt, got {0!r}".format(dt))
    steps = int(round(float(duration) / float(dt)))
    if steps < 1 or abs(steps * float(dt) - float(duration)) > 1e-9 * float(duration):
        raise ValueError("dt {0!r} does not divide the duration {1!r}".format(dt, duration))
    return steps


def plan(trial):
    """The solve a trial asks for: a fixed plan names the scan and its step count; an adaptive plan names the fused kernel, the tolerances, dt0 (gradsolve's own span/100 when the trial leaves it unset) and the suite's step bound. Raises ValueError on an algorithm gradsolve does not run under the trial's controller, and on step-size pins or gains, which the fused kernels do not take."""
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
    """The trial fields that select a compiled solve within a build."""
    return tuple(trial[field] for field in ("controller", "dt", "dt_min", "dt_max", "atol", "rtol", "gains"))


def finals_of(states, status, duration):
    """(finals, t_final, retcode): a trajectory whose kernel status is nonzero failed to reach the duration within the step bound and carries NaN states; gradsolve reports no end time, so t_final is the duration."""
    finals = np.array(states, copy=True)
    codes = np.asarray(status).reshape(-1)
    finals[codes != 0] = np.nan
    retcode = ["" if code == 0 else "status {0}: the step bound was reached".format(int(code)) for code in codes]
    return finals, np.full(finals.shape[0], float(duration)), retcode


def to_host(array):
    """A host copy of a Warp, jax or numpy array."""
    numpy = getattr(array, "numpy", None)
    return np.asarray(numpy() if callable(numpy) else array)


def as_memory_error(exc):
    """A MemoryError for a Warp allocation failure, else the exception itself."""
    if isinstance(exc, RuntimeError) and "allocate" in str(exc).lower():
        return MemoryError("OUT_OF_MEMORY: " + str(exc))
    return exc


class ScanProblem:
    """The gradsolve Problem protocol over a scalar system: name, dim, the span and f_jax."""

    is_stiff = False

    def __init__(self, name, dim, duration, f_jax):
        self.name, self.dim, self.t0, self.t1, self.f_jax = name, dim, 0.0, float(duration), f_jax


class Build:
    """One problem, precision and algorithm: the scalar right-hand side registered as a gradsolve field (adaptive) or wrapped as a Problem (fixed), a solve per plan, and the device inputs of the last upload."""

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
            # A cold build registers its field under a name of its own, so it never reuses a kernel the process already loaded.
            self.name = "suite_{0}_{1}".format(self.row["problem"], self.states)
            if cold:
                self.name += "_cold{0}".format(Build.serial)
        except BaseException:
            self.close()
            raise

    # ------------------------------------------------------------ fused
    def fused(self, plan):
        """(module, kernel, field_key) of an adaptive plan: the field registered once per build, refused above the kernel's GPU dimension ceiling."""
        from gradsolve import register_jax_field
        from gradsolve.warp import warp_ode, warp_rosenbrock
        module = warp_rosenbrock if plan["stiff"] else warp_ode
        ceiling = (warp_rosenbrock._GPU_MAT_DIM_CEILING if plan["stiff"]
                   else warp_ode._GPU_VEC_DIM_CEILING)
        if self.states > ceiling:
            raise RuntimeError("gradsolve's fused {0} kernel runs at most {1} states on a GPU; {2} has {3}"
                               .format(plan["module"], ceiling, self.row["problem"], self.states))
        key = self.name + ("_stiff" if plan["stiff"] else "")
        register_jax_field(key, self.f_jax, self.states, 1, stiff=plan["stiff"])
        if plan["stiff"]:
            kernel = module._precision_spec(key, self.states, self.precision)[4]
        else:
            kernel = module._precision_spec(key, self.states, self.precision, vecp_P=1)[3]
        return module, kernel, key

    def fused_inputs(self, plan, module, key, values):
        """The kernel's (y0, params) device arrays for the grid values, typed as the module's own driver types them."""
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
        """One launch on device inputs; returns the device (y_out, status) arrays after the device synchronizes."""
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
        """The jitted, vmapped fixed-step scan of a plan: grid values in, final states out."""
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
        """The solve of a trial's plan, built on first use."""
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
        """Compile the trial's solve: the Warp module of a fused kernel, or the lowered scan at the trial's n, refused as out of memory when its compiled footprint exceeds the device."""
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
        """One solve through host arrays: the upload is the h2d, the host copy of the finals the d2h; returns (states, status) on the host."""
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
        """One solve on the resident inputs, uploaded when they are not this grid's; the results stay on the device."""
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
    """The runner adapter of the gradsolve package on one machine."""

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
        """(finals, t_final, retcode) of a solve: the final state of every trajectory, the duration, and the kernel status of a failed one."""
        states, status = result
        states = to_host(states).reshape(-1, build.states)
        status = np.zeros(states.shape[0], dtype=np.int32) if status is None else to_host(status)
        return finals_of(states, status, build.duration)


def run(argv):
    """Entry point of the gradsolve suite: a GPU backend, the persistent jax and Warp caches, then the trial file through runner.main."""
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
