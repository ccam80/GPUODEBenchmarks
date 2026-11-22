#!/usr/bin/env python

"""Run gradsolve's solvers for runner.py.

Run an adaptive trial through gradsolve.solve on the engine gradsolve's own
router picks, after registering the problem with gradsolve.register_jax_field.
Declare the problem stiff for an implicit algorithm and non-stiff for an
explicit one, and pass the tolerances and nothing else, so that gradsolve
chooses the kernel, the first step and the step bound and refuses a problem
past its own state limits. Route a non-stiff problem of 16 states or fewer to
warp_ode. Print the engine each solve ran. gradsolve.solve takes and returns
host arrays, so time an adaptive Tsit5 solve on resident device inputs by
launching the warp_ode kernel that gradsolve.solve runs, with the same first
step and step bound, on inputs uploaded before the timing. Raise RuntimeError
for any other adaptive solve on resident inputs. Run a fixed-step trial on
gradsolve's lax.scan Tsit5 or RK4, jitted over the ensemble. Compile a cold
build into fresh Warp and jax caches.
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
from jax_bench import GIB, memory_shortfall, problem_row, set_cache_dir  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JAX_CACHE_DIR = os.path.join(REPO_ROOT, "generated", "gradsolve_jax_cache")
WARP_CACHE_DIR = os.path.join(REPO_ROOT, "generated", "gradsolve_warp_cache")

CONTROLLERS = ("fixed", "default")
PRECISIONS = ("float32", "float64")
# Map each adaptive algorithm to the stiffness we declare, which is the only routing input gradsolve takes from us.
STIFF = {"tsit5": False, "rosenbrock23_sciml": True}
# Map each fixed-step algorithm to its lax.scan solver module.
SCANS = {"tsit5": "fixed_step_tsit5", "classical-rk4": "fixed_step_explicit"}
DEVICE = "cuda"
# Bind these names to jax.numpy for the scalar right-hand sides.
JAX_FUNCTIONS = ("exp", "sin", "sqrt", "stack")
NO_FINAL_STATE = "gradsolve returned no finite final state"
# We call private warp_ode functions as they stand in this gradsolve release.
RESIDENT_GRADSOLVE = "0.2.1"
# warp_ode.solve starts every trajectory at a hundredth of the span and stops it after this many steps.
FIRST_STEP_FRACTION = 0.01
STEP_BOUND = 4096


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
    controller, return the declared stiffness and the tolerances. Raise ValueError if
    gradsolve does not run the algorithm under the trial's controller, or if the
    trial sets dt_min, dt_max or gains, which gradsolve.solve does not take.
    """
    controller, algorithm = trial["controller"], trial["algorithm"]
    if controller not in CONTROLLERS:
        raise ValueError("unknown controller " + controller)
    if controller == "fixed":
        if algorithm not in SCANS:
            raise ValueError("gradsolve has no fixed-step {0}".format(algorithm))
        return {"kind": "fixed", "module": SCANS[algorithm],
                "steps": step_count(float(trial["duration"]), trial["dt"])}
    if algorithm not in STIFF:
        raise ValueError("gradsolve has no adaptive {0}".format(algorithm))
    if _finite(trial["dt_min"]) or _finite(trial["dt_max"]):
        raise ValueError("gradsolve.solve takes no dt_min or dt_max")
    if trial["gains"] not in ("", "{}"):
        raise ValueError("gradsolve.solve takes no controller gains, got " + trial["gains"])
    return {"kind": "adaptive", "stiff": STIFF[algorithm],
            "atol": float(trial["atol"]), "rtol": float(trial["rtol"])}


def plan_key(trial):
    """Return the trial fields that select a compiled solve within a build."""
    return tuple(trial[field] for field in ("controller", "dt", "dt_min", "dt_max", "atol", "rtol", "gains"))


def finals_of(states, duration):
    """Return (finals, t_final, retcode) from final states.

    Give a failure message to every trajectory with a non-finite state, which is
    how gradsolve marks a trajectory that did not reach the duration. Report the
    duration as t_final, because gradsolve reports no end time.
    """
    finals = np.asarray(states)
    failed = ~np.isfinite(finals).all(axis=1)
    retcode = [NO_FINAL_STATE if flag else "" for flag in failed]
    return finals, np.full(finals.shape[0], float(duration)), retcode


def as_memory_error(exc):
    """Return a MemoryError for a Warp allocation failure, and any other exception unchanged."""
    if isinstance(exc, RuntimeError) and "allocate" in str(exc).lower():
        return MemoryError("OUT_OF_MEMORY: " + str(exc))
    return exc


def _host(array):
    """Return a Warp, jax or NumPy array as a NumPy array."""
    if type(array).__module__.startswith("warp"):
        return array.numpy()
    return np.asarray(array)


class ResidentResult:
    """Hold the final states and status of a resident solve on the device."""

    def __init__(self, states, status, dim):
        self.states, self.status, self.dim = states, status, dim

    def numpy(self):
        """Return the final states on the host, NaN for every trajectory with a nonzero status, as warp_ode.solve does."""
        states = np.array(_host(self.states)).reshape(-1, self.dim)
        states[np.asarray(_host(self.status)).reshape(-1) != 0] = np.nan
        return states


class ResidentTsit5:
    """Launch the warp_ode kernel that gradsolve.solve runs for a problem, on inputs held on the device.

    Raise RuntimeError if gradsolve's router sends the problem to another
    engine, if the installed gradsolve is not RESIDENT_GRADSOLVE, or if the
    problem has more states than warp_ode runs on a GPU.
    """

    def __init__(self, problem, plan, precision, device=DEVICE):
        import gradsolve
        from gradsolve import dispatch
        from gradsolve.warp import warp_ode
        if gradsolve.__version__ != RESIDENT_GRADSOLVE:
            raise RuntimeError("the resident Tsit5 solve calls private warp_ode functions of gradsolve {0}; "
                               "gradsolve {1} is installed".format(RESIDENT_GRADSOLVE, gradsolve.__version__))
        engine = dispatch.choose_engine(problem.dim, problem.is_stiff, False)
        if engine != warp_ode.name or not warp_ode.supports(problem):
            raise RuntimeError("gradsolve.solve runs {0} for this problem, which takes and returns host arrays, "
                               "so it has no solve on resident inputs".format(engine))
        self.warp_ode, self.problem, self.precision, self.device = warp_ode, problem, precision, device
        self.field_key, self.dim = warp_ode._field_for(problem)
        gpu = device in ("cuda", "gpu")
        if gpu and self.dim > warp_ode._GPU_VEC_DIM_CEILING:
            raise RuntimeError("warp fused GPU register-vector limit: dim={0} > {1}".format(
                self.dim, warp_ode._GPU_VEC_DIM_CEILING))
        # warp_ode.solve launches a 3-state field on a GPU through its jax FFI runner and every other field through wp.launch.
        self.ffi = gpu and self.dim == 3
        self.np_dtype = np.float32 if precision == "float32" else np.float64
        span = problem.t1 - problem.t0
        self.conf = np.array([problem.t0, problem.t1, plan["rtol"], plan["atol"], span * FIRST_STEP_FRACTION],
                             dtype=self.np_dtype)
        self.n = None
        self.inputs = None

    def upload(self, y0, params):
        """Copy the initial states, the parameters and the solver settings to the device."""
        warp_ode = self.warp_ode
        param = warp_ode._param_of(self.problem, params)
        self.inputs = None
        if self.ffi:
            import jax.numpy as jnp
            y0_np, param_np, n = warp_ode._validate_batch(y0, param, self.np_dtype, dim=3)
            self.run = warp_ode._ffi_runner(n, STEP_BOUND, 0, self.precision)
            self.inputs = (jnp.asarray(y0_np), jnp.asarray(param_np), jnp.asarray(self.conf))
        else:
            import warp as wp
            if self.field_key in warp_ode._warp_kernel._VECP_FIELDS:
                param_2d = np.ascontiguousarray(np.asarray(param))
                _, scalar, vec, self.kernel = warp_ode._precision_spec(self.field_key, self.dim, self.precision,
                                                                       vecp_P=param_2d.shape[1])
                vec_p = wp.types.vector(length=max(param_2d.shape[1], 1), dtype=scalar)
                param_wp = wp.array(param_2d.astype(self.np_dtype), dtype=vec_p, device=self.device)
                y0_np = np.ascontiguousarray(np.asarray(y0, dtype=self.np_dtype))
                n = y0_np.shape[0]
            else:
                _, scalar, vec, self.kernel = warp_ode._precision_spec(self.field_key, self.dim, self.precision)
                y0_np, param_np, n = warp_ode._validate_batch(y0, param, self.np_dtype, dim=self.dim)
                param_wp = wp.array(param_np, dtype=scalar, device=self.device)
            self.scalar, self.vec = scalar, vec
            self.inputs = (wp.array(y0_np, dtype=vec, device=self.device), param_wp,
                           wp.array(self.conf, dtype=scalar, device=self.device))
        self.n = n

    def launch(self):
        """Run the kernel on the resident inputs and return a ResidentResult left on the device."""
        if self.ffi:
            import jax
            outputs = jax.block_until_ready(self.run(*self.inputs))
            return ResidentResult(outputs[0], outputs[3], self.dim)
        import warp as wp
        n, device = self.n, self.device
        states = wp.zeros(n, dtype=self.vec, device=device)
        accepted = wp.zeros(n, dtype=wp.int32, device=device)
        rejected = wp.zeros(n, dtype=wp.int32, device=device)
        status = wp.zeros(n, dtype=wp.int32, device=device)
        steps = wp.zeros((1, 1), dtype=self.scalar, device=device)
        wp.launch(self.kernel, dim=n, inputs=[*self.inputs, STEP_BOUND, 0, states, accepted, rejected, status, steps],
                  device=device)
        wp.synchronize_device(device)
        return ResidentResult(states, status, self.dim)


class Problem:
    """Present a scalar system to gradsolve through its Problem protocol."""

    def __init__(self, name, dim, duration, f_jax, is_stiff=False):
        self.name, self.dim, self.t0, self.t1, self.f_jax = name, dim, 0.0, float(duration), f_jax
        self.is_stiff = is_stiff


class Build:
    """Hold the solves of one problem, precision and algorithm.

    Build one solve per set of solve settings: gradsolve.solve on the problem's
    registered gradsolve field, or a jitted scan. Keep the device inputs of the
    last scan upload for scans on resident inputs.
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
        self.resident_n = None
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
        """Register the problem's field and return the Problem that gradsolve.solve routes."""
        from gradsolve import dispatch, register_jax_field
        # Route the small non-stiff cell to warp_ode.
        dispatch.CUDA_TSIT5_ENABLED = False
        key = self.name + ("_stiff" if plan["stiff"] else "")
        register_jax_field(key, self.f_jax, self.states, 1, stiff=plan["stiff"])
        return Problem(key, self.states, self.duration, self.f_jax, is_stiff=plan["stiff"])

    def fused_inputs(self, values):
        """Return the (y0, params) host arrays gradsolve.solve takes for the grid values."""
        n = int(values.shape[0])
        y0 = np.ascontiguousarray(np.broadcast_to(self.y0, (n, self.states)))
        return y0, np.asarray(values, dtype=self.np_dtype).reshape(n, 1)

    def fused_solve(self, solve, values):
        """Solve the grid values through gradsolve.solve and return the final states on the host.

        Print the engine gradsolve ran the first time each solve runs it.
        """
        import gradsolve
        plan = solve["plan"]
        n = int(values.shape[0])
        y0, params = self.fused_inputs(values)
        try:
            result = gradsolve.solve(solve["problem"], y0, params, rtol=plan["rtol"], atol=plan["atol"],
                                     device=DEVICE)
        except RuntimeError as exc:
            raise as_memory_error(exc)
        if solve.get("engine") != result.solver:
            solve["engine"] = result.solver
            print("gradsolve engine: {0} {1} rtol={2:g} ran {3}".format(
                self.name, self.algorithm, plan["rtol"], result.solver), flush=True)
        return np.asarray(result.y_final).reshape(n, self.states)

    # ------------------------------------------------------------- scans
    def scan(self, plan):
        """Return the jitted fixed-step scan that maps grid values to final states."""
        import importlib

        import jax
        import jax.numpy as jnp
        module = importlib.import_module("gradsolve.solvers." + plan["module"])
        problem = Problem(self.name, self.states, self.duration, self.f_jax)
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
                self.solves[key] = {"plan": trial_plan, "problem": self.fused(trial_plan)}
        return self.solves[key]

    def compile(self, trial, values):
        """Compile the trial's solve.

        Lower and compile a scan at the trial's n, and raise MemoryError if it
        needs more memory than the device has. Compile a fused kernel by
        solving the first two grid points, because gradsolve.solve is the only
        public call that builds one.
        """
        solve = self.solve_of(trial)
        if "scan" in solve:
            import jax
            import jax.numpy as jnp
            compiled = solve["scan"].lower(jnp.asarray(values)).compile()
            limit = jax.local_devices()[0].memory_stats()["bytes_limit"]
            shortfall = memory_shortfall(compiled.memory_analysis(), limit)
            if shortfall is not None:
                raise MemoryError("RESOURCE_EXHAUSTED: the compiled solve needs {0:.2f} GiB; the device "
                                  "limit is {1:.2f} GiB".format(shortfall[0] / GIB, shortfall[1] / GIB))
        else:
            self.fused_solve(solve, np.ascontiguousarray(values[:2]))
        self.checked.add((plan_key(trial), int(values.shape[0])))

    def prepare(self, trial, values):
        if "scan" in self.solve_of(trial) and (plan_key(trial), int(values.shape[0])) not in self.checked:
            self.compile(trial, values)
        return self.solve_of(trial)

    def host_solve(self, trial, values):
        """Solve from host arrays and return the final states on the host, timing both copies."""
        solve = self.prepare(trial, values)
        host = np.ascontiguousarray(values)
        if "scan" not in solve:
            return self.fused_solve(solve, host)
        import jax
        import jax.numpy as jnp
        return jax.device_get(jax.block_until_ready(solve["scan"](jnp.asarray(host))))

    def device_solve(self, trial, values):
        """Solve on resident device inputs and leave the result on the device.

        Upload the inputs first when the resident ones belong to another grid.
        Launch an adaptive solve through ResidentTsit5, which raises
        RuntimeError for an engine other than warp_ode.
        """
        solve = self.prepare(trial, values)
        if "scan" not in solve:
            return self.resident_fused_solve(solve, values)
        import jax
        import jax.numpy as jnp
        n = int(values.shape[0])
        if self.resident_n != n:
            self.resident = None
            self.resident = jnp.asarray(np.ascontiguousarray(values))
            self.resident_n = n
        return jax.block_until_ready(solve["scan"](self.resident))

    def resident_fused_solve(self, solve, values):
        """Launch the solve's warp_ode kernel on resident inputs, uploading them when they belong to another grid."""
        resident = solve.get("resident")
        if resident is None:
            resident = solve["resident"] = ResidentTsit5(solve["problem"], solve["plan"], self.precision)
        if resident.n != int(values.shape[0]):
            resident.upload(*self.fused_inputs(values))
        return resident.launch()

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
        states = result.numpy() if isinstance(result, ResidentResult) else np.asarray(result)
        return finals_of(states.reshape(-1, build.states), build.duration)


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
