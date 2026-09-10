#!/usr/bin/env python

"""The cubie adapter for runner.py, shared by the CUBIE and CUBIE_MLIR suites: a build is one system and one Solver whose stepping follows each trial (in a fresh cache directory when cold); optimize runs Solver.optimize on the line's batch and records the winner; a solve runs through host arrays (`both`) or on the resident device inputs (`none`)."""

import gc
import importlib.metadata
import json
import math
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cubie_adapter as adapter  # noqa: E402
import runner  # noqa: E402
from cubie_systems import final_states, output_types, variable_order  # noqa: E402
from problems import as_problem  # noqa: E402

PRECISIONS = {"float32": np.float32, "float64": np.float64}
# The controller names a trial may carry: cubie's own step controllers plus the two shared tokens.
CONTROLLERS = ("fixed", "default", "i", "pi", "pid", "gustafsson")
# The trial fields that reach the Solver only when finite.
PINNED = ("dt", "dt_min", "dt_max", "newton_atol", "newton_rtol")
STEPPING = ("controller",) + PINNED + ("atol", "rtol", "gains")


def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def problem_row(trial):
    """The catalogue row of a trial, resized to its system_params states."""
    row = as_problem(trial["problem"])
    params = json.loads(trial["system_params"]) if trial["system_params"] else {}
    if "states" in params:
        row = row.resized(params["states"])
    return row


def stepping_kwargs(trial):
    """The Solver keywords a trial's stepping sets: the controller, atol and rtol when adaptive, and each pin that is not NaN; a NaN leaves cubie's own value in place."""
    kwargs = {}
    if trial["controller"] == "fixed":
        kwargs["step_controller"] = "fixed"
    else:
        kwargs["atol"] = float(trial["atol"])
        kwargs["rtol"] = float(trial["rtol"])
        if trial["controller"] != "default":
            kwargs["step_controller"] = trial["controller"]
    for key in PINNED:
        if _finite(trial[key]):
            kwargs[key] = float(trial[key])
    return kwargs


def gains_of(trial):
    """The explicit controller parameters of a trial as a dict."""
    return json.loads(trial["gains"]) if trial["gains"] else {}


def make_solver(system, trial, solver_class=None):
    """A Solver for a trial's algorithm and stepping; explicit gains are applied after construction."""
    if solver_class is None:
        import cubie
        solver_class = cubie.Solver
    kwargs = dict(algorithm=trial["algorithm"], save_every=float(trial["duration"]),
                  output_types=output_types(system), time_logging_level=None)
    kwargs.update(stepping_kwargs(trial))
    solver = solver_class(system, **kwargs)
    gains = gains_of(trial)
    if gains:
        solver.update(gains)
    return solver


def optimize_setting(trial):
    """(mode, setting) of the optimize row a trial records: its dt or tolerance."""
    return adapter.optimize_setting(trial)


class Build:
    """One system and one Solver; the grid arrays of the current n and the resident device inputs of the last host solve."""

    def __init__(self, package, key, root, trial, cold=False, solver_class=None):
        self.package, self.key, self.root = package, key, root
        self.solver_class = solver_class
        self.row = problem_row(trial)
        self.precision = PRECISIONS[trial["precision"]]
        self.duration = float(trial["duration"])
        self.states = len(variable_order(self.row))
        self.cache_dir = None
        self.saved_cache_root = None
        self.solver = None
        self.grid_n = None
        self.grid_arrays = None
        self.resident_n = None
        self.host_result = None
        if cold:
            from cubie.cache_root import get_cache_root_override, set_cache_root
            self.saved_cache_root = get_cache_root_override()
            self.cache_dir = tempfile.mkdtemp(prefix="cubie_cold_")
            set_cache_root(self.cache_dir)
        try:
            # A resized system keeps its own generated-code cache under a states suffix.
            params = json.loads(trial["system_params"]) if trial["system_params"] else {}
            self.system, self.initial_conditions = adapter.build_system(
                self.row, package, self.precision, states=params.get("states"))
            self.solver = make_solver(self.system, trial, solver_class)
        except BaseException:
            self.close()
            raise
        self.applied = {key: trial[key] for key in STEPPING}

    def apply(self, trial):
        """Follow a trial's stepping: a changed controller or gains rebuilds the solver, changed steps or tolerances update it; either drops the resident inputs with the kernel."""
        stepping = {key: trial[key] for key in STEPPING}
        if all(_same(stepping[key], self.applied[key]) for key in STEPPING):
            return
        self.host_result = None
        self.grid_arrays = None
        self.grid_n = None
        self.resident_n = None
        if stepping["controller"] != self.applied["controller"] \
                or stepping["gains"] != self.applied["gains"]:
            self.solver.close()
            self.solver = None
            gc.collect()
            self.solver = make_solver(self.system, trial, self.solver_class)
        else:
            self.solver.update(stepping_kwargs(trial))
        self.applied = stepping

    def grid(self, values):
        """(initial_values, parameters) arrays for a grid of the swept parameter; rebuilt only when n changes."""
        n = int(values.shape[0])
        if self.grid_n != n:
            self.grid_arrays = None
            parameters = {self.row["sweep_parameter"]: np.asarray(values, dtype=self.precision)}
            self.grid_arrays = self.solver.build_grid(initial_values=self.initial_conditions,
                                                      parameters=parameters)
            self.grid_n = n
        return self.grid_arrays

    def host_solve(self, values):
        """One solve through host arrays; its result is kept for the finals and its inputs stay resident."""
        self.host_result = None
        initials, parameters = self.grid(values)
        self.host_result = adapter.solve(self.solver, initials, parameters, self.duration)
        self.resident_n = int(values.shape[0])
        return self.host_result

    def device_solve(self, values):
        """One solve on the resident inputs, uploading through a host solve first when they are not this grid's; returns the host result whose finals the device run reproduces."""
        if self.resident_n != int(values.shape[0]) or self.host_result is None:
            self.host_solve(values)
        adapter.solve(self.solver, self.solver.device_initial_values,
                      self.solver.device_parameters, self.duration, on_device=True)
        return self.host_result

    def close(self):
        self.host_result = None
        self.grid_arrays = None
        if self.solver is not None:
            self.solver.close()
            self.solver = None
        gc.collect()
        if self.cache_dir is not None:
            from cubie.cache_root import set_cache_root
            set_cache_root(self.saved_cache_root)
            shutil.rmtree(self.cache_dir, ignore_errors=True)
            self.cache_dir = None


def _same(a, b):
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (math.isnan(a) and math.isnan(b))
    return a == b


class CubieAdapter:
    """The runner adapter of one cubie package on one machine."""

    controllers = CONTROLLERS

    def __init__(self, package, key, root, solver_class=None):
        self.package, self.key, self.root = package, key, root
        self.solver_class = solver_class

    def version(self):
        return importlib.metadata.version("cubie") + "+" + adapter.BACKENDS[self.package]

    def states(self, trial):
        return len(variable_order(problem_row(trial)))

    def build(self, trial, cold=False):
        return Build(self.package, self.key, self.root, trial, cold, self.solver_class)

    def compile(self, build, trial, values):
        build.apply(trial)
        initials, parameters = build.grid(values)
        build.solver.compile(initials, parameters, duration=build.duration)

    def optimize(self, build, trial, values):
        """The point's recorded settings from the same source applied to the solver and compiled, else Solver.optimize on the line's batch with the winner applied and recorded under the package, key and source."""
        build.apply(trial)
        build.host_result = None
        build.resident_n = None
        initials, parameters = build.grid(values)
        mode, setting = optimize_setting(trial)
        source = adapter.source_hash(build.solver)
        tuned = adapter.load_optimized(self.package, self.key, build.row, trial["algorithm"], mode, setting,
                                       states=build.row["states"], root=self.root,
                                       controller=trial["controller"], gains=trial["gains"], source=source,
                                       n=int(initials.shape[1]))
        if tuned is not None:
            adapter.apply_optimized(build.solver, tuned)
            build.solver.compile(initials, parameters, duration=build.duration)
            print("optimized {0}: recorded".format(runner.label(trial)), flush=True)
            return
        row = adapter.optimize_point(build.solver, build.row, initials, parameters, self.package,
                                     self.key, trial["algorithm"], mode, setting,
                                     states=build.row["states"], root=self.root, force=True,
                                     controller=trial["controller"], gains=trial["gains"], source=source)
        print("optimized {0}: {1}".format(runner.label(trial), row["label"]), flush=True)

    def solve(self, build, trial, values, transfers):
        build.apply(trial)
        if transfers == "both":
            return build.host_solve(values)
        return build.device_solve(values)

    def finals(self, build, result):
        """(finals, t_final, retcode) of a host result: the problem's variables in reference order, the duration where the run's status is clean and NaN otherwise, and the status flags joined by '|'."""
        from cubie.result_codes import decode_status_codes
        finals = np.array(final_states(build.system, result, build.row))
        codes = np.asarray(result.status_codes).reshape(-1)
        names = decode_status_codes(codes)
        retcode = ["|".join(names[index]) if index in names else "" for index in range(codes.shape[0])]
        t_final = np.where(codes == 0, build.duration, np.nan)
        return finals, t_final, retcode


def run(argv, package):
    """Entry point of a cubie suite: select the backend, then run the trial file through runner.main."""
    adapter.select_backend(package)
    from cubie.time_logger import default_timelogger
    default_timelogger.set_verbosity(None)
    return runner.main(argv, lambda key, root: CubieAdapter(package, key, root))
