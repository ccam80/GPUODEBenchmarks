#!/usr/bin/env python

"""The myokit_cuda adapter for runner.py: a build is one compiled Myokit CUDA model, its CellML picked or generated for the trial's problem and state count (compiled into a fresh CuPy kernel cache when cold); a solve runs the generated forward-Euler kernel for the trial's step count through host arrays (`both`) or on the resident device inputs (`none`), which the kernel integrates in place and `reset` restores before each repeat. The exported kernel is Euler in float32 at a fixed step, so `fixed` is the one controller a trial may name. The swept parameter reaches the kernel through the exporter's diffusion_current binding; the Fabbri-Linder model instead carries its two cascade inputs as zero-derivative states, set per trajectory from the grid's lattice index (fabbri.py), and its finals are the 35 model states in reference order."""

import json
import math
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fabbri  # noqa: E402
import runner  # noqa: E402
from problems import as_problem  # noqa: E402
from protocol import TRACE_EVERY_S, TRACE_SAMPLES  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGE_DIR = os.path.join(REPO_ROOT, "GPU_ODE_MYOKIT_CUDA")
MODELS_DIR = os.path.join(PACKAGE_DIR, "models")

CONTROLLERS = ("fixed",)
ALGORITHM = "euler"
PRECISION = "float32"
# The Fabbri-Linder states as Myokit names them, in CellML document order (fabbri.STATE_ORDER flattened).
FABBRI_QNAMES = (
    "Membrane.V_ode", "Nai_concentration.Nai_", "i_f_y_gate.y", "i_Na_m_gate.m", "i_Na_h_gate.h",
    "i_CaL_dL_gate.dL", "i_CaL_fL_gate.fL", "i_CaL_fCa_gate.fCa", "i_CaT_dT_gate.dT", "i_CaT_fT_gate.fT",
    "Ca_SR_release.R", "Ca_SR_release.O", "Ca_SR_release.I", "Ca_SR_release.RI", "Ca_buffering.fTMM",
    "Ca_buffering.fCMi", "Ca_buffering.fCMs", "Ca_buffering.fTC", "Ca_buffering.fTMC", "Ca_buffering.fCQ",
    "Ca_dynamics.Cai", "Ca_dynamics.Ca_nsr", "Ca_dynamics.Ca_jsr", "Ca_dynamics.Ca_sub",
    "i_Kur_rKur_gate.r_Kur", "i_Kur_sKur_gate.s_Kur", "i_to_q_gate.q", "i_to_r_gate.r", "i_Kr_pa_gate.paS",
    "i_Kr_pa_gate.paF", "i_Kr_pi_gate.piy", "i_Ks_n_gate.n", "i_KACh_a_gate.a", "cAMP.cAMP", "PLBp.PLBp",
)
# The two inputs promoted to states, appended to the model's own in this order.
FABBRI_INPUT_QNAMES = (fabbri.ACH_QNAME, fabbri.ISO_QNAME)
# problem -> (CellML component, ordered state variable names); the Fabbri-Linder model spans components.
MODELS = {
    "lorenz": ("lorenz", ("lorenz.x", "lorenz.y", "lorenz.z")),
    "lorenz96": ("lorenz96", tuple("lorenz96.x{0}".format(i) for i in range(1, 33))),
    "pleiades": ("pleiades", tuple("pleiades.{0}{1}".format(prefix, i)
                                   for prefix in ("x", "y", "u", "v") for i in range(1, 8))),
    fabbri.PROBLEM: (None, FABBRI_QNAMES),
}
RESIZABLE = "lorenz96"


def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def problem_row(trial):
    """The catalogue row of a trial, resized to its system_params states."""
    row = as_problem(trial["problem"])
    params = json.loads(trial["system_params"]) if trial["system_params"] else {}
    if "states" in params:
        row = row.resized(params["states"])
    return row


def check_trial(trial):
    """Refuses what the kernel cannot run: an algorithm other than euler, a precision other than float32, a controller other than fixed."""
    if trial["algorithm"] != ALGORITHM:
        raise ValueError("myokit_cuda runs {0} only, not {1}".format(ALGORITHM, trial["algorithm"]))
    if trial["precision"] != PRECISION:
        raise ValueError("myokit_cuda runs {0} only, not {1}".format(PRECISION, trial["precision"]))
    if trial["controller"] not in CONTROLLERS:
        raise ValueError("unknown controller " + trial["controller"])


def step_count(duration, dt):
    """The whole number of Euler steps that span the duration at dt."""
    if not _finite(dt) or dt <= 0.0:
        raise ValueError("myokit_cuda needs a finite positive dt, got {0!r}".format(dt))
    return int(round(float(duration) / float(dt)))


def steps_per_sample(dt):
    """The Euler steps between two trace samples; dt must divide the sample interval."""
    steps = step_count(TRACE_EVERY_S, dt)
    if steps < 1 or abs(steps * float(dt) - TRACE_EVERY_S) > 1e-9 * TRACE_EVERY_S:
        raise ValueError("dt {0!r} does not divide the {1} s sample interval".format(dt, TRACE_EVERY_S))
    return steps


def state_names(problem, states):
    """The state variable names the model must list, in order: the problem's own, then for the Fabbri-Linder model its two promoted inputs."""
    component, names = MODELS[problem]
    if problem == RESIZABLE:
        return tuple("{0}.x{1}".format(component, i) for i in range(1, states + 1))
    if problem == fabbri.PROBLEM:
        return names + FABBRI_INPUT_QNAMES
    return names


def model_path(problem, states, models_dir=MODELS_DIR):
    """The CellML file of a problem: the shipped model, a generated cyclic lorenz96 of another size, or the Fabbri-Linder file the packages share."""
    if problem not in MODELS:
        raise ValueError("no Myokit CellML model for problem '{0}'".format(problem))
    if problem == fabbri.PROBLEM:
        return fabbri.MODEL_PATH
    shipped = os.path.join(models_dir, problem + ".cellml")
    if problem == RESIZABLE and states != len(MODELS[problem][1]):
        return lorenz96_cellml(states, os.path.join(models_dir, "generated"))
    return shipped


def diffusion_variable(problem, parameter):
    """The qualified name of the swept parameter, bound to the exporter's diffusion_current; None for the Fabbri-Linder model, whose inputs travel as states."""
    if problem == fabbri.PROBLEM:
        return None
    return "{0}.{1}".format(MODELS[problem][0], parameter)


def prepare_fabbri(model):
    """Switch the Fabbri-Linder cAMP cascade on and promote its two analogue inputs to states with a zero derivative, so each trajectory carries its own ACh and Iso in the state array."""
    model.get(fabbri.ANS_QNAME).set_rhs(1)
    for qname in FABBRI_INPUT_QNAMES:
        variable = model.get(qname)
        variable.promote(0.0)
        variable.set_rhs(0)


def prepare_model(problem):
    """The model-preparation hook of a problem, or None."""
    return prepare_fabbri if problem == fabbri.PROBLEM else None


def lorenz96_cellml(n, outdir):
    """Write and return the path of a cyclic n-state lorenz96 CellML model."""
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "lorenz96_{0}.cellml".format(n))
    variables = "\n".join(
        '    <variable name="x{0}" units="dimensionless" '
        'initial_value="{1}"/>'.format(i, 9 if i == 1 else 8)
        for i in range(1, n + 1))
    rows = []
    for i in range(1, n + 1):
        ip1 = i % n + 1
        im1 = (i - 2) % n + 1
        im2 = (i - 3) % n + 1
        rows.append(
            "      <apply><eq/><apply><diff/><bvar><ci>time</ci></bvar>"
            "<ci>x{0}</ci></apply><apply><plus/><apply><minus/><apply>"
            "<times/><apply><minus/><ci>x{1}</ci><ci>x{2}</ci></apply>"
            "<ci>x{3}</ci></apply><ci>x{0}</ci></apply><ci>F</ci></apply>"
            "</apply>".format(i, ip1, im2, im1))
    text = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<model\n    name="lorenz96"\n'
        '    xmlns="http://www.cellml.org/cellml/1.1#"\n'
        '    xmlns:cellml="http://www.cellml.org/cellml/1.1#">\n'
        '  <component name="environment">\n'
        '    <variable\n        name="time"\n'
        '        units="dimensionless"\n'
        '        public_interface="out"/>\n'
        '  </component>\n\n'
        '  <component name="lorenz96">\n'
        '    <variable\n        name="time"\n'
        '        units="dimensionless"\n'
        '        public_interface="in"/>\n'
        + variables + '\n'
        '    <variable name="F" units="dimensionless" initial_value="8"/>\n\n'
        '    <math xmlns="http://www.w3.org/1998/Math/MathML">\n'
        + "\n".join(rows) + '\n'
        '    </math>\n'
        '  </component>\n\n'
        '  <connection>\n'
        '    <map_components component_1="environment" '
        'component_2="lorenz96"/>\n'
        '    <map_variables variable_1="time" variable_2="time"/>\n'
        '  </connection>\n'
        '</model>\n')
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return path


def finals_of(host_result, duration):
    """(finals, t_final, retcode) from a host (cells, states) result: every trajectory ends at the duration and the kernel reports no failure code."""
    finals = np.asarray(host_result)
    n = finals.shape[0]
    return finals, np.full(n, float(duration)), [""] * n


def cold_cache():
    """Point CuPy's kernel cache at a fresh directory; returns (previous backend, directory)."""
    from cupy.cuda import compiler
    directory = tempfile.mkdtemp(prefix="myokit_cold_")
    previous = compiler._kernel_cache_backend
    compiler._set_kernel_cache_backend(compiler._DiskKernelCacheBackend(directory))
    return previous, directory


def restore_cache(previous, directory):
    from cupy.cuda import compiler
    compiler._set_kernel_cache_backend(previous)
    shutil.rmtree(directory, ignore_errors=True)


def load_model_class():
    sys.path.insert(0, PACKAGE_DIR)
    from myokit_cuda import MyokitCudaModel
    return MyokitCudaModel


# ---------------------------------------------------------------------- build

class Build:
    """One compiled model; the host inputs (initial states and the per-cell binding value) and the resident device inputs of the current grid."""

    def __init__(self, trial, cold=False, model_class=None):
        check_trial(trial)
        self.row = problem_row(trial)
        self.problem = self.row["problem"]
        self.duration = float(trial["duration"])
        self.states = int(self.row["states"])
        self.cache = None
        self.model = None
        self.initial_n = None
        self.initial_grid = None
        self.initial = None
        self.resident_n = None
        self.resident_grid = None
        self.resident = None
        self.resident_initial = None
        if cold:
            self.cache = cold_cache()
        try:
            model_class = model_class or load_model_class()
            self.model = model_class(model_path(self.problem, self.states),
                                     diffusion_variable=diffusion_variable(self.problem,
                                                                           self.row["sweep_parameter"]),
                                     prepare=prepare_model(self.problem))
            expected = state_names(self.problem, self.states)
            if tuple(self.model.state_names) != expected:
                raise RuntimeError("unexpected {0} state order: {1}".format(
                    self.problem, self.model.state_names))
            # The rows of the promoted inputs, and the columns of the problem's own states in reference order.
            self.input_rows = tuple(expected.index(q) for q in FABBRI_INPUT_QNAMES) \
                if self.problem == fabbri.PROBLEM else ()
            self.columns = None if len(expected) == self.states \
                else [expected.index(q) for q in MODELS[self.problem][1]]
        except BaseException:
            self.close()
            raise

    def inputs(self, values):
        """(initial states (states, cells), the per-cell diffusion_current values) of a grid, rebuilt when the grid changes: the grid values themselves bind to the swept parameter, or for the Fabbri-Linder model set the two input states while the unused binding stays zero."""
        values = np.ascontiguousarray(values, dtype=np.float32)
        n = int(values.shape[0])
        if self.initial_n != n or not np.array_equal(self.initial_grid, values):
            self.initial = self.model.initial_states(n)
            if self.input_rows:
                ach, iso = fabbri.inputs(values, np.float32)
                self.initial[self.input_rows[0]] = ach
                self.initial[self.input_rows[1]] = iso
            self.initial_n = n
            self.initial_grid = values
        diffusion = np.zeros(n, dtype=np.float32) if self.input_rows else values
        return self.initial, diffusion

    def host_solve(self, trial, values):
        """One solve through host arrays: the uploads, the kernel and the copy back."""
        initial, diffusion = self.inputs(values)
        return self.model.solve(dt=float(trial["dt"]), step_count=step_count(self.duration, trial["dt"]),
                                initial_states=initial, diffusion_values=diffusion)

    def device_solve(self, trial, values):
        """One solve on the resident inputs, uploaded when they are not this grid's; the kernel integrates the resident states in place and they stay on the device."""
        values = np.ascontiguousarray(values, dtype=np.float32)
        n = int(values.shape[0])
        if self.resident_n != n or not np.array_equal(self.resident_grid, values):
            self.resident = None
            self.resident_initial = None
            self.resident = self.model.to_device(*self.inputs(values))
            self.resident_initial = self.resident[0].copy()
            self.resident_n = n
            self.resident_grid = values
        return self.model.solve_on_device(float(trial["dt"]), step_count(self.duration, trial["dt"]),
                                          *self.resident)

    def trace(self, trial, values):
        """States of the given grid points at every sample time, (cells, samples, states) in reference order."""
        initial, diffusion = self.inputs(values)
        samples = self.model.trace(float(trial["dt"]), steps_per_sample(trial["dt"]), TRACE_SAMPLES, initial, diffusion)
        if self.columns is not None:
            samples = np.ascontiguousarray(samples[:, :, self.columns])
        return samples

    def restore(self, n):
        """Put the initial states back into the resident buffer of this n, on the device."""
        if self.resident_n == n and self.resident is not None:
            self.resident[0][...] = self.resident_initial

    def finals(self, result):
        """(finals, t_final, retcode) of a host or device result: the problem's own states in reference order."""
        if not isinstance(result, np.ndarray):
            result = result.get().T
        finals, t_final, retcode = finals_of(result, self.duration)
        if self.columns is not None:
            finals = np.ascontiguousarray(finals[:, self.columns])
        return finals, t_final, retcode

    def close(self):
        self.resident = None
        self.resident_initial = None
        self.initial = None
        self.model = None
        if self.cache is not None:
            restore_cache(*self.cache)
            self.cache = None


class MyokitAdapter:
    """The runner adapter of the myokit_cuda package on one machine."""

    controllers = CONTROLLERS

    def __init__(self, key, root, model_class=None):
        self.key, self.root = key, root
        self.model_class = model_class

    def version(self):
        import myokit
        return myokit.__version__

    def states(self, trial):
        return int(problem_row(trial)["states"])

    def build(self, trial, cold=False):
        return Build(trial, cold, self.model_class)

    def compile(self, build, trial, values):
        """The kernel compiles when the model is built; nothing more to warm."""

    def optimize(self, build, trial):
        raise NotImplementedError("myokit_cuda has no launch geometry to optimize")

    def solve(self, build, trial, values, transfers):
        if transfers == "both":
            return build.host_solve(trial, values)
        return build.device_solve(trial, values)

    def trace(self, build, trial, values):
        return build.trace(trial, values)

    def reset(self, build, trial, values, transfers):
        """Before a repeated resident solve, put the initial states back; a host solve uploads its own."""
        if transfers == "none":
            build.restore(int(values.shape[0]))

    def finals(self, build, result):
        return build.finals(result)


def run(argv):
    """Entry point of the myokit_cuda suite: the trial file through runner.main."""
    return runner.main(argv, lambda key, root: MyokitAdapter(key, root))
