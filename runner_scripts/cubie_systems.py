"""Cubie system definitions, one builder per problem, shared by every cubie suite."""

import numpy as np
import cubie as qb

from problems import get_problem

# Ring modulator constants (Test Set for IVP Solvers, problem II-3).
RING_CONSTANTS = {
    "C": 1.6e-8,
    "Cp": 1.0e-8,
    "Lh": 4.45,
    "Ls1": 0.002,
    "Ls2": 5.0e-4,
    "Ls3": 5.0e-4,
    "gamma": 40.67286402e-9,
    "R": 25000.0,
    "Rp": 50.0,
    "Rg1": 36.3,
    "Rg2": 17.3,
    "Rg3": 17.3,
    "Ri": 50.0,
    "Rc": 600.0,
    "delta": 17.7493332,
    # Input angular frequencies: 2000*pi and 20000*pi.
    "w1": 6283.185307179586,
    "w2": 62831.85307179586,
}

RING_ORDER = ("U1", "U2", "U3", "U4", "U5", "U6", "U7",
              "I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8")
RING_STATES = {name: 0.0 for name in RING_ORDER}

# Variables torn out of the index-2 form; recorded as observables.
RING_INDEX2_OBSERVABLES = ["U3", "U4", "U6", "I3"]

# Diode voltages and currents, shared by both ring modulator formulations.
RING_AUXILIARIES = """
    Uin1 = Uin1_amplitude * sin(w1 * t)
    Uin2 = 2.0 * sin(w2 * t)
    UD1 = U3 - U5 - U7 - Uin2
    UD2 = -U4 + U6 - U7 - Uin2
    UD3 = U4 + U5 + U7 + Uin2
    UD4 = -U3 - U6 + U7 + Uin2
    qD1 = gamma * (exp(delta * UD1) - 1.0)
    qD2 = gamma * (exp(delta * UD2) - 1.0)
    qD3 = gamma * (exp(delta * UD3) - 1.0)
    qD4 = gamma * (exp(delta * UD4) - 1.0)
"""

# The rows that do not involve Cs.
RING_COMMON = """
    dU1 = (I1 - 0.5 * I3 + 0.5 * I4 + I7 - U1 / R) / C
    dU2 = (I2 - 0.5 * I5 + 0.5 * I6 + I8 - U2 / R) / C
    dU7 = (-U7 / Rp + qD1 + qD2 - qD3 - qD4) / Cp
    dI1 = -U1 / Lh
    dI2 = -U2 / Lh
    dI3 = (0.5 * U1 - U3 - Rg2 * I3) / Ls2
    dI4 = (-0.5 * U1 + U4 - Rg3 * I4) / Ls3
    dI5 = (0.5 * U2 - U5 - Rg2 * I5) / Ls2
    dI6 = (-0.5 * U2 + U6 - Rg3 * I6) / Ls3
    dI7 = (-U1 + Uin1 - (Ri + Rg1) * I7) / Ls1
    dI8 = (-U2 - (Rc + Rg1) * I8) / Ls1
"""


def _lorenz(problem, precision, name):
    system = qb.create_ODE_system(
        """
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        """,
        states={"x": 1.0, "y": 0.0, "z": 0.0},
        parameters={"rho": 21.0},
        constants={"sigma": 10.0, "beta": 8.0 / 3.0},
        name=name,
        precision=precision,
    )
    return system, {"x": 1.0, "y": 0.0, "z": 0.0}


def _ring_modulator(problem, precision, name):
    """Stiff 15-state ODE form; the swept Cs stays above zero."""
    equations = RING_AUXILIARIES + """
    dU3 = (I3 - qD1 + qD4) / Cs
    dU4 = (-I4 + qD2 - qD3) / Cs
    dU5 = (I5 + qD1 - qD3) / Cs
    dU6 = (-I6 - qD2 + qD4) / Cs
""" + RING_COMMON
    constants = dict(RING_CONSTANTS, Uin1_amplitude=0.5)
    system = qb.create_ODE_system(
        equations,
        states=dict(RING_STATES),
        parameters={"Cs": 2.0e-12},
        constants=constants,
        name=name,
        precision=precision,
    )
    return system, dict(RING_STATES)


def _ring_modulator_index2(problem, precision, name):
    """Cs = 0: the four capacitor rows become algebraic, giving index 2."""
    equations = RING_AUXILIARIES + """
    0 = I3 - qD1 + qD4
    0 = -I4 + qD2 - qD3
    0 = I5 + qD1 - qD3
    0 = -I6 - qD2 + qD4
""" + RING_COMMON
    system = qb.create_ODE_system(
        equations,
        states=dict(RING_STATES),
        parameters={"Uin1_amplitude": 0.5},
        constants=dict(RING_CONSTANTS),
        observables=list(RING_INDEX2_OBSERVABLES),
        name=name,
        precision=precision,
        simplify=True,
    )
    # Simplification tears states out, so the grid takes the retained set.
    return system, state_defaults(system)


# The problem's own variables, in the order the golden reference stores them.
_ORDER = {
    "lorenz": ("x", "y", "z"),
    "ring_modulator": RING_ORDER,
    "ring_modulator_index2": RING_ORDER,
}

_BUILDERS = {
    "lorenz": _lorenz,
    "ring_modulator": _ring_modulator,
    "ring_modulator_index2": _ring_modulator_index2,
}


def build_system(problem, precision=np.float32, name_suffix=""):
    """Return ``(system, initial_values)`` for a problem row or name.

    ``name_suffix`` separates one caller's generated-code cache from
    another's for the same equations."""
    row = problem if isinstance(problem, dict) else get_problem(problem)
    key = row["problem"]
    if key not in _BUILDERS:
        raise SystemExit("no cubie definition for problem '{0}'".format(key))
    name = row["display"].replace(" ", "_").replace("(", "").replace(")", "")
    return _BUILDERS[key](row, precision, name + name_suffix)


def variable_order(problem):
    """The problem's own variables, in golden-reference order."""
    row = problem if isinstance(problem, dict) else get_problem(problem)
    return _ORDER[row["problem"]]


def output_types(system):
    """Output types needed to recover every variable of the problem."""
    if getattr(system.sizes, "observables", 0):
        return ["state", "observables"]
    return ["state"]


def _names(index_map):
    return [str(symbol) for symbol in index_map]


def state_defaults(system):
    """Zero initial values for the states the system actually integrates."""
    return {name: 0.0 for name in _names(system.indices.states.index_map)}


def final_states(system, solution, problem):
    """Final values of the problem's variables, in golden-reference order; torn variables come from the observables."""
    order = variable_order(problem)
    state_names = _names(system.indices.states.index_map)
    finals = np.asarray(solution.state[-1, :, :]).T
    if len(state_names) == len(order) and list(state_names) == list(order):
        return finals
    observable_names = _names(system.indices.observables.index_map)
    observables = np.asarray(solution.observables[-1, :, :]).T
    columns = []
    for name in order:
        if name in state_names:
            columns.append(finals[:, state_names.index(name)])
        elif name in observable_names:
            columns.append(observables[:, observable_names.index(name)])
        else:
            raise SystemExit(
                "variable '{0}' is neither a state nor an observable".format(name))
    return np.stack(columns, axis=1)


def sweep_parameters(problem, n, precision=np.float32):
    """The ensemble parameter dict: the swept scalar over the problem range."""
    row = problem if isinstance(problem, dict) else get_problem(problem)
    return {row["sweep_parameter"]: row.sweep(n, dtype=precision)}
