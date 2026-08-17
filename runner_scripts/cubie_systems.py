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


def _lorenz96(problem, precision, name):
    """Cyclic 40-state Lorenz 96; the swept forcing F drives every row."""
    n = 40
    lines = []
    for i in range(1, n + 1):
        ip1 = i % n + 1
        im1 = (i - 2) % n + 1
        im2 = (i - 3) % n + 1
        lines.append("dx{0} = (x{1} - x{2}) * x{3} - x{0} + F".format(
            i, ip1, im2, im1))
    # Uniform state 8 with x1 perturbed to 9, so every swept F moves at t = 0.
    states = {"x{0}".format(i): 9.0 if i == 1 else 8.0
              for i in range(1, n + 1)}
    system = qb.create_ODE_system(
        "\n".join(lines),
        states=dict(states),
        parameters={"F": 8.0},
        name=name,
        precision=precision,
    )
    return system, dict(states)


# Pleiades initial state (Test Set for IVP Solvers): x, y, x', y' per star.
PLEIADES_X0 = (3.0, 3.0, -1.0, -3.0, 2.0, -2.0, 2.0)
PLEIADES_Y0 = (3.0, -3.0, 2.0, 0.0, 0.0, -4.0, 4.0)
PLEIADES_U0 = (0.0, 0.0, 0.0, 0.0, 0.0, 1.75, -1.5)
PLEIADES_V0 = (0.0, 0.0, 0.0, -1.25, 1.0, 0.0, 0.0)


def _pleiades(problem, precision, name):
    """Seven-body planar gravitation with masses (m1, 2, ..., 7); m1 swept."""
    lines = []
    for i in range(1, 8):
        for j in range(i + 1, 8):
            lines.append("q{0}{1} = (x{0} - x{1})**2 + (y{0} - y{1})**2"
                         .format(i, j))
            lines.append("w{0}{1} = q{0}{1} * sqrt(q{0}{1})".format(i, j))
    for i in range(1, 8):
        lines.append("dx{0} = u{0}".format(i))
        lines.append("dy{0} = v{0}".format(i))
    for i in range(1, 8):
        ax, ay = [], []
        for j in range(1, 8):
            if j == i:
                continue
            pair = "w{0}{1}".format(min(i, j), max(i, j))
            ax.append("m{0} * (x{0} - x{1}) / {2}".format(j, i, pair))
            ay.append("m{0} * (y{0} - y{1}) / {2}".format(j, i, pair))
        lines.append("du{0} = {1}".format(i, " + ".join(ax)))
        lines.append("dv{0} = {1}".format(i, " + ".join(ay)))
    states = {}
    for prefix, values in (("x", PLEIADES_X0), ("y", PLEIADES_Y0),
                           ("u", PLEIADES_U0), ("v", PLEIADES_V0)):
        for i, value in enumerate(values, start=1):
            states["{0}{1}".format(prefix, i)] = value
    system = qb.create_ODE_system(
        "\n".join(lines),
        states=dict(states),
        parameters={"m1": 1.0},
        constants={"m{0}".format(j): float(j) for j in range(2, 8)},
        name=name,
        precision=precision,
    )
    return system, dict(states)


# Pollution problem rate constants k2..k25 (k1 is swept).
POLLU_CONSTANTS = {
    "k2": 26.6, "k3": 1.23e4, "k4": 8.6e-4, "k5": 8.2e-4, "k6": 1.5e4,
    "k7": 1.3e-4, "k8": 2.4e4, "k9": 1.65e4, "k10": 9.0e3, "k11": 2.2e-2,
    "k12": 1.2e4, "k13": 1.88, "k14": 1.63e4, "k15": 4.8e6, "k16": 3.5e-4,
    "k17": 1.75e-2, "k18": 1.0e8, "k19": 4.44e11, "k20": 1.24e3, "k21": 2.1,
    "k22": 5.78, "k23": 4.74e-2, "k24": 1.78e3, "k25": 3.12,
}

POLLU_STATES = {"y{0}".format(i): 0.0 for i in range(1, 21)}
POLLU_STATES.update(y2=0.2, y4=0.04, y7=0.1, y8=0.3, y9=0.01, y17=0.007)


def _pollu(problem, precision, name):
    """Verwer's air pollution mechanism; the swept photolysis rate is k1."""
    system = qb.create_ODE_system(
        """
        r1 = k1 * y1
        r2 = k2 * y2 * y4
        r3 = k3 * y5 * y2
        r4 = k4 * y7
        r5 = k5 * y7
        r6 = k6 * y7 * y6
        r7 = k7 * y9
        r8 = k8 * y9 * y6
        r9 = k9 * y11 * y2
        r10 = k10 * y11 * y1
        r11 = k11 * y13
        r12 = k12 * y10 * y2
        r13 = k13 * y14
        r14 = k14 * y1 * y6
        r15 = k15 * y3
        r16 = k16 * y4
        r17 = k17 * y4
        r18 = k18 * y16
        r19 = k19 * y16
        r20 = k20 * y17 * y6
        r21 = k21 * y19
        r22 = k22 * y19
        r23 = k23 * y1 * y4
        r24 = k24 * y19 * y1
        r25 = k25 * y20
        dy1 = -r1 - r10 - r14 - r23 - r24 + r2 + r3 + r9 + r11 + r12 + r22 + r25
        dy2 = -r2 - r3 - r9 - r12 + r1 + r21
        dy3 = -r15 + r1 + r17 + r19 + r22
        dy4 = -r2 - r16 - r17 - r23 + r15
        dy5 = -r3 + 2.0 * r4 + r6 + r7 + r13 + r20
        dy6 = -r6 - r8 - r14 - r20 + r3 + 2.0 * r18
        dy7 = -r4 - r5 - r6 + r13
        dy8 = r4 + r5 + r6 + r7
        dy9 = -r7 - r8
        dy10 = -r12 + r7 + r9
        dy11 = -r9 - r10 + r8 + r11
        dy12 = r9
        dy13 = -r11 + r10
        dy14 = -r13 + r12
        dy15 = r14
        dy16 = -r18 - r19 + r16
        dy17 = -r20
        dy18 = r20
        dy19 = -r21 - r22 - r24 + r23 + r25
        dy20 = -r25 + r24
        """,
        states=dict(POLLU_STATES),
        parameters={"k1": 0.35},
        constants=dict(POLLU_CONSTANTS),
        name=name,
        precision=precision,
    )
    return system, dict(POLLU_STATES)


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
    "lorenz96": tuple("x{0}".format(i) for i in range(1, 41)),
    "pleiades": tuple("{0}{1}".format(prefix, i)
                      for prefix in ("x", "y", "u", "v")
                      for i in range(1, 8)),
    "pollu": tuple("y{0}".format(i) for i in range(1, 21)),
    "ring_modulator": RING_ORDER,
    "ring_modulator_index2": RING_ORDER,
}

# No nand_gate builder: cubie's DSL cannot express the C(y) y' left-hand side
# (linear combinations of derivatives with state-dependent coefficients).
_BUILDERS = {
    "lorenz": _lorenz,
    "lorenz96": _lorenz96,
    "pleiades": _pleiades,
    "pollu": _pollu,
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
