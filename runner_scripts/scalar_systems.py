"""Define the benchmark problems as Python source for the packages that compile it.

Write each problem as straight-line assignments over named scalar states, the
time t and the swept parameter p0, and turn it into a function with
right_hand_side(). gradsolve traces the function with jax.numpy and modax
compiles it with math, so both packages use the same definitions.
"""

import math

from problems import as_problem

# Pleiades initial state (Test Set for IVP Solvers): x, y, x', y' per star.
PLEIADES_X0 = (3.0, 3.0, -1.0, -3.0, 2.0, -2.0, 2.0)
PLEIADES_Y0 = (3.0, -3.0, 2.0, 0.0, 0.0, -4.0, 4.0)
PLEIADES_U0 = (0.0, 0.0, 0.0, 0.0, 0.0, 1.75, -1.5)
PLEIADES_V0 = (0.0, 0.0, 0.0, -1.25, 1.0, 0.0, 0.0)

# Pollution problem rate constants k2..k25 (k1 is swept).
POLLU_K = (26.6, 1.23e4, 8.6e-4, 8.2e-4, 1.5e4, 1.3e-4, 2.4e4, 1.65e4,
           9.0e3, 2.2e-2, 1.2e4, 1.88, 1.63e4, 4.8e6, 3.5e-4, 1.75e-2,
           1.0e8, 4.44e11, 1.24e3, 2.1, 5.78, 4.74e-2, 1.78e3, 3.12)

# Ring modulator constants (Test Set for IVP Solvers, problem II-3).
RING = {"C": 1.6e-8, "Cp": 1.0e-8, "Lh": 4.45, "Ls1": 0.002, "Ls2": 5.0e-4, "Ls3": 5.0e-4,
        "gamma": 40.67286402e-9, "R": 25000.0, "Rp": 50.0, "Rg1": 36.3, "Rg2": 17.3, "Rg3": 17.3,
        "Ri": 50.0, "Rc": 600.0, "delta": 17.7493332,
        "w1": 6283.185307179586, "w2": 62831.85307179586}


class ScalarSystem:
    """Hold one problem as source text.

    Give the state names, initial values, auxiliary assignments and one
    derivative expression per state, in state order. Write expressions in terms
    of the state names, the time `t` and the swept parameter `p0`.
    """

    def __init__(self, states, y0, derivatives, auxiliaries=()):
        if len(states) != len(y0) or len(states) != len(derivatives):
            raise ValueError("a system needs one initial value and one derivative per state")
        self.states = tuple(states)
        self.y0 = tuple(float(v) for v in y0)
        self.derivatives = tuple(derivatives)
        self.auxiliaries = tuple(auxiliaries)

    def source(self, name, pack, arguments="t, y, p"):
        """Return the text of `def <name>(<arguments>)`.

        Pass `arguments` naming t, y and p in the order the caller uses. The
        function unpacks y and p[0], evaluates the expressions and returns the
        derivatives formatted by `pack`.
        """
        lines = ["def {0}({1}):".format(name, arguments), "    p0 = p[0]"]
        lines += ["    {0} = y[{1}]".format(state, i) for i, state in enumerate(self.states)]
        lines += ["    " + line for line in self.auxiliaries]
        lines += ["    d{0} = {1}".format(i, expr) for i, expr in enumerate(self.derivatives)]
        lines.append("    return " + pack.format(", ".join("d{0}".format(i) for i in range(len(self.states)))))
        return "\n".join(lines) + "\n"


def _lorenz(row):
    return ScalarSystem(("x", "y_", "z"), (1.0, 0.0, 0.0),
                        ("10.0 * (y_ - x)", "p0 * x - y_ - x * z", "x * y_ - (8.0 / 3.0) * z"))


def _lorenz96(row):
    n = int(row["states"])
    names = ["x{0}".format(i) for i in range(n)]
    # Cyclic Lorenz 96: dx_i = (x_{i+1} - x_{i-2}) x_{i-1} - x_i + F.
    derivatives = ["({0} - {1}) * {2} - {3} + p0".format(names[(i + 1) % n], names[(i - 2) % n],
                                                          names[(i - 1) % n], names[i])
                   for i in range(n)]
    # Uniform state 8 with x1 perturbed to 9.
    return ScalarSystem(names, [9.0] + [8.0] * (n - 1), derivatives)


def _pleiades(row):
    stars = range(1, 8)
    states = ["{0}{1}".format(prefix, i) for prefix in ("x", "w", "u", "v") for i in stars]
    y0 = PLEIADES_X0 + PLEIADES_Y0 + PLEIADES_U0 + PLEIADES_V0
    masses = {1: "p0"}
    masses.update({i: repr(float(i)) for i in range(2, 8)})
    auxiliaries = []
    for i in stars:
        for j in range(i + 1, 8):
            auxiliaries.append("q{0}{1} = (x{0} - x{1}) * (x{0} - x{1}) + (w{0} - w{1}) * (w{0} - w{1})"
                               .format(i, j))
            auxiliaries.append("r{0}{1} = q{0}{1} * sqrt(q{0}{1})".format(i, j))
    accel_x, accel_y = [], []
    for i in stars:
        ax, ay = [], []
        for j in stars:
            if j == i:
                continue
            pair = "r{0}{1}".format(min(i, j), max(i, j))
            ax.append("{0} * (x{1} - x{2}) / {3}".format(masses[j], j, i, pair))
            ay.append("{0} * (w{1} - w{2}) / {3}".format(masses[j], j, i, pair))
        accel_x.append(" + ".join(ax))
        accel_y.append(" + ".join(ay))
    derivatives = (["u{0}".format(i) for i in stars] + ["v{0}".format(i) for i in stars]
                   + accel_x + accel_y)
    return ScalarSystem(states, y0, derivatives, auxiliaries)


def _pollu(row):
    states = ["c{0}".format(i) for i in range(20)]
    y0 = [0.0] * 20
    for index, value in ((1, 0.2), (3, 0.04), (6, 0.1), (7, 0.3), (8, 0.01), (16, 0.007)):
        y0[index] = value
    k = ["p0"] + [repr(value) for value in POLLU_K]
    # (rate, reactant states) of the 25 reactions; k[i] is the rate constant of reaction i + 1.
    reactants = ((0,), (1, 3), (4, 1), (6,), (6,), (6, 5), (8,), (8, 5), (10, 1), (10, 0), (12,),
                 (9, 1), (13,), (0, 5), (2,), (3,), (3,), (15,), (15,), (16, 5), (18,), (18,),
                 (0, 3), (18, 0), (19,))
    auxiliaries = ["r{0} = {1}".format(i + 1, " * ".join([k[i]] + ["c{0}".format(s) for s in species]))
                   for i, species in enumerate(reactants)]
    derivatives = (
        "-r1 - r10 - r14 - r23 - r24 + r2 + r3 + r9 + r11 + r12 + r22 + r25",
        "-r2 - r3 - r9 - r12 + r1 + r21",
        "-r15 + r1 + r17 + r19 + r22",
        "-r2 - r16 - r17 - r23 + r15",
        "-r3 + 2.0 * r4 + r6 + r7 + r13 + r20",
        "-r6 - r8 - r14 - r20 + r3 + 2.0 * r18",
        "-r4 - r5 - r6 + r13",
        "r4 + r5 + r6 + r7",
        "-r7 - r8",
        "-r12 + r7 + r9",
        "-r9 - r10 + r8 + r11",
        "r9",
        "-r11 + r10",
        "-r13 + r12",
        "r14",
        "-r18 - r19 + r16",
        "-r20",
        "r20",
        "-r21 - r22 - r24 + r23 + r25",
        "-r25 + r24",
    )
    return ScalarSystem(states, y0, derivatives, auxiliaries)


def _ring_modulator(row):
    c = {name: repr(value) for name, value in RING.items()}
    states = ["U{0}".format(i) for i in range(1, 8)] + ["I{0}".format(i) for i in range(1, 9)]
    auxiliaries = (
        "uin1 = 0.5 * sin({w1} * t)".format(**c),
        "uin2 = 2.0 * sin({w2} * t)".format(**c),
        "ud1 = U3 - U5 - U7 - uin2",
        "ud2 = -U4 + U6 - U7 - uin2",
        "ud3 = U4 + U5 + U7 + uin2",
        "ud4 = -U3 - U6 + U7 + uin2",
        "q1 = {gamma} * (exp({delta} * ud1) - 1.0)".format(**c),
        "q2 = {gamma} * (exp({delta} * ud2) - 1.0)".format(**c),
        "q3 = {gamma} * (exp({delta} * ud3) - 1.0)".format(**c),
        "q4 = {gamma} * (exp({delta} * ud4) - 1.0)".format(**c),
    )
    # The swept capacitance Cs divides rows 3 to 6.
    derivatives = (
        "(I1 - 0.5 * I3 + 0.5 * I4 + I7 - U1 / {R}) / {C}".format(**c),
        "(I2 - 0.5 * I5 + 0.5 * I6 + I8 - U2 / {R}) / {C}".format(**c),
        "(I3 - q1 + q4) / p0",
        "(-I4 + q2 - q3) / p0",
        "(I5 + q1 - q3) / p0",
        "(-I6 - q2 + q4) / p0",
        "(-U7 / {Rp} + q1 + q2 - q3 - q4) / {Cp}".format(**c),
        "-U1 / {Lh}".format(**c),
        "-U2 / {Lh}".format(**c),
        "(0.5 * U1 - U3 - {Rg2} * I3) / {Ls2}".format(**c),
        "(-0.5 * U1 + U4 - {Rg3} * I4) / {Ls3}".format(**c),
        "(0.5 * U2 - U5 - {Rg2} * I5) / {Ls2}".format(**c),
        "(-0.5 * U2 + U6 - {Rg3} * I6) / {Ls3}".format(**c),
        "(-U1 + uin1 - ({Ri} + {Rg1}) * I7) / {Ls1}".format(**c),
        "(-U2 - ({Rc} + {Rg1}) * I8) / {Ls1}".format(**c),
    )
    return ScalarSystem(states, [0.0] * 15, derivatives, auxiliaries)


BUILDERS = {
    "lorenz": _lorenz,
    "lorenz96": _lorenz96,
    "pleiades": _pleiades,
    "pollu": _pollu,
    "ring_modulator": _ring_modulator,
}

# Bind the functions the expressions call to the math module for modax.
MATH = {"exp": math.exp, "sin": math.sin, "sqrt": math.sqrt}


def system(problem):
    """Return the ScalarSystem of a problem row or name.

    Exit if the problem has no scalar definition.
    """
    row = as_problem(problem)
    if row["problem"] not in BUILDERS:
        raise SystemExit("no scalar definition for problem '{0}'".format(row["problem"]))
    return BUILDERS[row["problem"]](row)


def right_hand_side(scalar_system, namespace, pack, arguments="t, y, p", name="rhs"):
    """Compile a new function `name(<arguments>)` from the system's source.

    Pass the namespace that binds exp, sin and sqrt, and a `pack` format such as
    "({0},)" for a tuple of derivatives.
    """
    scope = dict(namespace)
    exec(compile(scalar_system.source(name, pack, arguments), "<{0}>".format(name), "exec"), scope)
    return scope[name]
