"""Cubie system definitions, one builder per problem, shared by every cubie suite."""

import numpy as np
import cubie as qb

from problems import get_problem


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


_BUILDERS = {
    "lorenz": _lorenz,
}


def build_system(problem, precision=np.float32, name_suffix=""):
    """Return ``(system, initial_values)`` for a problem row or name.

    ``name_suffix`` separates one caller's generated-code cache from
    another's for the same equations."""
    row = problem if isinstance(problem, dict) else get_problem(problem)
    key = row["problem"]
    if key not in _BUILDERS:
        raise SystemExit("no cubie definition for problem '{0}'".format(key))
    return _BUILDERS[key](row, precision, row["display"] + name_suffix)


def sweep_parameters(problem, n, precision=np.float32):
    """The ensemble parameter dict: the swept scalar over the problem range."""
    row = problem if isinstance(problem, dict) else get_problem(problem)
    return {row["sweep_parameter"]: row.sweep(n, dtype=precision)}
