"""Diffrax system definitions: each builder returns (vector_field, y0), where vector_field(p) closes the swept scalar into f(t, y, args)."""

import equinox as eqx
import jax.numpy as jnp

from problems import get_problem


class _Lorenz(eqx.Module):
    rho: float

    def __call__(self, t, y, args):
        f0 = 10.0 * (y[1] - y[0])
        f1 = self.rho * y[0] - y[1] - y[0] * y[2]
        f2 = y[0] * y[1] - (8 / 3) * y[2]
        return jnp.stack([f0, f1, f2])


def _lorenz(problem):
    return _Lorenz, jnp.array([1.0, 0.0, 0.0])


_BUILDERS = {
    "lorenz": _lorenz,
}


def build_problem(problem):
    """Return ``(vector_field, y0)`` for a problem row or name."""
    row = problem if isinstance(problem, dict) else get_problem(problem)
    key = row["problem"]
    if key not in _BUILDERS:
        raise SystemExit("no diffrax definition for problem '{0}'".format(key))
    return _BUILDERS[key](row)
