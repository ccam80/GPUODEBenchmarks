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


# Ring modulator constants (Test Set for IVP Solvers, problem II-3).
_RM_C = 1.6e-8
_RM_CP = 1.0e-8
_RM_LH = 4.45
_RM_LS1 = 0.002
_RM_LS2 = 5.0e-4
_RM_LS3 = 5.0e-4
_RM_GAMMA = 40.67286402e-9
_RM_R = 25000.0
_RM_RP = 50.0
_RM_RG1 = 36.3
_RM_RG2 = 17.3
_RM_RG3 = 17.3
_RM_RI = 50.0
_RM_RC = 600.0
_RM_DELTA = 17.7493332
_RM_W1 = 6283.185307179586
_RM_W2 = 62831.85307179586


class _RingModulator(eqx.Module):
    """Stiff 15-state form; the swept capacitance divides rows 3 to 6."""

    cs: float

    def __call__(self, t, y, args):
        uin1 = 0.5 * jnp.sin(_RM_W1 * t)
        uin2 = 2.0 * jnp.sin(_RM_W2 * t)
        ud1 = y[2] - y[4] - y[6] - uin2
        ud2 = -y[3] + y[5] - y[6] - uin2
        ud3 = y[3] + y[4] + y[6] + uin2
        ud4 = -y[2] - y[5] + y[6] + uin2
        q1 = _RM_GAMMA * (jnp.exp(_RM_DELTA * ud1) - 1.0)
        q2 = _RM_GAMMA * (jnp.exp(_RM_DELTA * ud2) - 1.0)
        q3 = _RM_GAMMA * (jnp.exp(_RM_DELTA * ud3) - 1.0)
        q4 = _RM_GAMMA * (jnp.exp(_RM_DELTA * ud4) - 1.0)
        return jnp.stack([
            (y[7] - 0.5 * y[9] + 0.5 * y[10] + y[13] - y[0] / _RM_R) / _RM_C,
            (y[8] - 0.5 * y[11] + 0.5 * y[12] + y[14] - y[1] / _RM_R) / _RM_C,
            (y[9] - q1 + q4) / self.cs,
            (-y[10] + q2 - q3) / self.cs,
            (y[11] + q1 - q3) / self.cs,
            (-y[12] - q2 + q4) / self.cs,
            (-y[6] / _RM_RP + q1 + q2 - q3 - q4) / _RM_CP,
            -y[0] / _RM_LH,
            -y[1] / _RM_LH,
            (0.5 * y[0] - y[2] - _RM_RG2 * y[9]) / _RM_LS2,
            (-0.5 * y[0] + y[3] - _RM_RG3 * y[10]) / _RM_LS3,
            (0.5 * y[1] - y[4] - _RM_RG2 * y[11]) / _RM_LS2,
            (-0.5 * y[1] + y[5] - _RM_RG3 * y[12]) / _RM_LS3,
            (-y[0] + uin1 - (_RM_RI + _RM_RG1) * y[13]) / _RM_LS1,
            (-y[1] - (_RM_RC + _RM_RG1) * y[14]) / _RM_LS1,
        ])


def _ring_modulator(problem):
    return _RingModulator, jnp.zeros(15)


_BUILDERS = {
    "lorenz": _lorenz,
    "ring_modulator": _ring_modulator,
}


def build_problem(problem):
    """Return ``(vector_field, y0)`` for a problem row or name."""
    row = problem if isinstance(problem, dict) else get_problem(problem)
    key = row["problem"]
    if key not in _BUILDERS:
        raise SystemExit("no diffrax definition for problem '{0}'".format(key))
    return _BUILDERS[key](row)
