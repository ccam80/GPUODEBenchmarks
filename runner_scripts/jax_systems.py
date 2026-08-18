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


class _Lorenz96(eqx.Module):
    """Cyclic 40-state Lorenz 96; the swept forcing F drives every row."""

    F: float

    def __call__(self, t, y, args):
        return (jnp.roll(y, -1) - jnp.roll(y, 2)) * jnp.roll(y, 1) - y + self.F


def _lorenz96(problem):
    # Uniform state 8 with x1 perturbed to 9.
    y0 = jnp.full(40, 8.0).at[0].set(9.0)
    return _Lorenz96, y0


class _Pleiades(eqx.Module):
    """Seven-body planar gravitation, y = (x, y, x', y'); the swept m1."""

    m1: float

    def __call__(self, t, y, args):
        x, yy = y[:7], y[7:14]
        masses = jnp.concatenate(
            [jnp.reshape(jnp.asarray(self.m1), (1,)), jnp.arange(2.0, 8.0)])
        dx = x[None, :] - x[:, None]
        dy = yy[None, :] - yy[:, None]
        r2 = dx * dx + dy * dy + jnp.eye(7)
        inv = (1.0 - jnp.eye(7)) / (r2 * jnp.sqrt(r2))
        ax = jnp.sum(masses[None, :] * dx * inv, axis=1)
        ay = jnp.sum(masses[None, :] * dy * inv, axis=1)
        return jnp.concatenate([y[14:28], ax, ay])


def _pleiades(problem):
    y0 = jnp.array([3.0, 3.0, -1.0, -3.0, 2.0, -2.0, 2.0,
                    3.0, -3.0, 2.0, 0.0, 0.0, -4.0, 4.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 1.75, -1.5,
                    0.0, 0.0, 0.0, -1.25, 1.0, 0.0, 0.0])
    return _Pleiades, y0


# Pollution problem rate constants k2..k25 (k1 is swept).
_POLLU_K = (26.6, 1.23e4, 8.6e-4, 8.2e-4, 1.5e4, 1.3e-4, 2.4e4, 1.65e4,
            9.0e3, 2.2e-2, 1.2e4, 1.88, 1.63e4, 4.8e6, 3.5e-4, 1.75e-2,
            1.0e8, 4.44e11, 1.24e3, 2.1, 5.78, 4.74e-2, 1.78e3, 3.12)


class _Pollu(eqx.Module):
    """Verwer's air pollution mechanism; the swept photolysis rate k1."""

    k1: float

    def __call__(self, t, y, args):
        k = _POLLU_K
        r1 = self.k1 * y[0]
        r2 = k[0] * y[1] * y[3]
        r3 = k[1] * y[4] * y[1]
        r4 = k[2] * y[6]
        r5 = k[3] * y[6]
        r6 = k[4] * y[6] * y[5]
        r7 = k[5] * y[8]
        r8 = k[6] * y[8] * y[5]
        r9 = k[7] * y[10] * y[1]
        r10 = k[8] * y[10] * y[0]
        r11 = k[9] * y[12]
        r12 = k[10] * y[9] * y[1]
        r13 = k[11] * y[13]
        r14 = k[12] * y[0] * y[5]
        r15 = k[13] * y[2]
        r16 = k[14] * y[3]
        r17 = k[15] * y[3]
        r18 = k[16] * y[15]
        r19 = k[17] * y[15]
        r20 = k[18] * y[16] * y[5]
        r21 = k[19] * y[18]
        r22 = k[20] * y[18]
        r23 = k[21] * y[0] * y[3]
        r24 = k[22] * y[18] * y[0]
        r25 = k[23] * y[19]
        return jnp.stack([
            -r1 - r10 - r14 - r23 - r24 + r2 + r3 + r9 + r11 + r12 + r22 + r25,
            -r2 - r3 - r9 - r12 + r1 + r21,
            -r15 + r1 + r17 + r19 + r22,
            -r2 - r16 - r17 - r23 + r15,
            -r3 + 2.0 * r4 + r6 + r7 + r13 + r20,
            -r6 - r8 - r14 - r20 + r3 + 2.0 * r18,
            -r4 - r5 - r6 + r13,
            r4 + r5 + r6 + r7,
            -r7 - r8,
            -r12 + r7 + r9,
            -r9 - r10 + r8 + r11,
            r9,
            -r11 + r10,
            -r13 + r12,
            r14,
            -r18 - r19 + r16,
            -r20,
            r20,
            -r21 - r22 - r24 + r23 + r25,
            -r25 + r24,
        ])


def _pollu(problem):
    y0 = jnp.zeros(20)
    y0 = y0.at[1].set(0.2).at[3].set(0.04).at[6].set(0.1)
    y0 = y0.at[7].set(0.3).at[8].set(0.01).at[16].set(0.007)
    return _Pollu, y0


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
    "lorenz96": _lorenz96,
    "pleiades": _pleiades,
    "pollu": _pollu,
    "ring_modulator": _ring_modulator,
}


def build_problem(problem):
    """Return ``(vector_field, y0)`` for a problem row or name."""
    row = problem if isinstance(problem, dict) else get_problem(problem)
    key = row["problem"]
    if key not in _BUILDERS:
        raise SystemExit("no diffrax definition for problem '{0}'".format(key))
    return _BUILDERS[key](row)
