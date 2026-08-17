"""torchdiffeq system definitions: each builder returns (module_factory, u0), where module_factory(p) closes the swept scalar into an nn.Module."""

import torch
import torch.nn as nn

from problems import get_problem


class _LorenzODE(torch.nn.Module):

    def __init__(self, rho=torch.tensor(21.0)):
        super(_LorenzODE, self).__init__()
        self.sigma = nn.Parameter(torch.as_tensor([10.0]))
        self.rho = nn.Parameter(rho)
        self.beta = nn.Parameter(torch.as_tensor([8 / 3]))

    def forward(self, t, u):
        x, y, z = u[0], u[1], u[2]
        du1 = self.sigma[0] * (y - x)
        du2 = x * (self.rho - z) - y
        du3 = x * y - self.beta[0] * z
        return torch.stack([du1, du2, du3])


def _lorenz(problem):
    return _LorenzODE, torch.tensor([1.0, 0.0, 0.0]).cuda()


class _Lorenz96ODE(torch.nn.Module):
    """Cyclic 40-state Lorenz 96; the swept forcing F drives every row."""

    def __init__(self, F=torch.tensor(8.0)):
        super(_Lorenz96ODE, self).__init__()
        self.F = nn.Parameter(torch.as_tensor(F))

    def forward(self, t, u):
        F = self.F if self.F.dim() == 0 else self.F[0]
        return ((torch.roll(u, -1) - torch.roll(u, 2)) * torch.roll(u, 1)
                - u + F)


def _lorenz96(problem):
    # Uniform state 8 with x1 perturbed to 9.
    u0 = torch.full((40,), 8.0)
    u0[0] = 9.0
    return _Lorenz96ODE, u0.cuda()


class _PleiadesODE(torch.nn.Module):
    """Seven-body planar gravitation, u = (x, y, x', y'); the swept m1."""

    def __init__(self, m1=torch.tensor(1.0)):
        super(_PleiadesODE, self).__init__()
        self.m1 = nn.Parameter(torch.as_tensor(m1))

    def forward(self, t, u):
        m1 = self.m1 if self.m1.dim() == 0 else self.m1[0]
        x, y = u[:7], u[7:14]
        masses = torch.cat([m1.reshape(1),
                            torch.arange(2.0, 8.0, device=u.device)])
        eye = torch.eye(7, device=u.device)
        dx = x[None, :] - x[:, None]
        dy = y[None, :] - y[:, None]
        r2 = dx * dx + dy * dy + eye
        inv = (1.0 - eye) / (r2 * torch.sqrt(r2))
        ax = torch.sum(masses[None, :] * dx * inv, dim=1)
        ay = torch.sum(masses[None, :] * dy * inv, dim=1)
        return torch.cat([u[14:28], ax, ay])


def _pleiades(problem):
    u0 = torch.tensor([3.0, 3.0, -1.0, -3.0, 2.0, -2.0, 2.0,
                       3.0, -3.0, 2.0, 0.0, 0.0, -4.0, 4.0,
                       0.0, 0.0, 0.0, 0.0, 0.0, 1.75, -1.5,
                       0.0, 0.0, 0.0, -1.25, 1.0, 0.0, 0.0])
    return _PleiadesODE, u0.cuda()


# Pollution problem rate constants k2..k25 (k1 is swept).
_POLLU_K = (26.6, 1.23e4, 8.6e-4, 8.2e-4, 1.5e4, 1.3e-4, 2.4e4, 1.65e4,
            9.0e3, 2.2e-2, 1.2e4, 1.88, 1.63e4, 4.8e6, 3.5e-4, 1.75e-2,
            1.0e8, 4.44e11, 1.24e3, 2.1, 5.78, 4.74e-2, 1.78e3, 3.12)


class _PolluODE(torch.nn.Module):
    """Verwer's air pollution mechanism; the swept photolysis rate k1."""

    def __init__(self, k1=torch.tensor(0.35)):
        super(_PolluODE, self).__init__()
        self.k1 = nn.Parameter(torch.as_tensor(k1))

    def forward(self, t, u):
        k = _POLLU_K
        k1 = self.k1 if self.k1.dim() == 0 else self.k1[0]
        r1 = k1 * u[0]
        r2 = k[0] * u[1] * u[3]
        r3 = k[1] * u[4] * u[1]
        r4 = k[2] * u[6]
        r5 = k[3] * u[6]
        r6 = k[4] * u[6] * u[5]
        r7 = k[5] * u[8]
        r8 = k[6] * u[8] * u[5]
        r9 = k[7] * u[10] * u[1]
        r10 = k[8] * u[10] * u[0]
        r11 = k[9] * u[12]
        r12 = k[10] * u[9] * u[1]
        r13 = k[11] * u[13]
        r14 = k[12] * u[0] * u[5]
        r15 = k[13] * u[2]
        r16 = k[14] * u[3]
        r17 = k[15] * u[3]
        r18 = k[16] * u[15]
        r19 = k[17] * u[15]
        r20 = k[18] * u[16] * u[5]
        r21 = k[19] * u[18]
        r22 = k[20] * u[18]
        r23 = k[21] * u[0] * u[3]
        r24 = k[22] * u[18] * u[0]
        r25 = k[23] * u[19]
        return torch.stack([
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
    u0 = torch.zeros(20)
    u0[1], u0[3], u0[6] = 0.2, 0.04, 0.1
    u0[7], u0[8], u0[16] = 0.3, 0.01, 0.007
    return _PolluODE, u0.cuda()


# Ring modulator constants (Test Set for IVP Solvers, problem II-3).
_RM = {
    "C": 1.6e-8, "Cp": 1.0e-8, "Lh": 4.45, "Ls1": 0.002, "Ls2": 5.0e-4,
    "Ls3": 5.0e-4, "gamma": 40.67286402e-9, "R": 25000.0, "Rp": 50.0,
    "Rg1": 36.3, "Rg2": 17.3, "Rg3": 17.3, "Ri": 50.0, "Rc": 600.0,
    "delta": 17.7493332, "w1": 6283.185307179586, "w2": 62831.85307179586,
}


class _RingModulatorODE(torch.nn.Module):
    """Stiff 15-state form; the swept capacitance divides rows 3 to 6."""

    def __init__(self, cs=torch.tensor(2.0e-12)):
        super(_RingModulatorODE, self).__init__()
        self.cs = nn.Parameter(cs)

    def forward(self, t, u):
        uin1 = 0.5 * torch.sin(_RM["w1"] * t)
        uin2 = 2.0 * torch.sin(_RM["w2"] * t)
        ud1 = u[2] - u[4] - u[6] - uin2
        ud2 = -u[3] + u[5] - u[6] - uin2
        ud3 = u[3] + u[4] + u[6] + uin2
        ud4 = -u[2] - u[5] + u[6] + uin2
        q1 = _RM["gamma"] * (torch.exp(_RM["delta"] * ud1) - 1.0)
        q2 = _RM["gamma"] * (torch.exp(_RM["delta"] * ud2) - 1.0)
        q3 = _RM["gamma"] * (torch.exp(_RM["delta"] * ud3) - 1.0)
        q4 = _RM["gamma"] * (torch.exp(_RM["delta"] * ud4) - 1.0)
        cs = self.cs if self.cs.dim() == 0 else self.cs[0]
        return torch.stack([
            (u[7] - 0.5 * u[9] + 0.5 * u[10] + u[13] - u[0] / _RM["R"]) / _RM["C"],
            (u[8] - 0.5 * u[11] + 0.5 * u[12] + u[14] - u[1] / _RM["R"]) / _RM["C"],
            (u[9] - q1 + q4) / cs,
            (-u[10] + q2 - q3) / cs,
            (u[11] + q1 - q3) / cs,
            (-u[12] - q2 + q4) / cs,
            (-u[6] / _RM["Rp"] + q1 + q2 - q3 - q4) / _RM["Cp"],
            -u[0] / _RM["Lh"],
            -u[1] / _RM["Lh"],
            (0.5 * u[0] - u[2] - _RM["Rg2"] * u[9]) / _RM["Ls2"],
            (-0.5 * u[0] + u[3] - _RM["Rg3"] * u[10]) / _RM["Ls3"],
            (0.5 * u[1] - u[4] - _RM["Rg2"] * u[11]) / _RM["Ls2"],
            (-0.5 * u[1] + u[5] - _RM["Rg3"] * u[12]) / _RM["Ls3"],
            (-u[0] + uin1 - (_RM["Ri"] + _RM["Rg1"]) * u[13]) / _RM["Ls1"],
            (-u[1] - (_RM["Rc"] + _RM["Rg1"]) * u[14]) / _RM["Ls1"],
        ])


def _ring_modulator(problem):
    return _RingModulatorODE, torch.zeros(15).cuda()


_BUILDERS = {
    "lorenz": _lorenz,
    "lorenz96": _lorenz96,
    "pleiades": _pleiades,
    "pollu": _pollu,
    "ring_modulator": _ring_modulator,
}


def build_problem(problem):
    """Return ``(module_factory, u0)`` for a problem row or name."""
    row = problem if isinstance(problem, dict) else get_problem(problem)
    key = row["problem"]
    if key not in _BUILDERS:
        raise SystemExit(
            "no torchdiffeq definition for problem '{0}'".format(key))
    return _BUILDERS[key](row)
