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
