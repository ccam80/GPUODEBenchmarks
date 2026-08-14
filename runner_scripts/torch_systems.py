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


_BUILDERS = {
    "lorenz": _lorenz,
}


def build_problem(problem):
    """Return ``(module_factory, u0)`` for a problem row or name."""
    row = problem if isinstance(problem, dict) else get_problem(problem)
    key = row["problem"]
    if key not in _BUILDERS:
        raise SystemExit(
            "no torchdiffeq definition for problem '{0}'".format(key))
    return _BUILDERS[key](row)
