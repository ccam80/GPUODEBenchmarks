#!/usr/bin/env python

"""The pytorch adapter for runner.py: a leg is one torchdiffeq module factory and initial state on the device; a solve runs vmapped over the parameters through host tensors (`both`) or on the resident device parameters (`none`). torchdiffeq under vmap is fixed-grid only, so `fixed` is the one controller a trial may name and there is nothing to warm."""

import importlib.metadata
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import runner  # noqa: E402
from problems import as_problem  # noqa: E402

CONTROLLERS = ("fixed",)
PRECISIONS = ("float32", "float64")
# Canonical algorithm name -> torchdiffeq method string; tsit5 is registered by tsit5_solver().
METHODS = {"euler": "euler", "classical-rk4": "rk4", "tsit5": "tsit5"}
COMMIT_CHARS = 12

# Fixed-grid Tsit5 from the Tsitouras 5(4) coefficients.
TSIT5_C = (0.161, 0.327, 0.9, 0.9800255409045097, 1.0, 1.0)
TSIT5_A = (
    (0.161,),
    (-0.008480655492356989, 0.335480655492357),
    (2.8971530571054935, -6.359448489975075, 4.3622954328695815),
    (5.325864828439257, -11.748883564062828, 7.4955393428898365, -0.09249506636175525),
    (5.86145544294642, -12.92096931784711, 8.159367898576159, -0.071584973281401,
     -0.028269050394068383),
    (0.09646076681806523, 0.01, 0.4798896504144996, 1.379008574103742, -3.290069515436081,
     2.324710524099774),
)
TSIT5_B = (0.09646076681806523, 0.01, 0.4798896504144996, 1.379008574103742, -3.290069515436081,
           2.324710524099774, 0.0)


def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def problem_row(trial):
    """The catalogue row of a trial, resized to its system_params states."""
    row = as_problem(trial["problem"])
    params = json.loads(trial["system_params"]) if trial["system_params"] else {}
    if "states" in params:
        row = row.resized(params["states"])
    return row


def fork_commit(direct_url):
    """The first COMMIT_CHARS of the commit a VCS install records in its direct_url.json text; None when the text names no commit."""
    if not direct_url:
        return None
    try:
        commit = json.loads(direct_url).get("vcs_info", {}).get("commit_id", "")
    except ValueError:
        return None
    return commit[:COMMIT_CHARS] or None


def package_version(torch_version, direct_url, fork_version):
    """torch's version, '+', and the torchdiffeq fork commit, or the fork's own version when it was not installed from git."""
    return torch_version + "+" + (fork_commit(direct_url) or fork_version)


def options(trial):
    """The odeint options of a trial's stepping: the step when dt is finite; an empty dict leaves torchdiffeq stepping the output grid."""
    if trial["controller"] != "fixed":
        raise ValueError("unknown controller " + trial["controller"])
    if _finite(trial["dt"]):
        return {"step_size": float(trial["dt"])}
    return {}


def method_of(algorithm):
    if algorithm not in METHODS:
        raise ValueError("no torchdiffeq method for {0}".format(algorithm))
    return METHODS[algorithm]


def finals_of(host_result, duration):
    """(finals, t_final, retcode) from a host (n, saves, states) result: the last save of every trajectory at the duration; torchdiffeq's fixed grid reports no failure code."""
    finals = np.asarray(host_result)[:, -1, :]
    n = finals.shape[0]
    return finals, np.full(n, float(duration)), [""] * n


def tsit5_solver():
    """torchdiffeq's fixed-grid solver class stepping with the Tsit5 tableau."""
    from torchdiffeq._impl.misc import Perturb
    from torchdiffeq._impl.solvers import FixedGridODESolver

    class Tsit5Fixed(FixedGridODESolver):
        order = 5

        def _step_func(self, func, t0, dt, t1, y0):
            f0 = func(t0, y0, perturb=Perturb.NEXT if self.perturb else Perturb.NONE)
            k = [f0]
            for ci, ai in zip(TSIT5_C, TSIT5_A):
                yi = y0
                for aij, kj in zip(ai, k):
                    yi = yi + dt * aij * kj
                k.append(func(t0 + ci * dt, yi))
            dy = None
            for bi, ki in zip(TSIT5_B, k):
                term = dt * bi * ki
                dy = term if dy is None else dy + term
            return dy, f0

    return Tsit5Fixed


def register_solvers():
    """Add tsit5 to torchdiffeq's solver table."""
    from torchdiffeq._impl.odeint import SOLVERS
    SOLVERS["tsit5"] = tsit5_solver()


# ---------------------------------------------------------------------- leg

class Leg:
    """One module factory and initial state on the device, in the trial's precision; the device parameters of the last upload."""

    def __init__(self, trial, cold=False):
        import torch
        from torch_systems import build_problem
        self.row = problem_row(trial)
        if trial["precision"] not in PRECISIONS:
            raise ValueError("precision '{0}' is not float32 or float64".format(trial["precision"]))
        self.duration = float(trial["duration"])
        self.method = method_of(trial["algorithm"])
        # The problem's tensors take the default dtype, so the precision is set for the leg's life.
        self.saved_dtype = torch.get_default_dtype()
        torch.set_default_dtype(getattr(torch, trial["precision"]))
        try:
            self.module_factory, self.u0 = build_problem(self.row)
            self.states = int(self.u0.numel())
            # Endpoints only: the benchmark scores the final state.
            self.t = torch.linspace(0.0, self.duration, 2).cuda()
        except BaseException:
            self.close()
            raise
        self.resident_n = None
        self.resident = None

    def solve_of(self, trial):
        """The per-trajectory solve of a trial's stepping."""
        import torch
        from torchdiffeq import odeint
        method, opts = self.method, options(trial)

        def solve(p):
            with torch.no_grad():
                return odeint(self.module_factory(p), self.u0, self.t, method=method, options=opts)

        return solve

    def _vmapped(self, trial, parameters):
        import torch
        try:
            out = torch.vmap(self.solve_of(trial))(parameters)
            torch.cuda.synchronize()
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise
        return out

    def host_solve(self, trial, values):
        """One solve through host tensors: .cuda() is the h2d, .cpu() the d2h."""
        import torch
        parameters = torch.from_numpy(np.ascontiguousarray(values)).cuda()
        out = self._vmapped(trial, parameters).cpu()
        torch.cuda.synchronize()
        return out

    def device_solve(self, trial, values):
        """One solve on the resident parameters, uploaded when they are not this grid's; the result stays on the device."""
        import torch
        n = int(values.shape[0])
        if self.resident_n != n:
            self.resident = None
            self.resident = torch.from_numpy(np.ascontiguousarray(values)).cuda()
            self.resident_n = n
        return self._vmapped(trial, self.resident)

    def close(self):
        import torch
        self.resident = None
        torch.set_default_dtype(self.saved_dtype)
        torch.cuda.empty_cache()


class TorchAdapter:
    """The runner adapter of the pytorch package on one machine."""

    controllers = CONTROLLERS

    def __init__(self, key, root):
        self.key, self.root = key, root

    def version(self):
        import torch
        distribution = importlib.metadata.distribution("torchdiffeq")
        return package_version(torch.__version__, distribution.read_text("direct_url.json"),
                              distribution.version)

    def states(self, trial):
        return int(problem_row(trial)["states"])

    def build_leg(self, trial, cold=False):
        return Leg(trial, cold)

    def compile(self, leg, trial, values):
        """torchdiffeq runs eagerly; there is nothing to warm."""

    def optimize(self, leg, trial, values):
        raise NotImplementedError("pytorch has no launch geometry to optimize")

    def solve(self, leg, trial, values, transfers):
        if transfers == "both":
            return leg.host_solve(trial, values)
        return leg.device_solve(trial, values)

    def finals(self, leg, result):
        return finals_of(result.cpu().numpy(), leg.duration)


def run(argv):
    """Entry point of the pytorch suite: a CUDA torch and the tsit5 solver, then the trial file through runner.main."""
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("torch sees no CUDA device; this is a GPU benchmark, so nothing is recorded")
    print("torch {0} on {1}".format(torch.__version__, torch.cuda.get_device_name(0)), flush=True)
    register_solvers()
    return runner.main(argv, lambda key, root: TorchAdapter(key, root))
