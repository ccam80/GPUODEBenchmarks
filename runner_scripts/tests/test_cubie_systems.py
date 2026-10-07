"""Cubie systems: a built system sweeps the problem's swept parameters and compiles the rest in."""

import os
import sys
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import cubie_systems  # noqa: E402


class FakeParameters:
    def __init__(self, defaults):
        self.as_float_dict = dict(defaults)


class FakeSystem:
    """Records every update; parameters in declaration order."""

    def __init__(self, defaults):
        self.parameters = FakeParameters(defaults)
        self.updates = []

    def update(self, **updates):
        self.updates.append(updates)


class BuildSweepTests(unittest.TestCase):
    def build(self, problem, defaults):
        system = FakeSystem(defaults)
        builder = mock.Mock(return_value=(system, {}))
        with mock.patch.dict(cubie_systems._BUILDERS, {problem: builder}):
            cubie_systems.build_system(problem)
        return system

    def test_a_built_system_sweeps_the_grid_parameter_and_fixes_the_rest(self):
        system = self.build("lorenz", {"rho": 21.0, "sigma": 10.0, "beta": 8.0 / 3.0})
        self.assertEqual(system.updates, [{
            "swept_parameters": ("rho",),
            "fixed_parameters": (("sigma", 10.0), ("beta", 8.0 / 3.0)),
        }])

    def test_the_fabbri_system_sweeps_both_inputs_in_ensemble_row_order(self):
        import fabbri
        defaults = {"g": 1.0, fabbri.ISO_PARAMETER: 0.0, fabbri.ACH_PARAMETER: 0.0}
        system = self.build("fabbri_linder", defaults)
        update, = system.updates
        self.assertEqual(update["swept_parameters"], cubie_systems.swept_parameters("fabbri_linder"))
        self.assertEqual(update["fixed_parameters"], (("g", 1.0),))


if __name__ == "__main__":
    unittest.main()
