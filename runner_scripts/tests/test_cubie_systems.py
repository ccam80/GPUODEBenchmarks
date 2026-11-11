"""Cubie systems: a built system sweeps the problem's swept parameters and compiles the rest in."""

import os
import sys
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import cubie_systems  # noqa: E402


class FakeSystem:
    """Record the names it is told to sweep; lay them out in `order` when one is given."""

    def __init__(self, order=None):
        self.order = order
        self.swept_parameters = ()

    def set_swept_parameters(self, names):
        self.swept_parameters = tuple(self.order or names)


class BuildSweepTests(unittest.TestCase):
    def build(self, problem, order=None):
        system = FakeSystem(order)
        builder = mock.Mock(return_value=(system, {}))
        with mock.patch.dict(cubie_systems._BUILDERS, {problem: builder}):
            cubie_systems.build_system(problem)
        return system

    def test_a_built_system_sweeps_the_grid_parameter(self):
        self.assertEqual(self.build("lorenz").swept_parameters, cubie_systems.swept_parameters("lorenz"))

    def test_the_fabbri_system_sweeps_both_inputs_in_ensemble_row_order(self):
        self.assertEqual(self.build("fabbri_linder").swept_parameters, cubie_systems.swept_parameters("fabbri_linder"))

    def test_a_system_that_reorders_the_swept_parameters_is_refused(self):
        reordered = tuple(reversed(cubie_systems.swept_parameters("fabbri_linder")))
        with self.assertRaises(ValueError):
            self.build("fabbri_linder", order=reordered)


if __name__ == "__main__":
    unittest.main()
