"""The scalar right-hand sides: one derivative per state from every builder, argument order, fresh functions per call, the cyclic Lorenz 96 and the pairwise Pleiades forces."""

import math
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import scalar_systems  # noqa: E402
from problems import get_problem  # noqa: E402

TUPLE = "({0},)"


def evaluate(problem, y, p, t=0.1, states=None):
    row = get_problem(problem) if states is None else get_problem(problem).resized(states)
    system = scalar_systems.system(row)
    rhs = scalar_systems.right_hand_side(system, scalar_systems.MATH, TUPLE)
    return np.array(rhs(t, list(y), [p]))


class ScalarSystemTests(unittest.TestCase):
    def test_every_builder_gives_one_derivative_and_initial_value_per_state(self):
        for name in scalar_systems.BUILDERS:
            row = get_problem(name)
            system = scalar_systems.system(row)
            self.assertEqual(len(system.states), row["states"], name)
            self.assertEqual(len(system.y0), row["states"], name)
            rhs = scalar_systems.right_hand_side(system, scalar_systems.MATH, TUPLE)
            values = rhs(0.0, list(system.y0), [float(row["sweep_max"])])
            self.assertEqual(len(values), row["states"], name)
            self.assertTrue(all(math.isfinite(v) for v in values), name)

    def test_the_argument_order_follows_the_caller(self):
        system = scalar_systems.system("ring_modulator")
        state = [0.01 * (i + 1) for i in range(15)]
        tyx = scalar_systems.right_hand_side(system, scalar_systems.MATH, TUPLE)
        ytx = scalar_systems.right_hand_side(system, scalar_systems.MATH, TUPLE, arguments="y, t, p")
        self.assertEqual(tyx(2.0e-4, state, [2.0e-10]), ytx(state, 2.0e-4, [2.0e-10]))

    def test_each_call_compiles_a_new_function(self):
        system = scalar_systems.system("lorenz")
        first = scalar_systems.right_hand_side(system, scalar_systems.MATH, TUPLE)
        second = scalar_systems.right_hand_side(system, scalar_systems.MATH, TUPLE)
        self.assertIsNot(first, second)
        self.assertEqual(first(0.0, [1.0, 2.0, 3.0], [21.0]), second(0.0, [1.0, 2.0, 3.0], [21.0]))

    def test_lorenz96_is_cyclic_at_every_size(self):
        rng = np.random.default_rng(1)
        for states in (4, 8, 33):
            y = rng.normal(size=states)
            base = evaluate("lorenz96", y, 8.0, states=states)
            self.assertEqual(base.shape, (states,))
            # Rotating the state rotates the derivative.
            shifted = evaluate("lorenz96", np.roll(y, 1), 8.0, states=states)
            np.testing.assert_allclose(shifted, np.roll(base, 1), rtol=0, atol=1e-12)
        self.assertEqual(scalar_systems.system(get_problem("lorenz96").resized(5)).y0, (9.0, 8.0, 8.0, 8.0, 8.0))

    def test_pleiades_forces_are_equal_and_opposite(self):
        rng = np.random.default_rng(2)
        y = np.concatenate([rng.normal(scale=3.0, size=14), rng.normal(size=14)])
        m1 = 1.7
        dy = evaluate("pleiades", y, m1)
        np.testing.assert_array_equal(dy[:14], y[14:])
        masses = np.array([m1, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        # Newton's third law: the mass-weighted accelerations sum to zero in each direction.
        self.assertAlmostEqual(float(np.dot(masses, dy[14:21])), 0.0, delta=1e-9)
        self.assertAlmostEqual(float(np.dot(masses, dy[21:28])), 0.0, delta=1e-9)

    def test_a_problem_without_a_scalar_definition_exits(self):
        with self.assertRaises(SystemExit):
            scalar_systems.system("nand_gate")
        with self.assertRaises(ValueError):
            scalar_systems.ScalarSystem(("a", "b"), (0.0,), ("1.0", "2.0"))


if __name__ == "__main__":
    unittest.main()
