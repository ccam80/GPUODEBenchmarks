"""errored_pct: the share of trajectories whose final state is not finite."""

import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from wp_common import errored_pct  # noqa: E402


class TestErroredPct(unittest.TestCase):
    def test_all_finite_is_zero(self):
        self.assertEqual(0.0, errored_pct(np.ones((8, 3))))

    def test_one_non_finite_component_errors_the_whole_trajectory(self):
        finals = np.ones((10, 3))
        finals[2, 1] = np.nan
        finals[7, 0] = np.inf
        self.assertEqual(20.0, errored_pct(finals))

    def test_a_flat_vector_counts_per_element(self):
        self.assertEqual(25.0, errored_pct([1.0, np.nan, 2.0, 3.0]))

    def test_empty_input_is_zero(self):
        self.assertEqual(0.0, errored_pct(np.zeros((0, 3))))


if __name__ == "__main__":
    unittest.main()
