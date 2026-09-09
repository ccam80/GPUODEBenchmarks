"""The finals comparison: a 1024-row finals paired by value with a 131072-row golden, errored trajectories left out of the RMS, pairing by value rather than index, NaN with no pairs, the golden lookup by system under any key, and the error of a row without finals."""

import math
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "analyses"))

import grid  # noqa: E402
import store  # noqa: E402
import errors  # noqa: E402

NAN = float("nan")
KEY = "windows_RTX-4070-SUPER"
OTHER_KEY = "linux_RTX-2060-SUPER"
N_GOLDEN = 131072
N_SWEEP = 1024
D = 2.0 ** -10


def spec(**overrides):
    fields = dict(problem="lorenz", system_params={}, duration=1.0, precision="float32",
                  parameter="rho", grid_scale="linear", grid_min=0.0, grid_max=21.0,
                  n=N_SWEEP, grid_dtype="float32", algorithm="kvaerno3", controller="fixed",
                  dt=2.0 ** -6, dt_min=NAN, dt_max=NAN, atol=NAN, rtol=NAN, gains={},
                  newton_atol=1e-6, newton_rtol=1e-6, transfers="none", package="cubie", key=KEY)
    fields.update(overrides)
    return fields


def golden_spec(**overrides):
    fields = spec(precision="float64", n=N_GOLDEN, algorithm="Vern9", controller="default",
                  dt=NAN, atol=1e-13, rtol=1e-13, newton_atol=NAN, newton_rtol=NAN,
                  package="julia_cpu")
    fields.update(overrides)
    return fields


def solution(values):
    """A final state that is exact in float32 for a float32 grid value: [v, 2v, -v]."""
    v = np.asarray(values, dtype=np.float64)
    return np.column_stack([v, 2.0 * v, -v])


class ErrorsCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="errors_test_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.store = store.Store(self.tmp)

    def record(self, fields, finals, t_final=None, retcode=None, **extra):
        """Record a row with its finals; returns the stored row."""
        n = finals.shape[0]
        t_final = np.full(n, fields["duration"]) if t_final is None else t_final
        relative = self.store.record_finals(fields, finals, t_final, retcode)
        pct = store.errored_pct(finals, t_final, retcode or [""] * n, fields["duration"])
        row = dict(fields, states=finals.shape[1], finals=relative, errored_pct=pct, **extra)
        return self.store.record(row)

    def golden(self, key=KEY, perturb=True):
        """The golden: state 1 offset by D on the first 512 trajectories, trajectory 12 flagged."""
        fields = golden_spec(key=key)
        values = grid.grid(fields)
        finals = solution(values)
        if perturb:
            finals[:512, 0] += D
        retcode = [""] * N_GOLDEN
        retcode[12] = "MaxIters"
        return self.record(fields, finals, retcode=retcode)

    def sweep(self, **overrides):
        """A 1024-row float32 finals on the prefix grid; trajectories 0..9 coded, 10 non-finite, 11 short."""
        fields = spec(grid_max=grid.grid_point("linear", 0.0, 21.0, N_GOLDEN, N_SWEEP - 1))
        fields.update(overrides)
        values = grid.grid(fields)
        finals = solution(values).astype(np.float32)
        finals[10, 1] = np.nan
        t_final = np.full(N_SWEEP, 1.0)
        t_final[11] = 0.5
        retcode = [""] * N_SWEEP
        for i in range(10):
            retcode[i] = "Unstable"
        return self.record(fields, finals, t_final, retcode)


class CompareTests(ErrorsCase):
    def test_a_1024_row_finals_pairs_by_value_with_the_131072_row_golden(self):
        golden = self.golden()
        sweep = self.sweep()
        errs = errors.Errors(self.store)
        values, states, errored = errs.finals(sweep)
        np.testing.assert_array_equal(values, grid.grid(golden_spec())[:N_SWEEP])
        self.assertEqual(states.dtype, np.float64)
        self.assertEqual(int(errored.sum()), 12)
        # 1011 unflagged pairs of 3 states; 512 - 13 of them carry the D offset in state 1.
        expected = math.sqrt((512 - 13) * D * D / (1011 * 3))
        self.assertAlmostEqual(errs.compare(sweep, golden), expected, places=15)
        self.assertAlmostEqual(errs.compare(golden, sweep), expected, places=15)
        self.assertAlmostEqual(errs.error(sweep), expected, places=15)
        self.assertEqual(errs.error(golden), 0.0)
        self.assertAlmostEqual(errors.error(sweep, self.store), expected, places=15)

    def test_pairing_is_by_grid_value_not_by_index(self):
        golden = self.golden()
        # A 1024-point grid over the full range shares only its two ends with the golden's grid.
        other = self.sweep(grid_max=21.0)
        errs = errors.Errors(self.store)
        shared, index_o, index_g = np.intersect1d(grid.grid(other), grid.grid(golden_spec()),
                                                  return_indices=True)
        self.assertEqual(shared[0], 0.0)
        self.assertEqual(shared[-1], 21.0)
        self.assertLess(len(shared), 8)
        # The sweep's trajectories 0..9 are coded, so v = 0 drops out; the rest pair at different indices.
        keep = index_o >= 10
        index_a, index_b = errs.paired(other, golden)
        self.assertEqual(index_a.tolist(), index_o[keep].tolist())
        self.assertEqual(index_b.tolist(), index_g[keep].tolist())
        self.assertTrue((index_a != index_b).all())
        offset = int((index_b < 512).sum())
        expected = math.sqrt(offset * D * D / (index_a.shape[0] * 3))
        self.assertAlmostEqual(errs.compare(other, golden), expected, places=15)
        # Two sweeps on the same grid pair every trajectory neither flags.
        twin = self.sweep(dt=2.0 ** -7)
        sweep = self.sweep()
        index_a, index_b = errs.paired(sweep, twin)
        self.assertEqual(index_a.shape[0], N_SWEEP - 12)
        np.testing.assert_array_equal(index_a, index_b)
        self.assertEqual(errs.compare(sweep, twin), 0.0)

    def test_no_pair_and_no_finals(self):
        golden = self.golden()
        disjoint = self.sweep(grid_min=30.0, grid_max=40.0)
        errs = errors.Errors(self.store)
        self.assertTrue(math.isnan(errs.compare(disjoint, golden)))
        untimed = self.store.record(dict(spec(dt=2.0 ** -9), states=3, min_ms=1.5))
        self.assertTrue(math.isnan(errs.error(untimed)))
        with self.assertRaises(ValueError):
            errs.compare(untimed, golden)

    def test_every_pair_flagged_is_nan(self):
        golden = self.golden()
        fields = spec(grid_max=grid.grid_point("linear", 0.0, 21.0, N_GOLDEN, N_SWEEP - 1))
        finals = solution(grid.grid(fields)).astype(np.float32)
        flagged = self.record(fields, finals, retcode=["Unstable"] * N_SWEEP)
        self.assertTrue(math.isnan(errors.compare(flagged, golden, self.store)))

    def test_state_count_mismatch_raises(self):
        golden = self.golden()
        fields = spec()
        finals = np.zeros((N_SWEEP, 2), dtype=np.float32)
        two = self.record(fields, finals)
        with self.assertRaises(ValueError):
            errors.compare(two, golden, self.store)


class GoldenLookupTests(ErrorsCase):
    def test_golden_of_matches_problem_system_params_and_duration_under_any_key(self):
        golden = self.golden(key=OTHER_KEY)
        sweep = self.sweep()
        errs = errors.Errors(self.store)
        found = errs.golden_of(sweep)
        self.assertEqual((found["run_id"], found["key"]), (golden["run_id"], OTHER_KEY))
        self.assertIsNone(errors.golden_of(dict(sweep, problem="lorenz96"), self.store))
        self.assertIsNone(errs.golden_of(dict(sweep, system_params='{"states":4}')))
        self.assertIsNone(errs.golden_of(dict(sweep, duration=2.0)))
        # A float32 julia_cpu finals row and a float64 row without finals are not goldens.
        fields = golden_spec(precision="float32", key=KEY)
        self.record(fields, solution(grid.grid(fields)).astype(np.float32))
        self.store.record(dict(golden_spec(key=KEY, atol=1e-12, rtol=1e-12), states=3))
        errs = errors.Errors(self.store)
        self.assertEqual(errs.golden_of(sweep)["run_id"], golden["run_id"])
        self.assertEqual(len(errs.goldens()), 1)

    def test_more_than_one_golden_raises_naming_the_rows(self):
        first = self.golden(key=KEY)
        second = self.golden(key=OTHER_KEY)
        errs = errors.Errors(self.store)
        with self.assertRaises(ValueError) as caught:
            errs.golden_of(self.sweep())
        for row in (first, second):
            self.assertIn(row["run_id"], str(caught.exception))
            self.assertIn(row["key"], str(caught.exception))


if __name__ == "__main__":
    unittest.main()
