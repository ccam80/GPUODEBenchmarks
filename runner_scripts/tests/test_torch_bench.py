"""The pytorch adapter's version string, odeint options, Tsit5 tableau and finals; the leg itself runs on a GPU through bench.py."""

import json
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import store  # noqa: E402
import torch_bench  # noqa: E402

NAN = float("nan")
COMMIT = "4f4524f719a619c9bd65b722e5f7bf699ff75f62"


def trial(n=8, **overrides):
    """A trial record: lorenz, fixed classical-rk4 at dt 2^-10, pytorch."""
    fields = dict(problem="lorenz", system_params="{}", duration=1.0, precision="float32",
                  parameter="rho", grid_scale="linear", grid_min=0.0, grid_max=21.0, n=n,
                  grid_dtype="float32", algorithm="classical-rk4", controller="fixed", dt=2.0 ** -10,
                  dt_min=NAN, dt_max=NAN, atol=NAN, rtol=NAN, gains="{}", newton_atol=NAN,
                  newton_rtol=NAN, package="pytorch")
    fields.update(overrides)
    fields["trial_id"] = store.trial_id(fields)
    return fields


class HostResult:
    """A (n, saves, states) result with torch's .cpu().numpy() surface."""

    def __init__(self, array):
        self.array = np.asarray(array)

    def cpu(self):
        return self

    def numpy(self):
        return self.array


class VersionTests(unittest.TestCase):
    def test_the_version_is_torch_plus_the_fork_commit(self):
        direct_url = json.dumps({"url": "https://github.com/utkarsh530/torchdiffeq.git",
                                 "vcs_info": {"vcs": "git", "commit_id": COMMIT}})
        self.assertEqual(torch_bench.fork_commit(direct_url), COMMIT[:12])
        self.assertEqual(torch_bench.package_version("2.13.0+cu132", direct_url, "0.2.3"),
                         "2.13.0+cu132+" + COMMIT[:12])
        # A wheel install records no commit, so the fork's own version stands in.
        self.assertIsNone(torch_bench.fork_commit(None))
        self.assertIsNone(torch_bench.fork_commit('{"url": "file:///x"}'))
        self.assertIsNone(torch_bench.fork_commit("not json"))
        self.assertEqual(torch_bench.package_version("2.13.0", None, "0.2.3"), "2.13.0+0.2.3")


class SteppingTests(unittest.TestCase):
    def test_only_fixed_is_a_controller_and_dt_sets_the_step(self):
        self.assertEqual(torch_bench.CONTROLLERS, ("fixed",))
        self.assertEqual(torch_bench.TorchAdapter.controllers, ("fixed",))
        self.assertEqual(torch_bench.options(trial()), {"step_size": 2.0 ** -10})
        self.assertEqual(torch_bench.options(trial(dt=NAN)), {})
        with self.assertRaises(ValueError):
            torch_bench.options(trial(controller="default", dt=NAN, atol=1e-5, rtol=1e-5))

    def test_methods_map_the_catalogue_names(self):
        self.assertEqual([torch_bench.method_of(a) for a in ("euler", "classical-rk4", "tsit5")],
                         ["euler", "rk4", "tsit5"])
        with self.assertRaises(ValueError):
            torch_bench.method_of("kvaerno3")

    def test_the_tsit5_tableau_is_consistent(self):
        self.assertEqual(len(torch_bench.TSIT5_C), 6)
        self.assertEqual([len(row) for row in torch_bench.TSIT5_A], [1, 2, 3, 4, 5, 6])
        for c, row in zip(torch_bench.TSIT5_C, torch_bench.TSIT5_A):
            self.assertAlmostEqual(sum(row), c, places=12)
        self.assertAlmostEqual(sum(torch_bench.TSIT5_B), 1.0, places=12)
        # The last stage reuses the solution weights, the FSAL property.
        self.assertEqual(torch_bench.TSIT5_A[-1], torch_bench.TSIT5_B[:-1])


class FinalsTests(unittest.TestCase):
    def test_finals_are_the_last_save_at_the_duration_with_no_retcode(self):
        result = np.zeros((4, 2, 3), dtype=np.float32)
        result[:, 1, :] = np.arange(12, dtype=np.float32).reshape(4, 3)
        result[3, 1, 0] = np.nan
        adapter = torch_bench.TorchAdapter("windows_RTX-4070-SUPER", "data")
        leg = type("Leg", (), {"duration": 3.0})()
        finals, t_final, retcode = adapter.finals(leg, HostResult(result))
        self.assertEqual(finals.shape, (4, 3))
        self.assertEqual(list(finals[1]), [3.0, 4.0, 5.0])
        self.assertEqual(list(t_final), [3.0] * 4)
        self.assertEqual(retcode, [""] * 4)
        self.assertEqual(store.errored_pct(finals, t_final, retcode, 3.0), 25.0)

    def test_states_follow_the_catalogue_row_and_system_params(self):
        adapter = torch_bench.TorchAdapter("windows_RTX-4070-SUPER", "data")
        self.assertEqual(adapter.states(trial()), 3)
        self.assertEqual(adapter.states(trial(problem="lorenz96", system_params='{"states":16}', parameter="F",
                                              grid_max=16.0)), 16)
        with self.assertRaises(NotImplementedError):
            adapter.optimize(None, trial())


if __name__ == "__main__":
    unittest.main()
