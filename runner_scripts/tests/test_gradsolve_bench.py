"""The gradsolve adapter's solve plans, step counts, finals, memory errors and the pieces that need no gradsolve; the solves run on a GPU through bench.py."""

import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import gradsolve_bench  # noqa: E402
import runner  # noqa: E402
import store  # noqa: E402
from jax_bench import ADAPTIVE_MAX_STEPS  # noqa: E402

NAN = float("nan")


def trial(n=8, **overrides):
    """A trial record: lorenz, fixed tsit5 at dt 2^-10, gradsolve."""
    fields = dict(problem="lorenz", system_params="{}", duration=1.0, precision="float32",
                  parameter="rho", grid_scale="linear", grid_min=0.0, grid_max=21.0, n=n,
                  grid_dtype="float32", algorithm="tsit5", controller="fixed", dt=2.0 ** -10,
                  dt_min=NAN, dt_max=NAN, atol=NAN, rtol=NAN, gains="{}", newton_atol=NAN,
                  newton_rtol=NAN, package="gradsolve")
    fields.update(overrides)
    fields["trial_id"] = store.trial_id(fields)
    return fields


def adaptive(**overrides):
    fields = dict(controller="default", dt=2.0 ** -10, atol=1e-5, rtol=1e-5)
    fields.update(overrides)
    return fields


class FakeArray:
    def __init__(self, values):
        self.values = values

    def numpy(self):
        return np.asarray(self.values)


class PlanTests(unittest.TestCase):
    def test_a_fixed_trial_runs_the_scan_of_its_algorithm_at_the_steps_that_span_the_duration(self):
        self.assertEqual(gradsolve_bench.plan(trial()),
                         {"kind": "fixed", "module": "fixed_step_tsit5", "steps": 1024})
        rk4 = gradsolve_bench.plan(trial(algorithm="classical-rk4", duration=60.0, dt=60.0 * 2.0 ** -13))
        self.assertEqual((rk4["module"], rk4["steps"]), ("fixed_step_explicit", 8192))
        for bad in (dict(dt=NAN), dict(dt=-1.0), dict(dt=0.3), dict(algorithm="rosenbrock23_sciml")):
            with self.assertRaises(ValueError, msg=bad):
                gradsolve_bench.plan(trial(**bad))

    def test_an_adaptive_trial_runs_a_fused_kernel_at_the_protocol_dt0_and_the_suite_step_bound(self):
        tsit5 = gradsolve_bench.plan(trial(**adaptive()))
        self.assertEqual(tsit5, {"kind": "adaptive", "module": "warp_ode", "stiff": False, "dt0": 2.0 ** -10,
                                 "atol": 1e-5, "rtol": 1e-5, "max_steps": ADAPTIVE_MAX_STEPS})
        rosenbrock = gradsolve_bench.plan(trial(algorithm="rosenbrock23_sciml", **adaptive()))
        self.assertEqual((rosenbrock["module"], rosenbrock["stiff"]), ("warp_rosenbrock", True))
        # An unset dt0 is gradsolve's own: a hundredth of the span.
        self.assertEqual(gradsolve_bench.plan(trial(duration=60.0, **adaptive(dt=NAN)))["dt0"], 0.6)

    def test_pins_gains_and_unknown_algorithms_or_controllers_are_refused(self):
        for bad in (adaptive(dt_min=1e-7), adaptive(dt_max=0.5), adaptive(gains='{"pcoeff":0.3}'),
                    dict(algorithm="classical-rk4", **adaptive()), dict(controller="pi", dt=NAN, atol=1e-5, rtol=1e-5)):
            with self.assertRaises(ValueError, msg=bad):
                gradsolve_bench.plan(trial(**bad))
        self.assertEqual(gradsolve_bench.GradsolveAdapter.controllers, ("fixed", "default"))

    def test_the_plan_key_follows_the_stepping_fields_not_the_ensemble(self):
        self.assertEqual(gradsolve_bench.plan_key(trial()), gradsolve_bench.plan_key(trial(n=32)))
        self.assertNotEqual(gradsolve_bench.plan_key(trial()), gradsolve_bench.plan_key(trial(dt=2.0 ** -11)))
        self.assertNotEqual(gradsolve_bench.plan_key(trial(**adaptive())),
                            gradsolve_bench.plan_key(trial(**adaptive(atol=1e-6, rtol=1e-6))))


class FinalsTests(unittest.TestCase):
    def test_a_nonzero_kernel_status_marks_the_trajectory_failed(self):
        states = np.arange(12, dtype=np.float32).reshape(4, 3)
        finals, t_final, retcode = gradsolve_bench.finals_of(states, np.array([0, 1, 0, 0]), 3.0)
        self.assertEqual(finals.dtype, np.float32)
        self.assertTrue(np.isnan(finals[1]).all())
        self.assertEqual(finals[2].tolist(), [6.0, 7.0, 8.0])
        self.assertEqual(states[1].tolist(), [3.0, 4.0, 5.0])
        self.assertEqual(list(t_final), [3.0] * 4)
        self.assertEqual([bool(code) for code in retcode], [False, True, False, False])
        self.assertEqual(store.errored_pct(finals, t_final, retcode, 3.0), 25.0)

    def test_the_adapter_reads_device_and_host_results_alike(self):
        adapter = gradsolve_bench.GradsolveAdapter("linux_RTX-4070-SUPER", "data")

        class Build:
            states = 3
            duration = 1.0

        device = (FakeArray(np.ones((2, 3), dtype=np.float32)), FakeArray(np.array([0, 2])))
        finals, _, retcode = adapter.finals(Build(), device)
        self.assertEqual(finals.shape, (2, 3))
        self.assertEqual([bool(code) for code in retcode], [False, True])
        host, _, retcode = adapter.finals(Build(), (np.zeros((2, 3)), None))
        self.assertEqual(host.shape, (2, 3))
        self.assertEqual(retcode, ["", ""])

    def test_a_warp_allocation_failure_is_recorded_as_out_of_memory(self):
        error = gradsolve_bench.as_memory_error(RuntimeError("Failed to allocate 1024 bytes on device 'cuda:0'"))
        self.assertIsInstance(error, MemoryError)
        self.assertEqual(runner.classify(error), "oom")
        other = RuntimeError("kernel launch failed")
        self.assertIs(gradsolve_bench.as_memory_error(other), other)

    def test_states_follow_the_catalogue_row_and_system_params(self):
        adapter = gradsolve_bench.GradsolveAdapter("linux_RTX-4070-SUPER", "data")
        self.assertEqual(adapter.states(trial()), 3)
        self.assertEqual(adapter.states(trial(problem="lorenz96", system_params='{"states":8}', parameter="F",
                                              grid_max=16.0)), 8)
        with self.assertRaises(NotImplementedError):
            adapter.optimize(None, trial())
        self.assertEqual(gradsolve_bench.step_count(3.0, 3.0 * 2.0 ** -10), 1024)


if __name__ == "__main__":
    unittest.main()
