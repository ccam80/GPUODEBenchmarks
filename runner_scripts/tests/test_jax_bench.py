"""The jax adapter's stepping plans, step bounds, memory check, finals and the pieces that need no jax; the leg itself runs on a GPU through bench.py."""

import math
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import jax_bench  # noqa: E402
import store  # noqa: E402

NAN = float("nan")


def trial(n=8, **overrides):
    """A trial record: lorenz, fixed tsit5 at dt 2^-10, jax."""
    fields = dict(problem="lorenz", system_params="{}", duration=1.0, precision="float32",
                  parameter="rho", grid_scale="linear", grid_min=0.0, grid_max=21.0, n=n,
                  grid_dtype="float32", algorithm="tsit5", controller="fixed", dt=2.0 ** -10,
                  dt_min=NAN, dt_max=NAN, atol=NAN, rtol=NAN, gains="{}", newton_atol=NAN,
                  newton_rtol=NAN, package="jax")
    fields.update(overrides)
    fields["trial_id"] = store.trial_id(fields)
    return fields


def adaptive(**overrides):
    fields = dict(controller="default", dt=2.0 ** -10, atol=1e-5, rtol=1e-5)
    fields.update(overrides)
    return fields


class Usage:
    def __init__(self, temp, argument, output, alias=0):
        self.temp_size_in_bytes = temp
        self.argument_size_in_bytes = argument
        self.output_size_in_bytes = output
        self.alias_size_in_bytes = alias


class SteppingTests(unittest.TestCase):
    def test_a_fixed_stepping_carries_dt0_and_a_power_of_two_step_bound(self):
        plan = jax_bench.stepping(trial())
        self.assertEqual(plan, {"kind": "fixed", "dt0": 2.0 ** -10, "max_steps": 4096, "newton": None})
        self.assertEqual(jax_bench.fixed_max_steps(1.0, 2.0 ** -10), 4096)
        self.assertEqual(jax_bench.fixed_max_steps(1.0, 2.0 ** -13), 16384)
        self.assertEqual(jax_bench.fixed_max_steps(1.0, 2.0 ** -17), 262144)
        self.assertEqual(jax_bench.fixed_max_steps(60.0, 60.0 * 2.0 ** -10), 4096)
        self.assertEqual(jax_bench.fixed_max_steps(3.0, 0.7), 4096)
        with self.assertRaises(ValueError):
            jax_bench.stepping(trial(dt=NAN))
        with self.assertRaises(ValueError):
            jax_bench.stepping(trial(dt=-1.0))

    def test_an_adaptive_stepping_carries_tolerances_pins_gains_and_the_adaptive_bound(self):
        plan = jax_bench.stepping(trial(**adaptive()))
        self.assertEqual(plan, {"kind": "adaptive", "dt0": 2.0 ** -10, "atol": 1e-5, "rtol": 1e-5,
                                "dtmin": None, "dtmax": None, "gains": {},
                                "max_steps": jax_bench.ADAPTIVE_MAX_STEPS, "newton": None})
        pinned = jax_bench.stepping(trial(**adaptive(dt=NAN, dt_min=1e-7, dt_max=0.5,
                                                     gains='{"pcoeff":0.3,"safety":0.8}')))
        self.assertIsNone(pinned["dt0"])
        self.assertEqual((pinned["dtmin"], pinned["dtmax"]), (1e-7, 0.5))
        self.assertEqual(pinned["gains"], {"pcoeff": 0.3, "safety": 0.8})

    def test_newton_tolerances_reach_the_plan_only_when_the_trial_carries_them(self):
        self.assertIsNone(jax_bench.stepping(trial(algorithm="kvaerno3"))["newton"])
        plan = jax_bench.stepping(trial(algorithm="kvaerno3", newton_atol=1e-6, newton_rtol=1e-7))
        self.assertEqual(plan["newton"], (1e-6, 1e-7))
        plan = jax_bench.stepping(trial(algorithm="kvaerno3", **adaptive(newton_atol=1e-5, newton_rtol=1e-5)))
        self.assertEqual(plan["newton"], (1e-5, 1e-5))

    def test_unknown_controllers_are_refused_and_the_key_follows_the_stepping_fields(self):
        self.assertEqual(jax_bench.CONTROLLERS, ("fixed", "default"))
        self.assertEqual(jax_bench.JaxAdapter.controllers, ("fixed", "default"))
        with self.assertRaises(ValueError):
            jax_bench.stepping(trial(controller="pi", dt=NAN, atol=1e-5, rtol=1e-5))
        self.assertEqual(jax_bench.stepping_key(trial()), jax_bench.stepping_key(trial(n=32)))
        self.assertNotEqual(jax_bench.stepping_key(trial()), jax_bench.stepping_key(trial(dt=2.0 ** -11)))
        self.assertNotEqual(jax_bench.stepping_key(trial(**adaptive())),
                            jax_bench.stepping_key(trial(**adaptive(gains='{"pcoeff":0.3}'))))


class MemoryAndFinalsTests(unittest.TestCase):
    def test_the_memory_check_sums_temporaries_arguments_and_outputs_less_aliases(self):
        limit = 10 * 2 ** 30
        self.assertIsNone(jax_bench.memory_shortfall(None, limit))
        self.assertIsNone(jax_bench.memory_shortfall(Usage(2 ** 30, 2 ** 20, 2 ** 20), limit))
        self.assertEqual(jax_bench.memory_shortfall(Usage(9 * 2 ** 30, 2 ** 30, 2 ** 30), limit),
                         (11 * 2 ** 30, limit))
        self.assertIsNone(jax_bench.memory_shortfall(Usage(9 * 2 ** 30, 2 ** 30, 2 ** 30, alias=2 ** 30), limit))

    def test_finals_carry_diffrax_messages_and_the_duration_of_successful_runs(self):
        ys = np.arange(12, dtype=np.float32).reshape(4, 3)
        ys[2] = np.inf
        messages = ["", "", "The maximum number of solver steps was reached.", ""]
        finals, t_final, retcode = jax_bench.finals_of(ys, messages, 60.0)
        self.assertEqual(finals.shape, (4, 3))
        self.assertEqual(finals.dtype, np.float32)
        self.assertEqual(list(t_final[[0, 1, 3]]), [60.0] * 3)
        self.assertTrue(math.isnan(t_final[2]))
        self.assertEqual(retcode, messages)
        self.assertEqual(store.errored_pct(finals, t_final, retcode, 60.0), 25.0)

    def test_states_follow_the_catalogue_row_and_system_params(self):
        adapter = jax_bench.JaxAdapter("windows_RTX-4070-SUPER", "data")
        self.assertEqual(adapter.states(trial()), 3)
        self.assertEqual(adapter.states(trial(problem="lorenz96", system_params='{"states":8}', parameter="F",
                                              grid_max=16.0)), 8)
        self.assertEqual(adapter.states(trial(problem="pleiades", parameter="m1", grid_min=0.5, grid_max=2.0,
                                              duration=3.0)), 28)
        with self.assertRaises(NotImplementedError):
            adapter.optimize(None, trial(), None)


if __name__ == "__main__":
    unittest.main()
