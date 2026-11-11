"""The modax adapter's trial checks, solve options, finals and the pieces that need no modax; the solves run on a GPU through bench.py."""

import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import modax_bench  # noqa: E402
import store  # noqa: E402

NAN = float("nan")


def trial(n=8, **overrides):
    """A trial record: lorenz, tsit5 under the default controller at tolerance 1e-5, float64, modax."""
    fields = dict(problem="lorenz", system_params="{}", duration=1.0, precision="float64",
                  parameter="rho", grid_scale="linear", grid_min=0.0, grid_max=21.0, n=n,
                  grid_dtype="float32", algorithm="tsit5", controller="default", dt=2.0 ** -10,
                  dt_min=NAN, dt_max=NAN, atol=1e-5, rtol=1e-5, gains="{}", newton_atol=NAN,
                  newton_rtol=NAN, package="modax")
    fields.update(overrides)
    fields["trial_id"] = store.trial_id(fields)
    return fields


class TrialTests(unittest.TestCase):
    def test_float32_fixed_steps_and_unknown_algorithms_are_refused(self):
        modax_bench.check_trial(trial())
        modax_bench.check_trial(trial(algorithm="rodas5p"))
        modax_bench.check_trial(trial(algorithm="rodas5p_lu64"))
        for bad in (dict(precision="float32"), dict(controller="fixed", atol=NAN, rtol=NAN),
                    dict(algorithm="kvaerno5")):
            with self.assertRaises(ValueError, msg=bad):
                modax_bench.check_trial(trial(**bad))
        self.assertEqual(modax_bench.ModaxAdapter.controllers, ("default",))

    def test_the_solve_options_carry_tolerances_and_gains_and_leave_the_steps_to_modax(self):
        self.assertEqual(modax_bench.solve_options(trial()), {"rtol": 1e-5, "atol": 1e-5})
        # The Rodas5P algorithms differ only in the precision of the LU factorisation.
        self.assertEqual(modax_bench.solve_options(trial(algorithm="rodas5p"))["lu_precision"], "fp32")
        self.assertEqual(modax_bench.solve_options(trial(algorithm="rodas5p_lu64"))["lu_precision"], "fp64")
        self.assertEqual(modax_bench.SOLVERS["rodas5p"][0], modax_bench.SOLVERS["rodas5p_lu64"][0])
        self.assertEqual(modax_bench.solve_options(trial(dt=NAN)), modax_bench.solve_options(trial()))
        gained = modax_bench.solve_options(trial(gains='{"icoeff":0.7,"pcoeff":0.4}'))
        self.assertEqual((gained["pcoeff"], gained["icoeff"]), (0.4, 0.7))
        for bad in (dict(dt_min=1e-7), dict(dt_max=0.5), dict(gains='{"safety":0.9}')):
            with self.assertRaises(ValueError, msg=bad):
                modax_bench.solve_options(trial(**bad))

    def test_the_step_bound_is_the_solve_functions_own_default(self):
        def solve(ode_fn, y0, t_span, params, *, rtol=1e-8, max_steps=321):
            return None

        self.assertEqual(modax_bench.step_bound(solve), 321)

    def test_the_stepping_key_follows_the_stepping_fields_not_the_ensemble(self):
        self.assertEqual(modax_bench.stepping_key(trial()), modax_bench.stepping_key(trial(n=32)))
        self.assertNotEqual(modax_bench.stepping_key(trial()), modax_bench.stepping_key(trial(atol=1e-6)))


class FinalsTests(unittest.TestCase):
    def test_a_trajectory_that_ran_to_the_step_bound_is_marked_failed(self):
        states = np.arange(12, dtype=np.float64).reshape(4, 3)
        finals, t_final, retcode = modax_bench.finals_of(states, np.array([10, 100, 99, 0]), 100, 60.0)
        self.assertTrue(np.isnan(finals[1]).all())
        self.assertEqual(finals[2].tolist(), [6.0, 7.0, 8.0])
        self.assertEqual(states[1].tolist(), [3.0, 4.0, 5.0])
        self.assertEqual(list(t_final), [60.0] * 4)
        self.assertEqual(retcode, ["", modax_bench.MAX_STEPS_RETCODE, "", ""])
        self.assertEqual(store.errored_pct(finals, t_final, retcode, 60.0), 25.0)

    def test_states_follow_the_catalogue_row_and_system_params(self):
        adapter = modax_bench.ModaxAdapter("linux_RTX-4070-SUPER", "data")
        self.assertEqual(adapter.states(trial()), 3)
        self.assertEqual(adapter.states(trial(problem="lorenz96", system_params='{"states":8}', parameter="F",
                                              grid_max=16.0)), 8)
        with self.assertRaises(NotImplementedError):
            adapter.optimize(None, trial())


if __name__ == "__main__":
    unittest.main()
