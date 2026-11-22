"""The gradsolve adapter's solve plans, step counts, finals, memory errors, the pieces that need no gradsolve, and the resident Tsit5 launch against gradsolve.solve on Warp's CPU device; the GPU solves run through bench.py."""

import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import gradsolve_bench  # noqa: E402
import grid  # noqa: E402
import runner  # noqa: E402
import store  # noqa: E402

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


class PlanTests(unittest.TestCase):
    def test_a_fixed_trial_runs_the_scan_of_its_algorithm_at_the_steps_that_span_the_duration(self):
        self.assertEqual(gradsolve_bench.plan(trial()),
                         {"kind": "fixed", "module": "fixed_step_tsit5", "steps": 1024})
        rk4 = gradsolve_bench.plan(trial(algorithm="classical-rk4", duration=60.0, dt=60.0 * 2.0 ** -13))
        self.assertEqual((rk4["module"], rk4["steps"]), ("fixed_step_explicit", 8192))
        for bad in (dict(dt=NAN), dict(dt=-1.0), dict(dt=0.3), dict(algorithm="rosenbrock23_sciml")):
            with self.assertRaises(ValueError, msg=bad):
                gradsolve_bench.plan(trial(**bad))

    def test_an_adaptive_trial_declares_its_stiffness_and_tolerances_and_leaves_the_engine_to_gradsolve(self):
        tsit5 = gradsolve_bench.plan(trial(**adaptive()))
        self.assertEqual(tsit5, {"kind": "adaptive", "stiff": False, "atol": 1e-5, "rtol": 1e-5})
        rosenbrock = gradsolve_bench.plan(trial(algorithm="rosenbrock23_sciml", **adaptive()))
        self.assertEqual(rosenbrock, {"kind": "adaptive", "stiff": True, "atol": 1e-5, "rtol": 1e-5})
        # gradsolve chooses its own first step, so the trial's does not reach the plan.
        self.assertEqual(gradsolve_bench.plan(trial(**adaptive(dt=NAN))), tsit5)

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


class FakeBuild(gradsolve_bench.Build):
    """A Build with one fused solve and no gradsolve behind it."""

    def __init__(self):
        self.states = 3
        self.duration = 1.0
        self.checked = set()
        self.fused_solves = []

    def solve_of(self, trial):
        return {"plan": gradsolve_bench.plan(trial), "problem": None}

    def fused_solve(self, solve, values):
        self.fused_solves.append(int(values.shape[0]))
        return np.zeros((values.shape[0], self.states), dtype=np.float32)


class SolveTests(unittest.TestCase):
    def test_a_trajectory_without_a_finite_state_is_marked_failed(self):
        states = np.arange(12, dtype=np.float32).reshape(4, 3)
        states[1, 2] = np.nan
        finals, t_final, retcode = gradsolve_bench.finals_of(states, 3.0)
        self.assertEqual(finals.dtype, np.float32)
        self.assertEqual(list(t_final), [3.0] * 4)
        self.assertEqual(retcode, ["", gradsolve_bench.NO_FINAL_STATE, "", ""])
        self.assertEqual(store.errored_pct(finals, t_final, retcode, 3.0), 25.0)

    def test_a_fused_solve_with_transfers_runs_from_host_arrays(self):
        adapter = gradsolve_bench.GradsolveAdapter("linux_RTX-4070-SUPER", "data")
        build = FakeBuild()
        values = np.linspace(0.0, 21.0, 8, dtype=np.float32)
        result = adapter.solve(build, trial(**adaptive()), values, "both")
        finals, _, retcode = adapter.finals(build, result)
        self.assertEqual((finals.shape, retcode), ((8, 3), [""] * 8))
        # A fused kernel compiles through a solve of two grid points.
        build.fused_solves = []
        build.compile(trial(**adaptive()), values)
        self.assertEqual(build.fused_solves, [2])

    def test_registering_a_field_routes_the_small_non_stiff_cell_to_warp_ode(self):
        from gradsolve import dispatch
        build = gradsolve_bench.Build.__new__(gradsolve_bench.Build)
        build.name, build.states = "suite_lorenz_3", 3
        build.duration, build.f_jax = 1.0, lambda t, y, p: y
        problem = build.fused({"stiff": False})
        self.assertEqual(dispatch.choose_engine(problem.dim, problem.is_stiff, False), "warp_ode")

    def test_a_resident_result_hands_back_host_finals_with_failed_trajectories_nan(self):
        states = np.arange(12, dtype=np.float32).reshape(4, 3)
        result = gradsolve_bench.ResidentResult(states.reshape(-1), np.array([0, 2, 0, 0], dtype=np.int32), 3)
        adapter = gradsolve_bench.GradsolveAdapter("linux_RTX-4070-SUPER", "data")
        finals, _, retcode = adapter.finals(FakeBuild(), result)
        self.assertTrue(np.isnan(finals[1]).all())
        np.testing.assert_array_equal(finals[[0, 2, 3]], states[[0, 2, 3]])
        self.assertEqual(retcode, ["", gradsolve_bench.NO_FINAL_STATE, "", ""])

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


class ResidentTests(unittest.TestCase):
    """The resident Tsit5 launch on Warp's CPU device against gradsolve.solve on the same device."""

    def resident_and_public(self, record):
        import gradsolve
        build = gradsolve_bench.Build(record)
        self.addCleanup(build.close)
        plan = gradsolve_bench.plan(record)
        problem = build.fused(plan)
        y0, params = build.fused_inputs(grid.grid(record))
        resident = gradsolve_bench.ResidentTsit5(problem, plan, record["precision"], device="cpu")
        resident.upload(y0, params)
        public = gradsolve.solve(problem, y0, params, rtol=plan["rtol"], atol=plan["atol"], device="cpu")
        return resident.launch().numpy(), np.asarray(public.y_final)

    def test_a_resident_tsit5_solve_matches_gradsolve_solve(self):
        lorenz96 = dict(problem="lorenz96", system_params='{"states":8}', parameter="F", grid_max=16.0)
        for record in (trial(n=16, **adaptive()), trial(n=16, precision="float64", **adaptive()),
                       trial(n=16, **adaptive(), **lorenz96)):
            got, want = self.resident_and_public(record)
            self.assertEqual(got.dtype, want.dtype)
            np.testing.assert_array_equal(got, want, err_msg=record["problem"] + " " + record["precision"])
            self.assertTrue(np.isfinite(got).all())

    def test_a_stiff_problem_has_no_resident_solve(self):
        record = trial(algorithm="rosenbrock23_sciml", **adaptive())
        build = gradsolve_bench.Build(record)
        self.addCleanup(build.close)
        plan = gradsolve_bench.plan(record)
        with self.assertRaisesRegex(RuntimeError, "warp_rosenbrock"):
            gradsolve_bench.ResidentTsit5(build.fused(plan), plan, "float32", device="cpu")


if __name__ == "__main__":
    unittest.main()
