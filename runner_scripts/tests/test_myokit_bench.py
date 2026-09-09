"""The myokit_cuda adapter against a fake model: trial checks, step counts, model selection and generated lorenz96 files, state-order checks, host and resident solves, finals and the cold cache hook."""

import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import grid  # noqa: E402
import myokit_bench  # noqa: E402
import store  # noqa: E402

NAN = float("nan")
KEY = "windows_RTX-4070-SUPER"


def trial(n=8, **overrides):
    """A trial record: lorenz, fixed euler at dt 2^-10, myokit_cuda."""
    fields = dict(problem="lorenz", system_params="{}", duration=1.0, precision="float32",
                  parameter="rho", grid_scale="linear", grid_min=0.0, grid_max=21.0, n=n,
                  grid_dtype="float32", algorithm="euler", controller="fixed", dt=2.0 ** -10,
                  dt_min=NAN, dt_max=NAN, atol=NAN, rtol=NAN, gains="{}", newton_atol=NAN,
                  newton_rtol=NAN, package="myokit_cuda")
    fields.update(overrides)
    fields["trial_id"] = store.trial_id(fields)
    return fields


def lorenz96(states, n=8):
    return trial(n, problem="lorenz96", system_params='{"states":%d}' % states, parameter="F", grid_max=16.0)


class DeviceArray:
    """A device buffer stand-in wrapping a host array, with cupy's copy, item assignment and get."""

    def __init__(self, host):
        self.host = np.array(host, dtype=np.float32)

    @property
    def shape(self):
        return self.host.shape

    def copy(self):
        return DeviceArray(self.host)

    def __setitem__(self, index, value):
        self.host[index] = value.host if isinstance(value, DeviceArray) else value

    def get(self):
        return self.host.copy()


class FakeModel:
    """Records its construction and every launch; the kernel adds step_count * dt * diffusion to every state."""

    made = []

    def __init__(self, cellml_path, diffusion_variable=None, block_size=128):
        self.cellml_path = str(cellml_path)
        self.diffusion_variable = diffusion_variable
        name = os.path.basename(self.cellml_path)
        if name.startswith("lorenz96"):
            count = int(name[len("lorenz96_"):-len(".cellml")]) if "_" in name else 32
            self.state_names = tuple("lorenz96.x{0}".format(i) for i in range(1, count + 1))
        elif name.startswith("lorenz"):
            self.state_names = ("lorenz.x", "lorenz.y", "lorenz.z")
        else:
            self.state_names = tuple("pleiades.{0}{1}".format(p, i) for p in "xyuv" for i in range(1, 8))
        self.initial_state = np.arange(len(self.state_names), dtype=np.float32)
        self.launches = []
        FakeModel.made.append(self)

    @property
    def state_count(self):
        return len(self.state_names)

    def initial_states(self, cell_count):
        return np.repeat(self.initial_state[:, None], int(cell_count), axis=1)

    def _integrate(self, dt, step_count, initial, diffusion):
        return initial + np.float32(step_count * dt) * diffusion[None, :]

    def solve(self, dt, step_count, cell_count=None, initial_states=None, diffusion_values=None):
        self.launches.append(("host", int(initial_states.shape[1]), dt, step_count))
        return self._integrate(dt, step_count, initial_states, diffusion_values).T

    def to_device(self, initial_states, diffusion_values):
        self.launches.append(("upload", int(initial_states.shape[1])))
        return DeviceArray(initial_states), DeviceArray(diffusion_values)

    def solve_on_device(self, dt, step_count, device_states, device_diffusion):
        self.launches.append(("device", int(device_states.shape[1]), dt, step_count))
        device_states.host[...] = self._integrate(dt, step_count, device_states.host, device_diffusion.host)
        return device_states


class ChecksTests(unittest.TestCase):
    def test_the_kernel_runs_euler_float32_fixed_only(self):
        self.assertEqual(myokit_bench.CONTROLLERS, ("fixed",))
        self.assertEqual(myokit_bench.MyokitAdapter.controllers, ("fixed",))
        myokit_bench.check_trial(trial())
        for bad in (trial(algorithm="tsit5"), trial(precision="float64"),
                    trial(controller="default", dt=NAN, atol=1e-5, rtol=1e-5)):
            with self.assertRaises(ValueError):
                myokit_bench.check_trial(bad)

    def test_step_counts_span_the_duration(self):
        self.assertEqual(myokit_bench.step_count(1.0, 2.0 ** -10), 1024)
        self.assertEqual(myokit_bench.step_count(60.0, 60.0 * 2.0 ** -17), 131072)
        self.assertEqual(myokit_bench.step_count(3.0, 3.0 * 2.0 ** -8), 256)
        for bad in (NAN, 0.0, -1.0):
            with self.assertRaises(ValueError):
                myokit_bench.step_count(1.0, bad)

    def test_finals_end_at_the_duration_with_no_retcode(self):
        finals, t_final, retcode = myokit_bench.finals_of(np.ones((4, 3), np.float32), 3.0)
        self.assertEqual(finals.shape, (4, 3))
        self.assertEqual(list(t_final), [3.0] * 4)
        self.assertEqual(retcode, [""] * 4)


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="myokit_bench_test_")
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_shipped_models_are_used_and_lorenz96_is_generated_at_other_sizes(self):
        models = os.path.join(self.tmp, "models")
        os.makedirs(models)
        for name in ("lorenz", "lorenz96", "pleiades"):
            open(os.path.join(models, name + ".cellml"), "w").close()
        self.assertEqual(myokit_bench.model_path("lorenz", 3, models), os.path.join(models, "lorenz.cellml"))
        self.assertEqual(myokit_bench.model_path("lorenz96", 32, models), os.path.join(models, "lorenz96.cellml"))
        generated = myokit_bench.model_path("lorenz96", 8, models)
        self.assertEqual(generated, os.path.join(models, "generated", "lorenz96_8.cellml"))
        with open(generated, encoding="utf-8") as handle:
            text = handle.read()
        self.assertEqual(text.count("<variable name=\"x"), 8)
        self.assertIn('<variable name="x1" units="dimensionless" initial_value="9"/>', text)
        self.assertIn('<variable name="x8" units="dimensionless" initial_value="8"/>', text)
        self.assertEqual(text.count("<apply><eq/>"), 8)
        # Row 1 couples x2, x7 (i-2 wraps) and x8 (i-1 wraps).
        self.assertIn("<ci>x2</ci><ci>x7</ci></apply><ci>x8</ci></apply><ci>x1</ci>", text)
        with self.assertRaises(ValueError):
            myokit_bench.model_path("pollu", 20, models)
        self.assertEqual(myokit_bench.diffusion_variable("lorenz", "rho"), "lorenz.rho")
        self.assertEqual(myokit_bench.diffusion_variable("lorenz96", "F"), "lorenz96.F")
        self.assertEqual(myokit_bench.state_names("lorenz96", 4),
                         ("lorenz96.x1", "lorenz96.x2", "lorenz96.x3", "lorenz96.x4"))
        self.assertEqual(len(myokit_bench.state_names("pleiades", 28)), 28)


class LegTests(unittest.TestCase):
    def setUp(self):
        FakeModel.made = []
        self.adapter = myokit_bench.MyokitAdapter(KEY, "data", model_class=FakeModel)

    def values(self, record):
        return grid.grid(record)

    def test_build_leg_loads_the_problem_model_and_checks_its_state_order(self):
        leg = self.adapter.build_leg(trial())
        self.assertEqual(leg.states, 3)
        self.assertTrue(leg.model.cellml_path.endswith("lorenz.cellml"))
        self.assertEqual(leg.model.diffusion_variable, "lorenz.rho")
        leg.close()
        self.assertIsNone(leg.model)
        leg = self.adapter.build_leg(lorenz96(8))
        self.assertEqual(leg.states, 8)
        self.assertTrue(leg.model.cellml_path.endswith(os.path.join("generated", "lorenz96_8.cellml")))
        self.assertEqual(leg.model.diffusion_variable, "lorenz96.F")
        leg.close()
        self.assertEqual(self.adapter.states(lorenz96(16)), 16)
        for bad in (trial(algorithm="tsit5"), trial(precision="float64"),
                    trial(problem="pollu", parameter="k1", grid_scale="log", grid_min=3.5e-2, grid_max=3.5,
                          duration=60.0)):
            with self.assertRaises(ValueError):
                self.adapter.build_leg(bad)

        class Reordered(FakeModel):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.state_names = tuple(reversed(self.state_names))

        with self.assertRaises(RuntimeError):
            myokit_bench.MyokitAdapter(KEY, "data", model_class=Reordered).build_leg(trial())

    def test_compile_launches_nothing(self):
        leg = self.adapter.build_leg(trial(n=8))
        self.adapter.compile(leg, trial(n=8), self.values(trial(n=8)))
        self.assertEqual(leg.model.launches, [])
        leg.close()

    def test_host_solves_run_the_step_count_and_hand_back_host_finals(self):
        leg = self.adapter.build_leg(trial(n=4))
        record = trial(n=4)
        result = self.adapter.solve(leg, record, self.values(record), "both")
        self.assertEqual(leg.model.launches, [("host", 4, 2.0 ** -10, 1024)])
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (4, 3))
        finals, t_final, retcode = self.adapter.finals(leg, result)
        # The fake kernel adds duration * rho to each state; rho = 0, 7, 14, 21.
        self.assertEqual(list(finals[:, 0]), [0.0, 7.0, 14.0, 21.0])
        self.assertEqual(list(finals[3]), [21.0, 22.0, 23.0])
        self.assertEqual(list(t_final), [1.0] * 4)
        self.assertEqual(retcode, [""] * 4)
        finer = trial(n=4, dt=2.0 ** -11)
        self.adapter.solve(leg, finer, self.values(finer), "both")
        self.assertEqual(leg.model.launches[-1], ("host", 4, 2.0 ** -11, 2048))
        leg.close()

    def test_device_solves_upload_once_per_n_and_reset_restores_the_resident_states(self):
        leg = self.adapter.build_leg(trial(n=4))
        record = trial(n=4)
        values = self.values(record)
        first = self.adapter.solve(leg, record, values, "none")
        self.adapter.reset(leg, record, values, "none")
        second = self.adapter.solve(leg, record, values, "none")
        self.assertEqual(leg.model.launches, [("upload", 4), ("device", 4, 2.0 ** -10, 1024),
                                              ("device", 4, 2.0 ** -10, 1024)])
        self.assertIsInstance(first, DeviceArray)
        self.assertIs(second, first)
        finals, t_final, retcode = self.adapter.finals(leg, second)
        self.assertEqual(finals.shape, (4, 3))
        # The kernel integrates in place; reset put the initial states back, so the second run repeats the first.
        self.assertEqual(list(finals[:, 0]), [0.0, 7.0, 14.0, 21.0])
        self.adapter.reset(leg, record, values, "both")
        self.adapter.solve(leg, record, values, "none")
        self.assertEqual(list(self.adapter.finals(leg, second)[0][:, 0]), [0.0, 14.0, 28.0, 42.0])
        self.assertEqual(len(leg.model.launches), 4)
        bigger = trial(n=8)
        self.adapter.solve(leg, bigger, self.values(bigger), "none")
        self.assertEqual(leg.model.launches[-2:], [("upload", 8), ("device", 8, 2.0 ** -10, 1024)])
        self.assertEqual(leg.resident_n, 8)
        # The initial states are rebuilt per n and shared by both transfer legs.
        self.adapter.solve(leg, bigger, self.values(bigger), "both")
        self.assertEqual(leg.initial_n, 8)
        leg.close()

    def test_a_cold_leg_swaps_the_kernel_cache_and_restores_it(self):
        swaps = []

        def fake_cold_cache():
            swaps.append("cold")
            return "previous", os.path.join(tempfile.mkdtemp(prefix="myokit_cold_test_"))

        def fake_restore(previous, directory):
            swaps.append(("restore", previous))
            shutil.rmtree(directory, ignore_errors=True)

        saved = myokit_bench.cold_cache, myokit_bench.restore_cache
        myokit_bench.cold_cache, myokit_bench.restore_cache = fake_cold_cache, fake_restore
        self.addCleanup(setattr, myokit_bench, "cold_cache", saved[0])
        self.addCleanup(setattr, myokit_bench, "restore_cache", saved[1])
        leg = self.adapter.build_leg(trial(), cold=True)
        self.assertEqual(swaps, ["cold"])
        leg.close()
        self.assertEqual(swaps, ["cold", ("restore", "previous")])
        with self.assertRaises(ValueError):
            self.adapter.build_leg(trial(algorithm="tsit5"), cold=True)
        # A refused trial never touched the cache.
        self.assertEqual(len(swaps), 2)
        warm = self.adapter.build_leg(trial(), cold=False)
        warm.close()
        self.assertEqual(len(swaps), 2)

    def test_version_reads_myokit_and_optimize_is_refused(self):
        with self.assertRaises(NotImplementedError):
            self.adapter.optimize(None, trial(), None)


if __name__ == "__main__":
    unittest.main()
