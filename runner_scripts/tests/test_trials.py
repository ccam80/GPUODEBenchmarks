"""trials.py: the expansion of every view, deduplication, ordinal order, the tiers, the JSONL round trip and the filters against a scratch store."""

import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import cubie_adapter  # noqa: E402
import store  # noqa: E402
import trials  # noqa: E402
from problems import get_problem  # noqa: E402
from protocol import (N_NE, N_WP, OPTIMIZE_N, OVERLAP_TOL, STATES_GRID,  # noqa: E402
                      STATES_N, TIMING_TOL, TOLS)

LORENZ = get_problem("lorenz")
NAN = float("nan")
SHIPPED = {"step_controller": "i", "integral_gain": 0.2, "safety": 0.9,
           "min_step_shrink": 0.2, "max_step_growth": 10.0}
PI = {"step_controller": "pi", "integral_gain": 0.3, "proportional_gain": 0.1,
      "safety": 0.9, "min_step_shrink": 0.2, "max_step_growth": 10.0}


def solves(items, **match):
    return [t for t in items if t.kind == "solve"
            and all(getattr(t, k) == v for k, v in match.items())]


class TrialCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="trials_test_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        for name, value in (("shipped_controller", lambda row: dict(SHIPPED)),
                            ("pi_controller", lambda row: dict(PI))):
            patcher = mock.patch.object(trials, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def request(self, packages=("cubie",), problems="lorenz", algorithms="tsit5",
                views=("perf",), nlist=(8, 32, 32768), **overrides):
        return trials.Request(packages=list(packages), problems=problems,
                              algorithms=algorithms, views=views, nlist=list(nlist),
                              key="test", store_root=self.tmp, **overrides)


class ExpansionTests(TrialCase):
    def test_perf_sweeps_n_at_the_timing_setting_with_both_legs_and_finals_at_32768(self):
        items = trials.expand(self.request())
        fixed = solves(items, mode="fixed")
        adaptive = solves(items, mode="adaptive")
        self.assertEqual(([t.n for t in fixed], [t.n for t in adaptive]), ([8, 32, 32768], [8, 32, 32768]))
        for trial in fixed:
            self.assertEqual((trial.setting_kind, trial.setting), ("dt", LORENZ.timing_dt))
            self.assertEqual((trial.transfers, trial.grid, trial.tier), (["both", "none"], "sweep", "default"))
            self.assertEqual(trial.leg, "lorenz/tsit5/fixed/n")
            self.assertEqual(trial.finals, 32768 if trial.n == 32768 else 0)
        self.assertEqual([t.ordinal for t in fixed], [0, 1, 2])
        self.assertEqual(adaptive[2].id, "cubie/lorenz/tsit5/adaptive/tol=1e-05/n=32768/s=3/default")
        self.assertEqual(adaptive[0].setting, TIMING_TOL)
        warm = [t for t in items if t.kind == "warm"]
        self.assertEqual(sorted(t.leg for t in warm), ["lorenz/tsit5/adaptive/n", "lorenz/tsit5/fixed/n"])
        self.assertEqual((warm[0].transfers, warm[0].finals, warm[0].n), ([], 0, 8))
        self.assertTrue(warm[0].id.startswith("warm/cubie/lorenz/tsit5/"))
        optimize = [t for t in items if t.kind == "optimize"]
        self.assertEqual(len(optimize), 2)
        for trial in optimize:
            self.assertEqual((trial.n, trial.transfers), (OPTIMIZE_N, []))
            self.assertEqual(trial.setting, trials.timing_setting(LORENZ, trial.mode)[1])

    def test_wp_walks_the_prefix_grid_at_n_wp_with_finals_for_cubie_ne_members(self):
        items = trials.expand(self.request(views=("wp",)))
        fixed = solves(items, mode="fixed")
        adaptive = solves(items, mode="adaptive")
        self.assertEqual([t.setting for t in fixed], LORENZ.dts("tsit5"))
        self.assertEqual([t.setting for t in adaptive], TOLS)
        for trial in fixed + adaptive:
            self.assertEqual((trial.n, trial.transfers, trial.grid), (N_WP, ["none"], "prefix"))
            self.assertTrue(trial.leg.endswith("/setting"))
        self.assertEqual({t.finals for t in fixed}, {0})
        self.assertEqual({t.finals for t in adaptive}, {N_NE})
        # dt descending and tol descending are cost ascending.
        self.assertEqual([t.ordinal for t in fixed], list(range(len(fixed))))
        self.assertEqual([t.ordinal for t in adaptive], list(range(len(adaptive))))
        jax = trials.expand(self.request(packages=("jax",), views=("wp",)))
        self.assertEqual({t.finals for t in solves(jax)}, {0})
        # Per-point families optimize at every setting, the rest once per leg.
        radau = trials.expand(self.request(algorithms="radau_iia_5", views=("wp",)))
        self.assertEqual(len([t for t in radau if t.kind == "optimize"]), len(solves(radau)))

    def test_ne_takes_the_ne_rows_at_n_wp_for_cubie_and_n_ne_for_julia_cpu(self):
        items = trials.expand(self.request(packages=("cubie", "julia_cpu", "jax"),
                                           algorithms="all", views=("ne",)))
        self.assertEqual(solves(items, package="jax"), [])
        cubie_fixed = solves(items, package="cubie", mode="fixed")
        julia_fixed = solves(items, package="julia_cpu", mode="fixed")
        self.assertEqual({t.n for t in cubie_fixed}, {N_WP})
        self.assertEqual({t.n for t in julia_fixed}, {N_NE})
        fixed_names = {t.algorithm for t in cubie_fixed}
        self.assertIn("backwards_euler", fixed_names)
        self.assertNotIn("tsit5", fixed_names)
        self.assertEqual(sorted({t.setting for t in cubie_fixed}), sorted(LORENZ.ne_dts()))
        adaptive = solves(items, package="julia_cpu", mode="adaptive")
        self.assertIn("tsit5", {t.algorithm for t in adaptive})
        self.assertNotIn("backwards_euler", {t.algorithm for t in adaptive})
        for trial in cubie_fixed + julia_fixed + adaptive:
            self.assertEqual((trial.transfers, trial.grid, trial.finals), (["none"], "prefix", N_NE))
        self.assertEqual([t for t in items if t.kind == "warm" and t.package == "julia_cpu"], [])
        self.assertTrue([t for t in items if t.kind == "warm" and t.package == "cubie"])

    def test_states_resizes_lorenz96_over_the_grid_at_n_states_without_warm_trials(self):
        items = trials.expand(self.request(packages=("cubie", "jax"), problems="all",
                                           views=("states",), states_grid=(4, 16, 8)))
        selected = solves(items)
        self.assertEqual({t.problem for t in selected}, {"lorenz96"})
        self.assertEqual({t.n for t in selected}, {STATES_N})
        cubie_fixed = solves(items, package="cubie", mode="fixed")
        self.assertEqual([t.states for t in cubie_fixed], [4, 8, 16])
        self.assertEqual([t.ordinal for t in cubie_fixed], [0, 1, 2])
        self.assertEqual(cubie_fixed[0].leg, "lorenz96/tsit5/fixed/states")
        self.assertEqual([t for t in items if t.kind == "warm"], [])
        self.assertEqual(len([t for t in items if t.kind == "optimize" and t.mode == "fixed"
                              and t.package == "cubie"]), 3)
        default = trials.expand(self.request(views=("states",), problems="all"))
        self.assertEqual({t.states for t in solves(default)}, set(STATES_GRID))

    def test_overlap_sweeps_n_and_the_prefix_grids_with_the_pi_tier_for_cubie(self):
        items = trials.expand(self.request(packages=("julia_gpu", "cubie", "jax"),
                                           algorithms="tsit5,kvaerno3,vern7", views=("overlap",),
                                           nlist=(8, 32), modes=("adaptive",)))
        self.assertEqual(solves(items, package="jax"), [])
        julia = solves(items, package="julia_gpu", algorithm="kvaerno3")
        sweep = [t for t in julia if t.axis == "n"]
        self.assertEqual([(t.n, t.setting) for t in sweep], [(8, OVERLAP_TOL), (32, OVERLAP_TOL)])
        wp = [t for t in julia if t.n == N_WP]
        self.assertEqual([t.setting for t in wp], TOLS)
        ne = [t for t in julia if t.n == N_NE]
        self.assertEqual(([t.setting for t in ne], {t.finals for t in ne}), (TOLS, {N_NE}))
        self.assertEqual({t.tier for t in julia}, {"default"})
        for trial in julia:
            self.assertEqual(trial.transfers, ["both", "none"])
        cubie = solves(items, package="cubie", algorithm="kvaerno3")
        self.assertEqual({t.tier for t in cubie}, {"default", "pi"})
        self.assertEqual([t for t in cubie if t.n == N_NE], [])
        pi = [t for t in cubie if t.tier == "pi"][0]
        self.assertEqual(pi.controller, PI)
        self.assertEqual(pi.id.split("/")[-1], "pi")
        self.assertTrue(solves(items, package="cubie", algorithm="vern7"))
        with mock.patch.object(trials, "pi_controller", lambda row: dict(SHIPPED)):
            same = trials.expand(self.request(packages=("cubie",), views=("overlap",),
                                              nlist=(8,), modes=("adaptive",)))
        self.assertEqual({t.tier for t in solves(same)}, {"default"})

    def test_matched_tier_comes_from_the_julia_cpu_controllers_table(self):
        path = os.path.join(self.tmp, "key=test", "package=julia_cpu", "controllers", "lorenz.csv")
        os.makedirs(os.path.dirname(path))
        with open(path, "w", newline="") as handle:
            handle.write("cubie_alias,controller,beta1,beta2,qmin,qmax,gamma,order\n"
                         "kvaerno3,PIController,0.23333333,0.13333334,0.2,10.0,0.9,3\n"
                         "radau_iia_5,PredictiveController,,,0.2,8.0,0.9,5\n"
                         "tsit5,Unknown,0.1,0.1,0.2,10.0,0.9,5\n")
        items = trials.expand(self.request(algorithms="kvaerno3,radau_iia_5,tsit5,rosenbrock23_sciml",
                                           views=("ne",), modes=("adaptive",)))
        tiers = {name: sorted({t.tier for t in solves(items, algorithm=name)})
                 for name in ("kvaerno3", "radau_iia_5", "tsit5", "rosenbrock23_sciml")}
        self.assertEqual(tiers, {"kvaerno3": ["default", "matched"],
                                 "radau_iia_5": ["default", "matched"],
                                 "tsit5": ["default"], "rosenbrock23_sciml": ["default"]})
        matched = [t for t in solves(items, algorithm="kvaerno3") if t.tier == "matched"]
        expected = cubie_adapter.matched_controller(
            trials.read_controller_constants(self.tmp, "test", "lorenz")["kvaerno3"], 3)[0]
        self.assertEqual(matched[0].controller, expected)
        self.assertEqual(matched[0].controller["step_controller"], "pi")
        self.assertEqual(len(matched), len(TOLS))
        # Matched and default trials interleave by cost, default first at each tolerance.
        ordered = sorted(solves(items, algorithm="kvaerno3"), key=lambda t: t.ordinal)
        self.assertEqual([(t.setting, t.tier) for t in ordered[:2]], [(TOLS[0], "default"), (TOLS[0], "matched")])
        # A matched controller equal to the shipped one adds no tier.
        with mock.patch.object(trials, "shipped_controller", lambda row: expected):
            same = trials.expand(self.request(algorithms="kvaerno3", views=("ne",), modes=("adaptive",)))
        self.assertEqual({t.tier for t in solves(same)}, {"default"})

    def test_views_merge_by_identity_keeping_the_first_leg(self):
        items = trials.expand(self.request(views=("perf", "wp"), nlist=(8, N_WP), modes=("adaptive",)))
        selected = solves(items)
        self.assertEqual(len(selected), 2 + len(TOLS) - 1)
        shared = [t for t in selected if t.n == N_WP and t.setting == TIMING_TOL]
        self.assertEqual(len(shared), 1)
        self.assertEqual((shared[0].transfers, shared[0].grid, shared[0].finals, shared[0].leg),
                         (["both", "none"], "prefix", N_NE, "lorenz/tsit5/adaptive/n"))
        ids = [t.id for t in selected]
        self.assertEqual(len(ids), len(set(ids)))

    def test_unknown_names_and_views_exit(self):
        for kwargs in ({"problems": "lorenz1000"}, {"algorithms": "rk9"}, {"views": ("plots",)},
                       {"packages": ("fortran",)}):
            with self.assertRaises(SystemExit):
                self.request(**kwargs)


class OrderTests(TrialCase):
    def test_cost_order_per_axis(self):
        def trial(axis, **kw):
            fields = dict(package="cubie", problem="lorenz", algorithm="tsit5", mode="fixed",
                          setting_kind="dt", setting=0.5, n=8, states=3, axis=axis)
            fields.update(kw)
            return trials.Trial(**fields)
        by_n = [trial("n", n=32, setting=0.25), trial("n", n=8, setting=0.25), trial("n", n=8, setting=0.5)]
        self.assertEqual([(t.n, t.setting) for t in sorted(by_n, key=trials.cost_key)],
                         [(8, 0.5), (8, 0.25), (32, 0.25)])
        by_setting = [trial("setting", setting=0.125), trial("setting", setting=0.5, tier="pi"),
                      trial("setting", setting=0.5)]
        self.assertEqual([(t.setting, t.tier) for t in sorted(by_setting, key=trials.cost_key)],
                         [(0.5, "default"), (0.5, "pi"), (0.125, "default")])
        by_states = [trial("states", states=16), trial("states", states=4)]
        self.assertEqual([t.states for t in sorted(by_states, key=trials.cost_key)], [4, 16])
        tols = [trial("setting", setting_kind="tol", setting=1e-8), trial("setting", setting_kind="tol", setting=1e-2)]
        self.assertEqual([t.setting for t in sorted(tols, key=trials.cost_key)], [1e-2, 1e-8])


class JsonlTests(TrialCase):
    def test_round_trip_and_line_order(self):
        items = trials.expand(self.request(nlist=(8, 32)))
        path = os.path.join(self.tmp, "cubie.jsonl")
        trials.write_jsonl(path, items)
        with open(path, encoding="utf-8") as handle:
            lines = [json.loads(line) for line in handle]
        self.assertEqual(list(lines[0]), ["id", "kind", "package", "problem", "algorithm", "mode",
                                          "setting_kind", "setting", "n", "states", "tier",
                                          "transfers", "grid", "finals", "controller", "leg",
                                          "ordinal"])
        kinds = [line["kind"] for line in lines]
        self.assertEqual(kinds[:3], ["warm", "optimize", "solve"])
        self.assertEqual(lines[2]["leg"], "lorenz/tsit5/fixed/n")
        self.assertEqual(lines[2]["ordinal"], 0)
        self.assertIsNone(lines[2]["controller"])
        back = trials.read_jsonl(path)
        self.assertEqual([t.to_json() for t in back], lines)
        self.assertEqual({t.leg_key for t in back}, {t.leg_key for t in items})
        with self.assertRaises(ValueError):
            trials.Trial.from_json(dict(lines[2], leg="lorenz/tsit5/fixed/sideways"))


class FilterTests(TrialCase):
    def items(self, **overrides):
        request = self.request(views=("perf", "wp"), nlist=(8, 32), **overrides)
        return trials.expand(request)

    def test_tier_transfers_setting_and_point_filters_take_their_services_along(self):
        items = self.items()
        both = trials.apply_filters(items, transfers=("both",))
        self.assertEqual({t.axis for t in solves(both)}, {"n"})
        self.assertEqual({tuple(t.transfers) for t in solves(both)}, {("both",)})
        self.assertEqual({t.leg for t in both if t.kind == "warm"},
                         {"lorenz/tsit5/fixed/n", "lorenz/tsit5/adaptive/n"})
        settings = trials.apply_filters(self.items(), settings=(TOLS[0], LORENZ.timing_dt * (1 + 1e-10)))
        self.assertEqual({t.setting for t in solves(settings, mode="adaptive")}, {TOLS[0], TIMING_TOL} - {TOLS[0]} | {TOLS[0]}
                         if TOLS[0] == TIMING_TOL else {TOLS[0]})
        self.assertEqual({t.setting for t in solves(settings, mode="fixed")}, {LORENZ.timing_dt})
        point = trials.apply_filters(self.items(), points=("cubie/lorenz/tsit5/adaptive",))
        self.assertEqual({t.mode for t in point}, {"adaptive"})
        exact = trials.apply_filters(self.items(), points=("cubie/lorenz/tsit5/fixed/dt=0.0009765625/n=32/s=3/default",))
        self.assertEqual([t.n for t in solves(exact)], [32])
        self.assertEqual([t.kind for t in exact if t.kind != "solve"], ["warm", "optimize"])
        none = trials.apply_filters(self.items(), tiers=("pi",))
        self.assertEqual(none, [])

    def test_resume_and_no_overwrite_read_the_store(self):
        items = self.items()
        rows = store.Store(self.tmp)
        fixed = solves(items, mode="fixed", axis="n")
        n8, n32 = fixed[0], fixed[1]
        rows.record(dict(n8.identity("test", "both"), min_ms=1.0))
        rows.record(dict(n8.identity("test", "none"), min_ms=0.5))
        rows.record(dict(n32.identity("test", "both"), min_ms=NAN, reason="error: x"))
        rows.record(dict(n32.identity("test", "none"), min_ms=NAN, reason="error: x"))
        adaptive = solves(items, mode="adaptive", axis="n")
        rows.record(dict(adaptive[0].identity("test", "both"), min_ms=2.0))
        index = trials.StoreIndex(self.tmp, "test")
        self.assertEqual(index.status(n8, "both"), "finite")
        self.assertEqual(index.status(n32, "none"), "nan")
        self.assertEqual(index.status(adaptive[0], "none"), "absent")
        resumed = trials.apply_filters(self.items(), resume=True, index=index)
        self.assertNotIn(n8.id, [t.id for t in resumed])
        self.assertNotIn(n32.id, [t.id for t in resumed])
        self.assertIn(adaptive[0].id, [t.id for t in resumed])
        self.assertNotIn("lorenz/tsit5/fixed/n", {t.leg for t in resumed if t.kind == "warm"})
        kept = trials.apply_filters(self.items(), no_overwrite=True, index=index)
        self.assertNotIn(n8.id, [t.id for t in kept])
        self.assertIn(n32.id, [t.id for t in kept])
        self.assertIn("lorenz/tsit5/fixed/n", {t.leg for t in kept if t.kind == "warm"})
        # Ordinals survive the filter, so a resumed leg keeps its cost positions.
        self.assertEqual([t.ordinal for t in solves(kept, mode="fixed", axis="n")], [1])

    def test_counts(self):
        summary = trials.counts(self.items())
        self.assertEqual(list(summary), ["cubie"])
        entry = summary["cubie"]
        # Four legs warm; the erk optimize point at the timing setting is shared by the n and setting legs.
        self.assertEqual((entry["warm"], entry["optimize"]), (4, 2))
        ids = [t.id for t in self.items()]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(entry["legs"]["lorenz/tsit5/fixed/n"], 2)
        self.assertEqual(entry["legs"]["lorenz/tsit5/adaptive/setting"], len(TOLS))
        self.assertEqual(entry["solve"], sum(entry["legs"].values()))


if __name__ == "__main__":
    unittest.main()
