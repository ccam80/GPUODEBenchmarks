import csv
import sys
import tempfile
import unittest
from pathlib import Path

SUITE = Path(__file__).resolve().parents[1]
ROOT = SUITE.parents[1]
sys.path.insert(0, str(SUITE))

import common  # noqa: E402
import analyze  # noqa: E402


class ProtocolTests(unittest.TestCase):
    def test_exact_overlap_inventory(self):
        rows = common.algorithms()
        self.assertEqual([r["cubie_alias"] for r in rows], [
            "tsit5", "vern7", "rosenbrock23_sciml", "kvaerno3", "kvaerno5"])
        self.assertEqual([r["order"] for r in rows], [5, 7, 2, 3, 5])

    def test_full_diffeqgpu_inventory(self):
        with (SUITE / "diffeqgpu_ode_inventory.csv").open(newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        self.assertEqual(reader.fieldnames, [
            "julia_algorithm", "order", "family", "cubie_alias", "status"])
        self.assertEqual(len(rows), 8)
        self.assertEqual(sum(r["status"] == "overlap" for r in rows), 5)
        self.assertEqual(sum(r["status"] == "no_overlap" for r in rows), 3)

    def test_scaling_grid(self):
        self.assertEqual(common.performance_ns(512), [8, 32, 128, 512])
        self.assertEqual(common.performance_ns(7), [])

    def test_from_n_continues_the_grid(self):
        self.assertEqual(common.performance_ns(512, 128), [128, 512])
        # Not a grid value: continue at the first point at or above it.
        self.assertEqual(common.performance_ns(512, 100), [128, 512])
        self.assertEqual(common.performance_ns(512, 0), [8, 32, 128, 512])
        self.assertEqual(
            common.profile_protocol("full", 512, 128)["performance_ns"],
            [128, 512])

    def test_smoke_keeps_every_metric_family(self):
        protocol = common.profile_protocol("smoke", 10_000)
        self.assertTrue(protocol["performance_ns"])
        self.assertTrue(protocol["ne_dts"] and protocol["ne_tols"])
        self.assertTrue(protocol["wp_dts"] and protocol["wp_tols"])
        self.assertLess(protocol["wp_n"], common.N_WP)

    def test_pi_controller_constants(self):
        settings = common.pi_controller(5)
        self.assertEqual(settings["step_controller"], "pi")
        self.assertAlmostEqual(settings["min_gain"], 0.2)
        self.assertAlmostEqual(settings["max_gain"], 10.0)
        self.assertAlmostEqual(settings["safety"], 0.9)


class PruneTests(unittest.TestCase):
    FIELDS = ["framework", "phase", "n", "time_ms"]

    def rows(self):
        return [
            {"framework": "cubie", "phase": "performance", "n": "8", "time_ms": "1"},
            {"framework": "cubie", "phase": "performance", "n": "2048", "time_ms": "2"},
            {"framework": "cubie", "phase": "numerical", "n": "1024", "time_ms": "3"},
            {"framework": "cubie", "phase": "work_precision", "n": "32768", "time_ms": "4"},
        ]

    def prune(self, phases, from_n=0):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cubie_timings.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.FIELDS)
                writer.writeheader()
                writer.writerows(self.rows())
            dropped = common.prune_csv(path, self.FIELDS, phases, from_n)
            with path.open(newline="") as handle:
                return dropped, list(csv.DictReader(handle))

    def test_one_leg_leaves_the_others(self):
        dropped, kept = self.prune(("performance",))
        self.assertEqual(dropped, 2)
        self.assertEqual([r["phase"] for r in kept], ["numerical", "work_precision"])

    def test_all_legs_clear_every_row(self):
        dropped, kept = self.prune(common.PHASES)
        self.assertEqual(dropped, 4)
        self.assertEqual(kept, [])

    def test_from_n_keeps_the_smaller_n(self):
        dropped, kept = self.prune(("performance",), from_n=2048)
        self.assertEqual(dropped, 1)
        self.assertEqual([r["n"] for r in kept], ["8", "1024", "32768"])

    def test_missing_file_is_created_with_a_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "julia_timings.csv"
            self.assertEqual(common.prune_csv(path, self.FIELDS, common.PHASES), 0)
            self.assertEqual(path.read_text().strip(), ",".join(self.FIELDS))


class AnalysisTests(unittest.TestCase):
    def test_timing_percentiles_and_speedup(self):
        base = {"algorithm": "tsit5", "phase": "performance", "mode": "fixed",
                "n": "8", "setting_kind": "dt", "setting": "0.1"}
        rows = []
        for framework, tier, samples in (("julia", "fixed", (4, 6)),
                                         ("cubie", "fixed", (2, 3))):
            for i, value in enumerate(samples):
                rows.append(dict(base, framework=framework, tier=tier,
                                 sample=str(i), time_ms=str(value)))
        summary = analyze.timing_summary(rows)
        boost = analyze.speedups(summary)
        self.assertEqual(len(boost), 1)
        self.assertAlmostEqual(boost[0]["julia_over_cubie_speedup"], 2.0)

    def test_invalid_points_are_excluded_from_timing_summary(self):
        timing = [{"framework": "cubie", "algorithm": "tsit5",
                   "phase": "performance", "mode": "adaptive",
                   "tier": "default", "n": "8", "setting_kind": "tol",
                   "setting": "1e-8", "sample": "0", "time_ms": "1.0"}]
        invalid = [{"framework": "cubie", "algorithm": "tsit5",
                    "phase": "performance", "mode": "adaptive",
                    "tier": "default", "n": "8", "setting_kind": "tol",
                    "setting": "1e-8", "finite_trajectories": "7",
                    "failed_trajectories": "1"}]
        self.assertEqual(analyze.timing_summary(timing, invalid), [])
        valid = [dict(invalid[0], finite_trajectories="8",
                      failed_trajectories="0")]
        self.assertEqual(len(analyze.timing_summary(timing, valid)), 1)

    def test_mutual_metrics_load_saved_finals(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for framework, values in (("julia", ((1, 2, 3), (4, 5, 6))),
                                      ("cubie", ((1, 2, 3), (4, 5, 7)))):
                path = root / (framework + ".csv")
                with path.open("w", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["traj", "x", "y", "z"])
                    for i, row in enumerate(values):
                        writer.writerow([i] + list(row))
            metrics = []
            for framework, tier in (("julia", "fixed"), ("cubie", "fixed")):
                metrics.append({"framework": framework, "algorithm": "tsit5",
                    "phase": "numerical", "mode": "fixed", "tier": tier,
                    "n": "2", "setting_kind": "dt", "setting": "0.1",
                    "finals_path": framework + ".csv"})
            result = analyze.numerical_comparisons(root, metrics)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["finite_pairs"], 2)
            self.assertAlmostEqual(result[0]["mutual_max_abs"], 1.0)


if __name__ == "__main__":
    unittest.main()
