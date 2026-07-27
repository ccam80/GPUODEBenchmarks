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

    def test_smoke_keeps_every_metric_family(self):
        protocol = common.profile_protocol("smoke", 10_000, 100, 20)
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
