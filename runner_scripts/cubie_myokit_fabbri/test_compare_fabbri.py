"""Dependency-light tests for Fabbri comparison mapping and statistics."""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parent / "compare_fabbri.py"
SPEC = importlib.util.spec_from_file_location(
    "compare_fabbri", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ComparisonTests(unittest.TestCase):
    """Check state mapping and accuracy calculations without GPU packages."""

    def test_component_state_name_mapping(self):
        self.assertEqual(
            MODULE.canonical_state_name("Membrane.V_ode"),
            "Membrane_V_ode",
        )

    def test_cubie_columns_are_reordered_to_myokit_order(self):
        cubie = np.asarray([[30.0, 10.0, 20.0]], dtype=np.float32)
        reordered, names = MODULE.mapped_cubie_states(
            ("a.x", "b.y", "c.z"),
            ("c_z", "a_x", "b_y"),
            cubie,
        )
        np.testing.assert_array_equal(
            reordered,
            np.asarray([[10.0, 20.0, 30.0]], dtype=np.float32),
        )
        self.assertEqual(names, ["a_x", "b_y", "c_z"])

    def test_accuracy_is_computed_in_float64(self):
        myokit = np.asarray([[1.0, 4.0]], dtype=np.float32)
        cubie = np.asarray([[0.5, 2.0]], dtype=np.float32)
        rows, summary = MODULE.accuracy_rows(
            ("x", "y"), myokit, cubie
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(summary["maximum_absolute_error"], 2.0)
        self.assertAlmostEqual(
            summary["root_mean_square_error"],
            np.sqrt((0.5 ** 2 + 2.0 ** 2) / 2.0),
        )

    def test_scaling_summary_contains_every_count(self):
        reports = []
        for count in (512, 2048):
            reports.append(
                {
                    "trajectories": count,
                    "dt": 1e-5,
                    "steps": 1000,
                    "duration": 0.01,
                    "repeats": 100,
                    "block_size": {
                        "myokit_cuda": 128,
                        "cubie": 64,
                    },
                    "timing_milliseconds": {
                        "myokit_cuda_minimum": count / 1000.0,
                        "cubie_minimum": count / 500.0,
                        "cubie_over_myokit_cuda": 2.0,
                        "myokit_cuda_over_cubie": 0.5,
                    },
                    "accuracy": {
                        "maximum_absolute_error": 1e-8,
                        "root_mean_square_error": 1e-9,
                    },
                    "allclose": {"result": True},
                }
            )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            MODULE.write_scaling_summary(output, reports)
            csv_text = (output / "scaling.csv").read_text()
            markdown = (output / "comparison.md").read_text()
        self.assertIn("512,", csv_text)
        self.assertIn("2048,", csv_text)
        self.assertIn("| 512 |", markdown)
        self.assertIn("| 2048 |", markdown)
        self.assertIn("CuBIE speedup", markdown)
        self.assertIn("100 synchronized repeats", markdown)

    def test_fabbri_normalization_is_metadata_only(self):
        source_text = """<?xml version="1.0"?>
<model xmlns="http://www.cellml.org/cellml/1.0#" name="fixture">
    <component name="Ca_buffering">
        <variable initial_value="0.217311" name="fCMi" units="dimensionless"/>
        KF_CM_VARIABLE
        <variable initial_value="542" name="kb_CM" units="per_second"/>
    </component>
    <connection>
        <map_components component_1="cAMP" component_2="ATPi"/>
        <map_variables variable_1="ATPi" variable_2="ATPi"/>
    </connection>
    <connection>
        <map_components component_1="ATPi" component_2="cAMP"/>
        <map_variables variable_1="cAMP" variable_2="cAMP"/>
    </connection>
</model>
"""
        source_text = source_text.replace(
            "KF_CM_VARIABLE",
            '<variable initial_value="1.642e6" name="kf_CM" '
            'units="per_millimolar_second"/>',
        )
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.cellml"
            output = Path(directory) / "output.cellml"
            source.write_text(source_text, encoding="utf-8")
            repairs = MODULE.normalized_fabbri_cellml(source, output)
            normalized = output.read_text(encoding="utf-8")
        self.assertEqual(repairs, MODULE.FABBRI_REPAIRS)
        self.assertEqual(
            normalized.count('public_interface="out"'), 3
        )
        self.assertEqual(
            normalized.count('component_1="cAMP" component_2="ATPi"'),
            1,
        )
        self.assertNotIn(
            'component_1="ATPi" component_2="cAMP"', normalized
        )
        self.assertEqual(
            normalized.count(
                'map_variables variable_1="cAMP" variable_2="cAMP"'
            ),
            1,
        )


if __name__ == "__main__":
    unittest.main()
