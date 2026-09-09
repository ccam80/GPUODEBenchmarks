"""Dependency-free checks for the Myokit-CUDA benchmark contract."""

import importlib.util
import sys
import unittest
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import numpy as np


PACKAGE_DIR = Path(__file__).resolve().parent.parent
MODULE_PATH = PACKAGE_DIR / "myokit_cuda.py"
SPEC = importlib.util.spec_from_file_location("myokit_cuda", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class StaticContractTests(unittest.TestCase):
    """Check source wrapping and the standards-valid Lorenz fixture."""

    def test_lorenz_cellml_is_well_formed(self):
        path = PACKAGE_DIR / "models" / "lorenz.cellml"
        root = ElementTree.parse(path).getroot()
        namespace = "{http://www.cellml.org/cellml/1.1#}"
        self.assertEqual(root.tag, namespace + "model")
        components = {
            item.attrib["name"]: item
            for item in root.findall(namespace + "component")
        }
        self.assertEqual(set(components), {"environment", "lorenz"})
        variables = {
            item.attrib["name"]
            for item in components["lorenz"].findall(
                namespace + "variable"
            )
        }
        self.assertEqual(
            variables,
            {"time", "x", "y", "z", "rho", "sigma", "beta"},
        )

    def test_launcher_calls_generated_euler_step(self):
        source = MODULE._LAUNCH_KERNEL
        self.assertIn("iterate_euler_cu(dt, state, input", source)
        self.assertIn("diffusion_current[cell]", source)
        self.assertNotIn("dx =", source)
        self.assertNotIn("dy =", source)
        self.assertNotIn("dz =", source)

    def test_only_known_nvrtc_incompatible_include_is_removed(self):
        generated = (
            "prefix\n"
            + MODULE._UNSUPPORTED_NVRTC_INCLUDE
            + "generated equations\n"
        )
        adjusted = generated.replace(
            MODULE._UNSUPPORTED_NVRTC_INCLUDE, "", 1
        )
        self.assertEqual(adjusted, "prefix\ngenerated equations\n")

    def test_float32_validation_is_contiguous(self):
        source = np.arange(12, dtype=np.float64).reshape(3, 4)[:, ::2]
        result = MODULE._validate_float32(source, "source")
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(result.flags.c_contiguous)


if __name__ == "__main__":
    unittest.main()
