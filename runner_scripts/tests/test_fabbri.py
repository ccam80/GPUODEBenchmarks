"""The Fabbri-Linder grid: the index rounding, the 32 x 32 head lattice with its corners, the bit-reversed fill, the nanomolar inputs in both precisions, the cubie parameter arrays, the reference state order against the CellML file, and fabbri.jl bit for bit."""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import cubie_systems  # noqa: E402
import fabbri  # noqa: E402
import grid  # noqa: E402
import protocol  # noqa: E402
from launch import julia_command  # noqa: E402
from problems import get_problem  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(HERE))


def reversed_bits(index):
    """The 17-bit reversal written out as text, independently of fabbri.py."""
    return int(format(index, "017b")[::-1], 2)


class LatticeTests(unittest.TestCase):
    def test_the_catalogue_row_is_the_lattice_index_grid(self):
        row = get_problem(fabbri.PROBLEM)
        self.assertEqual((row["sweep_parameter"], row["sweep_scale"], row["sweep_min"], row["sweep_max"]),
                         (fabbri.PARAMETER, "linear", 0.0, float(fabbri.LATTICE_POINTS - 1)))
        self.assertEqual((row["states"], row["duration"], row["golden_algorithm"]), (35, 2.0, "VCABM"))
        self.assertEqual(row["frameworks"], ("cubie", "cubie_mlir", "julia_cpu", "myokit_cuda"))
        values = grid.grid_values("linear", 0.0, 131071.0, fabbri.LATTICE_POINTS)
        np.testing.assert_array_equal(values, np.arange(fabbri.LATTICE_POINTS, dtype=np.float32))
        np.testing.assert_array_equal(fabbri.lattice_index(values), np.arange(fabbri.LATTICE_POINTS))

    def test_rounding_and_clamping(self):
        np.testing.assert_array_equal(fabbri.lattice_index([0.4, 0.5, 1.5, 2.5, 3.51, -7.0, 2.0e5]),
                                      [0, 0, 2, 2, 4, 0, 131071])
        self.assertEqual(fabbri.lattice_index(np.float32([131071.0])).dtype, np.int64)

    def test_the_head_of_the_grid_is_the_32_by_32_lattice_with_its_corners(self):
        index = np.arange(fabbri.TRACE_POINTS)
        ach, iso = fabbri.levels(index)
        np.testing.assert_array_equal(ach, (index // 32) / 31.0)
        np.testing.assert_array_equal(iso, (index % 32) / 31.0)
        ach_nm, iso_nm = fabbri.inputs(index, np.float64)
        corners = {(ach_nm[i], iso_nm[i]) for i in (0, 31, 992, 1023)}
        self.assertEqual(corners, {(0.0, 0.0), (0.0, 1000.0), (100.0, 0.0), (100.0, 1000.0)})
        self.assertEqual(len(set(zip(ach_nm.tolist(), iso_nm.tolist()))), fabbri.TRACE_POINTS)
        self.assertEqual(fabbri.TRACE_POINTS, protocol.TRACE_ROWS)

    def test_the_rest_of_the_grid_fills_the_plane_by_bit_reversal(self):
        index = np.arange(fabbri.TRACE_POINTS, fabbri.LATTICE_POINTS)
        expected = np.array([reversed_bits(i) for i in index])
        np.testing.assert_array_equal(fabbri.bit_reverse(index), expected)
        ach, iso = fabbri.levels(index)
        np.testing.assert_array_equal(ach, (expected >> 9) / 255.0)
        np.testing.assert_array_equal(iso, (expected & 511) / 511.0)
        # The fill reaches every ACh level, all but the four Iso levels whose partners sit in the head, and both range ends.
        self.assertEqual(len(set((expected >> 9).tolist())), 256)
        self.assertEqual(len(set((expected & 511).tolist())), 508)
        self.assertEqual((ach.max(), iso.max()), (1.0, 1.0))

    def test_inputs_in_both_precisions(self):
        values = np.float32([0.0, 1.0, 131071.0])
        ach32, iso32 = fabbri.inputs(values)
        self.assertEqual((ach32.dtype, iso32.dtype), (np.float32, np.float32))
        ach64, iso64 = fabbri.inputs(values, np.float64)
        self.assertEqual((ach64.dtype, iso64.dtype), (np.float64, np.float64))
        np.testing.assert_array_equal(ach32, ach64.astype(np.float32))
        # Index 0 is the unmodulated corner, 1 the next Iso level of the head lattice, 131071 a fill point.
        self.assertEqual((ach64[0], iso64[0]), (0.0, 0.0))
        self.assertEqual((ach64[1], iso64[1]), (0.0, 1000.0 / 31))
        self.assertEqual((ach64[2], iso64[2]), (100.0, 1000.0))
        parameters = fabbri.parameters(values)
        self.assertEqual(list(parameters), [fabbri.ACH_PARAMETER, fabbri.ISO_PARAMETER])
        np.testing.assert_array_equal(parameters[fabbri.ACH_PARAMETER], ach32)
        np.testing.assert_array_equal(parameters[fabbri.ISO_PARAMETER], iso32)
        ensemble = cubie_systems.ensemble_parameters("fabbri_linder", values)
        self.assertEqual(list(ensemble), list(parameters))
        for name in parameters:
            np.testing.assert_array_equal(ensemble[name], parameters[name])
            self.assertEqual(ensemble[name].dtype, np.float32)
        lorenz = cubie_systems.ensemble_parameters("lorenz", values, np.float64)
        self.assertEqual(list(lorenz), ["rho"])
        np.testing.assert_array_equal(lorenz["rho"], values.astype(np.float64))
        self.assertEqual(cubie_systems.variable_order("fabbri_linder"), fabbri.STATE_ORDER)

    def test_the_state_order_is_the_cellml_document_order(self):
        """Components in document order, and within one the differentiated variables in declaration order: Myokit's state order."""
        with open(fabbri.MODEL_PATH, encoding="utf-8") as handle:
            text = handle.read()
        states = []
        for component in re.finditer(r'<component name="([^"]+)">(.*?)</component>', text, re.S):
            body = component.group(2)
            differentiated = set(re.findall(r"<apply>\s*<eq/>\s*<apply>\s*<diff/>\s*<bvar>\s*<ci>\s*time\s*</ci>"
                                            r"\s*</bvar>\s*<ci>\s*([^<\s]+)\s*</ci>", body))
            for declared in re.findall(r'<variable\b[^>]*\bname="([^"]+)"', body):
                if declared in differentiated:
                    states.append("{0}_{1}".format(component.group(1), declared))
        self.assertEqual(tuple(states), fabbri.STATE_ORDER)
        self.assertEqual(len(fabbri.STATE_ORDER), 35)
        for qname in (fabbri.ANS_QNAME, fabbri.ACH_QNAME, fabbri.ISO_QNAME):
            component, name = qname.split(".")
            self.assertIn('name="{0}"'.format(name), text)
            self.assertEqual(fabbri.myokit_name(qname), "{0}_{1}".format(component, name))
        self.assertEqual(fabbri.myokit_name(fabbri.ACH_QNAME), fabbri.ACH_PARAMETER)
        self.assertEqual(fabbri.myokit_name(fabbri.ISO_QNAME), fabbri.ISO_PARAMETER)
        self.assertEqual(fabbri.myokit_name(fabbri.ANS_QNAME), fabbri.ANS_CONSTANT)

    def test_the_generated_julia_rhs_names_the_reference_order(self):
        generated = os.path.join(os.path.dirname(HERE), "generated")
        with open(os.path.join(generated, "fabbri_linder_rhs.jl"), encoding="utf-8") as handle:
            text = handle.read()
        names = re.search(r"const FABBRI_LINDER_STATES = \[(.*?)\]", text).group(1)
        self.assertEqual(tuple(json.loads("[" + names + "]")), fabbri.STATE_ORDER)
        with open(os.path.join(generated, "fabbri_linder_rhs_check.json"), encoding="utf-8") as handle:
            check = json.load(handle)
        self.assertEqual(tuple(check["states"]), fabbri.STATE_ORDER)
        self.assertEqual(len(check["du"]), len(check["t"]))
        self.assertTrue(np.isfinite(np.asarray(check["du"])).all())


class JuliaTwinTests(unittest.TestCase):
    """fabbri.jl reproduces fabbri.py's levels and inputs bit for bit over the whole lattice and at fractional values."""

    def test_julia_matches(self):
        launcher = julia_command()
        if shutil.which(launcher[0]) is None:
            self.skipTest("julia is not on PATH")
        tmp = tempfile.mkdtemp(prefix="fabbri_twin_")
        self.addCleanup(shutil.rmtree, tmp, True)
        values = np.concatenate([np.arange(fabbri.LATTICE_POINTS, dtype=np.float64),
                                 np.float64([0.4, 0.5, 1.5, 2.5, 3.51, -7.0, 2.0e5, 131070.6])])
        values_path = os.path.join(tmp, "values.txt")
        np.savetxt(values_path, values, fmt="%.17g")
        out_path = os.path.join(tmp, "out.txt")
        script = os.path.join(tmp, "twin.jl")
        with open(script, "w", encoding="utf-8") as handle:
            handle.write('include(raw"{0}")\n'.format(os.path.join(os.path.dirname(HERE), "fabbri.jl")))
            handle.write('values = parse.(Float64, readlines(raw"{0}"))\n'.format(values_path))
            handle.write('open(raw"{0}", "w") do io\n'.format(out_path))
            handle.write("    for v in values\n        a, i = fabbri_levels(v)\n"
                         "        a32, i32 = fabbri_inputs(v, Float32)\n        a64, i64 = fabbri_inputs(v, Float64)\n"
                         "        println(io, reinterpret(UInt64, a), ' ', reinterpret(UInt64, i), ' ', reinterpret(UInt32, a32), ' ',"
                         " reinterpret(UInt32, i32), ' ', reinterpret(UInt64, a64), ' ', reinterpret(UInt64, i64))\n    end\nend\n")
        run = subprocess.run(launcher + ["--startup-file=no", script], capture_output=True, text=True,
                             cwd=REPO_ROOT, timeout=600)
        self.assertEqual(run.returncode, 0, run.stderr)
        got = np.loadtxt(out_path, dtype=np.uint64).reshape(-1, 6)
        ach, iso = fabbri.levels(values)
        ach32, iso32 = fabbri.inputs(values, np.float32)
        ach64, iso64 = fabbri.inputs(values, np.float64)
        np.testing.assert_array_equal(got[:, 0], ach.view(np.uint64))
        np.testing.assert_array_equal(got[:, 1], iso.view(np.uint64))
        np.testing.assert_array_equal(got[:, 2], ach32.view(np.uint32).astype(np.uint64))
        np.testing.assert_array_equal(got[:, 3], iso32.view(np.uint32).astype(np.uint64))
        np.testing.assert_array_equal(got[:, 4], ach64.view(np.uint64))
        np.testing.assert_array_equal(got[:, 5], iso64.view(np.uint64))


if __name__ == "__main__":
    unittest.main()
