"""Tests for the native .x2d project file format (xdfem2d.file_io).

The .x2d writer/reader moved out of the GUI into the API so it can be tested
without PySide6. These cover the structure + results + view round-trip and the
restoration of NumPy diagram arrays.
"""
import os
import tempfile
import unittest

import numpy as np

from context import Structure2D  # noqa: F401  (adds src/ to sys.path)
from xdfem2d import save_x2d, load_x2d


def _model():
    s = Structure2D()
    s.add_material("M", elastic_modulus=30e6, unit_weight=25.0)
    s.add_section("S", "M", b=0.3, h=0.6)
    s.add_node("N1", 0.0, 0.0)
    s.add_node("N2", 5.0, 0.0)
    s.add_bar_element("E1", "N1", "N2", "S")
    s.add_support("PIN", ux=True, uy=True)
    s.add_support("ROLLER", ux=False, uy=True)
    s.assign_support("N1", "PIN")
    s.assign_support("N2", "ROLLER")
    s.add_load_case("LC")
    s.add_distributed_load("E1", "LC", fye=-10.0, fyd=-10.0)
    s.add_analysis_case("AC", "Linear", {"LC": 1.0})
    return s


class TestX2dRoundTrip(unittest.TestCase):
    def setUp(self):
        self.s = _model()
        self.results = self.s.calculate()

    def _save_load(self, results=None, view=None):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "model.x2d")
            save_x2d(self.s, results, path, view=view)
            self.assertGreater(os.path.getsize(path), 0)
            return load_x2d(path)

    def test_structure_only(self):
        struc, results, view = self._save_load(results=None)
        self.assertEqual(set(struc.nodes), set(self.s.nodes))
        self.assertEqual(len(struc.bar_elements), len(self.s.bar_elements))
        self.assertIsNone(results)
        self.assertIsNone(view)

    def test_structure_resolves_to_same_answer(self):
        struc, _, _ = self._save_load(results=None)
        r2 = struc.calculate()
        self.assertAlmostEqual(r2["reactions"]["LC"]["N1"][1],
                               self.results["reactions"]["LC"]["N1"][1], places=6)

    def test_view_roundtrip(self):
        _, _, view = self._save_load(results=self.results, view={"zoom": 1.5})
        self.assertEqual(view, {"zoom": 1.5})

    def test_steel_design_payload_survives(self):
        # The steel design is stored inside the results so it reopens with the
        # model (and its 'Steel design' view).
        self.results["steel_design"] = {
            "members": [{"member": "M1", "elements": "E1", "utilization": 0.9,
                         "passed": True}],
            "elements": {"E1": {"bending": 0.58, "shear": 0.3, "torsion": 0.0,
                                "buckling": 0.9, "combined": 0.9}},
            "header": "1 member — all verified.",
        }
        _, results, _ = self._save_load(results=self.results)
        sd = results.get("steel_design")
        self.assertIsNotNone(sd)
        self.assertEqual(sd["members"][0]["utilization"], 0.9)
        self.assertTrue(sd["members"][0]["passed"])
        self.assertEqual(sd["elements"]["E1"]["combined"], 0.9)

    def test_results_restored_with_numpy_diagrams(self):
        _, results, _ = self._save_load(results=self.results)
        self.assertIsNotNone(results)
        dist = results["analysis_cases"]["AC"].get("element_distribution", {})
        self.assertIn("E1", dist)
        self.assertIsInstance(dist["E1"]["M"], np.ndarray)

    def test_stored_stiffness_npy_roundtrip(self):
        """ANLG stiffness matrices are stored as .npy entries and restored."""
        self.results["stored_stiffness"] = {"NL1": np.eye(6) * 3.0}
        _, results, _ = self._save_load(results=self.results)
        self.assertIn("stored_stiffness", results)
        np.testing.assert_allclose(results["stored_stiffness"]["NL1"], np.eye(6) * 3.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
