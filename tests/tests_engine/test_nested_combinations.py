"""Tests for combinations that reference other combinations as inputs."""
import unittest

from context import Structure2D  # noqa: F401  (adds src/ to sys.path)
from xdfem2d.sap2000_io import to_s2k


def _model():
    s = Structure2D()
    s.add_material("C30", elastic_modulus=30e6, unit_weight=25.0)
    s.add_section("S1", "C30", b=0.3, h=0.6)
    s.add_node("N1", 0.0, 0.0)
    s.add_node("N2", 5.0, 0.0)
    s.add_bar_element("E1", "N1", "N2", "S1", hinge_j=True)
    s.add_support("PIN", ux=True, uy=True)
    s.add_support("ROLLER", ux=False, uy=True)
    s.assign_support("N1", "PIN")
    s.assign_support("N2", "ROLLER")
    s.add_load_case("G", self_weight_factor=1.0)
    s.add_load_case("Q")
    s.add_point_load("N2", "Q", fy=-8.0, mz=2.0)
    s.add_distributed_load("E1", "G", fye=-10.0, fyd=-10.0)
    return s


class TestExpandCombinationCoefficients(unittest.TestCase):
    def test_plain_combo_unchanged(self):
        s = _model()
        c = s.add_load_combination("CO", {"G": 1.35, "Q": 1.5})
        self.assertEqual(s.expand_combination_coefficients(c),
                         {"G": 1.35, "Q": 1.5})

    def test_nested_combo_expanded_and_scaled(self):
        s = _model()
        s.add_load_combination("BASE", {"G": 1.35, "Q": 1.5})
        top = s.add_load_combination("TOP", {"BASE": 2.0})
        self.assertEqual(s.expand_combination_coefficients(top),
                         {"G": 2.7, "Q": 3.0})

    def test_nested_combo_merges_with_direct_cases(self):
        s = _model()
        s.add_load_combination("BASE", {"G": 1.0})
        top = s.add_load_combination("TOP", {"BASE": 1.0, "Q": 1.5, "G": 0.35})
        # G appears via BASE (1.0) and directly (0.35) -> 1.35
        self.assertEqual(s.expand_combination_coefficients(top),
                         {"G": 1.35, "Q": 1.5})

    def test_multi_level_nesting(self):
        s = _model()
        s.add_load_combination("L1", {"G": 1.0})
        s.add_load_combination("L2", {"L1": 1.35, "Q": 1.5})
        top = s.add_load_combination("L3", {"L2": 2.0})
        self.assertEqual(s.expand_combination_coefficients(top),
                         {"G": 2.7, "Q": 3.0})

    def test_self_reference_rejected(self):
        s = _model()
        s.add_load_combination("A", {"G": 1.0})
        # mutate to create a self-reference and expand
        s.load_combinations[0].coefficients = {"A": 1.0}
        with self.assertRaises(ValueError):
            s.expand_combination_coefficients(s.load_combinations[0])

    def test_circular_reference_rejected(self):
        s = _model()
        s.add_load_combination("A", {"G": 1.0})
        s.add_load_combination("B", {"A": 1.0})
        # close the loop A -> B
        s.load_combinations[0].coefficients = {"B": 1.0}
        with self.assertRaises(ValueError):
            s.expand_combination_coefficients(s.load_combinations[1])

    def test_nonlinear_parent_can_reference_somalinear(self):
        # Envelope / SRSS / AbsSum may take LinearSum combos as inputs.
        s = _model()
        s.add_load_combination("B1", {"G": 1.35, "Q": 1.5})
        s.add_load_combination("B2", {"G": 1.0})
        env = s.add_load_combination("ENV", {"B1": 1.0, "B2": 1.0},
                                     combo_type="Envelope")
        self.assertEqual(env.combo_type, "Envelope")
        # enumeration still flattens for validation purposes
        self.assertEqual(s.expand_combination_coefficients(env),
                         {"G": 2.35, "Q": 1.5})

    def test_envelope_can_reference_any_combo_type(self):
        # Envelope may take SRSS / AbsSum / other Envelope as inputs.
        s = _model()
        s.add_load_combination("R", {"G": 1.0, "Q": 1.0}, combo_type="SRSS")
        s.add_load_combination("M", {"G": 1.0}, combo_type="AbsSum")
        env = s.add_load_combination("ENV", {"R": 1.0, "M": 1.0},
                                     combo_type="Envelope")
        self.assertEqual(env.combo_type, "Envelope")

    def test_somalinear_referencing_nonlinear_combo_rejected(self):
        # A non-Envelope parent may not take a non-linear combination as input.
        s = _model()
        s.add_load_combination("SUB", {"G": 1.0}, combo_type="SRSS")
        with self.assertRaises(ValueError):
            s.add_load_combination("TOP", {"SUB": 1.0})  # LinearSum parent
        with self.assertRaises(ValueError):
            s.add_load_combination("TOP2", {"SUB": 1.0}, combo_type="SRSS")


class TestNonLinearCombo(unittest.TestCase):
    def _nl_model(self):
        s = _model()
        s.add_analysis_case("NL", "NonLinear", {"G": 1.0})
        s.add_analysis_case("PD", "GeometricNonlinear", {"G": 1.0})
        return s

    def test_valid_nonlinearcombo(self):
        s = self._nl_model()
        c = s.add_load_combination("CNL", {"NL": 1.0}, combo_type="NonLinearCombo")
        self.assertEqual(c.combo_type, "NonLinearCombo")
        c2 = s.add_load_combination("CPD", {"PD": 1.0}, combo_type="NonLinearCombo")
        self.assertEqual(c2.combo_type, "NonLinearCombo")

    def test_nonlinear_case_requires_nonlinearcombo(self):
        s = self._nl_model()
        for ctype in ("LinearSum", "Envelope", "SRSS", "AbsSum"):
            with self.assertRaises(ValueError):
                s.add_load_combination(f"X_{ctype}", {"NL": 1.0}, combo_type=ctype)

    def test_nonlinearcombo_rejects_extra_or_factor(self):
        s = self._nl_model()
        with self.assertRaises(ValueError):  # extra case
            s.add_load_combination("A", {"NL": 1.0, "G": 1.0},
                                   combo_type="NonLinearCombo")
        with self.assertRaises(ValueError):  # factor != 1.0
            s.add_load_combination("B", {"NL": 1.5}, combo_type="NonLinearCombo")
        with self.assertRaises(ValueError):  # no non-linear case
            s.add_load_combination("C", {"G": 1.0}, combo_type="NonLinearCombo")

    def test_nonlinearcombo_result_is_passthrough(self):
        s = self._nl_model()
        s.add_load_combination("CNL", {"NL": 1.0}, combo_type="NonLinearCombo")
        res = s.calculate()
        ac = res["analysis_cases"]["NL"]
        cb = res["combinations"]["CNL"]
        for nid, vals in ac["displacements"].items():
            for k in range(3):
                self.assertAlmostEqual(vals[k], cb["displacements"][nid][k],
                                       places=9)

    def test_envelope_can_reference_nonlinearcombo(self):
        s = self._nl_model()
        s.add_load_combination("CNL", {"NL": 1.0}, combo_type="NonLinearCombo")
        s.add_load_combination("CG", {"G": 1.0})
        env = s.add_load_combination("ENV", {"CNL": 1.0, "CG": 1.0},
                                     combo_type="Envelope")
        res = s.calculate()
        self.assertIn("max", res["combinations"]["ENV"]["displacements"])


class TestNestedCombinationResults(unittest.TestCase):
    def test_nested_combo_matches_flat_combo(self):
        s = _model()
        s.add_load_combination("FLAT", {"G": 1.35, "Q": 1.5})
        s.add_load_combination("BASE", {"G": 1.35, "Q": 1.5})
        s.add_load_combination("NESTED", {"BASE": 1.0})
        res = s.calculate()
        flat = res["combinations"]["FLAT"]
        nested = res["combinations"]["NESTED"]
        for nid, vals in flat["displacements"].items():
            for k in range(3):
                self.assertAlmostEqual(vals[k],
                                       nested["displacements"][nid][k], places=9)

    def test_envelope_of_somalinear_unit(self):
        # An Envelope combo treats each referenced LinearSum combo as a single
        # input unit: the envelope max/min must bound the unit's own result.
        s = _model()
        s.add_load_combination("U1", {"G": 1.35, "Q": 1.5})
        s.add_load_combination("U2", {"G": 1.0})
        s.add_load_combination("ENV", {"U1": 1.0, "U2": 1.0},
                               combo_type="Envelope")
        res = s.calculate()
        u1 = res["combinations"]["U1"]
        env = res["combinations"]["ENV"]
        self.assertIn("max", env["displacements"])
        for nid, vals in u1["displacements"].items():
            for k in range(3):
                self.assertGreaterEqual(env["displacements"]["max"][nid][k] + 1e-9,
                                        vals[k])
                self.assertLessEqual(env["displacements"]["min"][nid][k] - 1e-9,
                                     vals[k])

    def test_envelope_diagram_has_minmax_bands(self):
        # An Envelope combo must produce true max/min N/V/M envelope diagrams.
        s = _model()
        s.add_load_combination("CG", {"G": 1.0})
        s.add_load_combination("CQ", {"Q": 1.0})
        s.add_load_combination("ENV", {"CG": 1.0, "CQ": 1.0},
                               combo_type="Envelope")
        res = s.calculate()
        dist = res["combo_distribution"]["ENV"]["E1"]
        self.assertTrue(dist.get("is_envelope"))
        for comp in ("N", "V", "M"):
            self.assertIn(comp + "_min", dist)
            # max band must dominate the min band everywhere
            self.assertTrue((dist[comp] >= dist[comp + "_min"] - 1e-9).all())

    def test_envelope_diagram_bounds_inputs(self):
        # The envelope band must bound each input diagram pointwise.
        s = _model()
        s.add_load_combination("CG", {"G": 1.0})
        s.add_load_combination("CQ", {"Q": 1.0})
        s.add_load_combination("ENV", {"CG": 1.0, "CQ": 1.0},
                               combo_type="Envelope")
        res = s.calculate()
        env = res["combo_distribution"]["ENV"]["E1"]
        for sub in ("CG", "CQ"):
            d = res["combo_distribution"][sub]["E1"]
            for comp in ("N", "V", "M"):
                self.assertTrue((env[comp] + 1e-9 >= d[comp]).all())
                self.assertTrue((env[comp + "_min"] - 1e-9 <= d[comp]).all())

    def test_single_case_envelope_collapses(self):
        # An envelope over a single (single-signed) input must collapse: max==min
        # (no spurious zero baseline band) — both the diagram and the reactions.
        s = _model()
        s.add_load_combination("CG", {"G": 1.0})
        s.add_load_combination("ENV", {"CG": 1.0}, combo_type="Envelope")
        res = s.calculate()
        d = res["combo_distribution"]["ENV"]["E1"]
        self.assertTrue((abs(d["M"] - d["M_min"]) < 1e-9).all())
        rc = res["combinations"]["ENV"]["reactions"]
        for nid in rc["max"]:
            for k in range(3):
                self.assertAlmostEqual(rc["max"][nid][k], rc["min"][nid][k],
                                       places=9)

    def test_envelope_reactions_governing_nonzero(self):
        # For single-signed reactions the 'max' band can be ~0 while 'min' holds
        # the real value; the governing magnitude must be non-zero so the GUI
        # can draw it (regression for envelope reactions not being drawn).
        s = _model()
        s.add_load_combination("CG", {"G": 1.0})
        s.add_load_combination("ENV", {"CG": 1.0}, combo_type="Envelope")
        res = s.calculate()
        reac = res["combinations"]["ENV"]["reactions"]
        self.assertIn("max", reac)
        gov = {nid: [a if abs(a) >= abs(b) else b
                     for a, b in zip(reac["max"][nid], reac["min"][nid])]
               for nid in reac["max"]}
        # vertical reaction at the pinned support N1 must be non-zero
        self.assertGreater(abs(gov["N1"][1]), 1e-6)

    def test_nested_combo_exports_flattened(self):
        s = _model()
        s.add_load_combination("BASE", {"G": 1.35, "Q": 1.5})
        s.add_load_combination("TOP", {"BASE": 1.0})
        txt = to_s2k(s)
        # TOP must export as references to analysis cases G and Q, not "BASE".
        self.assertIn('ComboName="TOP"', txt)
        self.assertIn('CaseName="G"   ScaleFactor=1.35', txt)
        self.assertNotIn('CaseName="BASE"', txt)


if __name__ == "__main__":
    unittest.main()
