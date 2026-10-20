"""Tests for the Phase-1 variants core (xdfem2d.variants).

Covers:
  - exact-sum equivalence (sum of two same-stiffness variants == single model
    load combination), cross-checked against the solver;
  - support-scenario envelope and the LinearSum stiffness guard;
  - stiffness_hash sensitivity;
  - derive_variant purity (does not mutate the base);
  - blocking of geometry-mismatch combinations.
"""
import unittest

from context import Structure2D, assert_close
from xdfem2d.models import SupportSet, SupportAssignment, Variant
from xdfem2d.variants import (
    solve_variants, combine_across_variants, CombTerm,
)


def _two_case_beam():
    """Simply-supported beam with two independent load cases A and B."""
    s = Structure2D()
    s.add_material("M", elastic_modulus=30e6, unit_weight=0.0)
    s.add_section("S", "M", b=0.3, h=0.6)
    s.add_node("N1", 0.0, 0.0)
    s.add_node("N2", 5.0, 0.0)
    s.add_node("N3", 10.0, 0.0)
    s.add_bar_element("E1", "N1", "N2", "S")
    s.add_bar_element("E2", "N2", "N3", "S")
    s.add_support("PIN", ux=True, uy=True)
    s.add_support("ROLLER", ux=False, uy=True)
    s.assign_support("N1", "PIN")
    s.assign_support("N3", "ROLLER")
    s.add_load_case("A")
    s.add_load_case("B")
    s.add_distributed_load("E1", "A", fye=-10.0, fyd=-10.0)
    s.add_distributed_load("E2", "B", fye=-8.0, fyd=-8.0)
    return s


class TestExactSum(unittest.TestCase):
    """Sum of two same-geometry variants equals the single-model A+B combo."""

    def setUp(self):
        self.s = _two_case_beam()
        # Reference: a real load combination A+B in the single model.
        self.s.add_load_combination("AB", {"A": 1.0, "B": 1.0},
                                    combo_type="LinearSum")
        self.ref = self.s.calculate()
        # Two variants with identical geometry/supports (same hash).
        v1 = Variant(id="V1")
        v2 = Variant(id="V2")
        self.vres = solve_variants(self.s, [v1, v2])

    def test_same_hash(self):
        self.assertEqual(self.vres["V1"].stiffness_hash,
                         self.vres["V2"].stiffness_hash)

    def test_linear_sum_matches_combo(self):
        out = combine_across_variants(
            self.vres,
            [CombTerm("V1", "A", 1.0), CombTerm("V2", "B", 1.0)],
            op="LinearSum",
        )
        ref_ef = self.ref["combinations"]["AB"]["element_forces"]
        for eid in ("E1", "E2"):
            for end in ("i", "j"):
                for k in range(3):
                    assert_close(self, out["element_forces"][eid][end][k],
                                 ref_ef[eid][end][k], rel=1e-9, abs_tol=1e-6,
                                 msg=f"{eid}.{end}[{k}]")

    def test_displacements_match_combo(self):
        out = combine_across_variants(
            self.vres,
            [CombTerm("V1", "A", 1.0), CombTerm("V2", "B", 1.0)],
            op="LinearSum",
        )
        ref_d = self.ref["combinations"]["AB"]["displacements"]
        for nid in ("N1", "N2", "N3"):
            for k in range(3):
                assert_close(self, out["displacements"][nid][k],
                             ref_d[nid][k], rel=1e-9, abs_tol=1e-9,
                             msg=f"{nid}[{k}]")


class TestSupportEnvelope(unittest.TestCase):
    """Same geometry, two support sets: envelope works, linear sum is blocked."""

    def setUp(self):
        self.s = _two_case_beam()
        # Base assigns PIN@N1, ROLLER@N3. Alternative: ROLLER@N1, PIN@N3.
        self.alt = SupportSet(
            id="ALT",
            assignments=[SupportAssignment(node_id="N1", support_name="ROLLER"),
                         SupportAssignment(node_id="N3", support_name="PIN")],
        )
        v_base = Variant(id="BASE")
        v_alt = Variant(id="ALT", support_set_id="ALT")
        self.vres = solve_variants(self.s, [v_base, v_alt],
                                   support_sets={"ALT": self.alt})

    def test_different_hash(self):
        self.assertNotEqual(self.vres["BASE"].stiffness_hash,
                            self.vres["ALT"].stiffness_hash)

    def test_linear_sum_allowed_on_bars(self):
        # Different stiffness no longer blocks LinearSum: bar internal forces are
        # always combinable; the result is flagged as not-same-stiffness so the
        # GUI can hide deformed/reactions.
        out = combine_across_variants(
            self.vres,
            [CombTerm("BASE", "A", 1.0), CombTerm("ALT", "A", 1.0)],
            op="LinearSum",
        )
        self.assertFalse(out["_same_stiffness"])
        self.assertIn("E1", out["element_forces"])
        self.assertIn("E1", out["combo_distribution"])

    def test_envelope_runs_and_bounds(self):
        out = combine_across_variants(
            self.vres,
            [CombTerm("BASE", "A", 1.0), CombTerm("ALT", "A", 1.0)],
            op="Envelope",
        )
        self.assertIn("max", out["element_forces"])
        self.assertIn("min", out["element_forces"])
        # max >= min component-wise on every element end.
        hi = out["element_forces"]["max"]
        lo = out["element_forces"]["min"]
        for eid in ("E1", "E2"):
            for end in ("i", "j"):
                for k in range(3):
                    self.assertGreaterEqual(hi[eid][end][k] + 1e-9,
                                            lo[eid][end][k])


class TestStiffnessHash(unittest.TestCase):
    def test_load_change_does_not_change_hash(self):
        s1 = _two_case_beam()
        s2 = _two_case_beam()
        s2.add_distributed_load("E1", "A", fye=-99.0, fyd=-99.0)   # extra load, same K
        self.assertEqual(s1.stiffness_hash(), s2.stiffness_hash())

    def test_section_change_changes_hash(self):
        s1 = _two_case_beam()
        s2 = _two_case_beam()
        s2.sections["S"].h = 0.9
        self.assertNotEqual(s1.stiffness_hash(), s2.stiffness_hash())

    def test_active_geometry_changes_hash(self):
        s = _two_case_beam()
        full = s.derive_variant(Variant(id="F"))
        part = s.derive_variant(Variant(id="P", active_elements={"E1"}))
        self.assertNotEqual(full.stiffness_hash(), part.stiffness_hash())


class TestDerivePurity(unittest.TestCase):
    def test_base_unchanged(self):
        s = _two_case_beam()
        n_nodes, n_elems = len(s.nodes), len(s.bar_elements)
        n_assign = len(s.support_assignments)
        _ = s.derive_variant(Variant(id="P", active_elements={"E1"}))
        self.assertEqual(len(s.nodes), n_nodes)
        self.assertEqual(len(s.bar_elements), n_elems)
        self.assertEqual(len(s.support_assignments), n_assign)

    def test_partial_drops_orphan(self):
        s = _two_case_beam()
        # Active only E1 (N1-N2); N3 is supported so it stays, but no orphan E2.
        sub = s.derive_variant(Variant(id="P", active_elements={"E1"}))
        self.assertEqual([e.id for e in sub.bar_elements], ["E1"])
        self.assertIn("N1", sub.nodes)
        self.assertIn("N2", sub.nodes)


class TestCombinedDistributions(unittest.TestCase):
    """Combine also produces N/V/M diagrams (combo_distribution) for the canvas."""

    def setUp(self):
        self.s = _two_case_beam()
        self.vres = solve_variants(self.s, [Variant(id="V1"), Variant(id="V2")])

    def test_linear_sum_distribution_doubles(self):
        single = combine_across_variants(
            self.vres, [CombTerm("V1", "A", 1.0)], op="LinearSum")
        double = combine_across_variants(
            self.vres, [CombTerm("V1", "A", 1.0), CombTerm("V2", "A", 1.0)],
            op="LinearSum")
        import numpy as np
        m1 = np.max(np.abs(single["combo_distribution"]["E1"]["M"]))
        m2 = np.max(np.abs(double["combo_distribution"]["E1"]["M"]))
        assert_close(self, m2, 2.0 * m1, rel=1e-9, abs_tol=1e-6)

    def test_envelope_distribution_has_bands(self):
        out = combine_across_variants(
            self.vres, [CombTerm("V1", "A", 1.0), CombTerm("V2", "A", 1.0)],
            op="Envelope")
        cd = out["combo_distribution"]["E1"]
        self.assertIn("M_min", cd)
        self.assertTrue(cd.get("is_envelope"))


class TestGeometryMismatch(unittest.TestCase):
    def test_mismatch_combines_on_bars(self):
        # Different active geometry is combinable on the shared bars; no block.
        s = _two_case_beam()
        v_full = Variant(id="F")
        v_part = Variant(id="P", active_elements={"E1"})
        vres = solve_variants(s, [v_full, v_part])
        out = combine_across_variants(
            vres, [CombTerm("F", "A", 1.0), CombTerm("P", "A", 1.0)],
            op="Envelope")
        self.assertIn("max", out["element_forces"])
        self.assertIn("E1", out["combo_distribution"])     # shared bar present
        self.assertFalse(out["_same_stiffness"])


if __name__ == "__main__":
    unittest.main()
