"""Fase 3 — workflow performance/robustness contracts.

run_variant_combination must reuse precomputed VariantResults (no double
solve), and combination_validity must judge model-backed variants without
deep-copying the structure (and without mutating it).
"""
import unittest

from context import Structure2D
from xdfem2d.models import Variant, SupportSet
from xdfem2d.variants import solve_variants
from xdfem2d.workflows import (make_variant_model, run_variant_combination,
                               combination_validity)


def _beam():
    s = Structure2D()
    s.add_material("M", elastic_modulus=30e6, unit_weight=0.0)
    s.add_section("S", "M", b=0.3, h=0.6)
    s.add_node("N1", 0.0, 0.0); s.add_node("N2", 5.0, 0.0); s.add_node("N3", 10.0, 0.0)
    s.add_bar_element("E1", "N1", "N2", "S"); s.add_bar_element("E2", "N2", "N3", "S")
    s.add_support("PIN", ux=True, uy=True); s.add_support("ROLLER", uy=True)
    s.assign_support("N1", "PIN"); s.assign_support("N3", "ROLLER")
    s.add_load_case("LC"); s.add_distributed_load("E1", "LC", fye=-10.0, fyd=-10.0)
    return s


class TestPrecomputedReuse(unittest.TestCase):
    def test_precomputed_result_is_used_not_resolved(self):
        base = _beam()
        v = Variant(id="V1", model=make_variant_model(base))
        base.add_variant(v)
        vres = solve_variants(base, [Variant(id="BASE"), v],
                              base.support_sets)
        # Tamper with a solved value: if the combination re-solved the
        # variants, the tampered value could not survive into the output.
        marker = 12345.0
        vres["V1"].results["element_forces"]["LC"]["E1"]["i"][2] = marker
        out = run_variant_combination(
            base, [("BASE", "LC", 1.0), ("V1", "LC", 1.0)], "Envelope",
            precomputed=vres)
        self.assertEqual(out["element_forces"]["max"]["E1"]["i"][2], marker)

    def test_missing_precomputed_variants_are_solved(self):
        base = _beam()
        v1 = Variant(id="V1", model=make_variant_model(base))
        v2 = Variant(id="V2", model=make_variant_model(base))
        base.add_variant(v1); base.add_variant(v2)
        pre = solve_variants(base, [v1], base.support_sets)  # V2 not included
        out = run_variant_combination(
            base, [("V1", "LC", 1.0), ("V2", "LC", 1.0)], "LinearSum",
            precomputed=pre)
        self.assertIn("E1", out["element_forces"])

    def test_unrelated_precomputed_entries_are_ignored(self):
        base = _beam()
        v = Variant(id="V1", model=make_variant_model(base))
        base.add_variant(v)
        pre = solve_variants(base, [Variant(id="OTHER")], base.support_sets)
        out = run_variant_combination(base, [("V1", "LC", 1.0)], "LinearSum",
                                      precomputed=pre)
        self.assertIn("E1", out["element_forces"])


class TestValidityWithoutCopy(unittest.TestCase):
    def test_same_stiffness_for_identical_variants(self):
        base = _beam()
        base.add_variant(Variant(id="V1", model=make_variant_model(base)))
        info = combination_validity(base, ["BASE", "V1"], "LinearSum")
        self.assertTrue(info["same_stiffness"])
        self.assertTrue(info["same_geometry"])

    def test_different_stiffness_when_supports_differ(self):
        base = _beam()
        base.add_support_set(SupportSet(
            id="SS1", restraints={"N1": (True, True, True),
                                  "N3": (True, True, False)}))
        base.add_variant(Variant(
            id="V1", support_set_id="SS1",
            model=make_variant_model(base,
                                     support_set=base.support_sets["SS1"])))
        info = combination_validity(base, ["BASE", "V1"], "LinearSum")
        self.assertFalse(info["same_stiffness"])
        self.assertTrue(info["same_geometry"])   # same bars/nodes

    def test_different_geometry_when_active_set_differs(self):
        base = _beam()
        base.add_variant(Variant(
            id="V1", active_elements={"E1"},
            model=make_variant_model(base, active_elements={"E1"})))
        info = combination_validity(base, ["BASE", "V1"], "Envelope")
        self.assertFalse(info["same_geometry"])

    def test_validity_does_not_mutate_the_base(self):
        base = _beam()
        base.add_variant(Variant(id="V1", model=make_variant_model(base)))
        before = base.stiffness_hash()
        combination_validity(base, ["BASE", "V1"], "LinearSum")
        self.assertEqual(base.stiffness_hash(), before)
        self.assertEqual(sorted(base.bar_elements_by_id), ["E1", "E2"])

    def test_validity_syncs_stale_model_first(self):
        base = _beam()
        v = Variant(id="V1", model=make_variant_model(base))
        base.add_variant(v)
        base.add_node("N4", 15.0, 0.0)
        base.add_bar_element("E3", "N3", "N4", "S")
        info = combination_validity(base, ["BASE", "V1"], "LinearSum")
        # After the implicit sync the variant matches the edited base again.
        self.assertTrue(info["same_geometry"])
        self.assertIn("E3", v.model.bar_elements_by_id)


if __name__ == "__main__":
    unittest.main()
