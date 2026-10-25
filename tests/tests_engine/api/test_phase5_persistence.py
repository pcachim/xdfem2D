"""Phase-5 backbone tests: persistence round-trip of variants/phasing overlays
and the headless workflow controller. No Qt is involved."""
import os
import tempfile
import unittest

from context import Structure2D, assert_close
from xdfem2d.models import (
    SupportSet, SupportAssignment, Variant,
    ConstructionPhase, ConstructionSequence, ElementInitialState,
)
from xdfem2d.workflows import (
    run_variant_combination, combination_validity, import_file_as_variant,
    run_sequence, list_overlays,
)


def _beam():
    s = Structure2D()
    s.add_material("M", elastic_modulus=30e6, unit_weight=0.0)
    s.add_section("S", "M", b=0.3, h=0.6)
    s.add_node("N1", 0.0, 0.0)
    s.add_node("N2", 5.0, 0.0)
    s.add_node("N3", 10.0, 0.0)
    s.add_bar_element("E1", "N1", "N2", "S")
    s.add_bar_element("E2", "N2", "N3", "S")
    s.add_support("PIN", ux=True, uy=True)
    s.add_support("ROLLER", uy=True)
    s.assign_support("N1", "PIN")
    s.assign_support("N3", "ROLLER")
    s.add_load_case("LC")
    s.add_distributed_load("E1", "LC", fye=-10.0, fyd=-10.0)
    return s


class TestRoundTrip(unittest.TestCase):
    def _save_load(self, s):
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            s.save(path)
            return Structure2D.load(path)
        finally:
            os.remove(path)

    def test_legacy_model_unchanged(self):
        s = _beam()
        r = self._save_load(s)
        self.assertEqual(r.support_sets, {})
        self.assertEqual(r.variants, {})
        self.assertEqual(r.construction_sequences, {})

    def test_support_set_and_variant_round_trip(self):
        s = _beam()
        s.add_support_set(SupportSet(
            id="ALT", assignments=[SupportAssignment("N1", "ROLLER"),
                                   SupportAssignment("N3", "PIN")]))
        s.add_variant(Variant(id="V1", support_set_id="ALT",
                              active_elements={"E1", "E2"}))
        r = self._save_load(s)
        self.assertIn("ALT", r.support_sets)
        self.assertEqual(len(r.support_sets["ALT"].assignments), 2)
        self.assertEqual(r.variants["V1"].support_set_id, "ALT")
        self.assertEqual(r.variants["V1"].active_elements, {"E1", "E2"})

    def test_sequence_round_trip(self):
        s = _beam()
        ph = ConstructionPhase(
            id="P1", active_elements={"E1"}, applied_cases=["LC"],
            initial_state={"E1": ElementInitialState(i=(1.0, 2.0, 3.0),
                                                     j=(-1.0, -2.0, -3.0))})
        s.add_construction_sequence(ConstructionSequence(
            id="SEQ", phases=[ph], displacement_method="cumulative"))
        r = self._save_load(s)
        seq = r.construction_sequences["SEQ"]
        self.assertEqual(seq.displacement_method, "cumulative")
        self.assertEqual(seq.phases[0].active_elements, {"E1"})
        self.assertEqual(seq.phases[0].initial_state["E1"].i, (1.0, 2.0, 3.0))


class TestWorkflows(unittest.TestCase):
    def test_run_variant_combination_with_base(self):
        s = _beam()
        s.add_variant(Variant(id="V"))           # full model
        out = run_variant_combination(
            s, [("BASE", "LC", 1.0), ("V", "LC", 1.0)], op="LinearSum")
        ref = s.calculate()
        for end in ("i", "j"):
            for k in range(3):
                assert_close(self, out["element_forces"]["E1"][end][k],
                             2.0 * ref["element_forces"]["LC"]["E1"][end][k],
                             rel=1e-9, abs_tol=1e-6)

    def test_combination_validity_warns(self):
        s = _beam()
        s.add_support_set(SupportSet(
            id="ALT", assignments=[SupportAssignment("N1", "ROLLER"),
                                   SupportAssignment("N3", "PIN")]))
        s.add_variant(Variant(id="BASE"))
        s.add_variant(Variant(id="ALT", support_set_id="ALT"))
        info = combination_validity(s, ["BASE", "ALT"], op="LinearSum")
        # Bar forces are always combinable now; the warning only flags that
        # displacements/reactions are not meaningful for different stiffness.
        self.assertTrue(info["linear_ok"])
        self.assertFalse(info["same_stiffness"])
        self.assertTrue(info["same_geometry"])

    def test_import_registers_overlay(self):
        base = _beam()
        other = _beam()
        unified, variant, report = import_file_as_variant(base, other, "IMP")
        self.assertIn("IMP", unified.variants)
        self.assertIn("IMP", unified.support_sets)
        self.assertIn("IMP:LC", unified.load_cases_by_id)

    def test_run_sequence_and_list(self):
        s = _beam()
        ph = ConstructionPhase(id="P", active_elements={"E1", "E2"},
                               applied_cases=["LC"])
        s.add_construction_sequence(ConstructionSequence(id="SEQ", phases=[ph]))
        out = run_sequence(s, "SEQ")
        self.assertIn("E1", out["final"]["element_forces"])
        summary = list_overlays(s)
        self.assertEqual(summary["sequences"][0]["id"], "SEQ")


if __name__ == "__main__":
    unittest.main()
