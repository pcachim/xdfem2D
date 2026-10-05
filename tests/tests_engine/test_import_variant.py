"""Tests for the Phase-2 import / node-welding foundation (xdfem2d.import_io)."""
import unittest

from context import Structure2D, assert_close
from xdfem2d.import_io import (
    import_model, import_as_variant, weld_nodes, reconcile_definitions,
    map_elements,
)
from xdfem2d.variants import solve_variants, combine_across_variants, CombTerm


def _beam(node_ids=("N1", "N2", "N3"), x0=0.0):
    """Two-element beam; node ids parametrised to simulate independent files."""
    s = Structure2D()
    s.add_material("M", elastic_modulus=30e6, unit_weight=0.0)
    s.add_section("S", "M", b=0.3, h=0.6)
    xs = [x0, x0 + 5.0, x0 + 10.0]
    for nid, x in zip(node_ids, xs):
        s.add_node(nid, x, 0.0)
    s.add_bar_element("E1", node_ids[0], node_ids[1], "S")
    s.add_bar_element("E2", node_ids[1], node_ids[2], "S")
    s.add_support("PIN", ux=True, uy=True)
    s.add_support("ROLLER", ux=False, uy=True)
    s.assign_support(node_ids[0], "PIN")
    s.assign_support(node_ids[2], "ROLLER")
    s.add_load_case("LC")
    s.add_distributed_load("E1", "LC", fye=-10.0, fyd=-10.0)
    return s


class TestWeldIdenticalGeometry(unittest.TestCase):
    """Same geometry, different node ids → welded to a single id space."""

    def setUp(self):
        self.base = _beam(("N1", "N2", "N3"))
        self.other = _beam(("A", "B", "C"))   # same coords, different ids
        self.unified, self.overlay, self.sset, self.report = import_model(
            self.base, self.other, overlay_id="IMP")

    def test_all_nodes_welded(self):
        # No new nodes: every 'other' node coincides with a base node.
        self.assertEqual(self.report.added_nodes, [])
        self.assertEqual(len(self.report.welded), 3)

    def test_no_new_elements(self):
        self.assertEqual(self.report.new_elements, [])

    def test_unified_node_count(self):
        self.assertEqual(len(self.unified.nodes), 3)

    def test_purity(self):
        self.assertEqual(len(self.base.nodes), 3)
        self.assertEqual(len(self.other.nodes), 3)
        self.assertNotIn("IMP:LC", self.base.load_cases_by_id)

    def test_same_stiffness_enables_exact_sum(self):
        # Imported variant has the same active geometry+supports as base ⇒ same
        # hash ⇒ LinearSum across base and import is exact (ties to Phase 1).
        from xdfem2d.models import Variant
        base_v = Variant(id="BASE")
        vres = solve_variants(self.unified, [base_v, self.overlay],
                              support_sets={"IMP": self.sset})
        self.assertEqual(vres["BASE"].stiffness_hash,
                         vres["IMP"].stiffness_hash)
        out = combine_across_variants(
            vres, [CombTerm("BASE", "LC", 1.0), CombTerm("IMP", "IMP:LC", 1.0)],
            op="LinearSum")
        # base LC and imported IMP:LC are identical loads → sum == 2× base LC.
        ref = self.unified.calculate()
        for end in ("i", "j"):
            for k in range(3):
                assert_close(self, out["element_forces"]["E1"][end][k],
                             2.0 * ref["element_forces"]["LC"]["E1"][end][k],
                             rel=1e-9, abs_tol=1e-6)


class TestImportExtend(unittest.TestCase):
    """import_extend grows the base (no variant); welds + merges loads by name."""

    def test_extends_and_merges(self):
        from xdfem2d.import_io import import_extend
        base = _beam(("N1", "N2", "N3"))            # E1,E2 ; loads on E1(LC)
        # other: one extra bar beyond N3, sharing N3, with its own load on LC.
        other = Structure2D()
        other.add_material("M", elastic_modulus=30e6, unit_weight=0.0)
        other.add_section("S", "M", b=0.3, h=0.6)
        other.add_node("P1", 10.0, 0.0)             # coincides with base N3
        other.add_node("P2", 15.0, 0.0)             # new
        other.add_bar_element("EX", "P1", "P2", "S")
        other.add_load_case("LC")
        other.add_distributed_load("EX", "LC", fye=-4.0, fyd=-4.0)
        unified, report = import_extend(base, other)
        # Geometry extended; no variant created.
        self.assertEqual(report.new_elements, ["EX"])
        self.assertEqual(sorted(e.id for e in unified.bar_elements), ["E1", "E2", "EX"])
        self.assertEqual(unified.bar_elements_by_id["EX"].node_i, "N3")   # welded
        self.assertEqual(unified.variants, {})
        # Same-named load case merged: LC now carries base + imported loads.
        lc_loads = [dl.element_id for dl in unified.distributed_loads
                    if dl.load_case_id == "LC"]
        self.assertIn("E1", lc_loads)
        self.assertIn("EX", lc_loads)
        # The unified model solves as one structure.
        unified.add_support("R", uy=True); unified.assign_support("P2", "R")
        res = unified.calculate()
        self.assertIn("EX", res["element_forces"]["LC"])

    def test_base_unchanged(self):
        from xdfem2d.import_io import import_extend
        base = _beam()
        other = _beam(("A", "B", "C"))
        n = len(base.bar_elements)
        import_extend(base, other)
        self.assertEqual(len(base.bar_elements), n)   # base not mutated


class TestExtendedGeometry(unittest.TestCase):
    """other shares the first node but extends the geometry."""

    def setUp(self):
        self.base = _beam(("N1", "N2", "N3"))
        # other: a single bar starting at the base's last node (x=10) going to x=15
        self.other = Structure2D()
        self.other.add_material("M", elastic_modulus=30e6, unit_weight=0.0)
        self.other.add_section("S", "M", b=0.3, h=0.6)
        self.other.add_node("P1", 10.0, 0.0)   # coincides with base N3
        self.other.add_node("P2", 15.0, 0.0)   # new
        self.other.add_bar_element("EX", "P1", "P2", "S")
        self.unified, self.overlay, self.sset, self.report = import_model(
            self.base, self.other, overlay_id="EXT")

    def test_shared_node_welded(self):
        self.assertEqual(len(self.report.welded), 1)        # P1 ↔ N3
        self.assertEqual(self.report.added_nodes, ["P2"])

    def test_new_element_added(self):
        self.assertEqual(self.report.new_elements, ["EX"])
        self.assertEqual(len(self.unified.bar_elements), 3)

    def test_connectivity_continuous(self):
        ex = self.unified.bar_elements_by_id["EX"]
        self.assertEqual(ex.node_i, "N3")     # welded onto base's id
        self.assertEqual(ex.node_j, "P2")


class TestConflicts(unittest.TestCase):
    def test_section_conflict_makes_distinct_element(self):
        base = _beam()
        other = _beam(("A", "B", "C"))
        other.sections["S"].h = 0.9          # same name, different props
        unified, overlay, sset, report = import_model(base, other, "IMP")
        self.assertIn("S", report.renamed_defs)
        # Imported bars use the renamed section and are distinct elements.
        self.assertEqual(len(report.new_elements), 2)

    def test_material_conflict_renamed(self):
        base = _beam()
        other = _beam(("A", "B", "C"))
        other.materials["M"].elastic_modulus = 99e6
        unified, overlay, sset, report = import_model(base, other, "IMP")
        self.assertIn("M", report.renamed_defs)


class TestNameMode(unittest.TestCase):
    def test_name_mode_assumes_shared_ids(self):
        base = _beam(("N1", "N2", "N3"))
        other = _beam(("N1", "N2", "N3"))    # already-coincident ids
        unified, overlay, sset, report = import_model(
            base, other, "IMP", weld="name")
        self.assertEqual(len(unified.nodes), 3)
        self.assertEqual(report.new_elements, [])


class TestAmbiguity(unittest.TestCase):
    def test_two_base_nodes_within_tol_flagged(self):
        base = Structure2D()
        base.add_material("M", 30e6, 0.0)
        base.add_section("S", "M", 0.3, 0.6)
        base.add_node("N1", 0.0, 0.0)
        base.add_node("N2", 0.0005, 0.0)     # very close to N1
        other = Structure2D()
        other.add_node("X", 0.0, 0.0)
        nmap, bits = weld_nodes(base, other, tol=0.001)
        self.assertTrue(bits['ambiguities'])


class TestAliasPurity(unittest.TestCase):
    def test_alias_and_purity(self):
        base = _beam()
        other = _beam(("A", "B", "C"))
        n_base = len(base.nodes)
        unified, overlay, sset, report = import_as_variant(base, other, "IMP")
        self.assertEqual(len(base.nodes), n_base)          # base untouched
        self.assertEqual(overlay.support_set_id, "IMP")


if __name__ == "__main__":
    unittest.main()
