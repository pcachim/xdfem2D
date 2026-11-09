"""Variants choose triangles, quads and geometry objects like bars, and the ids
of new entities are unique across the base and every variant's model."""
import unittest

from context import Structure2D
from xdfem2d.workflows import (new_variant, sync_variant_with_base,
                               variant_base_diff, variant_id_collisions,
                               area_selection)


def _plate():
    s = Structure2D()
    s.add_material("M", elastic_modulus=30e6, unit_weight=0.0)
    s.add_section("S", "M", b=0.3, h=0.6)
    s.add_tri_section("TS", "M", thickness=0.2)
    for nid, x, y in (("N1", 0, 0), ("N2", 4, 0), ("N3", 4, 3), ("N4", 0, 3),
                      ("N5", 8, 0), ("N6", 8, 3)):
        s.add_node(nid, x, y)
    s.add_bar_element("E1", "N1", "N2", "S")
    s.add_tri_element("T1", "N1", "N2", "N3", "TS")
    s.add_tri_element("T2", "N1", "N3", "N4", "TS")
    s.add_tri_element("T3", "N2", "N5", "N6", "TS")
    return s


def _variant(base, active=None, vid="V1"):
    v = new_variant(base, vid, active_elements=active)
    base.add_variant(v)
    return v


class TestFamilyIds(unittest.TestCase):
    def test_new_ids_avoid_the_base_and_other_variants(self):
        base = _plate()
        v1 = _variant(base, vid="V1")
        v2 = _variant(base, vid="V2")
        a = v1.model.create_node(1, 1)
        b = v2.model.create_node(2, 2)
        c = base.create_node(3, 3)
        self.assertEqual(len({a.id, b.id, c.id}), 3)
        self.assertNotIn(a.id, base.nodes)

    def test_bar_ids_too(self):
        base = _plate()
        v = _variant(base)
        n = v.model.create_node(5, 5)
        e = v.model.create_bar_element("S", "N2", n)
        e2 = base.create_bar_element("S", "N3", "N4")
        self.assertNotEqual(e.id, e2.id)

    def test_a_model_with_no_variants_numbers_as_before(self):
        s = _plate()
        self.assertEqual(s.create_node(9, 9).id, "N7")

    def test_copy_of_a_variant_model_does_not_copy_the_base(self):
        import copy
        base = _plate()
        v = _variant(base)
        sub = base.derive_variant(v)
        self.assertIs(sub._family_base, v.model._family_base)
        self.assertIsNot(copy.deepcopy(v.model)._family_base(), None)


class TestIdCollisions(unittest.TestCase):
    def test_no_collision_for_shared_ids(self):
        base = _plate(); v = _variant(base)
        self.assertEqual(variant_id_collisions(base, v), [])

    def test_a_base_id_taken_by_a_variant_own_bar(self):
        base = _plate(); v = _variant(base)
        v.model.add_node("N9", 4, 6)
        v.model.add_bar_element("X1", "N3", "N9", "S")
        sync_variant_with_base(base, v, force=True)
        base.add_node("N8", 0, 6)
        base.add_bar_element("X1", "N4", "N8", "S")
        self.assertIn(("bars", "X1"), variant_base_diff(base, v)["id_collisions"])


class TestAreaSelection(unittest.TestCase):
    def test_no_set_keeps_every_area_and_bars_only_keeps_none(self):
        base = _plate()
        self.assertIsNone(area_selection(base, None))
        self.assertEqual(area_selection(base, {"E1"}), (set(), set(), set()))

    def test_named_areas_are_chosen(self):
        base = _plate()
        t, q, o = area_selection(base, {"E1", "T1", "T3"})
        self.assertEqual((t, q, o), ({"T1", "T3"}, set(), set()))

    def test_new_variant_with_chosen_triangles(self):
        base = _plate()
        v = _variant(base, active={"E1", "T1"})
        self.assertEqual(sorted(v.model.tri_elements_by_id), ["T1"])
        self.assertEqual([e.id for e in v.model.bar_elements], ["E1"])

    def test_variant_with_bars_only_has_no_triangle(self):
        base = _plate()
        v = _variant(base, active={"E1"})
        self.assertEqual(sorted(v.model.tri_elements_by_id), [])

    def test_derived_without_model_also_filters(self):
        from xdfem2d.models import Variant
        base = _plate()
        sub = base.derive_variant(Variant(id="W", active_elements={"E1", "T2"}))
        self.assertEqual(sorted(sub.tri_elements_by_id), ["T2"])


class TestAreaSync(unittest.TestCase):
    def test_base_triangle_edit_reaches_untouched_variant(self):
        base = _plate(); v = _variant(base)
        base.tri_elements_by_id["T1"].section_name = "TS2"
        base.add_tri_section("TS2", "M", thickness=0.3)
        sync_variant_with_base(base, v, force=True)
        self.assertEqual(v.model.tri_elements_by_id["T1"].section_name, "TS2")
        self.assertIn("TS2", v.model.tri_sections)

    def test_variant_edit_of_a_triangle_stays(self):
        base = _plate(); v = _variant(base)
        v.model.add_tri_section("TS3", "M", thickness=0.4)
        v.model.tri_elements_by_id["T2"].section_name = "TS3"
        base.tri_elements_by_id["T1"].section_name = "TS"   # unrelated
        sync_variant_with_base(base, v, force=True)
        self.assertEqual(v.model.tri_elements_by_id["T2"].section_name, "TS3")

    def test_new_base_triangle_reaches_a_variant_that_keeps_all(self):
        base = _plate(); v = _variant(base)
        base.add_tri_element("T4", "N3", "N4", "N1", "TS")
        sync_variant_with_base(base, v, force=True)
        self.assertIn("T4", v.model.tri_elements_by_id)

    def test_new_base_triangle_does_not_reach_an_explicit_choice(self):
        base = _plate(); v = _variant(base, active={"E1", "T1"})
        base.add_tri_element("T4", "N3", "N4", "N1", "TS")
        sync_variant_with_base(base, v, force=True)
        self.assertNotIn("T4", v.model.tri_elements_by_id)

    def test_explicit_sync_prunes_areas_not_chosen(self):
        base = _plate(); v = _variant(base)
        v.active_elements = {"E1", "T1"}
        sync_variant_with_base(base, v, force=True, destructive=True)
        self.assertEqual(sorted(v.model.tri_elements_by_id), ["T1"])

    def test_diff_lists_areas(self):
        base = _plate(); v = _variant(base, active={"E1", "T1"})
        v.active_elements = {"E1", "T1", "T3"}
        d = variant_base_diff(base, v)
        self.assertEqual(d["add"], ["T3"])


class TestObjects(unittest.TestCase):
    def _with_object(self):
        base = _plate()
        base.add_geo_rectangle("R1", (10, 0), (12, 2), section_name="TS",
                               target_size=1.0)
        return base

    def test_object_follows_base_and_keeps_variant_edit(self):
        base = self._with_object(); v = _variant(base)
        self.assertIn("R1", v.model.geometry_objects)
        v.model.geometry_objects["R1"].target_size = 0.25
        base.geometry_objects["R1"].prefer_quad = False
        sync_variant_with_base(base, v, force=True)
        r = v.model.geometry_objects["R1"]
        self.assertEqual(r.target_size, 0.25)          # the variant's own
        self.assertFalse(r.prefer_quad)                # the base's change

    def test_object_chosen_out(self):
        base = self._with_object()
        v = _variant(base, active={"E1", "T1", "T2", "T3"})
        self.assertNotIn("R1", v.model.geometry_objects)
        self.assertEqual(sorted(v.model.nodes), ["N1", "N2", "N3", "N4", "N5", "N6"])


if __name__ == "__main__":
    unittest.main()
