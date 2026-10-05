"""The `create_*` convenience layer must be lossless.

dev/XDFEM2D_ENGINE.md's whole compatibility argument rests on one
invariant: a model built with `create_*` calls serialises to exactly the
same JSON as the same model built by hand with the underlying `add_*`
calls. If that ever stops being true, the "create_* is just permissive
sugar over add_*" story is broken and every claim in that document (old
files stay readable, nothing new is stored, etc.) stops holding.

This is checked structurally (dict equality on the serialised form from
xdfem2d.structure_io._to_dict), not by comparing ids/names — the two
models are built with matching explicit ids on the add_*/create_* side so
the auto-assigned ids on the create_* side land on the same names.
"""
from __future__ import annotations

import unittest

from context import Structure2D  # noqa: F401  (puts src/ on the path)

from xdfem2d.structure_io import _to_dict


class TestSectionsAndMaterialsAreLossless(unittest.TestCase):
    def test_create_rc_section_matches_add_rc_section(self):
        s_old = Structure2D()
        s_old.add_rc_section("B", 0.3, 0.5, concrete="C30/37", steel="B500B")

        s_new = Structure2D()
        s_new.create_rc_section("B", 0.3, 0.5, concrete="C30/37", steel="B500B")

        self.assertEqual(_to_dict(s_old), _to_dict(s_new))

    def test_create_steel_section_matches_add_steel_section(self):
        s_old = Structure2D()
        s_old.add_steel_section("St", "IPE300", grade="S275")

        s_new = Structure2D()
        s_new.create_steel_section("St", "IPE300", grade="S275")

        self.assertEqual(_to_dict(s_old), _to_dict(s_new))

    def test_create_bar_section_matches_add_section(self):
        """create_bar_section defaults shape to Rectangular (23/09/2026 --
        a plain b x h bar is one, and add_section's own bare default,
        Generic, was a display/maintenance trap: see
        rc_design._is_generic_concrete_section's docstring on why the
        design code never trusted shape=='Rectangular' in the first place).
        That is the one place this "lossless sugar over add_*" wrapper
        deliberately picks a smarter default than the call it wraps, so the
        hand-written side needs shape spelled out to still match."""
        s_old = Structure2D()
        s_old.add_material("M", 30e6, 25.0)
        s_old.add_section("S", "M", 0.3, 0.5, shape="Rectangular")

        s_new = Structure2D()
        s_new.add_material("M", 30e6, 25.0)
        s_new.create_bar_section("S", "M", 0.3, 0.5)

        self.assertEqual(_to_dict(s_old), _to_dict(s_new))

    def test_create_timber_bar_section_matches_manual_two_step(self):
        """No add_timber_section ever existed to compare against — the
        invariant here is that the one-call wrapper matches doing the same
        two steps (create_timber_material + create_bar_section) by hand."""
        s_old = Structure2D()
        mat = s_old.create_timber_material(timber="C24")
        s_old.create_bar_section("B", mat, 0.15, 0.30, shape="Rectangular",
                                 timber_service_class="SC1")

        s_new = Structure2D()
        s_new.create_timber_bar_section("B", 0.15, 0.30, timber="C24",
                                        service_class="SC1")

        self.assertEqual(_to_dict(s_old), _to_dict(s_new))

    def test_create_timber_bar_section_dedupes_material_across_sections(self):
        s = Structure2D()
        s.create_timber_bar_section("B1", 0.15, 0.30, timber="C24")
        s.create_timber_bar_section("B2", 0.10, 0.20, timber="C24")
        timber_materials = [m for m in s.materials.values()
                            if m.material_type.value == "Timber"]
        self.assertEqual(len(timber_materials), 1)

    def test_create_bar_section_auto_creates_material_from_eurocode_class(self):
        """create_bar_section("S", "C25/30", ...) with no such material yet
        must produce exactly the material+section a hand-written
        add_material(...) + add_section(...) pair would — same values, same
        derived material name (the one create_rc_material/_rc_material use)."""
        s_old = Structure2D()
        s_old.create_rc_material(concrete="C25/30", steel="B500B")
        mat_name = next(iter(s_old.materials))
        s_old.add_section("S", mat_name, 0.3, 0.5, shape="Rectangular")

        s_new = Structure2D()
        s_new.create_bar_section("S", "C25/30", 0.3, 0.5)

        self.assertEqual(_to_dict(s_old), _to_dict(s_new))

    def test_create_area_section_matches_add_tri_section_in_plane(self):
        s_old = Structure2D(domain="plane")
        s_old.add_material("M", 30e6, 25.0)
        s_old.add_tri_section("W", "M", thickness=0.2, formulation="CST")
        # create_area_section always pairs a QuadSection alongside the
        # TriSection now (add_area_section) — matching that pairing here,
        # not just the triangle half.
        s_old.add_quad_section("W", "M", thickness=0.2, formulation="QM6")

        s_new = Structure2D(domain="plane")
        s_new.add_material("M", 30e6, 25.0)
        s_new.create_area_section("W", "M", thickness=0.2)

        self.assertEqual(_to_dict(s_old), _to_dict(s_new))

    def test_create_area_section_matches_add_plate_section_in_plate(self):
        s_old = Structure2D(domain="plate")
        s_old.add_material("M", 30e6, 25.0)
        s_old.add_plate_section("W", "M", thickness=0.2)   # MITC3 default
        # Same pairing as the plane-domain test above, plate defaults.
        s_old.add_quad_section("W", "M", thickness=0.2, formulation="MITC4")

        s_new = Structure2D(domain="plate")
        s_new.add_material("M", 30e6, 25.0)
        s_new.create_area_section("W", "M", thickness=0.2)

        self.assertEqual(_to_dict(s_old), _to_dict(s_new))


class TestNodesAndElementsAreLossless(unittest.TestCase):
    def test_create_node_auto_id_is_N_prefixed(self):
        """create_node's auto id used to be a plain integer-as-string
        ("1", "2", ...) — the odd one out among the create_* layer, whose
        other auto-naming (create_bar_element -> "B1", create_load_case ->
        "LC1", ...) all go through auto_name(). It now does too."""
        s = Structure2D()
        n1 = s.create_node(0.0, 0.0)
        n2 = s.create_node(5.0, 0.0)
        self.assertEqual(n1.id, "N1")
        self.assertEqual(n2.id, "N2")

    def test_create_node_matches_add_node_with_matching_id(self):
        s_old = Structure2D()
        s_old.add_node("N1", 0.0, 0.0)
        s_old.add_node("N2", 5.0, 0.0)

        s_new = Structure2D()
        s_new.create_node(0.0, 0.0)   # auto id "N1"
        s_new.create_node(5.0, 0.0)   # auto id "N2"

        self.assertEqual(_to_dict(s_old), _to_dict(s_new))

    def test_create_node_list_matches_add_node_calls(self):
        s_old = Structure2D()
        s_old.add_node("N1", 0.0, 0.0)
        s_old.add_node("N2", 5.0, 0.0)
        s_old.add_node("N3", 5.0, 3.0)

        s_new = Structure2D()
        s_new.create_node_list([0, 0, 5, 0, 5, 3])

        self.assertEqual(_to_dict(s_old), _to_dict(s_new))

    def test_create_bar_element_matches_add_bar_element(self):
        s_old = Structure2D()
        s_old.add_node("N1", 0.0, 0.0)
        s_old.add_node("N2", 5.0, 0.0)
        s_old.add_material("M", 30e6, 25.0)
        s_old.add_section("S", "M", 0.3, 0.5, shape="Rectangular")
        s_old.add_bar_element("B1", "N1", "N2", "S")

        s_new = Structure2D()
        n1 = s_new.create_node(0.0, 0.0)
        n2 = s_new.create_node(5.0, 0.0)
        s_new.add_material("M", 30e6, 25.0)
        sec = s_new.create_bar_section("S", "M", 0.3, 0.5)
        s_new.create_bar_element(sec, n1, n2)   # auto id "B1", Node/Section objects

        self.assertEqual(_to_dict(s_old), _to_dict(s_new))

    def test_create_area_element_matches_add_tri_element(self):
        s_old = Structure2D()
        for i, (x, y) in enumerate([(0, 0), (4, 0), (4, 3)], start=1):
            s_old.add_node(f"N{i}", x, y)
        s_old.add_material("M", 30e6, 25.0)
        s_old.add_tri_section("W", "M", thickness=0.2, formulation="CST")
        # create_area_section (used on the s_new side via create_area_element's
        # sec) always pairs a QuadSection too — match it here as well.
        s_old.add_quad_section("W", "M", thickness=0.2, formulation="QM6")
        s_old.add_tri_element("T1", "N1", "N2", "N3", "W")

        s_new = Structure2D()
        nodes = s_new.create_node_list([0, 0, 4, 0, 4, 3])
        s_new.add_material("M", 30e6, 25.0)
        sec = s_new.create_area_section("W", "M", thickness=0.2)
        s_new.create_area_element(sec, *nodes)   # auto id "T1"

        self.assertEqual(_to_dict(s_old), _to_dict(s_new))


class TestSupportsWidenedArgumentsAreLossless(unittest.TestCase):
    def test_pin_on_a_list_matches_two_single_pin_calls(self):
        s_old = Structure2D()
        s_old.add_node("1", 0.0, 0.0)
        s_old.add_node("2", 5.0, 0.0)
        s_old.pin("1")
        s_old.pin("2")

        s_new = Structure2D()
        s_new.add_node("1", 0.0, 0.0)
        s_new.add_node("2", 5.0, 0.0)
        s_new.pin(["1", "2"])

        self.assertEqual(_to_dict(s_old), _to_dict(s_new))

    def test_pin_on_a_node_object_matches_pin_on_its_id(self):
        s_old = Structure2D()
        s_old.add_node("N1", 0.0, 0.0)
        s_old.pin("N1")

        s_new = Structure2D()
        n = s_new.create_node(0.0, 0.0)
        s_new.pin(n)

        self.assertEqual(_to_dict(s_old), _to_dict(s_new))

    def test_pin_single_ref_still_returns_a_scalar_not_a_list(self):
        s = Structure2D()
        n = s.create_node(0.0, 0.0)
        result = s.pin(n)
        self.assertFalse(isinstance(result, list))

    def test_pin_list_ref_returns_a_list(self):
        s = Structure2D()
        n1 = s.create_node(0.0, 0.0)
        n2 = s.create_node(5.0, 0.0)
        result = s.pin([n1, n2])
        self.assertTrue(isinstance(result, list))
        self.assertEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main()
