"""dev/IMPLEMENT_QUAD.md Phase 8, GUI item 2: rename_quad_section and
rename_quad_element — the quad counterparts of rename_tri_section /
rename_tri_element. Originally added because the (now-removed, see item 2b's
"Panel sections" unification) standalone QuadSectionPanel's edit flow needed
rename_quad_section; that job now belongs to the unified TriSectionPanel,
which calls rename_quad_section alongside rename_tri_section whenever a
paired "Panel" is renamed. QuadElementEditDialog's rename-on-name-change
still needs rename_quad_element directly (mirroring how
TriElementEditDialog uses rename_tri_element).
"""
import context  # noqa: F401
import pytest

from xdfem2d import Structure2D


def _quad_model():
    s = Structure2D()
    s.add_material("C", elastic_modulus=30e6, unit_weight=0.0)
    s.add_quad_section("QS1", "C", thickness=0.2, formulation="Q4")
    s.add_node("A", 0.0, 0.0); s.add_node("B", 1.0, 0.0)
    s.add_node("C", 1.0, 1.0); s.add_node("D", 0.0, 1.0)
    s.add_quad_element("Q1", "A", "B", "C", "D", "QS1")
    return s


class TestRenameQuadSection:

    def test_rename_updates_the_dict_key_and_referencing_quads(self):
        s = _quad_model()
        s.rename_quad_section("QS1", "Thick")
        assert "Thick" in s.quad_sections
        assert "QS1" not in s.quad_sections
        assert s.quad_sections["Thick"].name == "Thick"
        assert s.quad_elements_by_id["Q1"].section_name == "Thick"

    def test_rename_to_an_existing_name_is_a_no_op(self):
        s = _quad_model()
        s.add_quad_section("QS2", "C", thickness=0.3, formulation="Q4")
        s.rename_quad_section("QS1", "QS2")
        # Unlike rename_quad_element (which raises), rename_quad_section
        # mirrors rename_tri_section's silent no-op on a name collision.
        assert "QS1" in s.quad_sections
        assert s.quad_sections["QS2"].thickness == 0.3

    def test_rename_of_an_unknown_section_is_a_no_op(self):
        s = _quad_model()
        s.rename_quad_section("nope", "still_nope")
        assert "still_nope" not in s.quad_sections


class TestRenameQuadElement:

    def test_rename_updates_the_dict_key_and_the_elements_own_id(self):
        s = _quad_model()
        s.rename_quad_element("Q1", "Q2")
        assert "Q2" in s.quad_elements_by_id
        assert "Q1" not in s.quad_elements_by_id
        assert s.quad_elements_by_id["Q2"].id == "Q2"
        # The list and the dict share the same object (Phase 3 invariant,
        # same as tri_elements/tri_elements_by_id) — no stale duplicate.
        assert [q.id for q in s.quad_elements] == ["Q2"]

    def test_rename_to_an_existing_id_raises(self):
        s = _quad_model()
        s.add_node("E", 2.0, 0.0); s.add_node("F", 2.0, 1.0)
        s.add_quad_element("Q2", "B", "E", "F", "C", "QS1")
        with pytest.raises(ValueError):
            s.rename_quad_element("Q1", "Q2")

    def test_rename_of_a_missing_quad_raises(self):
        s = _quad_model()
        with pytest.raises(ValueError):
            s.rename_quad_element("nope", "still_nope")

    def test_rename_is_a_no_op_when_old_equals_new(self):
        s = _quad_model()
        s.rename_quad_element("Q1", "Q1")   # must not raise
        assert "Q1" in s.quad_elements_by_id

    def test_temperature_loads_are_carried_along(self):
        s = _quad_model()
        s.add_load_case("LC1")
        s.add_quad_temperature_load("Q1", "LC1", dt_i=5.0, dt_j=5.0,
                                    dt_k=5.0, dt_l=5.0)
        s.rename_quad_element("Q1", "Q2")
        assert len(s.quad_temperature_loads) == 1
        assert s.quad_temperature_loads[0].quad_id == "Q2"
