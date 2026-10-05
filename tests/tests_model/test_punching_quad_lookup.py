"""dev/IMPLEMENT_QUAD.md Phase 7: punching._slab_section_at / _rho_l_at
extended to quads — pure Python, no eurocodepy dependency, unlike the design
functions themselves (see test_quad_reinforcement_and_punching.py for those).
"""
import context  # noqa: F401

from xdfem2d import Structure2D
from xdfem2d.punching import _slab_section_at, _rho_l_at


def _plate_material(s):
    s.add_material("C", elastic_modulus=33e6, unit_weight=0.0, poisson=0.2)


class TestSlabSectionAtFindsQuads:

    def test_a_lone_quad_touching_the_node_is_found(self):
        s = Structure2D(domain="plate")
        _plate_material(s)
        s.add_quad_section("QS", "C", thickness=0.2, formulation="MITC4")
        s.add_node("A", 0, 0); s.add_node("B", 1, 0)
        s.add_node("C", 1, 1); s.add_node("D", 0, 1)
        s.add_quad_element("Q1", "A", "B", "C", "D", "QS")
        sec = _slab_section_at(s, "C")
        assert sec is not None and sec.name == "QS"

    def test_a_membrane_quad_q4_is_not_a_slab_section(self):
        s = Structure2D(domain="plate")
        _plate_material(s)
        s.add_quad_section("QS", "C", thickness=0.2, formulation="Q4")
        s.add_node("A", 0, 0); s.add_node("B", 1, 0)
        s.add_node("C", 1, 1); s.add_node("D", 0, 1)
        s.add_quad_element("Q1", "A", "B", "C", "D", "QS")
        assert _slab_section_at(s, "C") is None

    def test_mixed_tri_and_quad_agreeing_on_section_is_unambiguous(self):
        s = Structure2D(domain="plate")
        _plate_material(s)
        s.add_plate_section("S", "C", thickness=0.2, formulation="DKT")
        s.add_node("A", 0, 0); s.add_node("B", 1, 0)
        s.add_node("C", 1, 1); s.add_node("D", 0, 1); s.add_node("E", 2, 0.5)
        s.add_tri_element("T1", "B", "E", "C", "S")
        s.add_quad_element("Q1", "A", "B", "C", "D", "S")
        sec = _slab_section_at(s, "B")   # touched by both T1 and Q1
        assert sec is not None and sec.name == "S"

    def test_conflicting_sections_pick_the_majority(self):
        """Node N is touched by two DKT4 quads on section 'Thick' and one
        DKT triangle on section 'Thin' — the majority (Thick) wins,
        deterministically."""
        s = Structure2D(domain="plate")
        _plate_material(s)
        s.add_plate_section("Thin", "C", thickness=0.15, formulation="DKT")
        s.add_quad_section("Thick", "C", thickness=0.30, formulation="DKT4")
        s.add_node("N", 0, 0)
        s.add_node("A1", 1, 0); s.add_node("A2", 1, 1); s.add_node("A3", 0, 1)
        s.add_node("B1", -1, 0); s.add_node("B2", -1, -1); s.add_node("B3", 0, -1)
        s.add_node("T1", 0, -2)
        s.add_quad_element("Q1", "N", "A1", "A2", "A3", "Thick")
        s.add_quad_element("Q2", "N", "B1", "B2", "B3", "Thick")
        s.add_tri_element("T1e", "N", "B1", "T1", "Thin")
        sec = _slab_section_at(s, "N")
        assert sec is not None and sec.name == "Thick"

    def test_no_touching_plate_element_returns_none(self):
        s = Structure2D(domain="plate")
        _plate_material(s)
        assert _slab_section_at(s, "nowhere") is None


class TestRhoLAtIncludesQuadRows:

    def test_quad_row_contributes_to_the_average(self):
        s = Structure2D(domain="plate")
        _plate_material(s)
        s.add_quad_section("QS", "C", thickness=0.2, formulation="MITC4")
        s.add_node("A", 0, 0); s.add_node("B", 1, 0)
        s.add_node("C", 1, 1); s.add_node("D", 0, 1)
        s.add_quad_element("Q1", "A", "B", "C", "D", "QS")
        results = {"tri_reinforcement": {"rows": [
            {"triangle": "Q1", "Asx_bot": 0.001, "Asy_bot": 0.002},
        ]}}
        rho = _rho_l_at(s, results, "A", d_m=0.18)
        expected = (0.5 * (0.001 + 0.002)) / 0.18
        assert abs(rho - min(expected, 0.02)) < 1e-9

    def test_a_node_not_on_the_quad_falls_back_to_default(self):
        s = Structure2D(domain="plate")
        _plate_material(s)
        s.add_quad_section("QS", "C", thickness=0.2, formulation="MITC4")
        s.add_node("A", 0, 0); s.add_node("B", 1, 0)
        s.add_node("C", 1, 1); s.add_node("D", 0, 1)
        s.add_quad_element("Q1", "A", "B", "C", "D", "QS")
        results = {"tri_reinforcement": {"rows": [
            {"triangle": "Q1", "Asx_bot": 0.001, "Asy_bot": 0.002},
        ]}}
        rho = _rho_l_at(s, results, "far_away_node", d_m=0.18, default=0.005)
        assert rho == 0.005

    def test_tri_and_quad_rows_both_average_in_on_a_shared_node(self):
        s = Structure2D(domain="plate")
        _plate_material(s)
        s.add_plate_section("S", "C", thickness=0.2, formulation="DKT")
        s.add_node("A", 0, 0); s.add_node("B", 1, 0)
        s.add_node("C", 1, 1); s.add_node("D", 0, 1); s.add_node("E", 2, 0.5)
        s.add_tri_element("T1", "B", "E", "C", "S")
        s.add_quad_element("Q1", "A", "B", "C", "D", "S")
        results = {"tri_reinforcement": {"rows": [
            {"triangle": "T1", "Asx_bot": 0.0, "Asy_bot": 0.0},
            {"triangle": "Q1", "Asx_bot": 0.004, "Asy_bot": 0.004},
        ]}}
        rho = _rho_l_at(s, results, "B", d_m=0.2)   # touched by both
        expected = ((0.0 + 0.004) / 2.0) / 0.2
        assert abs(rho - min(expected, 0.02)) < 1e-9
