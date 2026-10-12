"""Dividing triangles / quads carries every load of the parent to its children
and leaves nothing behind that names the parent."""
import pytest

import context  # noqa: F401
from xdfem2d import Structure2D, subdivide


def _plate():
    s = Structure2D(domain="plate")
    s.add_material("M", elastic_modulus=3e7, unit_weight=0.0, poisson=0.2)
    s.add_area_section("A", "M", thickness=0.2)
    for i, (x, y) in enumerate([(0, 0), (4, 0), (4, 4), (0, 4), (8, 0), (8, 4)]):
        s.add_node(f"N{i}", x, y)
    s.add_tri_element("T0", "N0", "N1", "N2", "A")
    s.add_quad_element("Q0", "N1", "N4", "N5", "N2", "A")
    s.add_load_case("LC")
    return s


def test_dividing_a_triangle_carries_area_load_and_spring():
    s = _plate()
    s.add_area_load("T0", "LC", pz=-7.0)
    s.add_area_spring("T0", 1000.0)
    subdivide.divide_triangles(s, ["T0"], 3)
    assert "T0" not in s.tri_elements_by_id
    assert len(s.tri_elements) == 9
    assert len(s.tri_area_loads) == 9 and all(a.pz == -7.0 for a in s.tri_area_loads)
    assert len(s.tri_area_springs) == 9
    assert {a.tri_id for a in s.tri_area_loads} == set(s.tri_elements_by_id)
    assert s.reference_problems() == []


def test_dividing_a_triangle_leaves_no_orphan_with_the_parent_id():
    s = _plate()
    s.add_area_load("T0", "LC", pz=-7.0)
    subdivide.divide_triangles(s, ["T0"], 2)
    assert not any(a.tri_id == "T0" for a in s.tri_area_loads)


def test_dividing_a_quad_carries_edge_loads():
    s = _plate()
    s.add_quad_edge_load("QE", "Q0", "N1", "N4", "LC", fx=2.0, fy=-3.0)
    subdivide.divide_quads(s, ["Q0"], 2)
    assert "Q0" not in s.quad_elements_by_id
    assert len(s.quad_edge_loads) == 2           # one per sub-segment of the edge
    assert all(e.quad_id in s.quad_elements_by_id for e in s.quad_edge_loads)
    assert all((e.fx, e.fy) == (2.0, -3.0) for e in s.quad_edge_loads)
    assert not any(e.quad_id == "Q0" for e in s.quad_edge_loads)


def test_dividing_a_quad_still_carries_area_load():
    s = _plate()
    s.add_area_load("Q0", "LC", pz=-4.0)
    subdivide.divide_quads(s, ["Q0"], 2)
    assert len(s.quad_area_loads) == 4
    assert s.reference_problems() == []
