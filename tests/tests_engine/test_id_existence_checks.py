"""add_node_spring / add_element_spring / add_point_load /
add_distributed_load / add_element_point_load / add_tri_edge_load /
add_quad_edge_load used to store their record unconditionally, even when the
referenced node/element id did not exist in the model at all -- no
_require_area_object/_require_line_object-style dispatch touches these
(their target isn't a geometry object), so nothing rejected a typo'd or
since-removed id. The record was stored and silently never consumed at
assembly/solve time. This file locks in Structure2D._require_exists across
all seven.
"""
from __future__ import annotations

import unittest

from context import Structure2D


def _base():
    s = Structure2D()
    s.add_material("M", elastic_modulus=30e6, unit_weight=25.0)
    s.add_section("BS", "M", b=0.3, h=0.6)
    s.add_tri_section("S", "M", thickness=0.2)
    s.add_load_case("LC1")
    s.add_node("N1", 0.0, 0.0)
    s.add_node("N2", 1.0, 0.0)
    s.add_node("N3", 0.0, 1.0)
    s.add_node("N4", 1.0, 1.0)
    s.add_bar_element("B1", "N1", "N2", section_name="BS")
    s.add_tri_element("T1", "N1", "N2", "N3", section_name="S")
    s.add_quad_element("Q1", "N1", "N2", "N4", "N3", section_name="S")
    return s


class TestAddNodeSpring(unittest.TestCase):
    def test_unknown_node_is_rejected(self):
        s = _base()
        with self.assertRaises(ValueError):
            s.add_node_spring("NOPE", kx=100.0)

    def test_known_node_is_accepted(self):
        s = _base()
        sp = s.add_node_spring("N1", kx=100.0)
        self.assertEqual(sp.node_id, "N1")


class TestAddElementSpring(unittest.TestCase):
    def test_unknown_element_is_rejected(self):
        s = _base()
        with self.assertRaises(ValueError):
            s.add_element_spring("NOPE", kx=100.0)

    def test_non_bar_element_is_rejected(self):
        # element_springs/element_point_loads are consumed only via
        # bar_elements_by_id (assembly.py/solver.py) -- a tri/quad id must
        # be rejected the same as an unknown one.
        s = _base()
        with self.assertRaises(ValueError):
            s.add_element_spring("T1", kx=100.0)

    def test_known_bar_is_accepted(self):
        s = _base()
        es = s.add_element_spring("B1", kx=100.0)
        self.assertEqual(es.element_id, "B1")


class TestAddPointLoad(unittest.TestCase):
    def test_unknown_node_is_rejected(self):
        s = _base()
        with self.assertRaises(ValueError):
            s.add_point_load("NOPE", "LC1", fy=-10.0)

    def test_known_node_is_accepted(self):
        s = _base()
        pl = s.add_point_load("N1", "LC1", fy=-10.0)
        self.assertEqual(pl.node_id, "N1")


class TestAddDistributedLoad(unittest.TestCase):
    def test_unknown_element_is_rejected(self):
        s = _base()
        with self.assertRaises(ValueError):
            s.add_distributed_load("NOPE", "LC1", fye=-2.0, fyd=-2.0)

    def test_known_bar_is_accepted(self):
        s = _base()
        dl = s.add_distributed_load("B1", "LC1", fye=-2.0, fyd=-2.0)
        self.assertEqual(dl.element_id, "B1")


class TestAddElementPointLoad(unittest.TestCase):
    def test_unknown_element_is_rejected(self):
        s = _base()
        with self.assertRaises(ValueError):
            s.add_element_point_load("NOPE", "LC1", a=0.5, fy=-5.0)

    def test_non_bar_element_is_rejected(self):
        s = _base()
        with self.assertRaises(ValueError):
            s.add_element_point_load("T1", "LC1", a=0.5, fy=-5.0)

    def test_known_bar_is_accepted(self):
        s = _base()
        epl = s.add_element_point_load("B1", "LC1", a=0.5, fy=-5.0)
        self.assertEqual(epl.element_id, "B1")


class TestAddTriEdgeLoad(unittest.TestCase):
    def test_unknown_tri_is_rejected(self):
        s = _base()
        with self.assertRaises(ValueError):
            s.add_tri_edge_load("E1", "NOPE", "N1", "N2", "LC1", pn=5.0)

    def test_known_tri_is_accepted(self):
        s = _base()
        e = s.add_tri_edge_load("E1", "T1", "N1", "N2", "LC1", pn=5.0)
        self.assertEqual(e.tri_id, "T1")


class TestAddQuadEdgeLoad(unittest.TestCase):
    def test_unknown_quad_is_rejected(self):
        s = _base()
        with self.assertRaises(ValueError):
            s.add_quad_edge_load("E1", "NOPE", "N1", "N2", "LC1", pn=5.0)

    def test_known_quad_is_accepted(self):
        s = _base()
        e = s.add_quad_edge_load("E1", "Q1", "N1", "N2", "LC1", pn=5.0)
        self.assertEqual(e.quad_id, "Q1")


class TestAddNodeId(unittest.TestCase):
    """A node with id None was created by a model that wrote add_node(None, x, y):
    it sat in the model under the key None and broke the app's delete dialog."""

    def test_an_id_that_is_not_text_is_rejected(self):
        s = Structure2D()
        for bad in (None, "", "  ", 3):
            with self.assertRaises(ValueError):
                s.add_node(bad, 1.0, 2.0)
        self.assertEqual(len(s.nodes), 0)

    def test_the_message_points_to_create_node(self):
        with self.assertRaises(ValueError) as cm:
            Structure2D().add_node(None, 6.0, 6.0)
        self.assertIn("create_node", str(cm.exception))

    def test_create_node_still_numbers_automatically(self):
        s = Structure2D()
        n = s.create_node(0.0, 0.0)
        self.assertIsInstance(n.id, str)


if __name__ == '__main__':
    unittest.main()
