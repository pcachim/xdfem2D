"""object_id-taking add_*_load/spring functions must reject the wrong kind of
geometry object.

add_area_temperature_load used to accept ANY object_id string, including a
line object's (a GeoArc, say). The record was stored and never rejected, but
_apply_area_temperatures (geo_expand.py) only turns an AreaTemperatureLoad
into tri/quad thermal loads for an actual GeoRectangle/GeoPolygon — so the
load was a completely silent no-op: nothing raised, nothing warned, nothing
solved. This file locks in the fix (Structure2D._require_area_object /
_require_line_object) across every function that gained it.
"""
from __future__ import annotations

import unittest

from context import Structure2D


def _base():
    s = Structure2D()
    s.add_material("M", elastic_modulus=30e6, unit_weight=25.0)
    s.add_tri_section("S", "M", thickness=0.2)
    s.add_section("BS", "M", b=0.3, h=0.6)
    s.add_load_case("LC1")
    return s


def _with_rect_and_arc():
    s = _base()
    s.add_geo_rectangle("R1", (0.0, 0.0), (2.0, 2.0),
                        section_name="S", target_size=1.0, prefer_quad=False)
    s.add_geo_arc("A1", 0.0, 0.0, 5.0, 0.0, 90.0, section_name="BS")
    return s


class TestAddAreaTemperatureLoad(unittest.TestCase):

    def test_a_line_object_is_rejected(self):
        s = _with_rect_and_arc()
        with self.assertRaises(ValueError) as cm:
            s.add_area_temperature_load("A1", "LC1", dt_uniform=10.0)
        msg = str(cm.exception)
        self.assertIn("A1", msg)
        self.assertIn("line", msg.lower())

    def test_an_unknown_id_is_rejected(self):
        s = _with_rect_and_arc()
        with self.assertRaises(ValueError):
            s.add_area_temperature_load("NOPE", "LC1", dt_uniform=10.0)

    def test_an_area_object_is_accepted(self):
        s = _with_rect_and_arc()
        tl = s.add_area_temperature_load("R1", "LC1", dt_uniform=10.0)
        self.assertEqual(tl.object_id, "R1")


class TestAddLineTemperatureLoad(unittest.TestCase):

    def test_an_area_object_is_rejected(self):
        s = _with_rect_and_arc()
        with self.assertRaises(ValueError) as cm:
            s.add_line_temperature_load("R1", "LC1", dt_uniform=10.0)
        msg = str(cm.exception)
        self.assertIn("R1", msg)
        self.assertIn("surface", msg.lower())

    def test_a_line_object_is_accepted(self):
        s = _with_rect_and_arc()
        tl = s.add_line_temperature_load("A1", "LC1", dt_uniform=10.0)
        self.assertEqual(tl.object_id, "A1")


class TestAddLineDistributedLoad(unittest.TestCase):

    def test_an_area_object_is_rejected(self):
        s = _with_rect_and_arc()
        with self.assertRaises(ValueError):
            s.add_line_distributed_load("R1", "LC1", fy=-2.0)


class TestAddLineElementSpring(unittest.TestCase):

    def test_an_area_object_is_rejected(self):
        s = _with_rect_and_arc()
        with self.assertRaises(ValueError):
            s.add_line_element_spring("R1", kx=100.0)


class TestAddSurfaceEdgeLoad(unittest.TestCase):

    def test_a_line_object_is_rejected(self):
        s = _with_rect_and_arc()
        with self.assertRaises(ValueError):
            s.add_surface_edge_load("E1", "A1", "N1", "N2", "LC1", fy=-3.0)


if __name__ == '__main__':
    unittest.main()
