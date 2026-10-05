"""Regression tests for xdfem2d.geo_expand.bake_object_loads.

The desktop app's object-baking action copies a geometry object's
generated elements into the live model and deletes the object. Its own loads
(pressure/edge/temperature for a surface object, distributed/temperature for a
line object) must be translated into permanent per-element loads before the
object disappears — else expand_geometry() has nothing left to derive them
from on the next solve and the load is silently dropped.

bake_object_loads() is the Qt-free engine-level function that does this
translation; it is exercised here directly against a plain Structure2D, with
no GUI/Qt involved (this suite runs without PySide6, see
tests/tests_app/test_context_menu_phase3.py's docstring for why).
"""
from __future__ import annotations

import unittest

from context import Structure2D
from xdfem2d.geo_expand import expand_geometry, bake_object_loads


def _copy_object_mesh(struc, compiled, trace, object_id):
    """Copy one object's generated nodes/elements from *compiled* into
    *struc*, mirroring what _bake_objects() does before it bakes loads —
    bake_object_loads() only translates loads onto elements that already
    exist in the live model."""
    tr = trace[object_id]
    for nid in tr.get("nodes", []):
        if nid not in struc.nodes:
            nd = compiled.nodes[nid]
            struc.add_node(nid, nd.x, nd.y)
    for eid in tr.get("elems", []):
        if eid in struc.bar_elements_by_id:
            continue
        el = compiled.bar_elements_by_id[eid]
        struc.add_bar_element(eid, el.node_i, el.node_j, el.section_name)
    for tid in tr.get("tris", []):
        if tid in struc.tri_elements_by_id:
            continue
        t = compiled.tri_elements_by_id[tid]
        struc.add_tri_element(tid, t.node_i, t.node_j, t.node_k,
                              t.section_name)


class TestBakeSurfaceObjectLoads(unittest.TestCase):
    """Rectangle object: pressure + edge + temperature loads, all baked in."""

    def _model(self):
        s = Structure2D()
        s.add_material("M", 30e6, 25.0)
        s.add_tri_section("S", "M", thickness=0.2)
        s.add_load_case("LC1")
        s.add_geo_rectangle("R1", (0.0, 0.0), (2.0, 2.0),
                            section_name="S", target_size=1.0,
                            prefer_quad=False)
        s.add_area_load("R1", "LC1", pz=-5.0)
        s.add_area_temperature_load("R1", "LC1", dt_uniform=10.0,
                                    dt_gradient=2.0)
        obj = s.geometry_objects["R1"]
        s.add_surface_edge_load("E1", "R1", obj.node_ids[0], obj.node_ids[1],
                                "LC1", fy=-3.0)
        return s

    def test_loads_are_baked_into_elements_and_object_entries_removed(self):
        s = self._model()
        compiled, trace = expand_geometry(s)
        tr = trace["R1"]
        self.assertTrue(tr.get("tris"))
        _copy_object_mesh(s, compiled, trace, "R1")

        bake_object_loads(s, trace, "area", ["R1"])

        # Object-level entries are gone.
        self.assertFalse(any(a.object_id == "R1"
                             for a in s.surface_area_loads))
        self.assertFalse(any(a.object_id == "R1"
                             for a in s.area_temperature_loads))
        self.assertFalse(any(e.object_id == "R1"
                             for e in s.surface_edge_loads))

        # Permanent per-element loads exist for every generated triangle.
        tri_ids = set(tr["tris"])
        self.assertTrue(tri_ids)
        baked_tri_ids = {a.tri_id for a in s.tri_area_loads}
        self.assertTrue(tri_ids <= baked_tri_ids)
        for a in s.tri_area_loads:
            if a.tri_id in tri_ids:
                self.assertEqual(a.pz, -5.0)

        baked_temp_tri_ids = {t.tri_id for t in s.tri_temperature_loads}
        self.assertTrue(tri_ids <= baked_temp_tri_ids)

        # The edge load became consistent nodal point loads (translated by
        # _apply_surface_edge_loads via bake_object_loads).
        self.assertTrue(s.point_loads)


class TestBakeLineObjectLoads(unittest.TestCase):
    """Segment object: distributed + temperature loads, both baked in."""

    def _model(self):
        s = Structure2D()
        s.add_material("M", 30e6, 25.0)
        s.add_section("S", "M", 0.3, 0.5)
        s.add_load_case("LC1")
        s.add_geo_segment("L1", 0.0, 0.0, 4.0, 0.0, section_name="S",
                          divisions=4)
        s.add_line_distributed_load("L1", "LC1", fy=-2.0)
        s.add_line_temperature_load("L1", "LC1", dt_uniform=15.0,
                                    dt_gradient=1.0)
        return s

    def test_loads_are_baked_into_bars_and_object_entries_removed(self):
        s = self._model()
        compiled, trace = expand_geometry(s)
        tr = trace["L1"]
        elem_ids = set(tr["elems"])
        self.assertEqual(len(elem_ids), 4)
        _copy_object_mesh(s, compiled, trace, "L1")

        bake_object_loads(s, trace, "curve", ["L1"])

        # Object-level entries are gone.
        self.assertFalse(any(d.object_id == "L1"
                             for d in s.line_distributed_loads))
        self.assertFalse(any(t.object_id == "L1"
                             for t in s.line_temperature_loads))

        # Permanent per-bar loads exist for every generated bar.
        baked_dist_ids = {d.element_id for d in s.distributed_loads}
        self.assertTrue(elem_ids <= baked_dist_ids)
        for d in s.distributed_loads:
            if d.element_id in elem_ids:
                self.assertEqual(d.fye, -2.0)
                self.assertEqual(d.fyd, -2.0)

        baked_temp_ids = {t.element_id for t in s.temperature_loads}
        self.assertTrue(elem_ids <= baked_temp_ids)
        for t in s.temperature_loads:
            if t.element_id in elem_ids:
                self.assertEqual(t.delta_t_uniform, 15.0)
                self.assertEqual(t.delta_t_gradient, 1.0)


if __name__ == "__main__":
    unittest.main()
