"""Objects: serialisation round-trip and expansion (discretisation)."""
from __future__ import annotations

import math
import tempfile
import unittest

from context import Structure2D
from xdfem2d.structure_io import (save_structure_json, load_structure_json)
from xdfem2d.file_io import save_x2d, load_x2d
from xdfem2d.geo_expand import expand_geometry


class TestObjectSerialization(unittest.TestCase):
    def _model(self):
        s = Structure2D()
        s.add_material("M", 30e6, 25.0)
        s.add_section("S", "M", 0.3, 0.5)
        s.add_geo_arc("A1", 0.0, 0.0, 5.0, 0.0, 180.0,
                        section_name="S", divisions=12, max_chord=0.4)
        s.add_geo_polyline("P1", [[0, 0], [2, 3], [5, 3]], closed=True,
                             section_name="S", divisions=2)
        return s

    def test_json_roundtrip(self):
        import os
        s = self._model()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.json")
            save_structure_json(s, p)
            r = load_structure_json(p)
        a = r.geometry_objects["A1"]
        self.assertEqual((a.divisions, a.max_chord, a.section_name), (12, 0.4, "S"))
        # Node-driven: the arc is defined by 3 real nodes; geometry read back.
        self.assertEqual(len(a.node_ids), 3)
        pts = r.object_defining_points(a)
        self.assertAlmostEqual(pts[0][0], 5.0)   # start endpoint at angle 0
        self.assertAlmostEqual(pts[-1][0], -5.0, places=6)  # end at 180°
        p1 = r.geometry_objects["P1"]
        self.assertTrue(p1.closed)
        self.assertEqual(len(p1.node_ids), 3)
        self.assertEqual(r.object_defining_points(p1),
                         [(0.0, 0.0), (2.0, 3.0), (5.0, 3.0)])

    def test_x2d_roundtrip(self):
        import os
        s = self._model()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.x2d")
            save_x2d(s, None, p)
            r, _res, _view = load_x2d(p)
        self.assertEqual(sorted(r.geometry_objects), ["A1", "P1"])

    def test_backward_compatible(self):
        # A dict without 'geometry_objects' must load fine (no objects).
        import os
        s = Structure2D(); s.add_material("M", 30e6, 25.0)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "old.json")
            save_structure_json(s, p)
            import json
            data = json.load(open(p)); data.pop("geometry_objects", None)
            json.dump(data, open(p, "w"))
            r = load_structure_json(p)
        self.assertEqual(r.geometry_objects, {})


class TestObjectExpansion(unittest.TestCase):
    def test_arc_segment_count_and_geometry(self):
        s = Structure2D(); s.add_material("M", 30e6, 25.0)
        s.add_section("S", "M", 0.3, 0.5)
        s.add_geo_arc("A", 0.0, 0.0, 2.0, 0.0, 90.0, section_name="S",
                        divisions=6)
        comp, trace = expand_geometry(s)
        # 6 divisions → 7 nodes, 6 elements.
        self.assertEqual(len(trace["A"]["elems"]), 6)
        self.assertEqual(len(trace["A"]["nodes"]), 7)
        # Endpoints land on the arc: (2,0) and (0,2).
        n0 = comp.nodes[trace["A"]["nodes"][0]]
        n_last = comp.nodes[trace["A"]["nodes"][-1]]
        self.assertAlmostEqual(n0.x, 2.0); self.assertAlmostEqual(n0.y, 0.0)
        self.assertAlmostEqual(n_last.x, 0.0, places=9)
        self.assertAlmostEqual(n_last.y, 2.0, places=9)
        # Every generated node is at radius 2 from the centre.
        for nid in trace["A"]["nodes"]:
            nd = comp.nodes[nid]
            self.assertAlmostEqual(math.hypot(nd.x, nd.y), 2.0, places=9)

    def test_arc_max_chord_refines(self):
        s = Structure2D(); s.add_material("M", 30e6, 25.0)
        s.add_section("S", "M", 0.3, 0.5)
        # Quarter arc, r=2 → length = pi. max_chord 0.1 → ceil(pi/0.1)=32 segs,
        # which is finer than divisions=4, so 32 governs.
        s.add_geo_arc("A", 0.0, 0.0, 2.0, 0.0, 90.0, section_name="S",
                        divisions=4, max_chord=0.1)
        _comp, trace = expand_geometry(s)
        self.assertEqual(len(trace["A"]["elems"]), 32)

    def test_polygon_closed_counts_and_closes(self):
        s = Structure2D(); s.add_material("M", 30e6, 25.0)
        s.add_section("S", "M", 0.3, 0.5)
        s.add_geo_polyline("P", [[0, 0], [4, 0], [4, 3]], closed=True,
                             section_name="S", divisions=1)
        comp, trace = expand_geometry(s)
        # Triangle: 3 spans × 1 segment → 3 elements, 3 unique nodes.
        self.assertEqual(len(trace["P"]["elems"]), 3)
        self.assertEqual(len(trace["P"]["nodes"]), 3)

    def test_endpoint_merges_with_existing_node(self):
        s = Structure2D(); s.add_material("M", 30e6, 25.0)
        s.add_section("S", "M", 0.3, 0.5)
        s.add_node("BASE", 2.0, 0.0)     # coincides with the arc's start point
        s.add_geo_arc("A", 0.0, 0.0, 2.0, 0.0, 90.0, section_name="S",
                        divisions=4)
        comp, trace = expand_geometry(s)
        # The arc's first node must be the pre-existing BASE node (merged).
        self.assertEqual(trace["A"]["nodes"][0], "BASE")
        # First element connects BASE to the next generated node.
        first_e = comp.bar_elements_by_id[trace["A"]["elems"][0]]
        self.assertEqual(first_e.node_i, "BASE")

    def test_shared_endpoint_between_two_objects(self):
        s = Structure2D(); s.add_material("M", 30e6, 25.0)
        s.add_section("S", "M", 0.3, 0.5)
        # Two polylines sharing the point (2,0).
        s.add_geo_polyline("P1", [[0, 0], [2, 0]], section_name="S")
        s.add_geo_polyline("P2", [[2, 0], [2, 3]], section_name="S")
        comp, trace = expand_geometry(s)
        shared_1 = trace["P1"]["nodes"][-1]
        shared_2 = trace["P2"]["nodes"][0]
        self.assertEqual(shared_1, shared_2)      # same merged node

    def test_deterministic_ids(self):
        s = Structure2D(); s.add_material("M", 30e6, 25.0)
        s.add_section("S", "M", 0.3, 0.5)
        s.add_geo_arc("A", 0.0, 0.0, 2.0, 0.0, 90.0, section_name="S",
                        divisions=5)
        c1, t1 = expand_geometry(s)
        c2, t2 = expand_geometry(s)
        self.assertEqual(t1, t2)
        self.assertEqual(sorted(c1.nodes), sorted(c2.nodes))

    def test_original_model_untouched(self):
        s = Structure2D(); s.add_material("M", 30e6, 25.0)
        s.add_section("S", "M", 0.3, 0.5)
        s.add_geo_arc("A", 0.0, 0.0, 2.0, 0.0, 90.0, section_name="S",
                        divisions=4)
        # Arc is defined by 3 real nodes; still no bars in the model.
        self.assertEqual(list(s.geometry_objects), ["A"])
        self.assertEqual(len(s.nodes), 3)
        self.assertEqual(len(s.bar_elements), 0)
        expand_geometry(s)
        self.assertEqual(len(s.bar_elements), 0)   # editable model unchanged


class TestObjectSolve(unittest.TestCase):
    """The obj pipeline must give the same results as an equivalent explicit
    mesh, and calculate() must accept a obj-only geometry."""

    def _common(self, s):
        s.add_material("M", 30e6, 25.0)
        s.add_section("S", "M", 0.3, 0.5)
        s.add_support("Fix", ux=True, uy=True, tz=True)
        s.add_load_case("LC")

    def test_object_matches_explicit_mesh(self):
        L, P, n = 6.0, -10.0, 8

        # Explicit cantilever: n equal bars from (0,0) to (L,0).
        se = Structure2D(); self._common(se)
        for i in range(n + 1):
            se.add_node(f"N{i}", L * i / n, 0.0)
        for i in range(n):
            se.add_bar_element(f"E{i}", f"N{i}", f"N{i+1}", "S")
        se.assign_support("N0", "Fix")
        se.add_point_load(f"N{n}", "LC", fy=P)
        re = se.calculate()

        # Object cantilever: a polyline [0,0]->[L,0] with n divisions; base nodes
        # at the two ends (merged with the obj endpoints).
        sm = Structure2D(); self._common(sm)
        sm.add_node("A", 0.0, 0.0)
        sm.add_node("B", L, 0.0)
        sm.assign_support("A", "Fix")
        sm.add_point_load("B", "LC", fy=P)
        sm.add_geo_polyline("MB", [[0.0, 0.0], [L, 0.0]], section_name="S",
                              divisions=n)
        rm = sm.calculate()

        # Tip deflection (node B / N{n}) must match, and equal PL^3/3EI.
        E = 30e6; I = 0.3 * 0.5 ** 3 / 12.0
        analytic = P * L ** 3 / (3.0 * E * I)
        uy_explicit = re["analysis_cases"]["LC"]["displacements"][f"N{n}"][1]
        uy_obj = rm["analysis_cases"]["LC"]["displacements"]["B"][1]
        self.assertAlmostEqual(uy_explicit, uy_obj, places=9)
        self.assertAlmostEqual(uy_obj, analytic, places=4)
        # The trace is carried on the results.
        self.assertIn("object_trace", rm)
        self.assertEqual(len(rm["object_trace"]["MB"]["elems"]), n)

    def test_object_supports_and_distributed_load(self):
        # Simply-supported beam via a obj: supports at both endpoints and a
        # uniform load along it. Compare mid-span moment to wL^2/8.
        L, w, n = 8.0, -12.0, 8
        s = Structure2D(); self._common(s)
        s.add_support("Pin", ux=True, uy=True, tz=False)
        s.add_support("Roller", ux=False, uy=True, tz=False)
        m = s.add_geo_polyline("MB", [[0.0, 0.0], [L, 0.0]], section_name="S",
                                 divisions=n)
        # Supports go on the object's real endpoint nodes.
        s.assign_support(m.node_ids[0], "Pin")
        s.assign_support(m.node_ids[-1], "Roller")
        m.loads = [{"load_case": "LC", "wx": 0.0, "wy": w, "coord_sys": "global"}]
        res = s.calculate()
        # Peak |M| along the obj's generated elements ≈ w L^2 / 8.
        import numpy as np
        dist = res["element_distribution"]["LC"]
        peak = max(float(np.max(np.abs(dist[eid]["M"])))
                   for eid in res["object_trace"]["MB"]["elems"])
        self.assertAlmostEqual(peak, abs(w) * L ** 2 / 8.0, delta=abs(w) * L ** 2 / 8.0 * 0.02)

    def test_object_bc_roundtrip(self):
        import os
        s = Structure2D(); s.add_material("M", 30e6, 25.0)
        s.add_section("S", "M", 0.3, 0.5); s.add_support("Fix", True, True, True)
        s.add_load_case("LC")
        m = s.add_geo_polyline("MB", [[0, 0], [5, 0]], section_name="S")
        s.assign_support(m.node_ids[0], "Fix")
        m.loads = [{"load_case": "LC", "wx": 0.0, "wy": -7.0, "coord_sys": "global"}]
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.json")
            save_structure_json(s, p); r = load_structure_json(p)
        mb = r.geometry_objects["MB"]
        # The real endpoint node kept its support and the object its load.
        self.assertEqual(mb.node_ids[0], "MB.p0")
        self.assertTrue(any(a.node_id == "MB.p0" and a.support_name == "Fix"
                            for a in r.support_assignments))
        self.assertEqual(mb.loads[0]["wy"], -7.0)

    def test_object_only_geometry_solves(self):
        # No explicit bar_elements at all — geometry comes entirely from a obj.
        s = Structure2D(); self._common(s)
        s.add_node("A", 0.0, 0.0); s.add_node("B", 4.0, 0.0)
        s.assign_support("A", "Fix")
        s.add_point_load("B", "LC", fy=-5.0)
        s.add_geo_polyline("MB", [[0.0, 0.0], [4.0, 0.0]], section_name="S",
                             divisions=4)
        self.assertEqual(len(s.bar_elements), 0)   # nothing explicit
        res = s.calculate()                        # must not raise
        self.assertIn("analysis_cases", res)


class TestTeeJunctions(unittest.TestCase):
    """A node landing on a bar interior must split that bar so it connects."""

    def test_object_node_splits_existing_bar(self):
        s = Structure2D(); s.add_material("M", 30e6, 25.0)
        s.add_section("S", "M", 0.3, 0.5)
        # A horizontal beam N1—N2 (one explicit bar).
        s.add_node("N1", 0.0, 0.0); s.add_node("N2", 6.0, 0.0)
        s.add_bar_element("BEAM", "N1", "N2", "S")
        # A vertical object hanging from the beam mid-span (3,0)→(3,-2).
        s.add_geo_polyline("COL", [[3.0, 0.0], [3.0, -2.0]], section_name="S")
        comp, trace = expand_geometry(s)
        # BEAM must be split at (3,0) into two sub-bars; the column ties in.
        self.assertNotIn("BEAM", comp.bar_elements_by_id)
        subs = [e for e in comp.bar_elements if e.id.startswith("BEAM~")]
        self.assertEqual(len(subs), 2)
        # The split node is shared by the beam halves and the column's top.
        mid = [nid for nid, n in comp.nodes.items()
               if abs(n.x - 3.0) < 1e-9 and abs(n.y) < 1e-9]
        self.assertEqual(len(mid), 1)
        mid = mid[0]
        touching = [e for e in comp.bar_elements
                    if e.node_i == mid or e.node_j == mid]
        self.assertEqual(len(touching), 3)   # two beam halves + column top

    def test_buckling_overrides_and_column_flag_survive_the_split(self):
        """dev/BUCKLING_COLUMN_PERSISTENCE.md fase 2: an explicit bar marked
        as a column, with K overrides set, that happens to get split by an
        object touching its interior (T-junction) must not silently lose
        those on the resulting sub-bars."""
        s = Structure2D(); s.add_material("M", 30e6, 25.0)
        s.add_section("S", "M", 0.3, 0.5)
        s.add_node("N1", 0.0, 0.0); s.add_node("N2", 6.0, 0.0)
        beam = s.add_bar_element("BEAM", "N1", "N2", "S")
        beam.sd_ky = 2.0; beam.sd_kz = 1.5; beam.is_column = True
        s.add_geo_polyline("COL", [[3.0, 0.0], [3.0, -2.0]], section_name="S")
        comp, _ = expand_geometry(s)
        subs = [e for e in comp.bar_elements if e.id.startswith("BEAM~")]
        self.assertEqual(len(subs), 2)
        for e in subs:
            self.assertEqual(e.sd_ky, 2.0)
            self.assertEqual(e.sd_kz, 1.5)
            self.assertTrue(e.is_column)

    def test_object_level_design_fields_stamp_every_generated_bar(self):
        """dev/BUCKLING_COLUMN_PERSISTENCE.md fase 3 (estrutural): is_column/
        sd_ky/sd_kz/sd_klt/sd_ltb set directly on a GeoSegment/GeoArc/
        GeoMultisegment must be carried onto every bar expand_geometry()
        generates from it, and must survive repeated regeneration (the
        object dataclass is the single source of truth, never baked once
        onto a bar that a later expand_geometry() call would then wipe)."""
        s = Structure2D(); s.add_material("M", 30e6, 25.0)
        s.add_section("S", "M", 0.3, 0.5)
        col = s.add_geo_segment("COL", 0.0, 0.0, 0.0, 3.0, section_name="S", divisions=3)
        col.is_column = True
        col.sd_ky = 0.7; col.sd_kz = 0.85; col.sd_klt = 1.0; col.sd_ltb = False

        for _ in range(2):  # regenerate twice — must not depend on prior bake
            comp, _ = expand_geometry(s)
            subs = [e for e in comp.bar_elements if e.id.startswith("COL.")]
            self.assertEqual(len(subs), 3)
            for e in subs:
                self.assertTrue(e.is_column)
                self.assertEqual(e.sd_ky, 0.7)
                self.assertEqual(e.sd_kz, 0.85)
                self.assertEqual(e.sd_klt, 1.0)
                self.assertFalse(e.sd_ltb)

    def test_distributed_load_apportioned_on_split(self):
        s = Structure2D(); s.add_material("M", 30e6, 25.0)
        s.add_section("S", "M", 0.3, 0.5)
        s.add_load_case("LC")
        s.add_node("N1", 0.0, 0.0); s.add_node("N2", 4.0, 0.0)
        s.add_bar_element("BEAM", "N1", "N2", "S")
        s.add_distributed_load("BEAM", "LC", fye=-10.0, fyd=-10.0)  # uniform
        s.add_node("P", 1.0, 0.0)                 # existing node on the interior
        s.add_geo_polyline("HANG", [[1.0, 0.0], [1.0, -1.0]], section_name="S")
        comp, _ = expand_geometry(s)
        # Total distributed load must be conserved: 2 sub-bars still total 10 kN/m
        # over the original 4 m span (uniform → same intensity on each sub-bar).
        beam_dl = [dl for dl in comp.distributed_loads
                   if dl.element_id.startswith("BEAM~")]
        self.assertEqual(len(beam_dl), 2)
        for dl in beam_dl:
            self.assertAlmostEqual(dl.fye, -10.0)
            self.assertAlmostEqual(dl.fyd, -10.0)

    def test_pure_frame_tee_left_untouched(self):
        # Without objects, expand isn't even called; but calling it on a model
        # whose only geometry is explicit must not split explicit T-junctions.
        s = Structure2D(); s.add_material("M", 30e6, 25.0)
        s.add_section("S", "M", 0.3, 0.5)
        s.add_node("N1", 0.0, 0.0); s.add_node("N2", 6.0, 0.0)
        s.add_node("P", 3.0, 0.0)
        s.add_bar_element("BEAM", "N1", "N2", "S")   # P sits on its interior
        comp, trace = expand_geometry(s)             # no objects → no split
        self.assertIn("BEAM", comp.bar_elements_by_id)
        self.assertEqual(trace, {})


class TestDedup(unittest.TestCase):
    def test_object_duplicate_of_existing_bar_is_merged(self):
        s = Structure2D(); s.add_material("M", 30e6, 25.0)
        s.add_section("S", "M", 0.3, 0.5); s.add_load_case("LC")
        s.add_node("A", 0.0, 0.0); s.add_node("B", 5.0, 0.0)
        s.add_bar_element("BEAM", "A", "B", "S")
        # An object drawn exactly over the existing beam, carrying a load.
        m = s.add_geo_polyline("DUP", [[0.0, 0.0], [5.0, 0.0]], section_name="S")
        m.loads = [{"load_case": "LC", "wx": 0.0, "wy": -6.0,
                    "coord_sys": "global"}]
        comp, trace = expand_geometry(s)
        # Only one bar between A and B (no doubled stiffness).
        ab = [e for e in comp.bar_elements
              if {e.node_i, e.node_j} == {"A", "B"}]
        self.assertEqual(len(ab), 1)
        # The object's load survived on the kept bar.
        kept = ab[0].id
        self.assertTrue(any(dl.element_id == kept and dl.fye == -6.0
                            for dl in comp.distributed_loads))


class TestCrossings(unittest.TestCase):
    def _model(self, flag):
        s = Structure2D(); s.add_material("M", 30e6, 25.0)
        s.add_section("S", "M", 0.3, 0.5)
        # A horizontal beam as an explicit bar, and a diagonal object crossing
        # it at (2,0) without sharing a node.
        s.add_node("A", 0.0, 0.0); s.add_node("B", 4.0, 0.0)
        s.add_bar_element("BEAM", "A", "B", "S")
        m = s.add_geo_polyline("DIAG", [[2.0, -2.0], [2.0, 2.0]],
                               section_name="S")
        m.intersect_crossings = flag
        return s

    def test_crossing_connects_with_flag(self):
        comp, trace = expand_geometry(self._model(True))
        # A node must exist at the crossing (2,0), shared by beam halves + diag.
        at = [nid for nid, n in comp.nodes.items()
              if abs(n.x - 2.0) < 1e-9 and abs(n.y) < 1e-9]
        self.assertEqual(len(at), 1)
        touching = [e for e in comp.bar_elements
                    if e.node_i == at[0] or e.node_j == at[0]]
        self.assertEqual(len(touching), 4)   # 2 beam halves + 2 diag halves
        self.assertNotIn("BEAM", comp.bar_elements_by_id)   # beam was split

    def test_crossing_not_connected_without_flag(self):
        comp, trace = expand_geometry(self._model(False))
        # No node at (2,0); the beam stays whole and the members just overlap.
        at = [nid for nid, n in comp.nodes.items()
              if abs(n.x - 2.0) < 1e-9 and abs(n.y) < 1e-9]
        self.assertEqual(len(at), 0)
        self.assertIn("BEAM", comp.bar_elements_by_id)

    def test_intersect_flag_roundtrip(self):
        import os
        s = Structure2D(); s.add_material("M", 30e6, 25.0)
        s.add_section("S", "M", 0.3, 0.5)
        m = s.add_geo_polyline("D", [[0, 0], [2, 2]], section_name="S")
        m.intersect_crossings = True
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.json")
            save_structure_json(s, p); r = load_structure_json(p)
        self.assertTrue(r.geometry_objects["D"].intersect_crossings)


class TestOptionBCache(unittest.TestCase):
    def test_compiled_mesh_cached_and_restored(self):
        import os
        s = Structure2D(); s.add_material("M", 30e6, 25.0)
        s.add_section("S", "M", 0.3, 0.5); s.add_support("Fix", True, True, True)
        s.add_load_case("LC")
        m = s.add_geo_polyline("MB", [[0, 0], [6, 0]], section_name="S",
                               divisions=6)
        s.assign_support(m.node_ids[0], "Fix")
        s.assign_support(m.node_ids[-1], "Fix")
        m.loads = [{"load_case": "LC", "wx": 0, "wy": -8, "coord_sys": "global"}]
        res = s.calculate()
        from xdfem2d.geo_expand import expand_geometry
        mesh, _ = expand_geometry(s)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.x2d")
            save_x2d(s, res, p, object_mesh=mesh)
            import zipfile
            self.assertIn("object_mesh.json", zipfile.ZipFile(p).namelist())
            struc2, res2, _view = load_x2d(p)
        cached = getattr(struc2, "_cached_object_mesh", None)
        self.assertIsNotNone(cached)
        self.assertEqual(len(cached.bar_elements), len(mesh.bar_elements))

    def test_no_cache_without_objects(self):
        import os, zipfile
        s = Structure2D(); s.add_material("M", 30e6, 25.0)
        s.add_section("S", "M", 0.3, 0.5)
        s.add_node("A", 0, 0); s.add_node("B", 4, 0)
        s.add_bar_element("E", "A", "B", "S")
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.x2d")
            save_x2d(s, {"analysis_cases": {}}, p, object_mesh=None)
            self.assertNotIn("object_mesh.json", zipfile.ZipFile(p).namelist())


class TestRealEndpointNodes(unittest.TestCase):
    def test_endpoints_are_real_nodes(self):
        s = Structure2D(); s.add_material("M", 30e6, 25.0)
        s.add_section("S", "M", 0.3, 0.5)
        m = s.add_geo_line("L", 0, 0, 0, 9, section_name="S", divisions=3)
        self.assertEqual(len(m.node_ids), 2)
        self.assertTrue(all(nid in s.nodes for nid in m.node_ids))
        p0 = s.nodes[m.node_ids[0]]; p1 = s.nodes[m.node_ids[1]]
        self.assertEqual((p0.x, p0.y, p1.x, p1.y), (0, 0, 0, 9))

    def test_support_survives_edit(self):
        s = Structure2D(); s.add_material("M", 30e6, 25.0)
        s.add_section("S", "M", 0.3, 0.5); s.add_support("Fix", True, True, True)
        m = s.add_geo_line("L", 0, 0, 0, 6, section_name="S", divisions=2)
        base = m.node_ids[0]
        s.assign_support(base, "Fix")
        # Edit in place (change divisions + move the far end via new points).
        m.divisions = 3
        s.materialize_object_nodes(m, [(0, 0), (0, 9)])
        # The base node id and its support are preserved.
        self.assertEqual(m.node_ids[0], base)
        self.assertTrue(any(a.node_id == base and a.support_name == "Fix"
                            for a in s.support_assignments))

    def test_delete_object_removes_unused_endpoint_nodes(self):
        s = Structure2D(); s.add_material("M", 30e6, 25.0)
        s.add_section("S", "M", 0.3, 0.5)
        m = s.add_geo_line("L", 0, 0, 4, 0, section_name="S")
        ids = list(m.node_ids)
        self.assertEqual(len(s.nodes), 2)
        s.remove_geo_object("L")
        self.assertEqual(len(s.nodes), 0)   # both endpoint nodes removed
        self.assertNotIn("L", s.geometry_objects)

    def test_delete_object_keeps_shared_node(self):
        s = Structure2D(); s.add_material("M", 30e6, 25.0)
        s.add_section("S", "M", 0.3, 0.5)
        # Two lines sharing the point (4,0).
        a = s.add_geo_line("A", 0, 0, 4, 0, section_name="S")
        b = s.add_geo_line("B", 4, 0, 4, 3, section_name="S")
        shared = a.node_ids[-1]
        self.assertEqual(shared, b.node_ids[0])   # reused coincident node
        s.remove_geo_object("A")
        self.assertIn(shared, s.nodes)            # still used by B


class TestNodeDrivenArc(unittest.TestCase):
    def test_arc_by_3_points(self):
        s = Structure2D(); s.add_material("M", 30e6, 25.0)
        s.add_section("S", "M", 0.3, 0.5)
        # A semicircle through (1,0),(0,1),(-1,0) → radius 1 about origin.
        a = s.add_geo_arc_3pts("A", (1, 0), (0, 1), (-1, 0), section_name="S",
                               divisions=12)
        self.assertEqual(len(a.node_ids), 3)
        comp, tr = expand_geometry(s)
        for nid in tr["A"]["nodes"]:
            n = comp.nodes[nid]
            self.assertAlmostEqual(math.hypot(n.x, n.y), 1.0, places=6)

    def test_collinear_arc_blocked(self):
        s = Structure2D(); s.add_material("M", 30e6, 25.0)
        s.add_section("S", "M", 0.3, 0.5)
        with self.assertRaises(ValueError):
            s.add_geo_arc_3pts("A", (0, 0), (1, 0), (2, 0), section_name="S")

    def test_moving_a_node_reshapes_arc(self):
        # Two-way: the arc is derived from its 3 nodes, so moving one reshapes it.
        s = Structure2D(); s.add_material("M", 30e6, 25.0)
        s.add_section("S", "M", 0.3, 0.5)
        a = s.add_geo_arc_3pts("A", (1, 0), (0, 1), (-1, 0), section_name="S",
                               divisions=8)
        mid = a.node_ids[1]
        # Move the mid node further out → larger arc; all points still on the new
        # circle through the (unchanged) endpoints and the moved mid.
        s.nodes[mid].x = 0.0; s.nodes[mid].y = 2.0
        comp, tr = expand_geometry(s)
        from xdfem2d.geo_expand import circle_from_3pts
        cx, cy, r = circle_from_3pts((1, 0), (0, 2), (-1, 0))
        for nid in tr["A"]["nodes"]:
            n = comp.nodes[nid]
            self.assertAlmostEqual(math.hypot(n.x - cx, n.y - cy), r, places=6)

    def test_add_geo_arc_center_matches_3pts(self):
        s = Structure2D(); s.add_material("M", 30e6, 25.0)
        s.add_section("S", "M", 0.3, 0.5)
        a = s.add_geo_arc("A", 0, 0, 3.0, 0.0, 90.0, section_name="S")
        pts = s.object_defining_points(a)
        self.assertAlmostEqual(pts[0][0], 3.0); self.assertAlmostEqual(pts[0][1], 0.0)
        self.assertAlmostEqual(pts[2][0], 0.0, places=6)
        self.assertAlmostEqual(pts[2][1], 3.0)


class TestModelCheck(unittest.TestCase):
    def _types(self, issues):
        return sorted({i["type"] for i in issues})

    def test_free_node_and_crossing_and_tee_detected(self):
        from xdfem2d.model_check import model_check
        s = Structure2D(); s.add_material("M", 30e6, 25.0)
        s.add_section("S", "M", 0.3, 0.5)
        # explicit frame (no objects) → check runs on it directly
        s.add_node("A", 0, 0); s.add_node("B", 4, 0)
        s.add_node("C", 2, -2); s.add_node("D", 2, 2)
        s.add_node("P", 2, 0)          # on AB interior (tee) and isolated (free)
        s.add_bar_element("AB", "A", "B", "S")
        s.add_bar_element("CD", "C", "D", "S")   # crosses AB at (2,0), no node
        issues = model_check(s)
        t = self._types(issues)
        self.assertIn("free_node", t)
        self.assertIn("tee_junction", t)
        self.assertIn("crossing", t)

    def test_object_model_is_clean_after_resolution(self):
        from xdfem2d.model_check import model_check
        # A beam + a crossing diagonal flagged to intersect → after expansion the
        # crossing is connected and the check finds nothing.
        s = Structure2D(); s.add_material("M", 30e6, 25.0)
        s.add_section("S", "M", 0.3, 0.5)
        s.add_node("A", 0, 0); s.add_node("B", 4, 0)
        s.add_bar_element("BEAM", "A", "B", "S")
        m = s.add_geo_polyline("DIAG", [[2, -2], [2, 2]], section_name="S")
        m.intersect_crossings = True
        self.assertEqual(model_check(s), [])

    def test_triangles_connect_their_nodes(self):
        """A meshed wall has no bars at all, and every one of its nodes was
        reported isolated.

        The count came back as '357 free nodes, none connected to any
        element', which reads as a mesh that failed to generate — alarming and
        false. 'used' was collected from the bars alone.
        """
        from xdfem2d.model_check import model_check
        s = Structure2D(); s.add_material("M", 30e6, 25.0)
        s.add_tri_section("T", "M", thickness=0.2)
        s.add_geo_rectangle("R", (0, 0), (4, 3), section_name="T",
                            target_size=0.5)
        issues = model_check(s)
        self.assertNotIn("free_node", self._types(issues))

    def test_a_node_on_nothing_is_still_free_in_a_meshed_model(self):
        """The fix must not go the other way and silence the real finding."""
        from xdfem2d.model_check import model_check
        s = Structure2D(); s.add_material("M", 30e6, 25.0)
        s.add_tri_section("T", "M", thickness=0.2)
        s.add_geo_rectangle("R", (0, 0), (4, 3), section_name="T",
                            target_size=0.5)
        s.add_node("LOOSE", 10.0, 10.0)
        free = [i for i in model_check(s) if i["type"] == "free_node"]
        self.assertEqual(len(free), 1)
        self.assertIn("LOOSE", free[0]["msg"])

    def test_a_triangle_built_by_hand_counts_too(self):
        """Not only the meshed path: explicit triangles connect nodes as well."""
        from xdfem2d.model_check import model_check
        s = Structure2D(); s.add_material("M", 30e6, 25.0)
        s.add_tri_section("T", "M", thickness=0.2)
        s.add_node("A", 0, 0); s.add_node("B", 1, 0); s.add_node("C", 0, 1)
        s.add_tri_element("T1", "A", "B", "C", "T")
        self.assertEqual(model_check(s), [])

    def test_object_unconnected_crossing_reported(self):
        from xdfem2d.model_check import model_check
        s = Structure2D(); s.add_material("M", 30e6, 25.0)
        s.add_section("S", "M", 0.3, 0.5)
        s.add_node("A", 0, 0); s.add_node("B", 4, 0)
        s.add_bar_element("BEAM", "A", "B", "S")
        s.add_geo_polyline("DIAG", [[2, -2], [2, 2]], section_name="S")  # no flag
        issues = model_check(s)
        self.assertIn("crossing", self._types(issues))


if __name__ == "__main__":
    unittest.main()
