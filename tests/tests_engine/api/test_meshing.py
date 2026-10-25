"""Polygon meshing (Delaunay) + CST surface-object expansion."""
from __future__ import annotations

import unittest

from context import Structure2D
from xdfem2d.meshing import mesh_polygon, polygon_area, point_in_poly
from xdfem2d.geo_expand import expand_geometry
from xdfem2d.structure_io import _to_dict, _from_dict


def _tri_area(pts, a, b, c):
    (x1, y1), (x2, y2), (x3, y3) = pts[a], pts[b], pts[c]
    return abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)) / 2.0


class TestMesher(unittest.TestCase):
    def test_rectangle_area_conserved(self):
        poly = [(0, 0), (4, 0), (4, 2), (0, 2)]
        pts, tris = mesh_polygon(poly, 0.5)
        self.assertGreater(len(tris), 0)
        A = sum(_tri_area(pts, *t) for t in tris)
        self.assertAlmostEqual(A, abs(polygon_area(poly)), places=6)

    def test_concave_L_shape(self):
        poly = [(0, 0), (4, 0), (4, 2), (2, 2), (2, 4), (0, 4)]
        pts, tris = mesh_polygon(poly, 0.5)
        A = sum(_tri_area(pts, *t) for t in tris)
        self.assertAlmostEqual(A, abs(polygon_area(poly)), places=6)
        # No triangle centroid may fall outside the (concave) polygon.
        for a, b, c in tris:
            cx = (pts[a][0] + pts[b][0] + pts[c][0]) / 3
            cy = (pts[a][1] + pts[b][1] + pts[c][1]) / 3
            self.assertTrue(point_in_poly(cx, cy, poly))

    def test_quality_pass_lifts_min_angle_on_a_curved_boundary(self):
        # A faceted disc: the plain Delaunay leaves slivers where the chords
        # meet the interior grid; the smoothing + edge-flip pass raises the
        # minimum element angle well above the sliver range.
        import math
        poly = [(5 * math.cos(a), 5 * math.sin(a))
                for a in [i * 2 * math.pi / 24 for i in range(24)]]
        # Target ≈ facet length (the regime the curved templates use, so the
        # boundary chords are not sub-split into boundary-locked slivers).
        facet = math.dist(poly[0], poly[1])
        pts, tris = mesh_polygon(poly, facet * 1.15)
        self.assertGreater(len(tris), 0)

        def _min_angle(a, b, c):
            P = [pts[a], pts[b], pts[c]]
            best = 180.0
            for o, x, y in ((0, 1, 2), (1, 0, 2), (2, 0, 1)):
                v1 = (P[x][0] - P[o][0], P[x][1] - P[o][1])
                v2 = (P[y][0] - P[o][0], P[y][1] - P[o][1])
                d = ((v1[0] * v2[0] + v1[1] * v2[1])
                     / (math.hypot(*v1) * math.hypot(*v2) + 1e-30))
                best = min(best, math.degrees(math.acos(max(-1.0, min(1.0, d)))))
            return best

        self.assertGreater(min(_min_angle(*t) for t in tris), 25.0)
        # Area is still conserved (the boundary and point count are untouched).
        A = sum(_tri_area(pts, *t) for t in tris)
        self.assertAlmostEqual(A, abs(polygon_area(poly)), places=6)
        # No degenerate (zero-area) triangle.
        self.assertGreater(min(_tri_area(pts, *t) for t in tris), 1e-9)

    def test_forced_nodes_present(self):
        poly = [(0, 0), (4, 0), (4, 2), (0, 2)]
        pts, tris = mesh_polygon(poly, 1.0, forced=[(2, 0), (1, 1)])
        self.assertTrue(any(abs(x - 2) < 1e-6 and abs(y) < 1e-6 for x, y in pts))
        self.assertTrue(any(abs(x - 1) < 1e-6 and abs(y - 1) < 1e-6 for x, y in pts))

    # ── structured (transfinite) branch ──────────────────────────────

    @staticmethod
    def _mirror_report(pts, tris, mirror):
        """(#nodes, #triangles) of *tris* whose image under *mirror* is absent."""
        k = lambda x, y: (round(x, 9), round(y, 9))          # noqa: E731
        have = {k(*p) for p in pts}
        n_bad = sum(1 for p in pts if k(*mirror(*p)) not in have)
        tset = {frozenset(k(*pts[i]) for i in t) for t in tris}
        t_bad = sum(1 for t in tset
                    if frozenset(k(*mirror(*c)) for c in t) not in tset)
        return n_bad, t_bad

    def test_structured_branch_is_symmetric(self):
        """A rectangle with no forced node takes the structured branch, whose
        quadrant diagonal pattern must be invariant under reflection about both
        mid-planes and under the 180° rotation. This is the property a plain
        Delaunay triangulation of the same grid does NOT have — its cocircular
        cells are tie-broken the same way everywhere — which showed up as
        asymmetric displacements under a symmetric load."""
        poly = [(0.0, 0.0), (4.0, 0.0), (4.0, 2.0), (0.0, 2.0)]
        pts, tris = mesh_polygon(poly, 0.5)
        self.assertGreater(len(tris), 0)
        for name, mirror in (
                ("horizontal", lambda x, y: (x, 2.0 - y)),
                ("vertical",   lambda x, y: (4.0 - x, y)),
                ("rotation",   lambda x, y: (4.0 - x, 2.0 - y))):
            n_bad, t_bad = self._mirror_report(pts, tris, mirror)
            self.assertEqual((n_bad, t_bad), (0, 0),
                             f"{name} symmetry broken: {n_bad} nodes, "
                             f"{t_bad} triangles unmatched")

    def test_structured_branch_conserves_area(self):
        """The transfinite map must tile a general convex quad exactly."""
        poly = [(0.0, 0.0), (3.0, 0.0), (3.4, 1.6), (0.4, 1.2)]
        pts, tris = mesh_polygon(poly, 0.25)
        A = sum(_tri_area(pts, *t) for t in tris)
        self.assertAlmostEqual(A, abs(polygon_area(poly)), places=6)

    def test_structured_divisions_match_target(self):
        """Divisions are simply rounded to the nearest count for the target
        element size — no parity requirement. A 4 x 2 rectangle at target 0.5
        gives an 8 x 4 grid (already even here, so this is unchanged from
        before parity was dropped)."""
        poly = [(0.0, 0.0), (4.0, 0.0), (4.0, 2.0), (0.0, 2.0)]
        pts, tris = mesh_polygon(poly, 0.5)
        self.assertEqual(len(pts), 9 * 5)
        self.assertEqual(len(tris), 8 * 4 * 2)

    def test_structured_divisions_can_be_odd(self):
        """An odd division count (once rejected/rounded away) is now honoured
        directly: the middle row/column, which is its own mirror image, is
        resolved with a small triangle fan around a new centre point instead
        of a diagonal, and the mesh stays exactly area-conserving."""
        from xdfem2d.meshing import mesh_quad_structured, _structured_grid

        rect = [(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0)]
        _, nu, nv = _structured_grid(rect, 1.0)
        self.assertEqual((nu, nv), (5, 5))  # fully odd grid, was impossible before

        # every cell on the middle row/column is its own mirror image and
        # gets a fan centre instead of a diagonal (5x5 grid: the middle row
        # and middle column together are 9 such cells, one shared corner).
        pts, tris = mesh_quad_structured(rect, 1.0)
        self.assertEqual(len(pts), (nu + 1) * (nv + 1) + 9)

        def tri_area(t):
            (x1, y1), (x2, y2), (x3, y3) = (pts[t[0]], pts[t[1]], pts[t[2]])
            return abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)) / 2.0

        self.assertAlmostEqual(sum(tri_area(t) for t in tris), 25.0)

    def test_dispatch_falls_back_to_delaunay(self):
        """The structured branch cannot honour a node that is not already on its
        grid, so those outlines go to the Delaunay branch (which conforms)."""
        from xdfem2d.meshing import mesh_quad_structured
        rect = [(0.0, 0.0), (4.0, 0.0), (4.0, 2.0), (0.0, 2.0)]
        concave = [(0, 0), (4, 0), (4, 2), (2, 2), (2, 4), (0, 4)]
        ok = lambda forced: mesh_quad_structured(rect, 0.5, forced) is not None
        self.assertTrue(ok([]))
        self.assertTrue(ok([(0.0, 0.0)]))          # a corner is a grid node
        self.assertTrue(ok([(1.5, 0.0)]))          # falls on the grid
        self.assertTrue(ok([(2.0, 1.0)]))          # interior, but on the grid
        self.assertFalse(ok([(1.37, 0.0)]))        # off-grid, on an edge
        self.assertFalse(ok([(1.9, 1.0)]))         # off-grid, inside
        self.assertIsNone(mesh_quad_structured(concave, 0.5, []))   # not a quad

    def test_refine_boundary_does_not_oversplit(self):
        """A boundary piece exactly one target long must not be split in two.

        Each cut parameter comes from a projection, so such a piece can be
        reconstructed as target*(1+1e-15); a bare ceil() then plants a node at
        its midpoint. On an edge shared with an already-meshed neighbour that
        node has no partner on the other side — a hanging node and a crack.
        """
        from xdfem2d.meshing import refine_boundary
        poly = [(-1.0, 5.0), (6.0, 5.0), (6.0, 9.0), (-1.0, 9.0)]
        # Forced nodes every 0.5 along the bottom edge, as a neighbouring
        # structured panel would leave them.
        forced = [(-1.0 + 0.5 * i, 5.0) for i in range(15)]
        b = refine_boundary(poly, 0.5, forced)
        on_edge = sorted({round(x, 6) for x, y in b if abs(y - 5.0) < 1e-9})
        self.assertEqual(on_edge, [round(-1.0 + 0.5 * i, 6) for i in range(15)],
                         "boundary gained a node the neighbour does not have")

    def test_adjacent_surfaces_conform(self):
        """Two surfaces sharing an edge must mesh conformingly: no node of one
        may land inside an edge of the other."""
        from xdfem2d.meshing import mesh_quad_structured
        rect = [(-1.0, 1.0), (6.0, 1.0), (6.0, 5.0), (-1.0, 5.0)]
        # A polygon sitting on the rectangle's top edge, as in the reported model.
        cap = [(-1.0, 5.0), (6.0, 5.0), (7.0, 7.5), (4.25, 7.75), (1.0, 6.25)]
        p_rect, t_rect = mesh_quad_structured(rect, 0.5)
        p_cap, t_cap = mesh_polygon(cap, 0.5, forced=list(p_rect))
        k = lambda p: (round(p[0], 6), round(p[1], 6))       # noqa: E731
        rect_on_edge = {k(p) for p in p_rect if abs(p[1] - 5.0) < 1e-9}
        cap_on_edge = {k(p) for p in p_cap if abs(p[1] - 5.0) < 1e-9}
        self.assertEqual(cap_on_edge, rect_on_edge,
                         "the two meshes disagree on the shared edge")

    def test_adjacent_panels_stay_structured(self):
        """Two panels sharing an edge are the normal case in a real model. The
        second one sees the first one's nodes along the shared edge; because
        they fall exactly on its own grid, it must still mesh structured rather
        than degrade to Delaunay."""
        from xdfem2d.meshing import mesh_quad_structured
        left = [(0.0, 0.0), (10.0, 0.0), (10.0, 4.0), (0.0, 4.0)]
        right = [(10.0, 0.0), (20.0, 0.0), (20.0, 4.0), (10.0, 4.0)]
        pts_l, _ = mesh_quad_structured(left, 0.5)
        shared = [p for p in pts_l if abs(p[0] - 10.0) < 1e-9]
        self.assertGreater(len(shared), 2)
        self.assertIsNotNone(mesh_quad_structured(right, 0.5, shared))
        # A neighbour meshed at a MODERATELY different size (0.3 vs 0.5, a
        # 4-division difference along the 4 m shared edge here) still lines
        # up: _fit_divisions searches division counts within max_steps=6 of
        # this quad's own target-derived count for one under which every
        # forced point lands on an exact grid line, and 0.3's own division
        # count is within that window — this is the documented "gently
        # adjusted, still structured" accommodation (_fit_divisions'
        # docstring), not a coincidence.
        pts_c, _ = mesh_quad_structured(left, 0.3)
        shared_c = [p for p in pts_c if abs(p[0] - 10.0) < 1e-9]
        self.assertIsNotNone(mesh_quad_structured(right, 0.5, shared_c))
        # A neighbour meshed at a GENUINELY different size — far outside
        # _fit_divisions' bounded search window — does NOT line up and must
        # fall back to the unstructured branch.
        pts_f, _ = mesh_quad_structured(left, 0.1)
        shared_f = [p for p in pts_f if abs(p[0] - 10.0) < 1e-9]
        self.assertIsNone(mesh_quad_structured(right, 0.5, shared_f))


class TestSurfaceExpansion(unittest.TestCase):
    def _wall(self):
        s = Structure2D()
        s.add_material("C", 30e6, 25.0, poisson=0.2)
        s.add_tri_section("W", "C", thickness=0.2)
        s.add_load_case("G").self_weight_factor = 1.0
        s.add_node("BL", 0, 0); s.add_node("BR", 4, 0)
        s.add_geo_rectangle("R1", (0, 0), (4, 2),
                            section_name="W", target_size=0.5)
        s.add_support("Pin", ux=True, uy=True)
        s.assign_support("BL", "Pin"); s.assign_support("BR", "Pin")
        return s

    def test_rectangle_expands_to_triangles(self):
        s = self._wall()
        compiled, trace = expand_geometry(s)
        self.assertGreater(len(compiled.tri_elements), 0)
        self.assertEqual(len(compiled.bar_elements), 0)
        self.assertEqual(len(trace["R1"]["tris"]), len(compiled.tri_elements))

    def test_surface_self_weight_equilibrium(self):
        s = self._wall()
        res = s.calculate()
        case = next(iter(res["analysis_cases"]))
        reac = res["analysis_cases"][case]["reactions"]
        ry = sum(r[1] for r in reac.values())
        self.assertAlmostEqual(ry, 25.0 * 0.2 * 8.0, places=6)   # γ·t·A
        ts = res["analysis_cases"][case]["tri_stress"]
        self.assertGreater(max(d["vm"] for d in ts.values()), 0.0)

    def test_boundary_conforms_to_existing_node(self):
        # A node placed on the boundary must survive expansion (conformity).
        s = self._wall()
        s.add_node("MID", 2, 0)
        compiled, _ = expand_geometry(s)
        self.assertIn("MID", compiled.nodes)

    def test_surface_roundtrip(self):
        s = self._wall()
        s2 = _from_dict(_to_dict(s))
        m = s2.geometry_objects["R1"]
        self.assertEqual(m.tri_section_name, "W")
        self.assertAlmostEqual(m.target_size, 0.5)
        # And it still expands after the round-trip.
        compiled, _ = expand_geometry(s2)
        self.assertGreater(len(compiled.tri_elements), 0)


if __name__ == "__main__":
    unittest.main()
