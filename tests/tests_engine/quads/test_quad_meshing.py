"""Quad meshing for surface objects (dev/IMPLEMENT_QUAD.md Phase 6).

A surface object (``GeoRectangle``/``GeoPolygon``) whose ``tri_section_name``
resolves to a ``QuadSection`` now meshes into quads through the structured
(transfinite) branch — ``xdfem2d.meshing.mesh_quad_structured_cells`` — with a
per-cell hard-validity + quality fallback to triangles, and a whole-object
fallback for a non-quad-eligible outline or a feature quads cannot carry
(area spring, area temperature). The unstructured/Delaunay branch stays
triangle-only throughout.

These are full-pipeline tests (``Structure2D`` -> ``expand_geometry`` ->
``calculate()``), not kernel tests — Phase 1/2/4/5's kernel- and
assembly-level coverage of the quad formulations themselves is unaffected;
this file is specifically the meshing/expansion wiring.
"""
from __future__ import annotations

import unittest

from context import Structure2D, assert_close
from xdfem2d.geo_expand import expand_geometry
from xdfem2d.meshing import mesh_quad_structured_cells, quad_cell_quality


class TestRectangularSlabMeshesIntoQuads(unittest.TestCase):
    """A plain rectangular surface referencing a QuadSection meshes entirely
    into quads (no per-cell fallback needed on an unskewed rectangular grid),
    and the full pipeline reproduces the MITC4 SS-plate benchmark from
    test_quad_assembly.py's TestMITC4PlateFullPipeline — the same physics,
    now reached through the surface-object path instead of hand-built
    QuadElements."""

    A, T, NU, Q, N = 4.0, 0.1, 0.3, 10_000.0, 8
    E = 30.0e9
    ALPHA = 0.00406

    def _model(self):
        s = Structure2D(domain='plate')
        s.add_material('M', elastic_modulus=self.E, unit_weight=0.0,
                       poisson=self.NU)
        s.add_quad_section('QS', 'M', thickness=self.T, formulation='MITC4')
        s.add_geo_rectangle('R1', (0, 0), (self.A, self.A),
                            section_name='QS', target_size=self.A / self.N)
        s.add_support('SS', ux=True, uy=False, tz=False)
        for nid in s.geometry_objects['R1'].node_ids:
            s.assign_support(nid, 'SS')
        s.add_load_case('LC')
        s.add_area_load('R1', 'LC', pz=-self.Q)
        s.add_analysis_case('AC', 'Linear', {'LC': 1.0})
        return s

    def test_meshes_all_quads_no_triangles(self):
        s = self._model()
        compiled, trace = expand_geometry(s)
        self.assertEqual(len(compiled.tri_elements), 0)
        self.assertEqual(len(compiled.quad_elements), self.N * self.N)
        self.assertEqual(len(trace['R1']['quads']), self.N * self.N)
        self.assertEqual(trace['R1']['tris'], [])

    def test_full_pipeline_matches_benchmark(self):
        s = self._model()
        compiled, _ = expand_geometry(s)
        r = s.calculate()
        disp = r['analysis_cases']['AC']['displacements']
        best = min(compiled.nodes.items(),
                  key=lambda kv: (kv[1].x - self.A / 2) ** 2
                                + (kv[1].y - self.A / 2) ** 2)
        w_center = disp[best[0]][0]
        D = self.E * self.T ** 3 / (12.0 * (1.0 - self.NU ** 2))
        exact = -self.ALPHA * self.Q * self.A ** 4 / D
        assert_close(self, w_center, exact, rel=0.01,
                     msg="surface-object MITC4 SS-plate centre w")

    def test_domain_consistent(self):
        s = self._model()
        self.assertEqual(s.domain_problems(), [])
        self.assertEqual(s.reference_problems(), [])


class TestMembraneQuadMesh(unittest.TestCase):
    """A plane-domain rectangle referencing a QM6 QuadSection meshes into
    quads and solves consistently — the membrane analogue of the plate case
    above, and a check that quad meshing isn't plate-domain-only."""

    def test_reaction_balances_self_weight(self):
        s = Structure2D(domain='plane')
        s.add_material('M', elastic_modulus=30.0e9, unit_weight=25.0,
                       poisson=0.2)
        s.add_quad_section('QS', 'M', thickness=0.2, formulation='QM6')
        s.add_geo_rectangle('R1', (0, 0), (2.0, 1.0),
                            section_name='QS', target_size=0.5)
        s.add_support('FIX', ux=True, uy=True, tz=True)
        ids = s.geometry_objects['R1'].node_ids
        s.assign_support(ids[0], 'FIX')
        s.assign_support(ids[1], 'FIX')
        s.add_load_case('SW', self_weight_factor=1.0)
        s.add_analysis_case('AC', 'Linear', {'SW': 1.0})
        compiled, _ = expand_geometry(s)
        self.assertGreater(len(compiled.quad_elements), 0)
        self.assertEqual(len(compiled.tri_elements), 0)
        r = s.calculate()
        reac = r['analysis_cases']['AC']['reactions']
        total_uy = sum(v[1] for v in reac.values())
        expected = 25.0 * 0.2 * (2.0 * 1.0)   # gamma * t * area
        assert_close(self, total_uy, expected, rel=1e-6,
                     msg="surface-object QM6 self-weight reaction")


class TestNonQuadOutlineFallsBackToTriangles(unittest.TestCase):
    """A GeoPolygon whose outline is not itself a convex quadrilateral (here,
    5 vertices) cannot use the structured branch at all — the whole surface
    falls back to the plain triangle mesher, exactly as before Phase 6, even
    though its section is a QuadSection."""

    def test_pentagon_all_triangles(self):
        s = Structure2D(domain='plane')
        s.add_material('M', elastic_modulus=30.0e9, unit_weight=0.0,
                       poisson=0.2)
        s.add_quad_section('QS', 'M', thickness=0.15, formulation='QM6')
        pts = [(0, 0), (2, 0), (2.5, 1.0), (1.0, 2.0), (-0.5, 1.0)]
        s.add_geo_polygon('P1', pts, section_name='QS', target_size=0.5)
        compiled, trace = expand_geometry(s)
        self.assertEqual(len(compiled.quad_elements), 0)
        self.assertGreater(len(compiled.tri_elements), 0)
        self.assertEqual(trace['P1']['quads'], [])
        # The fallback section is derived from QS (QM6 -> CST), not QS itself.
        used = {t.section_name for t in compiled.tri_elements}
        self.assertEqual(used, {'QS~tri'})
        ts = compiled.tri_sections['QS~tri']
        self.assertEqual(ts.formulation, 'CST')
        self.assertAlmostEqual(ts.thickness, 0.15)


class TestSkewedQuadOutlinePerCellFallback(unittest.TestCase):
    """A convex-but-skewed 4-vertex GeoPolygon (still eligible for the
    structured branch) mixes quad and triangle cells: the well-shaped
    interior cells stay quads, the badly-skewed ones near the tapered end
    fall back to triangles cell-by-cell, per mesh_quad_structured_cells'
    documented skew/aspect thresholds."""

    def test_mixed_quad_and_triangle_cells(self):
        s = Structure2D(domain='plane')
        s.add_material('M', elastic_modulus=30.0e9, unit_weight=0.0,
                       poisson=0.2)
        s.add_quad_section('QS', 'M', thickness=0.1, formulation='QM6')
        # A long, strongly tapered quadrilateral: mesh_quad_structured_cells
        # is expected (and separately unit-tested below) to keep some cells
        # as quads and split others into triangles.
        pts = [(0, 0), (10, 0), (9, 1), (0, 1)]
        s.add_geo_polygon('P1', pts, section_name='QS', target_size=1.0)
        compiled, trace = expand_geometry(s)
        self.assertGreater(len(compiled.quad_elements), 0)
        self.assertGreater(len(compiled.tri_elements), 0)
        self.assertEqual(len(trace['P1']['quads']), len(compiled.quad_elements))
        self.assertEqual(len(trace['P1']['tris']), len(compiled.tri_elements))
        self.assertEqual(s.domain_problems(), [])
        self.assertEqual(s.reference_problems(), [])

    def test_kernel_level_mixed_cells_directly(self):
        """The meshing-only check behind the model-level test above: on this
        outline mesh_quad_structured_cells must actually produce both quad
        and triangle cells (not silently degrade to all-triangle), and every
        emitted quad cell must be within the documented quality thresholds."""
        pts = [(0, 0), (10, 0), (9, 1), (0, 1)]
        result = mesh_quad_structured_cells(pts, 1.0)
        self.assertIsNotNone(result)
        points, quads, tris = result
        self.assertGreater(len(quads), 0)
        self.assertGreater(len(tris), 0)
        for (a, b, c, d) in quads:
            coords = [points[a], points[b], points[c], points[d]]
            skew, aspect = quad_cell_quality(coords)
            self.assertLessEqual(skew, 30.0)
            self.assertLessEqual(aspect, 4.0)

    def test_matches_plain_structured_grid(self):
        """The point grid mesh_quad_structured_cells builds is identical to
        mesh_quad_structured's — the two share _structured_grid — so a
        surface split between a quad-eligible and a triangle-only neighbour
        still conforms node-for-node along their shared edge."""
        from xdfem2d.meshing import mesh_quad_structured
        rect = [(0, 0), (4, 0), (4, 4), (0, 4)]
        p1, quads, tris = mesh_quad_structured_cells(rect, 1.0)
        p2, t2 = mesh_quad_structured(rect, 1.0)
        self.assertEqual(p1, p2)
        # A plain square grid is all well-shaped: every cell stays a quad,
        # none fall back to the triangle split mesh_quad_structured used.
        self.assertEqual(len(quads), 16)
        self.assertEqual(len(tris), 0)


class TestAreaSpringAppliesToQuadCells(unittest.TestCase):
    """A surface carrying an area spring now DOES mesh into quads
    (dev/IMPLEMENT_QUAD.md Phase 6 follow-up: ``QuadAreaSpring`` closes the
    gap that used to force these surfaces to an all-triangle mesh), and every
    quad cell gets its own QuadAreaSpring, exactly as every triangle cell
    gets a TriAreaSpring."""

    def test_quads_created_and_area_spring_applied_per_cell(self):
        s = Structure2D(domain='plate')
        s.add_material('M', elastic_modulus=30.0e9, unit_weight=0.0,
                       poisson=0.2)
        s.add_quad_section('QS', 'M', thickness=0.2, formulation='MITC4')
        s.add_geo_rectangle('R1', (0, 0), (2, 2), section_name='QS',
                            target_size=1.0)
        s.add_area_spring('R1', kz=1000.0)
        compiled, trace = expand_geometry(s)
        self.assertGreater(len(compiled.quad_elements), 0)
        self.assertEqual(trace['R1']['quads'], [q.id for q in compiled.quad_elements])
        self.assertEqual(len(compiled.quad_area_springs), len(compiled.quad_elements))
        self.assertEqual(len(compiled.tri_area_springs), len(compiled.tri_elements))
        for asp in compiled.quad_area_springs:
            self.assertEqual(asp.kz, 1000.0)


class TestAreaTemperatureAppliesToQuadCells(unittest.TestCase):
    """Same as the area-spring case, for an area temperature load — quads now
    carry a real thermal-load feature (``QuadTemperatureLoad``), so a surface
    with an area temperature also meshes into quads."""

    def test_quads_created_and_area_temperature_applied_per_cell(self):
        s = Structure2D(domain='plane')
        s.add_material('M', elastic_modulus=30.0e9, unit_weight=0.0,
                       poisson=0.2)
        s.add_quad_section('QS', 'M', thickness=0.1, formulation='QM6')
        s.add_geo_rectangle('R1', (0, 0), (2, 2), section_name='QS',
                            target_size=1.0)
        s.add_load_case('LC')
        s.add_area_temperature_load('R1', 'LC', dt_uniform=10.0)
        compiled, trace = expand_geometry(s)
        self.assertGreater(len(compiled.quad_elements), 0)
        self.assertEqual(trace['R1']['quads'], [q.id for q in compiled.quad_elements])
        self.assertEqual(len(compiled.quad_temperature_loads), len(compiled.quad_elements))
        for tl in compiled.quad_temperature_loads:
            self.assertEqual(tl.dt_mean, 10.0)


class TestQuadSectionInPlaneDomainStillFlagged(unittest.TestCase):
    """The domain_problems() fix (dev/IMPLEMENT_QUAD.md Phase 6): a plate
    QuadSection (MITC4) referenced by a surface object in a plane-domain
    model must still be flagged, now that the surface-derived section name
    is checked against the RIGHT table (quad_sections, not just
    tri_sections) — before the fix this fell into tri_sections.get(name),
    returned None, and passed silently."""

    def test_plate_quad_section_via_surface_flagged(self):
        s = Structure2D(domain='plane')
        s.add_material('M', elastic_modulus=30.0e9, unit_weight=0.0,
                       poisson=0.2)
        s.add_quad_section('QSp', 'M', thickness=0.2, formulation='MITC4')
        s.add_geo_rectangle('R1', (0, 0), (2, 2), section_name='QSp',
                            target_size=1.0)
        probs = s.domain_problems()
        self.assertTrue(any('MITC4' in p for p in probs), probs)
        with self.assertRaises(ValueError):
            s.check_references()

    def test_membrane_tri_section_via_surface_still_fine(self):
        """Sanity check that the split doesn't over-trigger: a surface using
        a plain CST TriSection in a plane model is unaffected."""
        s = Structure2D(domain='plane')
        s.add_material('M', elastic_modulus=30.0e9, unit_weight=0.0,
                       poisson=0.2)
        s.add_tri_section('TS', 'M', thickness=0.2, formulation='CST')
        s.add_geo_rectangle('R1', (0, 0), (2, 2), section_name='TS',
                            target_size=1.0)
        self.assertEqual(s.domain_problems(), [])


class TestSurfaceEdgeLoadThicknessFix(unittest.TestCase):
    """Regression for the thickness-lookup bug found during Phase 6's
    investigation: _apply_surface_edge_loads used to resolve the section only
    against tri_sections, silently defaulting to thickness=1.0 for a surface
    referencing a QuadSection. A pz-independent way to see the fix: an edge
    load's resultant nodal force is proportional to the resolved thickness,
    so comparing it against the QuadSection's real (non-1.0) thickness
    catches a regression back to the old default."""

    def test_edge_load_uses_quad_section_thickness(self):
        s = Structure2D(domain='plane')
        s.add_material('M', elastic_modulus=30.0e9, unit_weight=0.0,
                       poisson=0.2)
        thickness = 0.35   # deliberately far from the old silent default (1.0)
        s.add_quad_section('QS', 'M', thickness=thickness, formulation='QM6')
        s.add_geo_rectangle('R1', (0, 0), (2, 1), section_name='QS',
                            target_size=0.5)
        ids = s.geometry_objects['R1'].node_ids   # [a, b, c, d] CCW
        s.add_load_case('LC')
        # A pressure (pn, normal to the edge) load along one boundary side.
        s.add_surface_edge_load('EL1', 'R1', ids[0], ids[1], 'LC',
                                coord_sys='local', pn=100.0)
        compiled, _ = expand_geometry(s)
        total_fy = sum(p.fy for p in compiled.point_loads
                      if p.load_case_id == 'LC')
        # Edge a->b is the bottom side (0,0)->(2,0); an outward normal
        # pressure pushes in -y, magnitude pn * length * thickness.
        expected = -100.0 * 2.0 * thickness
        assert_close(self, total_fy, expected, rel=1e-9,
                     msg="surface edge load resultant uses the resolved "
                         "QuadSection thickness, not the old 1.0 default")


if __name__ == '__main__':
    unittest.main()
