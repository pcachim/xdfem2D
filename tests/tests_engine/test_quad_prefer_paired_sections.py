"""dev/IMPLEMENT_QUAD.md Phase 8, GUI item 2b revision: GeoRectangle/
GeoPolygon.prefer_quad.

The GUI's "Panel sections" dialog now always creates a same-named TriSection
AND QuadSection pair together (one logical "Panel", two engine sections) —
so a surface object's tri_section_name can no longer signal which kind to
mesh into by itself (a TriSection under that name always exists too, so the
old tri-first-else-quad resolution in geo_expand._expand_surface would
always pick triangles). prefer_quad breaks the tie explicitly. This is the
engine-level counterpart of test_quad_meshing.py, focused specifically on
the paired-section / tie-breaking behaviour that file's cases never exercise
(they all use a section name that resolves in exactly one dict).
"""
from __future__ import annotations

import unittest

from context import Structure2D
from xdfem2d.geo_expand import expand_geometry


def _paired_model(prefer_quad=False, domain='plate'):
    s = Structure2D(domain=domain)
    s.add_material('M', elastic_modulus=30.0e9, unit_weight=0.0, poisson=0.2)
    formulation_tri, formulation_quad = (
        ('DKT', 'DKT4') if domain == 'plate' else ('CST', 'Q4'))
    s.add_tri_section('Panel1', 'M', thickness=0.2, formulation=formulation_tri)
    s.add_quad_section('Panel1', 'M', thickness=0.2, formulation=formulation_quad)
    s.add_geo_rectangle('R1', (0, 0), (4.0, 4.0), section_name='Panel1',
                        target_size=1.0, prefer_quad=prefer_quad)
    return s


class TestPreferQuadDefaultsToTriangles(unittest.TestCase):
    """Unchanged pre-Phase-8-revision behaviour: with prefer_quad left at
    its default (False), a paired same-named section meshes into triangles,
    exactly as if the QuadSection sibling didn't exist."""

    def test_paired_section_meshes_triangles_by_default(self):
        s = _paired_model(prefer_quad=False)
        compiled, trace = expand_geometry(s)
        self.assertEqual(len(compiled.quad_elements), 0)
        self.assertGreater(len(compiled.tri_elements), 0)
        self.assertEqual(trace['R1']['quads'], [])
        self.assertTrue(all(t.section_name == 'Panel1'
                            for t in compiled.tri_elements))


class TestPreferQuadTruePicksQuads(unittest.TestCase):
    """prefer_quad=True checks quad_sections first: on a plain rectangular
    grid (no fallback needed), the whole surface meshes into quads."""

    def test_paired_section_meshes_quads_when_preferred(self):
        s = _paired_model(prefer_quad=True)
        compiled, trace = expand_geometry(s)
        self.assertEqual(len(compiled.tri_elements), 0)
        self.assertGreater(len(compiled.quad_elements), 0)
        self.assertEqual(trace['R1']['tris'], [])
        self.assertTrue(all(q.section_name == 'Panel1'
                            for q in compiled.quad_elements))


class TestFallbackCellsReuseTheRealPairedTriSection(unittest.TestCase):
    """When prefer_quad=True picks the quad path but some cells still need a
    triangle fallback (skewed outline), those fallback triangles must use
    the REAL user-defined 'Panel1' TriSection — not a derived 'Panel1~tri'
    section auto-built from the quad's formulation mapping (the pre-existing
    _tri_fallback_section_for path, still correct for a genuinely unpaired
    QuadSection, but wrong/wasteful here since a real paired section already
    carries the user's own chosen tri formulation)."""

    def test_fallback_triangles_use_the_paired_name_not_a_derived_one(self):
        s = Structure2D(domain='plane')
        s.add_material('M', elastic_modulus=30.0e9, unit_weight=0.0,
                       poisson=0.2)
        # Allman on purpose (not CST, the auto-derived mapping's target) —
        # if fallback cells used the derived section instead of the real
        # paired one, this formulation choice would silently not survive.
        s.add_tri_section('Panel1', 'M', thickness=0.1, formulation='Allman')
        s.add_quad_section('Panel1', 'M', thickness=0.1, formulation='QM6')
        pts = [(0, 0), (10, 0), (9, 1), (0, 1)]   # same tapered outline
        s.add_geo_polygon('P1', pts, section_name='Panel1',
                          target_size=1.0, prefer_quad=True)
        compiled, trace = expand_geometry(s)
        self.assertGreater(len(compiled.quad_elements), 0)
        self.assertGreater(len(compiled.tri_elements), 0)
        section_names = {t.section_name for t in compiled.tri_elements}
        self.assertEqual(section_names, {'Panel1'})
        self.assertNotIn('Panel1~tri', compiled.tri_sections)


class TestPreferQuadTrueStillFallsBackWithNoQuadSection(unittest.TestCase):
    """prefer_quad=True with no matching QuadSection at all (an ordinary,
    unpaired TriSection) must still mesh into triangles — the tie-break only
    matters when both dicts actually have a same-named entry."""

    def test_no_quad_section_under_that_name_still_meshes_triangles(self):
        s = Structure2D(domain='plane')
        s.add_material('M', elastic_modulus=30.0e9, unit_weight=0.0,
                       poisson=0.2)
        s.add_tri_section('OnlyTri', 'M', thickness=0.15, formulation='CST')
        s.add_geo_rectangle('R1', (0, 0), (2.0, 2.0),
                            section_name='OnlyTri', target_size=1.0,
                            prefer_quad=True)
        compiled, trace = expand_geometry(s)
        self.assertEqual(len(compiled.quad_elements), 0)
        self.assertGreater(len(compiled.tri_elements), 0)


class TestPreferQuadRoundTripsThroughSaveLoad(unittest.TestCase):

    def test_prefer_quad_survives_save_and_load(self):
        import os
        import tempfile
        s = _paired_model(prefer_quad=True)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.json")
            s.save(path)
            s2 = Structure2D.load(path)
        self.assertTrue(s2.geometry_objects['R1'].prefer_quad)

    def test_prefer_quad_false_is_the_default_on_load_of_older_files(self):
        """A file saved before this field existed has no 'prefer_quad' key
        at all — must default to False, not raise or default to True."""
        import os
        import tempfile
        s = _paired_model(prefer_quad=False)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.json")
            s.save(path)
            import json
            with open(path) as f:
                data = json.load(f)
            for obj in data.get('geometry_objects', []):
                obj.pop('prefer_quad', None)
            with open(path, 'w') as f:
                json.dump(data, f)
            s2 = Structure2D.load(path)
        self.assertFalse(s2.geometry_objects['R1'].prefer_quad)


if __name__ == '__main__':
    unittest.main()
