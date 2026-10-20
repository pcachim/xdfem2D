"""Bug report: switching an existing rectangle/polygon geometry object created
by "New from Template" to "Mesh as: Quads" produced the just-added
unpaired-section warning ("Section 'Slab' has no quad formulation paired with
it — meshing will fall back to triangles everywhere...") and, if continued
anyway, still meshed 100% triangles — because every rectangle/polygon-object
template (templates.py) created only a TriSection/plate section ('Wall' /
'Slab' / 'CST'), never the paired QuadSection a "Panel section"
(dev/IMPLEMENT_QUAD.md Phase 8's TriSectionPanel) is supposed to be.

Fixed by adding a same-named ``add_quad_section`` call right after every
``add_tri_section``/``add_plate_section`` call in every template builder —
including the plain explicit-mesh builders like ``_tpl_wall``/``_tpl_slab``
(per the user's decision that every template always creates the slab/wall
pair, so no section is ever a dead end for a later "Mesh as: Quads") and
``_default_new_structure`` (the bare "File ▸ New" seed section).
Quad formulation picked to match the plane/plate domain defaults set
alongside this fix (domain_adapter.PLANE/PLATE.quad_formulations[0]): QM6
for plane ('Wall'/'CST'), MITC4 for plate ('Slab').

This does not change what any template PRODUCES by default — prefer_quad on
every GeoRectangle/GeoPolygon these builders create is still left at its
default False, so the mesh is still 100% triangles unless the user
explicitly switches "Mesh as" to Quads afterwards (in the GUI) or sets
``m.prefer_quad = True`` directly. It only makes that switch actually work.
"""
import unittest

from context import Structure2D  # noqa: F401  (adds src/ to sys.path)
from xdfem2d import templates as T
from xdfem2d.geo_expand import expand_geometry


def _obj_id(struc):
    """The single geometry object every builder below creates (or the first
    one, for _tpl_wall_frame_obj/_tpl_flat_slab_obj which create several)."""
    return next(iter(struc.geometry_objects))


class TestTemplateObjectsHaveAPairedQuadSection(unittest.TestCase):
    """Every _obj template's section is a real "Panel" (tri+quad pair under
    the same name) — checked directly against tri_sections/quad_sections,
    not just that meshing happens to produce quads (that is the next class)."""

    def test_wall_obj(self):
        s = T._tpl_wall_obj(4.0, 3.0, 4, 3, 0.20, 3.0e7, 25.0)
        self.assertIn('Wall', s.tri_sections)
        self.assertIn('Wall', s.quad_sections)

    def test_wall_beam_obj(self):
        s = T._tpl_wall_beam_obj(4.0, 3.0, 4, 3, 0.20, 3.0e7, 25.0)
        self.assertIn('Wall', s.tri_sections)
        self.assertIn('Wall', s.quad_sections)

    def test_wall_frame_obj(self):
        s = T._tpl_wall_frame_obj(2, 2, 5.0, 3.0, 3.0, 3,
                                  0.3, 0.5, 3.0e7, 25.0, 0.20)
        self.assertIn('Wall', s.tri_sections)
        self.assertIn('Wall', s.quad_sections)

    def test_slab_obj(self):
        s = T._tpl_slab_obj()
        self.assertIn('Slab', s.tri_sections)
        self.assertIn('Slab', s.quad_sections)

    def test_slab_edges_obj(self):
        s = T._tpl_slab_edges_obj({'bottom': 'clamped', 'top': 'simply'})
        self.assertIn('Slab', s.tri_sections)
        self.assertIn('Slab', s.quad_sections)

    def test_slab_ribbed_obj(self):
        s = T._tpl_slab_ribbed_obj()
        self.assertIn('Slab', s.tri_sections)
        self.assertIn('Slab', s.quad_sections)

    def test_flat_slab_obj(self):
        s = T._tpl_flat_slab_obj()
        self.assertIn('Slab', s.tri_sections)
        self.assertIn('Slab', s.quad_sections)

    def test_slab_on_grade_obj(self):
        s = T._tpl_slab_on_grade_obj()
        self.assertIn('Slab', s.tri_sections)
        self.assertIn('Slab', s.quad_sections)

    def test_slab_sector_obj(self):
        s = T._tpl_slab_sector_obj()
        self.assertIn('Slab', s.tri_sections)
        self.assertIn('Slab', s.quad_sections)

    def test_default_new_structure_plane_and_plate(self):
        plane = T._default_new_structure('plane')
        self.assertIn('CST', plane.tri_sections)
        self.assertIn('CST', plane.quad_sections)
        plate = T._default_new_structure('plate')
        self.assertIn('Slab', plate.tri_sections)
        self.assertIn('Slab', plate.quad_sections)

    def test_plain_explicit_mesh_templates_are_also_paired(self):
        # Per the user's decision: every template creates the slab/wall pair,
        # including the plain explicit-mesh builders — a section is always a
        # real "Panel" (tri+quad under one name), so it is never a dead end if
        # the user later adds a surface object referencing it and meshes as
        # quads.
        wall = T._tpl_wall(4.0, 3.0, 4, 3, 0.20, 3.0e7, 25.0)
        self.assertIn('Wall', wall.tri_sections)
        self.assertIn('Wall', wall.quad_sections)
        slab = T._tpl_slab()
        self.assertIn('Slab', slab.tri_sections)
        self.assertIn('Slab', slab.quad_sections)


class TestPreferQuadNowActuallyMeshesQuads(unittest.TestCase):
    """The end-to-end regression the bug report described: flip prefer_quad
    on a template-created object and the mesh must actually contain quads,
    not silently stay 100% triangles."""

    def test_slab_obj_prefer_quad(self):
        s = T._tpl_slab_obj()
        s.geometry_objects[_obj_id(s)].prefer_quad = True
        mesh, _ = expand_geometry(s)
        self.assertEqual(len(mesh.tri_elements), 0)
        self.assertGreater(len(mesh.quad_elements), 0)

    def test_wall_obj_prefer_quad(self):
        s = T._tpl_wall_obj(4.0, 3.0, 4, 3, 0.20, 3.0e7, 25.0)
        s.geometry_objects[_obj_id(s)].prefer_quad = True
        mesh, _ = expand_geometry(s)
        self.assertEqual(len(mesh.tri_elements), 0)
        self.assertGreater(len(mesh.quad_elements), 0)

    def test_default_new_structure_prefer_quad(self):
        # No geometry object exists yet on a bare new model — this just
        # confirms the seed section itself resolves as a quad section when
        # asked to (mirrors what add_geo_rectangle(..., prefer_quad=True)
        # would do once the user adds a surface).
        s = T._default_new_structure('plate')
        self.assertEqual(s.quad_sections['Slab'].formulation, 'MITC4')
        s2 = T._default_new_structure('plane')
        self.assertEqual(s2.quad_sections['CST'].formulation, 'QM6')

    def test_slab_obj_without_prefer_quad_is_still_all_triangles(self):
        # Pairing the section must not change a template's default output —
        # only make the explicit "Mesh as: Quads" switch work.
        s = T._tpl_slab_obj()
        mesh, _ = expand_geometry(s)
        self.assertGreater(len(mesh.tri_elements), 0)
        self.assertEqual(len(mesh.quad_elements), 0)


if __name__ == '__main__':
    unittest.main()
