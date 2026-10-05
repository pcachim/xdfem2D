"""Tests for the IFC exporter (xdfem2d.ifc_io) — Phase A skeleton."""
import os
import tempfile
import unittest

from context import Structure2D  # noqa: F401  (adds src/ to sys.path)

try:
    import ifcopenshell  # noqa: F401
    HAVE_IFC = True
except ImportError:
    HAVE_IFC = False


def _model(domain="plane"):
    s = Structure2D(domain=domain)
    s.add_material("C30", elastic_modulus=30e6, unit_weight=25.0, poisson=0.2)
    s.add_section("S1", "C30", b=0.3, h=0.6)
    for i, (x, y) in enumerate([(0, 0), (1, 0), (1, 1), (0, 1), (2, 0)]):
        s.add_node(f"N{i}", float(x), float(y))
    s.add_bar_element("B1", "N1", "N4", "S1")
    s.add_tri_section("T", "C30", thickness=0.2)
    s.add_tri_element("t1", "N0", "N1", "N2", "T")
    s.add_quad_section("Q", "C30", thickness=0.25)
    s.add_quad_element("q1", "N0", "N1", "N2", "N3", "Q")
    s.add_support("FIX", ux=True, uy=True, tz=True)
    s.assign_support("N0", "FIX")
    s.add_load_case("G", self_weight_factor=1.0)
    s.add_load_case("Q")
    s.add_point_load("N2", "Q", fx=5.0, fy=-8.0, mz=2.0)
    if domain == "plate":
        s.add_area_load("t1", "G", pz=-4.0)
        s.add_area_load("q1", "G", pz=-4.0)
    else:
        s.add_distributed_load("B1", "G", fye=-10.0, fyd=-10.0)
    return s


@unittest.skipUnless(HAVE_IFC, "ifcopenshell not installed")
class TestIfcExport(unittest.TestCase):
    def test_entity_counts(self):
        from xdfem2d.ifc_io import to_ifc
        f = to_ifc(_model())
        self.assertEqual(len(f.by_type("IfcStructuralPointConnection")), 5)
        self.assertEqual(len(f.by_type("IfcStructuralCurveMember")), 1)
        self.assertEqual(len(f.by_type("IfcStructuralSurfaceMember")), 2)
        self.assertEqual(len(f.by_type("IfcStructuralAnalysisModel")), 1)
        self.assertEqual(len(f.by_type("IfcBoundaryNodeCondition")), 1)

    def test_roundtrip_file(self):
        from xdfem2d.ifc_io import save_ifc
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.ifc")
            save_ifc(_model(), path)
            self.assertTrue(os.path.exists(path))
            g = ifcopenshell.open(path)
            self.assertEqual(g.schema, "IFC4")
            self.assertEqual(len(g.by_type("IfcStructuralPointConnection")), 5)

    def test_plane_coordinates_xz(self):
        from xdfem2d.ifc_io import to_ifc
        f = to_ifc(_model("plane"))
        c = next(c for c in f.by_type("IfcStructuralPointConnection")
                 if c.Name == "N2")
        xyz = c.Representation.Representations[0].Items[0].VertexGeometry.Coordinates
        self.assertEqual(tuple(xyz), (1.0, 0.0, 1.0))

    def test_plate_coordinates_xy(self):
        from xdfem2d.ifc_io import to_ifc
        f = to_ifc(_model("plate"))
        c = next(c for c in f.by_type("IfcStructuralPointConnection")
                 if c.Name == "N2")
        xyz = c.Representation.Representations[0].Items[0].VertexGeometry.Coordinates
        self.assertEqual(tuple(xyz), (1.0, 1.0, 0.0))

    def test_plate_support_maps_to_bending_dofs(self):
        from xdfem2d.ifc_io import to_ifc
        f = to_ifc(_model("plate"))
        bc = f.by_type("IfcBoundaryNodeCondition")[0]
        # plate: w→Z, θx→RX, θy→RY are fixed; X/Y translation free.
        self.assertIsNone(bc.TranslationalStiffnessX)
        self.assertIsNotNone(bc.TranslationalStiffnessZ)
        self.assertIsNotNone(bc.RotationalStiffnessX)
        self.assertIsNotNone(bc.RotationalStiffnessY)

    def test_thickness_carried(self):
        from xdfem2d.ifc_io import to_ifc
        f = to_ifc(_model())
        thicks = sorted(m.Thickness for m in f.by_type("IfcStructuralSurfaceMember"))
        self.assertEqual(thicks, [0.2, 0.25])

    # ── Phase B ───────────────────────────────────────────────────────
    def test_material_and_profile(self):
        from xdfem2d.ifc_io import to_ifc
        f = to_ifc(_model())
        self.assertEqual(len(f.by_type("IfcMaterial")), 1)
        # Rectangular bar section → rectangle profile with the right dims.
        rects = f.by_type("IfcRectangleProfileDef")
        self.assertTrue(any(abs(r.XDim - 0.3) < 1e-9 and abs(r.YDim - 0.6) < 1e-9
                            for r in rects))
        self.assertTrue(f.by_type("IfcMaterialProperties"))

    def test_load_cases_and_actions(self):
        from xdfem2d.ifc_io import to_ifc
        f = to_ifc(_model("plane"))
        self.assertEqual(len(f.by_type("IfcStructuralLoadCase")), 2)
        self.assertEqual(len(f.by_type("IfcStructuralPointAction")), 1)
        self.assertEqual(len(f.by_type("IfcStructuralLinearAction")), 1)
        model = f.by_type("IfcStructuralAnalysisModel")[0]
        self.assertEqual(len(model.LoadedBy), 2)

    def test_plate_area_loads(self):
        from xdfem2d.ifc_io import to_ifc
        f = to_ifc(_model("plate"))
        planar = f.by_type("IfcStructuralPlanarAction")
        self.assertEqual(len(planar), 2)
        # Native kN/m² (SAP2000 convention): -4.
        self.assertEqual(planar[0].AppliedLoad.PlanarForceZ, -4.0)

    def test_plane_point_load_maps_to_xz(self):
        from xdfem2d.ifc_io import to_ifc
        f = to_ifc(_model("plane"))
        force = f.by_type("IfcStructuralLoadSingleForce")[0]
        # Native kN / kNm (SAP2000 convention).
        self.assertEqual(force.ForceX, 5.0)      # fx → X
        self.assertEqual(force.ForceZ, -8.0)     # fy → Z
        self.assertEqual(force.MomentY, -2.0)    # mz → -My (X–Z sign)

    def test_units_declared(self):
        from xdfem2d.ifc_io import to_ifc
        f = to_ifc(_model())
        derived = {d.UnitType for d in f.by_type("IfcDerivedUnit")}
        self.assertIn("MODULUSOFELASTICITYUNIT", derived)
        self.assertIn("MASSDENSITYUNIT", derived)
        self.assertIn("PLANARFORCEUNIT", derived)
        # SAP2000 unit convention: force in kN, mass in tonnes.
        force_unit = next(u for u in f.by_type("IfcSIUnit")
                          if u.UnitType == "FORCEUNIT")
        self.assertEqual(force_unit.Prefix, "KILO")
        mass_unit = next(u for u in f.by_type("IfcSIUnit")
                         if u.UnitType == "MASSUNIT")
        self.assertEqual(mass_unit.Prefix, "MEGA")

    def test_orphan_node_not_exported(self):
        # An unconnected node must not become a point connection (IFC requires
        # every connection to join at least one member — SAP2000 chokes on the
        # empty member list otherwise).
        from xdfem2d.ifc_io import to_ifc
        s = _model("plane")
        s.add_node("ORPHAN", 9.0, 9.0)
        f = to_ifc(s)
        names = {c.Name for c in f.by_type("IfcStructuralPointConnection")}
        self.assertNotIn("ORPHAN", names)
        # Every point connection is joined to a member.
        for c in f.by_type("IfcStructuralPointConnection"):
            self.assertTrue(c.ConnectsStructuralMembers)


@unittest.skipUnless(HAVE_IFC, "ifcopenshell not installed")
class TestIfcImport(unittest.TestCase):
    def _roundtrip(self, domain):
        from xdfem2d.ifc_io import save_ifc, load_ifc
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.ifc")
            save_ifc(_model(domain), path)
            return load_ifc(path)

    def test_roundtrip_plane(self):
        r = self._roundtrip("plane")
        self.assertEqual(r.domain, "plane")
        self.assertEqual(len(r.nodes), 5)
        self.assertEqual(len(r.bar_elements), 1)
        self.assertEqual(len(r.tri_elements), 1)
        self.assertEqual(len(r.quad_elements), 1)
        self.assertEqual(len(r.support_assignments), 1)
        self.assertEqual(len(r.point_loads), 1)
        self.assertEqual(len(r.distributed_loads), 1)
        # N2 at (1, 1) in the model plane.
        self.assertEqual((r.nodes["N2"].x, r.nodes["N2"].y), (1.0, 1.0))

    def test_roundtrip_plate(self):
        r = self._roundtrip("plate")
        self.assertEqual(r.domain, "plate")
        self.assertEqual(len(r.tri_area_loads) + len(r.quad_area_loads), 2)
        self.assertEqual((r.nodes["N2"].x, r.nodes["N2"].y), (1.0, 1.0))

    def test_point_load_roundtrip_values(self):
        r = self._roundtrip("plane")
        pl = r.point_loads[0]
        self.assertEqual((pl.fx, pl.fy, pl.mz), (5.0, -8.0, 2.0))

    def test_material_roundtrip(self):
        r = self._roundtrip("plane")
        mat = next(iter(r.materials.values()))
        self.assertAlmostEqual(mat.elastic_modulus, 30e6, delta=1.0)
        self.assertAlmostEqual(mat.poisson, 0.2, places=6)

    def test_infer_domain_plane_and_plate(self):
        from xdfem2d.ifc_io import _infer_domain
        self.assertEqual(_infer_domain([(0, 0, 0), (1, 0, 1)]), "plane")
        self.assertEqual(_infer_domain([(0, 0, 0), (1, 1, 0)]), "plate")

    def test_infer_domain_ambiguous(self):
        from xdfem2d.ifc_io import _infer_domain, AmbiguousDomainError
        with self.assertRaises(AmbiguousDomainError):
            _infer_domain([(0, 0, 0), (1, 0, 0)])

    def test_infer_domain_nonplanar(self):
        from xdfem2d.ifc_io import _infer_domain, NonPlanarModelError
        with self.assertRaises(NonPlanarModelError):
            _infer_domain([(0, 0, 0), (1, 1, 1)])


def _physical_file():
    """A minimal physical IFC model: one beam axis + one slab footprint."""
    import ifcopenshell
    from ifcopenshell import guid
    f = ifcopenshell.file(schema="IFC4")
    ctx = f.create_entity(
        "IfcGeometricRepresentationContext", ContextType="Model",
        CoordinateSpaceDimension=3, Precision=1e-6,
        WorldCoordinateSystem=f.create_entity(
            "IfcAxis2Placement3D",
            Location=f.create_entity("IfcCartesianPoint", Coordinates=(0., 0., 0.))))

    def place(x, y, z):
        return f.create_entity(
            "IfcLocalPlacement", RelativePlacement=f.create_entity(
                "IfcAxis2Placement3D",
                Location=f.create_entity("IfcCartesianPoint", Coordinates=(x, y, z))))

    def poly(pts):
        return f.create_entity("IfcPolyline", Points=[
            f.create_entity("IfcCartesianPoint", Coordinates=p) for p in pts])

    def rep(ident, items):
        return f.create_entity("IfcProductDefinitionShape", Representations=[
            f.create_entity(
                "IfcShapeRepresentation", ContextOfItems=ctx,
                RepresentationIdentifier=ident, RepresentationType="Curve3D",
                Items=items)])

    f.create_entity(
        "IfcBeam", GlobalId=guid.new(), Name="B1", ObjectPlacement=place(0., 0., 3.),
        Representation=rep("Axis", [poly([(0., 0., 0.), (5., 0., 0.)])]))
    f.create_entity(
        "IfcSlab", GlobalId=guid.new(), Name="S1", ObjectPlacement=place(0., 0., 0.),
        Representation=rep("FootPrint", [poly([
            (0., 0., 0.), (5., 0., 0.), (5., 0., 3.), (0., 0., 3.), (0., 0., 0.)])]))
    return f


@unittest.skipUnless(HAVE_IFC, "ifcopenshell not installed")
class TestIfcPhysicalImport(unittest.TestCase):
    def test_beam_and_slab_geometry(self):
        from xdfem2d.ifc_io import from_ifc_physical
        s = from_ifc_physical(_physical_file())
        self.assertEqual(s.domain, "plane")
        kinds = sorted(type(o).__name__ for o in s.geometry_objects.values())
        self.assertEqual(kinds, ["GeoMultisegment", "GeoPolygon"])

    def test_beam_axis_world_coords(self):
        from xdfem2d.ifc_io import from_ifc_physical
        from xdfem2d.models import GeoMultisegment
        s = from_ifc_physical(_physical_file())
        seg = next(o for o in s.geometry_objects.values()
                   if isinstance(o, GeoMultisegment))
        # Placement z=3 → world (0,0,3)..(5,0,3); plane domain → (x, z).
        nodes = [s.nodes[nid] for nid in seg.node_ids]
        xs = sorted(n.x for n in nodes)
        self.assertEqual(xs, [0.0, 5.0])
        self.assertTrue(all(abs(n.y - 3.0) < 1e-9 for n in nodes))

    def test_empty_raises(self):
        import ifcopenshell
        from xdfem2d.ifc_io import from_ifc_physical
        with self.assertRaises(ValueError):
            from_ifc_physical(ifcopenshell.file(schema="IFC4"))

    def test_explicit_domain_plate(self):
        from xdfem2d.ifc_io import from_ifc_physical
        s = from_ifc_physical(_physical_file(), domain="plate")
        self.assertEqual(s.domain, "plate")


if __name__ == "__main__":
    unittest.main()
