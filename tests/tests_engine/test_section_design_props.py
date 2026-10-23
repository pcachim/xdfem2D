"""Steel-section design properties: a catalogue profile stores every property
(minor inertia, torsion, elastic/plastic moduli, shear areas) from the
catalogue and reports its designation; a manually-defined section computes all
of them from the shape and is flagged as non-catalogue.
"""
import math

import context  # noqa: F401
from xdfem2d import Structure2D
from xdfem2d.models import (
    Section, SectionShape, section_plastic_moduli, section_shear_areas,
)
from xdfem2d.structure_io import _from_dict, _to_dict

# IPE300 (metres) with its catalogue design properties (SI).
B, H, TW, TF = 0.15, 0.30, 0.0071, 0.0107
CAT = dict(
    inertia_override=8356e-8, inertia_minor_override=603.8e-8,
    wel_y_override=557.1e-6, wpl_y_override=628.4e-6,
    wel_z_override=80.5e-6, wpl_z_override=125.2e-6,
    av_y_override=32.1e-4, av_z_override=25.68e-4,
    torsion_override=19.75e-8,
)


def test_manual_section_computes_every_property_from_the_shape():
    m = Section("M", "St", B, H, shape=SectionShape.I, tw=TW, tf=TF)
    assert m.is_catalogue_profile is False
    assert m.profile_name is None
    # computed from the shape formulas (no override present)
    wply, wplz = section_plastic_moduli(SectionShape.I, B, H, TW, TF)
    avy, avz = section_shear_areas(SectionShape.I, B, H, TW, TF)
    assert m.wpl_y == wply and m.wpl_z == wplz
    assert m.av_y == avy and m.av_z == avz
    assert math.isclose(m.wel_y, m.inertia_major / (H / 2), rel_tol=1e-12)
    assert math.isclose(m.radius_gyration_y, (m.inertia_major / m.area) ** 0.5,
                        rel_tol=1e-12)


def test_catalogue_section_reports_the_catalogue_values():
    c = Section("C", "St", B, H, shape=SectionShape.I, tw=TW, tf=TF,
                profile_name="IPE300", **CAT)
    assert c.is_catalogue_profile is True
    assert c.profile_name == "IPE300"
    assert math.isclose(c.wpl_y, 628.4e-6, rel_tol=1e-9)
    assert math.isclose(c.wel_z, 80.5e-6, rel_tol=1e-9)
    assert math.isclose(c.av_z, 25.68e-4, rel_tol=1e-9)
    assert math.isclose(c.inertia_minor, 603.8e-8, rel_tol=1e-9)
    assert math.isclose(c.torsion, 19.75e-8, rel_tol=1e-9)


def test_catalogue_values_differ_from_the_shape_fallback():
    # The catalogue Wpl,y (with fillets) exceeds the flat-plate estimate.
    m = Section("M", "St", B, H, shape=SectionShape.I, tw=TW, tf=TF)
    c = Section("C", "St", B, H, shape=SectionShape.I, tw=TW, tf=TF,
                profile_name="IPE300", **CAT)
    assert c.wpl_y > m.wpl_y
    assert c.av_z > m.av_z


def test_catalogue_minor_inertia_feeds_the_angle_rotation():
    # Rotating a catalogue profile 90° uses the accurate catalogue Iz.
    c = Section("C", "St", B, H, shape=SectionShape.I, tw=TW, tf=TF,
                profile_name="IPE300", angle=90.0, **CAT)
    assert math.isclose(c.inertia, 603.8e-8, rel_tol=1e-9)


def test_design_properties_survive_save_load():
    s = Structure2D()
    s.add_material("St", 210e6, 78.5)
    s.add_section("B", "St", B, H, shape="I", tw=TW, tf=TF,
                  profile_name="IPE300", **CAT)
    sc = _from_dict(_to_dict(s)).sections["B"]
    assert sc.profile_name == "IPE300"
    assert math.isclose(sc.wpl_y, 628.4e-6, rel_tol=1e-9)
    assert math.isclose(sc.av_z, 25.68e-4, rel_tol=1e-9)
    assert math.isclose(sc.inertia_minor, 603.8e-8, rel_tol=1e-9)
