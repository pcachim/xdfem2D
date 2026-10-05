"""Steel-section orientation: the ``angle`` property rotates the profile in the
frame plane, so the in-plane bending inertia interpolates between the strong
(y-y, θ=0) and weak (z-z, θ=90°) axes as Iy·cos²θ + Iz·sin²θ.
"""
import math

import context  # noqa: F401
from xdfem2d import Structure2D
from xdfem2d.models import (
    Section, SectionShape, section_area_inertia, section_inertia_minor,
)
from xdfem2d.structure_io import _from_dict, _to_dict

# IPE300-like I-section, in metres.
B, H, TW, TF = 0.15, 0.30, 0.0071, 0.0107


def _iy_iz():
    iy = section_area_inertia(SectionShape.I, B, H, TW, TF)[1]
    iz = section_inertia_minor(SectionShape.I, B, H, TW, TF)
    return iy, iz


def test_minor_inertia_is_smaller_for_a_tall_i():
    iy, iz = _iy_iz()
    assert iy > iz > 0.0
    # I-section: Iz = 2·(tf·b³/12) + hw·tw³/12
    hw = H - 2 * TF
    expected = 2.0 * (TF * B**3 / 12.0) + hw * TW**3 / 12.0
    assert math.isclose(iz, expected, rel_tol=1e-12)


def test_angle_zero_is_the_major_axis():
    iy, _ = _iy_iz()
    sec = Section("S", "M", B, H, shape=SectionShape.I, tw=TW, tf=TF, angle=0.0)
    assert math.isclose(sec.inertia, iy, rel_tol=1e-15)


def test_angle_ninety_is_the_minor_axis():
    _, iz = _iy_iz()
    sec = Section("S", "M", B, H, shape=SectionShape.I, tw=TW, tf=TF, angle=90.0)
    assert math.isclose(sec.inertia, iz, rel_tol=1e-12)


def test_angle_interpolates_cos2_sin2():
    iy, iz = _iy_iz()
    for ang in (15.0, 30.0, 45.0, 60.0):
        sec = Section("S", "M", B, H, shape=SectionShape.I, tw=TW, tf=TF,
                      angle=ang)
        th = math.radians(ang)
        expected = iy * math.cos(th) ** 2 + iz * math.sin(th) ** 2
        assert math.isclose(sec.inertia, expected, rel_tol=1e-12)


def test_area_and_torsion_are_orientation_invariant():
    a0 = Section("S", "M", B, H, shape=SectionShape.I, tw=TW, tf=TF, angle=0.0)
    a45 = Section("S", "M", B, H, shape=SectionShape.I, tw=TW, tf=TF, angle=45.0)
    assert math.isclose(a0.area, a45.area, rel_tol=1e-15)
    assert math.isclose(a0.torsion, a45.torsion, rel_tol=1e-15)


def test_angle_survives_save_load_round_trip():
    s = Structure2D()
    s.add_material("St", 210e6, 78.5)
    s.add_section("B", "St", B, H, shape="I", tw=TW, tf=TF, angle=30.0)
    s2 = _from_dict(_to_dict(s))
    sc = s2.sections["B"]
    assert math.isclose(sc.angle, 30.0, rel_tol=1e-12)
    assert math.isclose(sc.inertia, s.sections["B"].inertia, rel_tol=1e-12)
