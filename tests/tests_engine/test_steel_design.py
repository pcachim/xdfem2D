"""Integration test for the EC3 steel design (xdfem2d.steel_design).

It exercises member recognition → section properties → the §6.2 + §6.3.3 checks end to end
(no solver needed), for both the plane and the plate (grillage) domains.
"""
import pytest

import context  # noqa: F401
from xdfem2d import Structure2D

import eurocodepy  # noqa: F401

from xdfem2d.steel_design import design_steel_members  # noqa: E402

PREFS = {"gamma_M1": 1.0, "steel_ky": 1.0, "steel_kz": 1.0,
         "steel_klt": 1.0, "steel_ltb_rolled": True,
         "steel_class4_effective": True}


def _plane_column():
    s = Structure2D()
    s.add_material("S275", 210e6, 78.5, material_type="Steel", design={"fy": 275.0})
    s.add_section("C", "S275", 0.15, 0.30, shape="I", tw=0.0071, tf=0.0107)
    for i, y in enumerate((0.0, 3.0, 6.0)):
        s.add_node(f"N{i}", 0.0, y)
    s.add_bar_element("B0", "N0", "N1", "C")
    s.add_bar_element("B1", "N1", "N2", "C")
    s.add_support("Pin", ux=True, uy=True, tz=False)
    s.assign_support("N0", "Pin")
    return s


def _plate_grillage_beam():
    s = Structure2D(domain="plate")
    s.add_material("S275", 210e6, 78.5, material_type="Steel", design={"fy": 275.0})
    s.add_section("C", "S275", 0.15, 0.30, shape="I", tw=0.0071, tf=0.0107)
    s.add_node("N0", 0.0, 0.0)
    s.add_node("N1", 3.0, 0.0)
    s.add_bar_element("B0", "N0", "N1", "C")
    return s


def _results(slot, v, m):
    ef = {"i": [slot, v, m], "j": [-slot, -v, -m]}
    return {"combinations": {"ULS": {"element_forces": {"B0": ef, "B1": ef}}}}


def test_plane_returns_member_and_element_ratios():
    s = _plane_column()
    out = design_steel_members(s, _results(-300.0, 40.0, 50.0), prefs=PREFS)
    members, elems = out["members"], out["elements"]
    assert len(members) == 1
    assert members[0]["length"] == pytest.approx(6.0)     # both bars
    r = elems["B0"]
    for k in ("bending", "shear", "buckling", "combined"):
        assert k in r and r[k] >= 0.0
    assert r["combined"] == pytest.approx(
        max(r["bending"], r["shear"], r.get("torsion", 0.0), r["buckling"]))


def test_plate_grillage_uses_torsion_from_the_n_slot():
    s = _plate_grillage_beam()
    # In the plate domain the 'N' slot carries torsion.
    out = design_steel_members(s, _results(2.0, 30.0, 40.0), prefs=PREFS)
    r = out["elements"]["B0"]
    assert r["torsion"] > 0.0            # torsion picked up from the N slot
    assert r["shear"] > 0.0
    assert r["bending"] > 0.0


def test_higher_load_raises_utilization():
    s = _plane_column()
    lo = design_steel_members(s, _results(-200.0, 20.0, 30.0), prefs=PREFS)
    hi = design_steel_members(s, _results(-600.0, 20.0, 30.0), prefs=PREFS)
    assert hi["elements"]["B0"]["combined"] > lo["elements"]["B0"]["combined"]


def test_non_steel_members_are_skipped():
    s = Structure2D()
    s.add_material("C30", 33e6, 25.0, material_type="Concrete")
    s.add_section("R", "C30", 0.3, 0.5, shape="Rectangular")
    s.add_node("N0", 0, 0); s.add_node("N1", 0, 3)
    s.add_bar_element("B0", "N0", "N1", "R")
    out = design_steel_members(s, _results(-300.0, 20.0, 50.0), prefs=PREFS)
    assert out["members"] == [] and out["elements"] == {}
