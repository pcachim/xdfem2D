"""Hollow-section member checks against the EurocodeApplied worked example
RHS 90x50/8, S235, L=2 m, N=-30 kN, My=8 kNm (uniform), plus CHS consistency.

Hollow sections are not susceptible to LTB/torsional deformation, so Annex B
uses Table B.1 (kzy = 0.6·kyy) — reference Eq. 6.62: A 0.550, B 0.533.
"""
import pytest

from xdfem2d import Structure2D
from xdfem2d.steel_design import design_steel_members


def _member(profile, length, n, m, method):
    s = Structure2D()
    s.add_node("N1", 0, 0)
    s.add_node("N2", length, 0)
    s.create_steel_section(profile, profile, grade="S235")
    s.add_bar_element("E1", "N1", "N2", profile)
    s.add_support("PIN", ux=True, uy=True, tz=False)
    s.add_support("ROL", ux=False, uy=True, tz=False)
    s.assign_support("N1", "PIN")
    s.assign_support("N2", "ROL")
    s.add_load_case("LC1", self_weight_factor=0.0)
    s.add_point_load("N2", "LC1", fx=-n, mz=m)
    s.add_point_load("N1", "LC1", mz=-m)
    s.add_load_combination("ULS", {"LC1": 1.0})
    res = s.calculate()
    prefs = dict(steel_cmy=1, steel_cmz=1, steel_cmlt=1, steel_c1=1,
                 steel_G=80769.0, steel_interaction_method=method)
    return design_steel_members(s, res, prefs=prefs)["members"][0]


@pytest.mark.parametrize("method,u61,u62", [("A", 0.782, 0.550),
                                            ("B", 0.768, 0.533)])
def test_rhs_90x50x8(method, u61, u62):
    m = _member("RHS90x50x8", 2.0, 30.0, 8.0, method)
    assert m["section_class"] == 1
    assert m["chi_y"] == pytest.approx(0.844, abs=1e-3)
    assert m["util_6_61"] == pytest.approx(u61, abs=2e-3)
    assert m["util_6_62"] == pytest.approx(u62, abs=2e-3)


@pytest.mark.parametrize("method,u61", [("A", 0.748), ("B", 0.746)])
def test_chs_48_3x5(method, u61):
    """CHS 48.3/5, S235H, L=2 m, N=-3 kN, M=1.5 kNm (uniform): reference
    chi=0.427, Method 1 u=0.748, Method 2 u=0.746 (Eq. 6.61, single plane).
    The catalogue rounds A/W slightly, hence the 2e-3 tolerance."""
    m = _member("CHS48_3x5", 2.0, 3.0, 1.5, method)
    assert m["section_class"] == 1
    assert m["chi_y"] == pytest.approx(0.427, abs=2e-3)
    assert m["chi_y"] == pytest.approx(m["chi_z"], rel=1e-9)  # axisymmetric
    assert m["util_6_61"] == pytest.approx(u61, abs=2e-3)


# ── grillage (plate domain): the N slot carries the St-Venant torsion ───────

def _cantilever_plate(profile, length, v, t):
    s = Structure2D(domain="plate")
    s.create_steel_section(profile, profile, grade="S235")
    for k in range(5):
        s.add_node(f"N{k}", length * k / 4, 0.0)
    for k in range(4):
        s.add_bar_element(f"E{k}", f"N{k}", f"N{k + 1}", profile)
    s.add_support("FIX", w=True, tx=True, ty=True)
    s.assign_support("N0", "FIX")
    s.add_load_case("LC1", self_weight_factor=0.0)
    s.add_point_load("N4", "LC1", fz=-v, mx=t)
    s.add_load_combination("ULS", {"LC1": 1.0})
    res = s.calculate()
    prefs = dict(steel_G=80769.0)
    return design_steel_members(s, res, prefs=prefs)["elements"]["E0"]


def test_rhs_torsion_and_shear_plate():
    """RHS 90x50/8: V=10, My=8 (root of a 0.8 m cantilever), T=3 kNm.
    Reference: bending 0.662, torsion 0.512, shear+torsion 0.123."""
    r = _cantilever_plate("RHS90x50x8", 0.8, 10.0, 3.0)
    assert r["bending"] == pytest.approx(0.662, abs=2e-3)
    assert r["torsion"] == pytest.approx(0.512, abs=2e-3)
    assert r["shear"] == pytest.approx(0.123, abs=2e-3)


def test_chs_torsion_and_shear_plate():
    """CHS 48.3/5: V=3, M=1.5 (0.5 m cantilever), T=1 kNm.
    Reference: bending 0.679, torsion 0.551, shear+torsion 0.114."""
    r = _cantilever_plate("CHS48_3x5", 0.5, 3.0, 1.0)
    assert r["bending"] == pytest.approx(0.679, abs=2e-3)
    assert r["torsion"] == pytest.approx(0.551, abs=2e-3)
    assert r["shear"] == pytest.approx(0.114, abs=2e-3)
