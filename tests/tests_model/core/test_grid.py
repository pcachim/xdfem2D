"""Grillage (plate-domain bar) validation against closed-form solutions.

Every test here is an analytic benchmark: cantilever bending and torsion,
fixed-fixed beam under a uniform load, a two-beam orthogonal grillage, and a
bending release. Units: m, kN, kNm.
"""
import math

import numpy as np
import pytest

import context  # noqa: F401
from xdfem2d import Structure2D

E = 33e6          # kN/m²
NU = 0.2
B, H = 0.3, 0.5   # m


def _props():
    from xdfem2d.models import section_torsion_constant, SectionShape
    I = B * H ** 3 / 12.0
    J = section_torsion_constant(SectionShape.RECTANGULAR, B, H)
    G = E / (2.0 * (1.0 + NU))
    return I, J, G


def _base(domain="plate"):
    s = Structure2D(domain=domain)
    s.add_material('C', E, 25.0, poisson=NU)
    s.add_section('S', 'C', B, H, shape='Rectangular')
    return s


def _cantilever(L=4.0):
    s = _base()
    s.add_node('N1', 0.0, 0.0)
    s.add_node('N2', L, 0.0)
    s.add_bar_element('E1', 'N1', 'N2', 'S')
    s.add_support('ENC', w=True, tx=True, ty=True)
    s.assign_support('N1', 'ENC')
    s.add_load_case('LC')
    return s


def test_cantilever_tip_force():
    L, P = 4.0, 10.0
    I, J, G = _props()
    s = _cantilever(L)
    s.add_point_load('N2', 'LC', fz=-P)
    r = s.calculate()

    w2, tx2, ty2 = r['displacements']['LC']['N2']
    assert w2 == pytest.approx(-P * L ** 3 / (3 * E * I), rel=1e-9)
    # ψ = −dw/dx: the tip slope dw/dx = −PL²/2EI, so ty = +PL²/2EI.
    assert ty2 == pytest.approx(P * L ** 2 / (2 * E * I), rel=1e-9)
    assert tx2 == pytest.approx(0.0, abs=1e-12)

    # Reactions: Rz = +P, My = −P·L (moment of the load about N1 is +P·L
    # about Y, right-hand rule), no torsion.
    rz, mx, my = r['reactions']['LC']['N1']
    assert rz == pytest.approx(P, rel=1e-9)
    assert mx == pytest.approx(0.0, abs=1e-8)
    assert my == pytest.approx(-P * L, rel=1e-9)

    # Diagram: M(0) = −P·L rising to 0 at the tip; V constant; T zero.
    d = r['element_distribution']['LC']['E1']
    assert d['M'][0] == pytest.approx(-P * L, rel=1e-9)
    assert d['M'][-1] == pytest.approx(0.0, abs=1e-8)
    assert np.allclose(d['N'], 0.0, atol=1e-9)          # torsion slot
    assert d['V'][0] == pytest.approx(d['V'][-1], rel=1e-9)


def test_cantilever_tip_torque():
    L, T = 4.0, 7.0
    I, J, G = _props()
    s = _cantilever(L)
    # Bar along X: a moment about X at the tip is pure torsion.
    s.add_point_load('N2', 'LC', mx=T)
    r = s.calculate()

    w2, tx2, ty2 = r['displacements']['LC']['N2']
    assert tx2 == pytest.approx(T * L / (G * J), rel=1e-9)
    assert w2 == pytest.approx(0.0, abs=1e-12)
    assert ty2 == pytest.approx(0.0, abs=1e-12)

    rz, mx, my = r['reactions']['LC']['N1']
    assert mx == pytest.approx(-T, rel=1e-9)
    assert rz == pytest.approx(0.0, abs=1e-9)

    # Torsion rides in the first slot of the member forces (the plane bar's
    # axial slot); constant along the bar.
    d = r['element_distribution']['LC']['E1']
    assert abs(d['N'][0]) == pytest.approx(T, rel=1e-9)
    assert np.allclose(d['N'], d['N'][0])
    assert np.allclose(d['M'], 0.0, atol=1e-9)


def test_inclined_cantilever_tip_force():
    """Same cantilever rotated 30° in plan — w and |M| must not change."""
    L, P, ang = 4.0, 10.0, math.radians(30.0)
    I, J, G = _props()
    s = _base()
    s.add_node('N1', 0.0, 0.0)
    s.add_node('N2', L * math.cos(ang), L * math.sin(ang))
    s.add_bar_element('E1', 'N1', 'N2', 'S')
    s.add_support('ENC', w=True, tx=True, ty=True)
    s.assign_support('N1', 'ENC')
    s.add_load_case('LC')
    s.add_point_load('N2', 'LC', fz=-P)
    r = s.calculate()

    w2 = r['displacements']['LC']['N2'][0]
    assert w2 == pytest.approx(-P * L ** 3 / (3 * E * I), rel=1e-9)
    d = r['element_distribution']['LC']['E1']
    assert d['M'][0] == pytest.approx(-P * L, rel=1e-9)
    assert np.allclose(d['N'], 0.0, atol=1e-8)          # no spurious torsion
    # The two reaction moment components recombine to the full P·L.
    rz, mx, my = r['reactions']['LC']['N1']
    assert math.hypot(mx, my) == pytest.approx(P * L, rel=1e-9)


def test_fixed_fixed_uniform_load():
    L, q = 6.0, 12.0    # kN/m downward
    I, J, G = _props()
    s = _base()
    s.add_node('N1', 0.0, 0.0)
    s.add_node('N2', L, 0.0)
    s.add_bar_element('E1', 'N1', 'N2', 'S')
    s.add_support('ENC', w=True, tx=True, ty=True)
    s.assign_support('N1', 'ENC')
    s.assign_support('N2', 'ENC')
    s.add_load_case('LC')
    s.add_distributed_load('E1', 'LC', fze=-q, fzd=-q)
    r = s.calculate()

    d = r['element_distribution']['LC']['E1']
    xs = d['x']
    # End moments −qL²/12, midspan +qL²/24 (sagging positive).
    assert d['M'][0] == pytest.approx(-q * L ** 2 / 12.0, rel=1e-6)
    assert d['M'][-1] == pytest.approx(-q * L ** 2 / 12.0, rel=1e-6)
    mid = np.argmin(np.abs(xs - L / 2))
    assert d['M'][mid] == pytest.approx(q * L ** 2 / 24.0, rel=1e-6)
    # Shear ±qL/2 at the ends.
    assert abs(d['V'][0]) == pytest.approx(q * L / 2.0, rel=1e-6)
    # Reactions: qL/2 each, no net torsion.
    assert r['reactions']['LC']['N1'][0] == pytest.approx(q * L / 2, rel=1e-9)
    assert r['reactions']['LC']['N2'][0] == pytest.approx(q * L / 2, rel=1e-9)


def test_two_beam_orthogonal_grillage():
    """Two equal simply supported beams crossing at their midpoints: each
    carries half the central load; the centre deflection is (P/2)·L³/48EI."""
    L, P = 6.0, 20.0
    I, J, G = _props()
    s = _base()
    s.add_node('A1', -L / 2, 0.0); s.add_node('A2', L / 2, 0.0)
    s.add_node('B1', 0.0, -L / 2); s.add_node('B2', 0.0, L / 2)
    s.add_node('C', 0.0, 0.0)
    s.add_bar_element('EX1', 'A1', 'C', 'S')
    s.add_bar_element('EX2', 'C', 'A2', 'S')
    s.add_bar_element('EY1', 'B1', 'C', 'S')
    s.add_bar_element('EY2', 'C', 'B2', 'S')
    s.add_support('SS', w=True)
    for nid in ('A1', 'A2', 'B1', 'B2'):
        s.assign_support(nid, 'SS')
    s.add_load_case('LC')
    s.add_point_load('C', 'LC', fz=-P)
    r = s.calculate()

    wc = r['displacements']['LC']['C'][0]
    assert wc == pytest.approx(-(P / 2) * L ** 3 / (48 * E * I), rel=1e-9)
    # Each support carries P/4 by symmetry.
    for nid in ('A1', 'A2', 'B1', 'B2'):
        assert r['reactions']['LC'][nid][0] == pytest.approx(P / 4, rel=1e-9)
    # By symmetry no beam twists.
    assert r['displacements']['LC']['C'][1] == pytest.approx(0.0, abs=1e-12)
    assert r['displacements']['LC']['C'][2] == pytest.approx(0.0, abs=1e-12)


def test_hinge_releases_bending_moment():
    """Fixed-fixed beam with a bending release at the j end under uniform
    load — the propped-cantilever moment −qL²/8 appears at the fixed end and
    zero at the released one."""
    L, q = 6.0, 12.0
    s = _base()
    s.add_node('N1', 0.0, 0.0)
    s.add_node('N2', L, 0.0)
    s.add_bar_element('E1', 'N1', 'N2', 'S', hinge_j=True)
    s.add_support('ENC', w=True, tx=True, ty=True)
    s.assign_support('N1', 'ENC')
    s.assign_support('N2', 'ENC')
    s.add_load_case('LC')
    s.add_distributed_load('E1', 'LC', fze=-q, fzd=-q)
    r = s.calculate()

    ef = r['element_forces']['LC']['E1']
    assert ef['i'][2] == pytest.approx(-q * L ** 2 / 8.0, rel=1e-6)
    assert ef['j'][2] == pytest.approx(0.0, abs=1e-6)


def test_self_weight_matches_equivalent_uniform_load():
    """A load case with self_weight_factor=1 equals fz = −γ·A applied by hand."""
    L = 5.0
    gA = 25.0 * B * H
    s1 = _cantilever(L)
    s1.load_cases_by_id['LC'].self_weight_factor = 1.0
    r1 = s1.calculate()

    s2 = _cantilever(L)
    s2.add_distributed_load('E1', 'LC', fze=-gA, fzd=-gA)
    r2 = s2.calculate()

    d1 = r1['displacements']['LC']['N2']
    d2 = r2['displacements']['LC']['N2']
    assert d1 == pytest.approx(d2, rel=1e-9)
    m1 = r1['element_distribution']['LC']['E1']['M']
    m2 = r2['element_distribution']['LC']['E1']['M']
    assert np.allclose(m1, m2)


def test_element_point_load_at_midspan():
    """Simply supported grillage bar, P at midspan: M_max = PL/4."""
    L, P = 6.0, 16.0
    s = _base()
    s.add_node('N1', 0.0, 0.0)
    s.add_node('N2', L, 0.0)
    s.add_bar_element('E1', 'N1', 'N2', 'S')
    s.add_support('SS', w=True, tx=True)   # torsion held, bending free
    s.assign_support('N1', 'SS')
    s.assign_support('N2', 'SS')
    s.add_load_case('LC')
    s.add_element_point_load('E1', 'LC', a=L / 2, fz=-P)
    r = s.calculate()

    d = r['element_distribution']['LC']['E1']
    assert d['M'].max() == pytest.approx(P * L / 4.0, rel=1e-6)
    assert r['reactions']['LC']['N1'][0] == pytest.approx(P / 2, rel=1e-9)


def test_thermal_gradient_free_cantilever_no_forces():
    """A gradient on a cantilever (statically determinate) bends it freely:
    curvature α·ΔT/h, tip deflection κL²/2, and zero member forces."""
    L, dT = 4.0, 30.0
    s = _cantilever(L)
    s.add_temperature_load('E1', 'LC', delta_t_uniform=0.0,
                           delta_t_gradient=dT)
    r = s.calculate()

    kappa = 1e-5 * dT / H
    w2 = r['displacements']['LC']['N2'][0]
    assert abs(w2) == pytest.approx(kappa * L ** 2 / 2.0, rel=1e-6)
    ef = r['element_forces']['LC']['E1']
    assert np.allclose(ef['i'], 0.0, atol=1e-6)
    assert np.allclose(ef['j'], 0.0, atol=1e-6)


def test_winkler_spring_carries_load():
    """A grillage bar floating on a uniform Winkler bed under uniform load
    settles w = q/k with no bending (rigid-body sink)."""
    L, q, k = 6.0, 12.0, 5000.0     # kN/m, kN/m²
    s = _base()
    s.add_node('N1', 0.0, 0.0)
    s.add_node('N2', L, 0.0)
    s.add_bar_element('E1', 'N1', 'N2', 'S')
    s.add_element_spring('E1', ky=k)
    # Hold the meaningless rigid modes (torsion about the bar) without
    # touching w: restrain tx at both ends.
    s.add_support('TOR', tx=True)
    s.assign_support('N1', 'TOR')
    s.assign_support('N2', 'TOR')
    s.add_load_case('LC')
    s.add_distributed_load('E1', 'LC', fze=-q, fzd=-q)
    r = s.calculate()

    w1 = r['displacements']['LC']['N1'][0]
    w2 = r['displacements']['LC']['N2'][0]
    assert w1 == pytest.approx(-q / k, rel=1e-6)
    assert w2 == pytest.approx(-q / k, rel=1e-6)


def test_settlement_alias():
    """A prescribed w settlement at the fixed end shifts the tip rigidly."""
    L, dz = 4.0, -0.01
    s = _cantilever(L)
    s.create_support_settlement('N1', 'LC', w=dz)
    r = s.calculate()
    assert r['displacements']['LC']['N1'][0] == pytest.approx(dz, rel=1e-9)
    assert r['displacements']['LC']['N2'][0] == pytest.approx(dz, rel=1e-6)
    ef = r['element_forces']['LC']['E1']
    assert np.allclose(ef['i'], 0.0, atol=1e-5)
