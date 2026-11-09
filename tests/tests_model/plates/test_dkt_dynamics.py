"""Plate-domain dynamics (phase 3): the out-of-plane vibration of DKT slabs and
grillages. The mass sits on the transverse DOF w, so the modes are vertical and
the participation/spectrum machinery reads them through the "x" slot.

Validated against the Kirchhoff closed form for a simply supported square plate
and through the mass bookkeeping (a lumped, diagonal mass matrix makes the
distributed-mass frequency converge to the exact value FROM BELOW). Units: m,
kN, kNm, t.
"""
import math

import numpy as np
import pytest

import context  # noqa: F401
from xdfem2d import Structure2D

E, NU, T = 33e6, 0.2, 0.2
RHO = 2.5486                       # mass density [t/m³] (≈ 25 kN/m³ ÷ g)
D = E * T ** 3 / (12.0 * (1.0 - NU * NU))
G = 9.81


def _grid(s, L, n, sec='P'):
    ids = {}
    for i in range(n + 1):
        for j in range(n + 1):
            nid = f'N{i}_{j}'
            ids[(i, j)] = nid
            s.add_node(nid, L * i / n, L * j / n)
    k = 0
    for i in range(n):
        for j in range(n):
            a, b, c, d = (ids[(i, j)], ids[(i + 1, j)],
                          ids[(i + 1, j + 1)], ids[(i, j + 1)])
            s.add_tri_element(f'T{k}', a, b, c, sec); k += 1
            s.add_tri_element(f'T{k}', a, c, d, sec); k += 1
    return ids


def _ss_plate(L, n, with_mass=True):
    s = Structure2D(domain='plate')
    s.add_material('C', E, 25.0, poisson=NU, unit_mass=RHO)
    s.add_plate_section('P', 'C', thickness=T, formulation='DKT')
    s.add_load_case('SW')
    if with_mass:
        s.load_cases_by_id['SW'].self_weight_factor = 1.0
    ids = _grid(s, L, n)
    s.add_support('SS', w=True)
    for (i, j), nid in ids.items():
        if i in (0, n) or j in (0, n):
            s.assign_support(nid, 'SS')
    return s, ids


def _f_kirchhoff(L, m, n):
    """Natural frequency [Hz] of a simply supported square plate, mode (m, n)."""
    return (math.pi / 2.0) * ((m / L) ** 2 + (n / L) ** 2) \
        * math.sqrt(D / (RHO * T))


def test_ss_square_plate_fundamental_frequency():
    """f₁ of a simply supported square plate against Kirchhoff, converging from
    below (the lumped mass under-stiffens the inertia)."""
    L, n = 5.0, 12
    s, _ = _ss_plate(L, n)
    s.add_analysis_case('M', 'Mass', coefficients={'SW': 1.0})
    s.add_analysis_case('MOD', 'Modal', modal_case_id='M', num_modes=6)
    mod = s.calculate()['analysis_cases']['MOD']
    f1 = mod['modal_info'][0]['frequency']
    fa = _f_kirchhoff(L, 1, 1)
    assert f1 == pytest.approx(fa, rel=0.02)
    assert f1 <= fa * 1.001            # lumped distributed mass → from below
    assert mod.get('plate') is True


def test_ss_square_plate_mode_spectrum_degeneracy():
    """The (1,2) and (2,1) modes of a square plate are a degenerate pair, and
    (2,2) sits well above them — the DKT must reproduce that ordering."""
    L, n = 5.0, 12
    s, _ = _ss_plate(L, n)
    s.add_analysis_case('M', 'Mass', coefficients={'SW': 1.0})
    s.add_analysis_case('MOD', 'Modal', modal_case_id='M', num_modes=4)
    info = s.calculate()['analysis_cases']['MOD']['modal_info']
    f = [mi['frequency'] for mi in info]
    assert f == sorted(f)
    assert f[1] == pytest.approx(f[2], rel=0.03)          # degenerate pair
    assert f[1] == pytest.approx(_f_kirchhoff(L, 1, 2), rel=0.04)
    assert f[3] > 1.4 * f[1]                               # (2,2) well above


def test_effective_mass_sums_to_free_vertical_mass():
    """Σ meff over the modes approaches the free (participating) vertical mass —
    the completeness identity, on the "x" slot the plate uses for vertical."""
    L, n = 4.0, 10
    s, _ = _ss_plate(L, n)
    s.add_analysis_case('M', 'Mass', coefficients={'SW': 1.0})
    # Enough modes to capture most of the mass.
    s.add_analysis_case('MOD', 'Modal', modal_case_id='M', num_modes=40)
    mod = s.calculate()['analysis_cases']['MOD']
    meff = sum(mi['meff_x'] for mi in mod['modal_info'])
    assert meff == pytest.approx(mod['total_mass_x'], rel=0.05)
    # The vertical direction carries the mass; the (unused) "y" slot is empty.
    assert mod['total_mass_y'] == pytest.approx(0.0, abs=1e-9)


def test_mass_from_vertical_load_bookkeeping():
    """A Mass case built from a vertical load lumps F/g on the w DOF, and the
    reported nodal mass equals the load divided by g."""
    L, n, P = 4.0, 4, 30.0
    s, ids = _ss_plate(L, n, with_mass=False)
    s.add_load_case('Q')
    c = ids[(n // 2, n // 2)]
    s.add_point_load(c, 'Q', fz=-P)
    s.add_analysis_case('M', 'Mass', coefficients={'Q': 1.0})
    massres = s.calculate()['analysis_cases']['M']
    assert massres['nodal_masses'][c]['mx'] == pytest.approx(P / G, rel=1e-9)
    assert massres['total_mass'] == pytest.approx(P / G, rel=1e-9)


def test_nodal_rotational_inertia_is_not_translational_mass():
    """In the plate domain a NodalMass carries vertical mass (mx) plus rotational
    inertias (my→θx, mtz→θy). Only the vertical part counts as translational
    mass, so adding rotary inertia leaves total_mass_x unchanged."""
    L, n = 4.0, 6
    s, ids = _ss_plate(L, n, with_mass=False)
    c = ids[(n // 2, n // 2)]
    s.add_nodal_mass(c, 'M', mx=2.0, my=0.5, mtz=0.7)   # 2 t + rotary inertias
    s.add_analysis_case('M', 'Mass')
    s.add_analysis_case('MOD', 'Modal', modal_case_id='M', num_modes=3)
    mod = s.calculate()['analysis_cases']['MOD']
    assert mod['model_mass_x'] == pytest.approx(2.0, rel=1e-9)
    assert mod['model_mass_y'] == pytest.approx(0.0, abs=1e-12)


def test_vertical_response_spectrum_runs_and_scales():
    """A vertical response spectrum produces a transverse response, and a
    uniformly larger spectrum scales it up proportionally."""
    L, n = 4.0, 8
    s, _ = _ss_plate(L, n)
    s.add_analysis_case('M', 'Mass', coefficients={'SW': 1.0})
    s.add_analysis_case('MOD', 'Modal', modal_case_id='M', num_modes=8)
    s.add_spectral_function('SP', damping=0.05,
                            points=[[0.0, 1.0], [0.2, 2.5], [1.0, 2.5],
                                    [3.0, 0.5]])
    s.add_analysis_case('SPEC', 'Spectrum', modal_case_id='MOD',
                        spectrum_id='SP', combination_rule='CQC',
                        direction='Z', damping=0.05)
    r = s.calculate()
    sp = r['analysis_cases']['SPEC']
    assert 'error' not in sp
    wmax = max(abs(v[0]) for v in sp['displacements'].values())
    assert wmax > 0.0

    # Double the spectral ordinates → double the (linear) response.
    s.add_spectral_function('SP2', damping=0.05,
                            points=[[0.0, 2.0], [0.2, 5.0], [1.0, 5.0],
                                    [3.0, 1.0]])
    s.add_analysis_case('SPEC2', 'Spectrum', modal_case_id='MOD',
                        spectrum_id='SP2', combination_rule='CQC',
                        direction='Z', damping=0.05)
    sp2 = s.calculate()['analysis_cases']['SPEC2']
    wmax2 = max(abs(v[0]) for v in sp2['displacements'].values())
    assert wmax2 == pytest.approx(2.0 * wmax, rel=1e-6)


def test_pdelta_stays_blocked_in_plate():
    """P-Delta needs in-plane axial force, which a plate does not carry — it must
    stay refused, by both the domain gate and the solve path."""
    L, n = 2.0, 2
    s, _ = _ss_plate(L, n)
    s.add_analysis_case('PD', 'GeometricNonlinear', coefficients={'SW': 1.0})
    assert any('P-Delta' in p for p in s.domain_problems())
    with pytest.raises(ValueError):
        s.calculate()


def test_load_combination_in_plate_is_unchanged():
    """Combinations operate on result vectors and labels, so they must work in
    the plate domain untouched: a 1.35·G + 1.5·Q envelope equals the hand-summed
    static cases."""
    L, n = 4.0, 6
    s, ids = _ss_plate(L, n, with_mass=False)
    s.add_load_case('G'); s.add_load_case('Q')
    for tid in list(s.tri_elements_by_id):
        s.add_area_load(tid, 'G', pz=-5.0)
        s.add_area_load(tid, 'Q', pz=-3.0)
    s.add_load_combination('ULS', coefficients={'G': 1.35, 'Q': 1.5})
    r = s.calculate()
    c = ids[(n // 2, n // 2)]
    wg = r['displacements']['G'][c][0]
    wq = r['displacements']['Q'][c][0]
    wc = r['combinations']['ULS']['displacements'][c][0]
    assert wc == pytest.approx(1.35 * wg + 1.5 * wq, rel=1e-9)
