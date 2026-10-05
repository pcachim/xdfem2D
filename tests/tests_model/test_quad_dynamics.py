"""Plate-domain dynamics with quads (dev/refactor_area_path.md item C).

Mass / Modal / Spectrum are element-shape-agnostic (mass comes from the load
vector + NodalMass, the modal solve consumes the already-assembled global K
which includes quads since Phase 4). This validates that they not only *run*
with quads but give the right numbers, against the Kirchhoff closed form for a
simply supported square plate — the quad (MITC4) analogue of test_dkt_dynamics.
Units: m, kN, kNm, t.
"""
import math

import pytest

import context  # noqa: F401
from xdfem2d import Structure2D

E, NU, T = 33e6, 0.2, 0.2
RHO = 2.5486                       # mass density [t/m³] (≈ 25 kN/m³ ÷ g)
D = E * T ** 3 / (12.0 * (1.0 - NU * NU))
G = 9.81


def _quad_grid(s, L, n, sec='P'):
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
            s.add_quad_element(f'Q{k}', a, b, c, d, sec); k += 1
    return ids


def _ss_plate(L, n, with_mass=True):
    s = Structure2D(domain='plate')
    s.add_material('C', E, 25.0, poisson=NU, unit_mass=RHO)
    s.add_quad_section('P', 'C', thickness=T, formulation='MITC4')
    s.add_load_case('SW')
    if with_mass:
        s.load_cases_by_id['SW'].self_weight_factor = 1.0
    ids = _quad_grid(s, L, n)
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
    """f₁ of a MITC4-meshed simply supported square plate against Kirchhoff,
    converging from below (lumped mass + MITC shear under-stiffens)."""
    L, n = 5.0, 12
    s, _ = _ss_plate(L, n)
    s.add_analysis_case('M', 'Mass', coefficients={'SW': 1.0})
    s.add_analysis_case('MOD', 'Modal', modal_case_id='M', num_modes=6)
    mod = s.calculate()['analysis_cases']['MOD']
    f1 = mod['modal_info'][0]['frequency']
    fa = _f_kirchhoff(L, 1, 1)
    assert f1 == pytest.approx(fa, rel=0.03)
    assert f1 <= fa * 1.001            # from below
    assert mod.get('plate') is True


def test_ss_square_plate_mode_ordering_and_degeneracy():
    """(1,2)/(2,1) are a degenerate pair and (2,2) sits well above — MITC4 must
    reproduce the ordering."""
    L, n = 5.0, 12
    s, _ = _ss_plate(L, n)
    s.add_analysis_case('M', 'Mass', coefficients={'SW': 1.0})
    s.add_analysis_case('MOD', 'Modal', modal_case_id='M', num_modes=4)
    f = [mi['frequency'] for mi in
         s.calculate()['analysis_cases']['MOD']['modal_info']]
    assert f == sorted(f)
    assert f[1] == pytest.approx(f[2], rel=0.03)
    assert f[1] == pytest.approx(_f_kirchhoff(L, 1, 2), rel=0.04)
    assert f[3] > 1.4 * f[1]


def test_effective_mass_completeness():
    """Σ meff over enough modes approaches the free vertical mass (the "x" slot
    the plate uses for vertical)."""
    L, n = 4.0, 10
    s, _ = _ss_plate(L, n)
    s.add_analysis_case('M', 'Mass', coefficients={'SW': 1.0})
    s.add_analysis_case('MOD', 'Modal', modal_case_id='M', num_modes=40)
    mod = s.calculate()['analysis_cases']['MOD']
    meff = sum(mi['meff_x'] for mi in mod['modal_info'])
    assert meff == pytest.approx(mod['total_mass_x'], rel=0.05)
    assert mod['total_mass_y'] == pytest.approx(0.0, abs=1e-9)


def test_mass_from_vertical_load_bookkeeping():
    """A Mass case from a vertical load lumps F/g on the w DOF of a quad mesh."""
    L, n, P = 4.0, 4, 30.0
    s, ids = _ss_plate(L, n, with_mass=False)
    s.add_load_case('Q')
    c = ids[(n // 2, n // 2)]
    s.add_point_load(c, 'Q', fz=-P)
    s.add_analysis_case('M', 'Mass', coefficients={'Q': 1.0})
    massres = s.calculate()['analysis_cases']['M']
    assert massres['nodal_masses'][c]['mx'] == pytest.approx(P / G, rel=1e-9)
    assert massres['total_mass'] == pytest.approx(P / G, rel=1e-9)


def test_vertical_response_spectrum_runs_and_scales():
    """A vertical response spectrum on a quad slab produces a transverse
    response, and a uniformly doubled spectrum doubles it."""
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
    sp = s.calculate()['analysis_cases']['SPEC']
    assert 'error' not in sp
    wmax = max(abs(v[0]) for v in sp['displacements'].values())
    assert wmax > 0.0

    s.add_spectral_function('SP2', damping=0.05,
                            points=[[0.0, 2.0], [0.2, 5.0], [1.0, 5.0],
                                    [3.0, 1.0]])
    s.add_analysis_case('SPEC2', 'Spectrum', modal_case_id='MOD',
                        spectrum_id='SP2', combination_rule='CQC',
                        direction='Z', damping=0.05)
    sp2 = s.calculate()['analysis_cases']['SPEC2']
    wmax2 = max(abs(v[0]) for v in sp2['displacements'].values())
    assert wmax2 == pytest.approx(2.0 * wmax, rel=1e-6)


def test_mixed_tri_quad_modal_and_spectrum_run():
    """A slab meshed half in MITC4 quads, half in DKT triangles, solves Modal
    and Spectrum through the full pipeline with a sensible fundamental mode and
    no error — the two element kinds coexist in the dynamic solve."""
    L, n = 4.0, 8
    s = Structure2D(domain='plate')
    s.add_material('C', E, 25.0, poisson=NU, unit_mass=RHO)
    s.add_quad_section('Q', 'C', thickness=T, formulation='MITC4')
    s.add_plate_section('Tsec', 'C', thickness=T, formulation='DKT')
    s.add_load_case('SW'); s.load_cases_by_id['SW'].self_weight_factor = 1.0
    ids = {}
    for i in range(n + 1):
        for j in range(n + 1):
            nid = f'N{i}_{j}'; ids[(i, j)] = nid
            s.add_node(nid, L * i / n, L * j / n)
    k = 0
    for i in range(n):
        for j in range(n):
            a, b, c, d = (ids[(i, j)], ids[(i + 1, j)],
                          ids[(i + 1, j + 1)], ids[(i, j + 1)])
            if i < n // 2:
                s.add_quad_element(f'Q{k}', a, b, c, d, 'Q'); k += 1
            else:
                s.add_tri_element(f'T{k}', a, b, c, 'Tsec')
                s.add_tri_element(f'T{k}b', a, c, d, 'Tsec'); k += 1
    s.add_support('SS', w=True)
    for (i, j), nid in ids.items():
        if i in (0, n) or j in (0, n):
            s.assign_support(nid, 'SS')
    s.add_analysis_case('M', 'Mass', coefficients={'SW': 1.0})
    s.add_analysis_case('MOD', 'Modal', modal_case_id='M', num_modes=4)
    s.add_spectral_function('SP', damping=0.05,
                            points=[[0.0, 1.0], [0.2, 2.5], [1.0, 2.5],
                                    [3.0, 0.5]])
    s.add_analysis_case('SPEC', 'Spectrum', modal_case_id='MOD',
                        spectrum_id='SP', combination_rule='CQC',
                        direction='Z', damping=0.05)
    r = s.calculate()
    mod, sp = r['analysis_cases']['MOD'], r['analysis_cases']['SPEC']
    assert mod.get('plate') is True
    assert mod['modal_info'][0]['frequency'] > 0.0
    assert 'error' not in sp
    assert max(abs(v[0]) for v in sp['displacements'].values()) > 0.0


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-q']))
