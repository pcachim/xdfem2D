"""Phase 8B — the MITC3 shear-deformable plate triangle.

MITC3 is the Mindlin-Reissner counterpart of the DKT: same 3 nodes and 9 DOFs
(w, θx, θy), but with transverse-shear deformation, so it deflects more than
thin-plate theory for a thick slab (which the DKT cannot) while still converging
to the thin solution as the slab gets thin. These check the element in
isolation (symmetry, rigid-body modes) and through the engine (convergence to
Navier, thick-plate shear deflection, equilibrium, a formulation-agnostic mass).
"""
import math

import context  # noqa: F401

import numpy as np

from xdfem2d import Structure2D
from xdfem2d.tri_elements_mitc3 import mitc3_matrices


# ── The element in isolation ────────────────────────────────────────────────

def test_stiffness_is_symmetric_with_three_rigid_body_modes():
    k, Bb, Bs_c, Db, Ds, area = mitc3_matrices([(0.0, 0.0), (1.3, 0.1),
                                                (0.2, 1.1)], 30e6, 0.2, 0.2)
    assert np.allclose(k, k.T)
    ev = np.linalg.eigvalsh(k)
    # A plate element has exactly three zero-energy modes: the transverse
    # translation and the two rigid rotations.
    assert int(np.sum(np.abs(ev) < 1e-6 * abs(ev).max())) == 3


def test_rigid_body_displacements_store_no_energy():
    coords = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
    k, *_ = mitc3_matrices(coords, 30e6, 0.2, 0.2)
    # Uniform w translation, and the two rigid rotations (w = y with θx = 1;
    # w = x with θy = −1) — all curvature- and shear-free.
    for u in (np.array([1, 0, 0, 1, 0, 0, 1, 0, 0.0]),
              np.array([0, 1, 0, 0, 1, 0, 1, 1, 0.0]),
              np.array([0, 0, -1, 1, 0, -1, 0, 0, -1.0])):
        assert abs(float(u @ k @ u)) < 1e-6


# ── Helpers: a simply supported square plate and the Navier deflection ───────

def _ss_plate(n, formulation, t, L=5.0, E=30e6, nu=0.2, pz=-10.0):
    s = Structure2D(domain='plate')
    s.add_material('M', elastic_modulus=E, unit_weight=0.0, poisson=nu)
    s.add_tri_section('S', 'M', thickness=t, formulation=formulation)
    ids = {}
    for j in range(n + 1):
        for i in range(n + 1):
            nid = f'N{i}_{j}'
            ids[(i, j)] = nid
            s.add_node(nid, L * i / n, L * j / n)
    e = 0
    for j in range(n):
        for i in range(n):
            a, b = ids[(i, j)], ids[(i + 1, j)]
            c, d = ids[(i + 1, j + 1)], ids[(i, j + 1)]
            s.add_tri_element(f'T{e}', a, b, c, 'S'); e += 1
            s.add_tri_element(f'T{e}', a, c, d, 'S'); e += 1
    s.add_support('SS', w=True)
    for (i, j), nid in ids.items():
        if i in (0, n) or j in (0, n):
            s.assign_support(nid, 'SS')
    s.add_load_case('G')
    for tid in list(s.tri_elements_by_id):
        s.add_area_load(tid, 'G', pz=pz)
    return s, ids


def _navier_wmax(q, L, E, nu, t):
    """Thin-plate (Kirchhoff) centre deflection of a SS square plate under a
    uniform load, by the Navier double series."""
    D = E * t ** 3 / (12.0 * (1.0 - nu ** 2))
    w = 0.0
    for m in range(1, 60, 2):
        for k in range(1, 60, 2):
            w += (math.sin(m * math.pi / 2) * math.sin(k * math.pi / 2)) \
                / (m * k * (m * m + k * k) ** 2)
    return w * 16.0 * q / (math.pi ** 6 * D) * L ** 4


def _centre_w(s, ids, n):
    r = s.calculate()
    disp = r['analysis_cases']['G']['displacements']
    return abs(disp[ids[(n // 2, n // 2)]][0])


# ── Convergence to the thin solution ────────────────────────────────────────

def test_thin_plate_converges_to_navier():
    t, L = 0.05, 5.0
    wn = _navier_wmax(10.0, L, 30e6, 0.2, t)
    ratios = []
    for n in (16, 24, 32):
        s, ids = _ss_plate(n, 'MITC3', t, L=L)
        ratios.append(_centre_w(s, ids, n) / wn)
    # Monotone convergence from below, within ~1% by n = 32.
    assert ratios[0] < ratios[1] < ratios[2]
    assert 0.99 < ratios[-1] < 1.02


# ── Thick plate: MITC3 adds the shear deflection the DKT misses ──────────────

def test_thick_plate_deflects_more_than_thin_theory_and_more_than_dkt():
    t, L = 0.5, 5.0                      # span/thickness = 10 → genuinely thick
    wn = _navier_wmax(10.0, L, 30e6, 0.2, t)
    s_m, ids = _ss_plate(16, 'MITC3', t, L=L)
    s_d, _ = _ss_plate(16, 'DKT', t, L=L)
    w_mitc = _centre_w(s_m, ids, 16)
    w_dkt = _centre_w(s_d, ids, 16)
    # The DKT tracks thin theory whatever the thickness; MITC3 adds a few percent
    # of transverse-shear deflection on top, which is the whole point.
    assert w_mitc > w_dkt
    assert 1.03 < w_mitc / wn < 1.20


# ── Results and equilibrium through the engine ──────────────────────────────

def test_reports_shear_field_and_balances_the_load():
    t = 0.3
    s, ids = _ss_plate(6, 'MITC3', t, pz=-5.0)
    r = s.calculate()
    ts = r['analysis_cases']['G']['tri_stress']
    one = next(iter(ts.values()))
    assert one['formulation'] == 'MITC3'
    assert {'mx', 'my', 'mxy', 'vx', 'vy', 'mx_bot', 'my_top'}.issubset(one)
    # ΣRz balances the applied pressure (25 m² · 5 kN/m² = 125 kN up).
    reac = r['analysis_cases']['G']['reactions']
    assert abs(sum(v[0] for v in reac.values()) - 125.0) < 1e-6


# ── The plumbing is formulation-agnostic (mass depends on geometry, not the
#    element): swapping DKT ↔ MITC3 leaves the nodal masses identical ─────────

def test_self_weight_is_identical_between_dkt_and_mitc3():
    # The self-weight that seeds the plate mass is density·area, not stiffness,
    # so it does not depend on the element. Same geometry and material → same
    # total self-weight reaction whichever plate formulation is used.
    s_d, _ = _ss_plate(4, 'DKT', 0.3, pz=0.0)
    s_m, _ = _ss_plate(4, 'MITC3', 0.3, pz=0.0)
    for s in (s_d, s_m):
        s.materials['M'].unit_weight = 25.0
        s.add_load_case('SW', self_weight_factor=1.0)
    rd = s_d.calculate()['analysis_cases']['SW']['reactions']
    rm = s_m.calculate()['analysis_cases']['SW']['reactions']
    assert abs(sum(v[0] for v in rd.values())
               - sum(v[0] for v in rm.values())) < 1e-9
    assert abs(sum(v[0] for v in rd.values())) > 0.0   # there is weight
