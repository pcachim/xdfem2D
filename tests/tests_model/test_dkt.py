"""DKT plate-bending validation: patch test, Navier series, clamped plate,
cantilever strip (pins the rotation sign convention), self-weight, edge
loads, surface meshing, and the domain gates. Units: m, kN, kNm.
"""
import math

import numpy as np
import pytest

import context  # noqa: F401
from xdfem2d import Structure2D
from xdfem2d.tri_elements_dkt import wood_armer


def test_wood_armer_simple_case_no_sign_crossing():
    # |mxy| < mx, my → the simple m ± |mxy| rule holds, no correction.
    mx, my, mxy = 40.0, 30.0, 10.0
    mx_bot, my_bot, mx_top, my_top = wood_armer(mx, my, mxy)
    assert mx_bot == pytest.approx(50.0) and my_bot == pytest.approx(40.0)
    # top would be +30/+20 > 0 → clamped to zero (no top steel where it sags)
    assert mx_top == 0.0 and my_top == 0.0


def test_wood_armer_bottom_correction_uses_mxy_squared_over_m():
    # m*x,bot = mx + |mxy| goes negative → set to 0, correct m*y with mxy²/mx.
    mx, my, mxy = -20.0, 20.0, 12.0
    mx_bot, my_bot, mx_top, my_top = wood_armer(mx, my, mxy)
    assert mx_bot == 0.0
    assert my_bot == pytest.approx(my + abs(mxy * mxy / mx))   # 20 + 144/20
    assert mx_bot >= 0.0 and my_bot >= 0.0


def test_wood_armer_top_correction_uses_mxy_squared_over_m():
    # m*y,top = my − |mxy| stays positive → set to 0, correct m*x with mxy²/my.
    mx, my, mxy = -30.0, 20.0, 12.0
    mx_bot, my_bot, mx_top, my_top = wood_armer(mx, my, mxy)
    assert my_top == 0.0
    assert mx_top == pytest.approx(mx - abs(mxy * mxy / my))   # -30 − 144/20
    assert mx_top <= 0.0 and my_top <= 0.0


def test_wood_armer_symmetry_between_x_and_y():
    # Swapping x↔y swaps the two components of each pair.
    a = wood_armer(10.0, -4.0, 9.0)
    b = wood_armer(-4.0, 10.0, 9.0)
    assert (a[0], a[1], a[2], a[3]) == pytest.approx((b[1], b[0], b[3], b[2]))

E, NU, T = 33e6, 0.2, 0.2
D = E * T ** 3 / (12.0 * (1.0 - NU * NU))


def _base():
    s = Structure2D(domain='plate')
    s.add_material('C', E, 25.0, poisson=NU)
    s.add_plate_section('P', 'C', thickness=T, formulation='DKT')
    s.add_load_case('LC')
    return s


def _grid_mesh(s, L, n, ids=None):
    """n×n structured mesh of the square [0,L]²; returns {(i,j): node_id}."""
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
            s.add_tri_element(f'T{k}', a, b, c, 'P'); k += 1
            s.add_tri_element(f'T{k}', a, c, d, 'P'); k += 1
    return ids


def _navier(L, q, terms=39):
    """Exact centre deflection and centre mx of a simply supported square
    plate under uniform q (Navier double series). q negative = downward;
    returns (w_centre, mx_centre) in the program's conventions (w up
    positive, sagging moment positive)."""
    w = mx = 0.0
    for m in range(1, terms + 1, 2):
        for n in range(1, terms + 1, 2):
            f = 16.0 * (-q) / (math.pi ** 6 * m * n
                               * (m ** 2 / L ** 2 + n ** 2 / L ** 2) ** 2)
            s = math.sin(m * math.pi / 2) * math.sin(n * math.pi / 2)
            w += f / D * s
            mx += f * ((m * math.pi / L) ** 2
                       + NU * (n * math.pi / L) ** 2) * s
    return -w, mx


def test_constant_curvature_patch():
    """The patch test: boundary DOFs prescribed from the quadratic field
    w = (x² + xy + y²)/2 (constant curvature w,xx = w,yy = 1, 2w,xy = 1).
    The interior node must land exactly on the field and every element must
    report the same, exact moments m = D_b·(1, 1, 1) — positive, because the
    program's sagging-positive convention is m ∝ +w,xx (concave up)."""
    s = _base()
    pts = {'N1': (0.0, 0.0), 'N2': (2.0, 0.0), 'N3': (2.2, 1.8),
           'N4': (0.0, 1.6), 'N5': (0.9, 0.7)}   # N5 interior, irregular
    for nid, (x, y) in pts.items():
        s.add_node(nid, x, y)
    for k, (a, b, c) in enumerate((('N1', 'N2', 'N5'), ('N2', 'N3', 'N5'),
                                   ('N3', 'N4', 'N5'), ('N4', 'N1', 'N5'))):
        s.add_tri_element(f'T{k}', a, b, c, 'P')

    def field(x, y):
        w = (x * x + x * y + y * y) / 2.0
        wx = (2 * x + y) / 2.0
        wy = (x + 2 * y) / 2.0
        return w, wy, -wx          # (w, tx, ty): tx = w,y ; ty = −w,x

    s.add_support('ALL', w=True, tx=True, ty=True)
    for nid in ('N1', 'N2', 'N3', 'N4'):
        s.assign_support(nid, 'ALL')
        w, tx, ty = field(*pts[nid])
        s.create_support_settlement(nid, 'LC', w=w, tx=tx, ty=ty)
    r = s.calculate()

    w5 = r['displacements']['LC']['N5']
    assert w5 == pytest.approx(list(field(*pts['N5'])), rel=1e-8, abs=1e-10)

    # κ = (w,xx, w,yy, 2w,xy) = (1, 1, 1) in the sagging-positive convention.
    Db = D * np.array([[1, NU, 0], [NU, 1, 0], [0, 0, (1 - NU) / 2]])
    m_exact = Db @ np.array([1.0, 1.0, 1.0])
    for tid, mres in r['tri_stress']['LC'].items():
        assert mres['mx'] == pytest.approx(m_exact[0], rel=1e-8)
        assert mres['my'] == pytest.approx(m_exact[1], rel=1e-8)
        assert mres['mxy'] == pytest.approx(m_exact[2], rel=1e-8)
        assert mres['vx'] == pytest.approx(0.0, abs=1e-5)
        assert mres['vy'] == pytest.approx(0.0, abs=1e-5)


def test_navier_simply_supported_square():
    L, q, n = 6.0, -10.0, 12
    s = _base()
    ids = _grid_mesh(s, L, n)
    s.add_support('SS', w=True)
    for (i, j), nid in ids.items():
        if i in (0, n) or j in (0, n):
            s.assign_support(nid, 'SS')
    for tid in list(s.tri_elements_by_id):
        s.add_area_load(tid, 'LC', pz=q)
    r = s.calculate()

    w_ex, mx_ex = _navier(L, q)
    wc = r['displacements']['LC'][ids[(n // 2, n // 2)]][0]
    assert wc == pytest.approx(w_ex, rel=0.01)

    cen = ids[(n // 2, n // 2)]
    tris = s.tri_elements_by_id
    mxs = [v['mx'] for tid, v in r['tri_stress']['LC'].items()
           if cen in (tris[tid].node_i, tris[tid].node_j, tris[tid].node_k)]
    assert np.mean(mxs) == pytest.approx(mx_ex, rel=0.05)
    # Sagging positive at the centre; Wood–Armer bottom ≥ mx.
    assert np.mean(mxs) > 0
    from xdfem2d.tri_elements_dkt import wood_armer
    for tid, v in r['tri_stress']['LC'].items():
        wa = wood_armer(v['mx'], v['my'], v['mxy'])
        assert (v['mx_bot'], v['my_bot'],
                v['mx_top'], v['my_top']) == pytest.approx(wa)
        # Bottom clamps at ≥ 0, top at ≤ 0 (W-A sign convention).
        assert v['mx_bot'] >= -1e-9 and v['my_bot'] >= -1e-9
        assert v['mx_top'] <= 1e-9 and v['my_top'] <= 1e-9

    # Equilibrium: total reaction equals the total load q·L².
    tot = sum(v[0] for v in r['reactions']['LC'].values())
    assert tot == pytest.approx(-q * L * L, rel=1e-6)


def test_clamped_square():
    """Clamped square plate, uniform load: w_c = 0.00126·qL⁴/D (Timoshenko)."""
    L, q, n = 6.0, -10.0, 12
    s = _base()
    ids = _grid_mesh(s, L, n)
    s.add_support('ENC', w=True, tx=True, ty=True)
    for (i, j), nid in ids.items():
        if i in (0, n) or j in (0, n):
            s.assign_support(nid, 'ENC')
    for tid in list(s.tri_elements_by_id):
        s.add_area_load(tid, 'LC', pz=q)
    r = s.calculate()
    wc = r['displacements']['LC'][ids[(n // 2, n // 2)]][0]
    assert wc == pytest.approx(0.00126 * q * L ** 4 / D, rel=0.02)


def test_cantilever_strip_matches_beam_and_rotation_sign():
    """A slab strip clamped along x = 0 under an edge line load behaves as a
    cantilever of rigidity D per unit width: w = −PL³/3D, ty = +PL²/2D —
    the ty sign is what pins the DKT convention to the grillage's."""
    L, W, P, n = 4.0, 1.0, 5.0, 8      # P per metre of width
    s = _base()
    ids = {}
    for i in range(n + 1):
        for j in (0, 1):
            nid = f'N{i}_{j}'
            ids[(i, j)] = nid
            s.add_node(nid, L * i / n, W * j)
    k = 0
    for i in range(n):
        a, b, c, d = (ids[(i, 0)], ids[(i + 1, 0)],
                      ids[(i + 1, 1)], ids[(i, 1)])
        s.add_tri_element(f'T{k}', a, b, c, 'P'); k += 1
        s.add_tri_element(f'T{k}', a, c, d, 'P'); k += 1
    s.add_support('ENC', w=True, tx=True, ty=True)
    s.assign_support(ids[(0, 0)], 'ENC')
    s.assign_support(ids[(0, 1)], 'ENC')
    # Free strip edges curl by ν; suppress tx along the strip to keep the
    # cylindrical-bending (beam) analogy exact.
    s.add_support('CYL', tx=True)
    for i in range(1, n + 1):
        s.assign_support(ids[(i, 0)], 'CYL')
        s.assign_support(ids[(i, 1)], 'CYL')
    s.add_point_load(ids[(n, 0)], 'LC', fz=-P * W / 2)
    s.add_point_load(ids[(n, 1)], 'LC', fz=-P * W / 2)
    r = s.calculate()

    w_tip = r['displacements']['LC'][ids[(n, 0)]][0]
    ty_tip = r['displacements']['LC'][ids[(n, 0)]][2]
    assert w_tip == pytest.approx(-P * L ** 3 / (3 * D), rel=0.02)
    assert ty_tip == pytest.approx(P * L ** 2 / (2 * D), rel=0.02)
    # Root moment per metre ≈ −P·L (hogging).
    root = [v['mx'] for tid, v in r['tri_stress']['LC'].items()
            if tid in ('T0', 'T1')]
    assert np.mean(root) == pytest.approx(-P * L, rel=0.15)


def test_self_weight_equals_equivalent_pressure():
    """self_weight_factor=1 on a slab equals pz = −γ·t applied by hand."""
    L, n = 4.0, 6
    gt = 25.0 * T

    def build(sw):
        s = _base()
        ids = _grid_mesh(s, L, n)
        s.add_support('SS', w=True)
        for (i, j), nid in ids.items():
            if i in (0, n) or j in (0, n):
                s.assign_support(nid, 'SS')
        if sw:
            s.load_cases_by_id['LC'].self_weight_factor = 1.0
        else:
            for tid in list(s.tri_elements_by_id):
                s.add_area_load(tid, 'LC', pz=-gt)
        return s, ids

    s1, ids = build(True)
    s2, _ = build(False)
    r1, r2 = s1.calculate(), s2.calculate()
    c = ids[(n // 2, n // 2)]
    assert r1['displacements']['LC'][c] == pytest.approx(
        r2['displacements']['LC'][c], rel=1e-9)


def test_edge_line_load_total_reaction():
    """A transverse line load on a triangle edge (a wall on the slab) enters
    through the fx slot [kN/m] and lands whole on the supports."""
    L, n, fz = 4.0, 4, -8.0
    s = _base()
    ids = _grid_mesh(s, L, n)
    s.add_support('SS', w=True)
    for (i, j), nid in ids.items():
        if i in (0, n) or j in (0, n):
            s.assign_support(nid, 'SS')
    # Load the mesh edge from (1,1) to (2,1) of triangle T? — find a tri
    # containing both nodes.
    na, nb = ids[(1, 1)], ids[(2, 1)]
    tri = next(t for t in s.tri_elements
               if {na, nb} <= {t.node_i, t.node_j, t.node_k})
    s.add_tri_edge_load('EL1', tri.id, na, nb, 'LC', fx=fz)
    r = s.calculate()
    seg = L / n
    tot = sum(v[0] for v in r['reactions']['LC'].values())
    assert tot == pytest.approx(-fz * seg, rel=1e-6)


def test_surface_object_slab():
    """A meshed surface region with an area load: the pressure reaches the
    generated triangles and equilibrium holds."""
    L, q = 6.0, -10.0
    s = _base()
    obj = s.add_geo_surface('S1', [(0.0, 0.0), (L, 0.0), (L, L), (0.0, L)],
                            section_name='P', target_size=0.75)
    s.add_support('SS', w=True)
    for nid in obj.node_ids:
        s.assign_support(nid, 'SS')   # + propagation along the edges
    s.add_area_load('S1', 'LC', pz=q)
    r = s.calculate()
    tot = sum(v[0] for v in r['reactions']['LC'].values())
    assert tot == pytest.approx(-q * L * L, rel=1e-6)


def test_domain_gates():
    # DKT in a plane model is refused.
    p = Structure2D()
    p.add_material('C', E, 25.0, poisson=NU)
    p.add_plate_section('P', 'C', thickness=T, formulation='DKT')
    p.add_node('N1', 0, 0); p.add_node('N2', 1, 0); p.add_node('N3', 0, 1)
    p.add_tri_element('T1', 'N1', 'N2', 'N3', 'P')
    assert any('DKT' in x for x in p.domain_problems())
    # A membrane section in a plate model is refused.
    s = _base()
    s.add_tri_section('M', 'C', thickness=T, formulation='CST')
    s.add_node('N1', 0, 0); s.add_node('N2', 1, 0); s.add_node('N3', 0, 1)
    s.add_tri_element('T1', 'N1', 'N2', 'N3', 'M')
    assert any('CST' in x for x in s.domain_problems())


def test_slab_with_stiffening_beam():
    """Laje vigada — the cross-consistency test between DKT and the grillage
    bar: both must share the (w, tx, ty) convention, or the beam would fight
    the slab instead of stiffening it. The beam must reduce the centre
    deflection and carry a sagging-positive midspan moment."""
    L, q, n = 6.0, -10.0, 8

    def build(with_beam):
        s = _base()
        s.add_section('V', 'C', 0.3, 0.6, shape='Rectangular')
        ids = _grid_mesh(s, L, n)
        if with_beam:
            for i in range(n):
                s.add_bar_element(f'B{i}', ids[(i, n // 2)],
                                  ids[(i + 1, n // 2)], 'V')
        s.add_support('SS', w=True)
        for (i, j), nid in ids.items():
            if i in (0, n) or j in (0, n):
                s.assign_support(nid, 'SS')
        for tid in list(s.tri_elements_by_id):
            s.add_area_load(tid, 'LC', pz=q)
        return s, ids[(n // 2, n // 2)]

    s0, c = build(False)
    w0 = s0.calculate()['displacements']['LC'][c][0]
    s1, c = build(True)
    r1 = s1.calculate()
    w1 = r1['displacements']['LC'][c][0]
    assert w1 < 0 and abs(w1) < 0.75 * abs(w0)
    # Sagging-positive moment in the beam segment next to midspan.
    assert r1['element_forces']['LC'][f'B{n//2 - 1}']['j'][2] > 5.0


def test_area_load_round_trip():
    from xdfem2d.structure_io import _to_dict, _from_dict
    s = _base()
    _grid_mesh(s, 2.0, 1)
    s.add_area_load('T0', 'LC', pz=-7.5)
    q = _from_dict(_to_dict(s))
    assert q.tri_area_loads[0].pz == -7.5
    assert q.tri_area_loads[0].tri_id == 'T0'


# ---------------------------------------------------------------------------
# Thermal through-thickness gradient (ΔT/t → curvature κ₀)
# ---------------------------------------------------------------------------
ALPHA = 1e-5


def _base_thermal():
    """A plate model whose material carries an explicit expansion coefficient."""
    s = Structure2D(domain='plate')
    s.add_material('C', E, 25.0, poisson=NU, alpha=ALPHA)
    s.add_plate_section('P', 'C', thickness=T, formulation='DKT')
    s.add_load_case('LC')
    return s


def test_thermal_gradient_free_plate_reports_zero_moment():
    """A plate free to curve under a uniform through-thickness gradient adopts
    the free thermal curvature and reports *zero* moment everywhere — the plate
    analogue of a freely expanding CST reporting zero stress. This pins the
    consistency between the gradient load and the moment recovery."""
    s = _base_thermal()
    L, n = 2.0, 4
    ids = _grid_mesh(s, L, n)
    # Statically determinate restraint: kill the three rigid-body modes only
    # (three w DOFs), leaving the plate free to curve.
    s.add_support('W', w=True)
    for corner in ((0, 0), (n, 0), (0, n)):
        s.assign_support(ids[corner], 'W')
    for tid in list(s.tri_elements_by_id):
        s.add_tri_temperature_load(tid, 'LC', dt_gradient=20.0)
    r = s.calculate()
    for v in r['tri_stress']['LC'].values():
        assert v['mx'] == pytest.approx(0.0, abs=1e-6)
        assert v['my'] == pytest.approx(0.0, abs=1e-6)
        assert v['mxy'] == pytest.approx(0.0, abs=1e-6)
    # The free curvature is spherical: w = ½·κ·(x²+y²) + a + bx + cy. The
    # physical curvature is w,xx = +α·ΔT/t (a positive top−bottom gradient bows
    # the free plate concave-up here, the sign that makes the *restrained* plate
    # hog — see the clamped test). The three pinned corners (0,0), (L,0), (0,L)
    # fix the rigid body: a = 0, b = c = −½·κ·L. Check the centre node.
    kphys = ALPHA * 20.0 / T
    w = r['displacements']['LC']

    def w_field(x, y):
        return 0.5 * kphys * (x * x + y * y) - 0.5 * kphys * L * (x + y)

    xc = yc = L / 2.0
    assert w[ids[(n // 2, n // 2)]][0] == pytest.approx(w_field(xc, yc),
                                                        rel=1e-6, abs=1e-12)


def test_thermal_gradient_clamped_plate_matches_analytic_moment():
    """A fully clamped plate under a uniform gradient stays flat (w ≡ 0 is the
    exact solution) and carries the uniform restraint moment
    m = −(α·ΔT/t)·D·(1+ν) — Timoshenko's thermal-moment result."""
    s = _base_thermal()
    L, n, dt = 2.0, 8, 20.0
    ids = _grid_mesh(s, L, n)
    s.add_support('ENC', w=True, tx=True, ty=True)
    for (i, j), nid in ids.items():
        if i in (0, n) or j in (0, n):
            s.assign_support(nid, 'ENC')
    for tid in list(s.tri_elements_by_id):
        s.add_tri_temperature_load(tid, 'LC', dt_gradient=dt)
    r = s.calculate()
    m_exact = -(ALPHA * dt / T) * D * (1 + NU)
    for v in r['tri_stress']['LC'].values():
        assert v['mx'] == pytest.approx(m_exact, rel=1e-6)
        assert v['my'] == pytest.approx(m_exact, rel=1e-6)
        assert v['mxy'] == pytest.approx(0.0, abs=1e-6)
    wmax = max(abs(d[0]) for d in r['displacements']['LC'].values())
    assert wmax == pytest.approx(0.0, abs=1e-9)


def test_thermal_gradient_round_trip_and_script():
    from xdfem2d.structure_io import _to_dict, _from_dict
    from xdfem2d.script_export import to_python
    s = _base_thermal()
    _grid_mesh(s, 2.0, 1)
    s.add_tri_temperature_load('T0', 'LC', dt_gradient=15.0)
    q = _from_dict(_to_dict(s))
    assert q.tri_temperature_loads[0].dt_gradient == 15.0
    assert 'dt_gradient=15.0' in to_python(s).replace(' ', '')


# ---------------------------------------------------------------------------
# Winkler area spring (slab on grade)
# ---------------------------------------------------------------------------

def test_winkler_area_spring_uniform_bed():
    """A slab resting fully on a Winkler bed under uniform pressure settles as a
    rigid body w = pz/kz with zero bending — the lumped kz·A/3 balances the
    lumped pz·A/3 at every node. Rotations are pinned (the bed gives no
    rotational stiffness)."""
    s = Structure2D(domain='plate')
    s.add_material('C', E, 25.0, poisson=NU)
    s.add_plate_section('P', 'C', thickness=T, formulation='DKT')
    s.add_load_case('LC')
    L, n, pz, kz = 4.0, 6, -8.0, 5000.0
    ids = _grid_mesh(s, L, n)
    for tid in list(s.tri_elements_by_id):
        s.add_area_load(tid, 'LC', pz=pz)
        s.add_area_spring(tid, kz=kz)
    s.add_support('R', tx=True, ty=True)   # bed has no rotational stiffness
    for nid in ids.values():
        s.assign_support(nid, 'R')
    r = s.calculate()
    ws = [d[0] for d in r['displacements']['LC'].values()]
    assert min(ws) == pytest.approx(pz / kz, rel=1e-6)
    assert max(ws) - min(ws) == pytest.approx(0.0, abs=1e-9)
    for v in r['tri_stress']['LC'].values():
        assert abs(v['mx']) + abs(v['my']) + abs(v['mxy']) == pytest.approx(
            0.0, abs=1e-6)


def test_winkler_area_spring_stiffens_slab():
    """Adding a Winkler bed under a simply supported loaded slab reduces the
    centre deflection — the foundation shares the load."""
    L, q, n, kz = 6.0, -10.0, 8, 2000.0

    def build(bed):
        s = _base()
        ids = _grid_mesh(s, L, n)
        s.add_support('SS', w=True)
        for (i, j), nid in ids.items():
            if i in (0, n) or j in (0, n):
                s.assign_support(nid, 'SS')
        for tid in list(s.tri_elements_by_id):
            s.add_area_load(tid, 'LC', pz=q)
            if bed:
                s.add_area_spring(tid, kz=kz)
        return s, ids[(n // 2, n // 2)]

    s0, c = build(False)
    s1, c = build(True)
    w0 = s0.calculate()['displacements']['LC'][c][0]
    w1 = s1.calculate()['displacements']['LC'][c][0]
    assert w1 < 0 and abs(w1) < abs(w0)


def test_surface_area_spring_and_gradient_propagate():
    """An area spring and a temperature gradient set on a surface object reach
    the triangles it meshes into at solve time."""
    L = 4.0
    s = _base_thermal()
    s.add_geo_surface('S1', [(0.0, 0.0), (L, 0.0), (L, L), (0.0, L)],
                      section_name='P', target_size=1.0)
    s.add_area_spring('S1', kz=3000.0)
    s.add_area_temperature_load('S1', 'LC', dt_gradient=12.0)
    from xdfem2d.geo_expand import expand_geometry
    comp, _trace = expand_geometry(s)
    assert comp.tri_area_springs, "surface area spring did not propagate"
    assert all(a.kz == 3000.0 for a in comp.tri_area_springs)
    grads = [t.dt_gradient for t in comp.tri_temperature_loads]
    assert grads and all(g == 12.0 for g in grads)


def test_area_spring_round_trip_and_domain_gate():
    from xdfem2d.structure_io import _to_dict, _from_dict
    s = _base()
    _grid_mesh(s, 2.0, 1)
    s.add_area_spring('T0', kz=4200.0)
    q = _from_dict(_to_dict(s))
    assert q.tri_area_springs[0].kz == 4200.0
    assert q.tri_area_springs[0].tri_id == 'T0'
    # A Winkler area spring in a plane model is refused.
    p = Structure2D()
    p.add_material('C', E, 25.0, poisson=NU)
    p.add_tri_section('M', 'C', thickness=T, formulation='CST')
    p.add_node('N1', 0, 0); p.add_node('N2', 1, 0); p.add_node('N3', 0, 1)
    p.add_tri_element('T1', 'N1', 'N2', 'N3', 'M')
    p.add_area_spring('T1', kz=100.0)
    assert any('Winkler' in x for x in p.domain_problems())
