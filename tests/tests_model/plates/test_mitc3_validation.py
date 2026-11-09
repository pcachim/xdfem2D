"""Phase 8C — formal validation of the MITC3 plate element.

Four independent checks, each targeting a property the element must have:

* the constant-moment **patch test** on an irregular mesh (exactness);
* convergence to the **exact Mindlin** analytical solution for a thick plate
  under the matching (hard) support (quantitative accuracy of the shear);
* the **boundary-layer** effect — a soft simple support deflects more than a
  hard one, the physical reason the FE and a textbook series can disagree;
* robustness on a **distorted** mesh.
"""
import math

import context  # noqa: F401

import numpy as np

from xdfem2d import Structure2D
from xdfem2d.tri_elements_mitc3 import _entry, mitc3_matrices


# ── 1. Constant-moment patch test ───────────────────────────────────────────

def test_constant_moment_patch_test_on_an_irregular_mesh():
    """A patch of four MITC3 triangles around one interior node, its boundary
    DOFs prescribed from an exact constant-curvature field. The solved interior
    node and every element's moment must be exact — the defining patch test."""
    E, nu, t = 30e6, 0.25, 0.2
    nodes = {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (1.0, 1.0),
             3: (0.0, 1.0), 4: (0.42, 0.37)}          # 4 is the interior node
    tris = [(0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)]
    kx, ky, kxy = 2e-4, -1e-4, 0.5e-4

    def field(x, y):
        return [0.5 * (kx * x * x + ky * y * y) + kxy * x * y,   # w
                ky * y + kxy * x,                                # θx = ∂w/∂y
                -(kx * x + kxy * y)]                             # θy = −∂w/∂x

    ndof = 15
    K = np.zeros((ndof, ndof))
    Db = None
    for (i, j, k) in tris:
        coords = [nodes[i], nodes[j], nodes[k]]
        ke, _, _, Db, _, _ = mitc3_matrices(coords, E, nu, t)
        dofs = [3 * n + c for n in (i, j, k) for c in range(3)]
        K[np.ix_(dofs, dofs)] += ke

    u = np.zeros(ndof)
    for n in (0, 1, 2, 3):
        u[3 * n:3 * n + 3] = field(*nodes[n])
    free = [12, 13, 14]
    presc = [d for d in range(ndof) if d not in free]
    u[free] = np.linalg.solve(K[np.ix_(free, free)],
                              -K[np.ix_(free, presc)] @ u[presc])

    # The interior node comes out exactly on the field.
    assert np.allclose(u[free], field(*nodes[4]), atol=1e-12)

    # Every element reports the one exact constant moment m = Db·[κx,κy,2κxy]
    # (sagging-positive convention — see tri_elements_mitc3._entry).
    m_exact = Db @ np.array([kx, ky, 2 * kxy])
    for (i, j, k) in tris:
        coords = [nodes[i], nodes[j], nodes[k]]
        dofs = [3 * n + c for n in (i, j, k) for c in range(3)]
        e = _entry(coords, E, nu, t, u[dofs])
        assert np.allclose([e['mx'], e['my'], e['mxy']], m_exact, atol=1e-8)


# ── Helpers for the plate-level checks ──────────────────────────────────────

def _square(n, t, support='hard', a=5.0, E=30e6, nu=0.2, pz=-10.0,
            jitter=0.0):
    s = Structure2D(domain='plate')
    s.add_material('M', elastic_modulus=E, unit_weight=0.0, poisson=nu)
    s.add_tri_section('S', 'M', thickness=t, formulation='MITC3')
    rng = np.random.default_rng(0)
    ids = {}
    for j in range(n + 1):
        for i in range(n + 1):
            x, y = a * i / n, a * j / n
            if jitter and 0 < i < n and 0 < j < n:
                x += jitter * a / n * (rng.random() - 0.5)
                y += jitter * a / n * (rng.random() - 0.5)
            nid = f'N{i}_{j}'
            ids[(i, j)] = nid
            s.add_node(nid, x, y)
    e = 0
    for j in range(n):
        for i in range(n):
            aa, bb = ids[(i, j)], ids[(i + 1, j)]
            cc, dd = ids[(i + 1, j + 1)], ids[(i, j + 1)]
            s.add_tri_element(f'T{e}', aa, bb, cc, 'S'); e += 1
            s.add_tri_element(f'T{e}', aa, cc, dd, 'S'); e += 1
    # Soft SS: w only. Hard SS: also the edge-tangential rotation (θx on the
    # vertical edges, θy on the horizontal ones) — the support the Navier-Mindlin
    # series assumes.
    s.add_support('W', w=True)
    s.add_support('WX', w=True, tx=True)
    s.add_support('WY', w=True, ty=True)
    s.add_support('WXY', w=True, tx=True, ty=True)
    for (i, j), nid in ids.items():
        onv, onh = i in (0, n), j in (0, n)
        if not (onv or onh):
            continue
        if support == 'soft':
            s.assign_support(nid, 'W')
        elif onv and onh:
            s.assign_support(nid, 'WXY')
        elif onv:
            s.assign_support(nid, 'WX')
        else:
            s.assign_support(nid, 'WY')
    s.add_load_case('G')
    for tid in list(s.tri_elements_by_id):
        s.add_area_load(tid, 'G', pz=pz)
    return s, ids


def _centre(s, ids, n):
    r = s.calculate()
    return abs(r['analysis_cases']['G']['displacements'][ids[(n // 2, n // 2)]][0])


def _mindlin_uniform(q, a, E, nu, t, kappa=5.0 / 6.0):
    """Exact first-order-shear (Mindlin) centre deflection of a hard-SS square
    plate under a uniform load, by the Navier double series."""
    D = E * t ** 3 / (12.0 * (1.0 - nu ** 2))
    G = E / (2.0 * (1.0 + nu))
    w = 0.0
    for m in range(1, 80, 2):
        for n in range(1, 80, 2):
            lam = math.pi ** 2 * (m * m / a ** 2 + n * n / a ** 2)
            term = (16.0 * q / (math.pi ** 2 * m * n)) / (D * lam ** 2)
            term *= math.sin(m * math.pi / 2) * math.sin(n * math.pi / 2)
            term *= (1.0 + lam * D / (kappa * G * t))
            w += term
    return w


# ── 2. Thick plate vs the exact Mindlin solution (hard support) ──────────────

def test_thick_plate_converges_to_exact_mindlin():
    a = 5.0
    for t in (0.5, 1.0):                     # span/thickness 10 and 5
        wM = _mindlin_uniform(10.0, a, 30e6, 0.2, t)
        s, ids = _square(40, t, support='hard', a=a)
        ratio = _centre(s, ids, 40) / wM
        assert 0.99 < ratio < 1.01, (t, ratio)


# ── 3. The boundary layer: soft support deflects more than hard ──────────────

def test_soft_support_deflects_more_than_hard():
    """A hard simple support also holds the tangential rotation; a soft one does
    not, so the plate deflects more — the shear boundary layer, and the reason a
    soft-support FE result overshoots a hard-support textbook series."""
    s_hard, ids = _square(24, 1.0, support='hard')
    s_soft, _ = _square(24, 1.0, support='soft')
    assert _centre(s_soft, ids, 24) > _centre(s_hard, ids, 24)


# ── 4. Distorted mesh stays accurate ────────────────────────────────────────

def test_distorted_mesh_still_converges():
    """Interior nodes randomly jittered: MITC3 must stay close to the thin
    Navier value for a thin plate (it is not sensitive to mesh distortion)."""
    t, a = 0.05, 5.0
    # Thin Navier reference.
    D = 30e6 * t ** 3 / (12.0 * (1.0 - 0.2 ** 2))
    wn = 0.0
    for m in range(1, 60, 2):
        for n in range(1, 60, 2):
            wn += (math.sin(m * math.pi / 2) * math.sin(n * math.pi / 2)) \
                / (m * n * (m * m + n * n) ** 2)
    wn *= 16.0 * 10.0 / (math.pi ** 6 * D) * a ** 4
    s, ids = _square(24, t, support='soft', a=a, jitter=0.4)
    ratio = _centre(s, ids, 24) / wn
    assert 0.9 < ratio < 1.05, ratio
