"""Allman triangle — CST plane element with drilling DOFs (vertex rotations).

Three-node plane-stress/plane-strain membrane element, like the CST in
``tri_elements.py``, but with 3 DOF per node (ux, uy, tz) instead of 2. The
extra "drilling" rotation at each vertex:

  * lets the element transfer moment through a node shared with a
    :class:`~xdfem2d.models.BarElement` (a plain CST has zero stiffness in
    tz, so no moment crosses a CST-only node — see
    dev/allman_triangle_plan.md §0);
  * improves in-plane bending accuracy on coarse meshes, since (unlike CST)
    the strain field is linear over the element rather than constant.

IMPORTANT — the tz stiffness here is a numerical "drilling" device, not a
real plate-bending stiffness. This is still a plane-stress/strain membrane
element, not a plate or shell element. See dev/allman_triangle_plan.md §4.

Formulation
-----------
Displacements are the standard linear (CST) field plus a quadratic
"bubble" along each edge, driven by the difference of the two vertex
drilling rotations at its ends (Allman 1984; the L/8 midside-tangential-
displacement form, e.g. Cook, *Concepts and Applications of Finite Element
Analysis*). In area coordinates (L1, L2, L3):

    u = sum_i Li*ui - 0.5 * sum_(edges i,j) Li*Lj*(ti - tj)*(yj - yi)
    v = sum_i Li*vi + 0.5 * sum_(edges i,j) Li*Lj*(ti - tj)*(xj - xi)

(edges taken as (1,2), (2,3), (3,1); ti is the drilling rotation at vertex
i). Note the bubble amplitude pairs u with -(yj - yi) and v with (xj - xi):
a rotation displaces the edge midpoint *perpendicularly* to the edge, so the
amplitude follows the 90°-rotated edge vector. Pairing u with (xj - xi) and
v with (yj - yi) instead — displacement along the edge — still passes the
rigid-body and constant-strain patch tests, but makes the element
non-covariant under reflection, so a mirror-symmetric mesh under a symmetric
load returns asymmetric displacements.

Because the bubble term vanishes identically whenever t1 = t2 = t3, a
uniform rigid-body rotation (ui, vi consistent with a rotation omega about
some centre, ti = omega for all i) produces exactly zero strain, same as
CST — required for element consistency and for exact rigid-body modes.

Since the strain field is linear (not constant), the stiffness integral is
evaluated with the standard 3-point Gauss rule for triangles (exact for
polynomials up to degree 2 — sufficient here).

Cook stabilization
-------------------
The raw formulation above has a well-known spurious zero-energy mode: with
*zero* corner translations and *equal, nonzero* drilling rotations at all
three vertices (t1 = t2 = t3 != 0), every (ti - tj) term is zero, so the
bubble displacement field — and therefore the strain and the stiffness
matrix's action on that mode — is identically zero. This is a genuine rank
deficiency of the raw element, not a numerical-precision artefact (see
dev/allman_triangle_plan.md §3.1), and left unfixed it can produce a
singular or near-singular global stiffness matrix.

The fix (Cook, 1986) adds a small stiffness term that penalises the
*deviation* of each drilling DOF from the mean rotation implied by the
element's own corner-translation field:

    omega_bar = 0.5 * (dv_lin/dx - du_lin/dy)   (constant over the element)
    r_i = t_i - omega_bar                        for i = 1, 2, 3
    E_stab = alpha * G * t * A * sum_i r_i**2

This is zero whenever t_i == omega_bar for all i — which holds exactly both
for the spurious mode's opposite (a genuine rigid rotation, where
omega_bar the linear field reproduces equals the imposed omega) *and* for
any constant-strain (patch-test) state (omega_bar there equals the
element's own uniform local rotation) — so the patch test and the exact
rigid-body modes are unaffected. It is *not* zero for the spurious mode
itself (translations zero => omega_bar = 0, but t_i != 0), which is exactly
what removes the singularity. ``alpha`` is a small dimensionless factor
(default 1e-3, a conservative value within the commonly cited 1e-3-to-1
range) so the added stiffness does not measurably contaminate real drilling
behaviour.
"""
from __future__ import annotations

import numpy as np

from .tri_elements import plane_D, principal_stresses

# See "Cook stabilization" in the module docstring.
_COOK_STABILIZATION_ALPHA = 1.0e-3

# 3-point Gauss rule for triangles (area coordinates), exact for polynomials
# complete to degree 2 in (x, y) — the Allman B matrix is linear in (x, y),
# so B^T D B is (at most) quadratic, and this rule integrates it exactly.
_GAUSS_L = np.array([
    [2 / 3, 1 / 6, 1 / 6],
    [1 / 6, 2 / 3, 1 / 6],
    [1 / 6, 1 / 6, 2 / 3],
])
_GAUSS_W = np.array([1 / 3, 1 / 3, 1 / 3])  # each point's weight (sums to 1)


def _edge_geometry(coords):
    """Triangle geometry pieces shared by B, K, and the stabilization term.

    Mirrors the b_i/c_i/a2 convention in ``tri_elements.cst_B_area`` exactly,
    so an Allman element and a CST element built from the same *coords*
    agree on area, sign, and node ordering."""
    (x1, y1), (x2, y2), (x3, y3) = coords
    a2 = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)   # signed 2*area
    area = abs(a2) / 2.0
    b1, b2, b3 = y2 - y3, y3 - y1, y1 - y2
    c1, c2, c3 = x3 - x2, x1 - x3, x2 - x1
    return dict(
        a2=a2, area=area, b=(b1, b2, b3), c=(c1, c2, c3),
        dx=(x2 - x1, x3 - x2, x1 - x3),   # dx12, dx23, dx31
        dy=(y2 - y1, y3 - y2, y1 - y3),   # dy12, dy23, dy31
    )


def allman_B(coords, L1: float, L2: float, L3: float) -> np.ndarray:
    """3x9 strain-displacement matrix at area coordinates (L1, L2, L3).

    Column order: [u1, v1, tz1, u2, v2, tz2, u3, v3, tz3]. Unlike the CST's
    B (constant over the element), this B varies with position — see the
    module docstring.
    """
    g = _edge_geometry(coords)
    a2 = g['a2']
    b1, b2, b3 = g['b']
    c1, c2, c3 = g['c']
    dx12, dx23, dx31 = g['dx']
    dy12, dy23, dy31 = g['dy']
    L = (L1, L2, L3)
    b = (b1, b2, b3)
    c = (c1, c2, c3)

    # A drilling rotation displaces the edge midpoint PERPENDICULARLY to the
    # edge, so the bubble amplitude pairs u with the 90°-rotated edge vector
    # (-dy, +dx), not with the edge vector (dx, dy) itself. Pairing u with dx
    # and v with dy makes the drilling term produce a displacement *along* the
    # edge, which leaves the element non-covariant under reflection: mirroring
    # a triangle no longer mirrors its stiffness (the coupling block keeps the
    # wrong sign), so a symmetric mesh under symmetric load gives asymmetric
    # results. Both rigid-body modes and the constant-strain patch test are
    # insensitive to this choice, which is why it went unnoticed.
    eu12, eu23, eu31 = -dy12, -dy23, -dy31
    ev12, ev23, ev31 = dx12, dx23, dx31

    def dLiLj_dx(i, j):
        return (L[i] * b[j] + L[j] * b[i]) / a2

    def dLiLj_dy(i, j):
        return (L[i] * c[j] + L[j] * c[i]) / a2

    # 0-based (i, j, k) <-> nodes (1, 2, 3); edges (0,1)=12, (1,2)=23, (2,0)=31.
    dx01, dx12_, dx20 = dLiLj_dx(0, 1), dLiLj_dx(1, 2), dLiLj_dx(2, 0)
    dy01, dy12_, dy20 = dLiLj_dy(0, 1), dLiLj_dy(1, 2), dLiLj_dy(2, 0)

    # Bubble_u = sum_i theta_i * P_i_u(L);  P_i_u picks up +edge(i,*) and
    # -edge(*,i) contributions from the (ti - tj) differences. eu/ev are the
    # perpendicular edge factors defined above.
    dP1u_dx = 0.5 * (eu12 * dx01 - eu31 * dx20)
    dP2u_dx = 0.5 * (-eu12 * dx01 + eu23 * dx12_)
    dP3u_dx = 0.5 * (-eu23 * dx12_ + eu31 * dx20)

    dP1u_dy = 0.5 * (eu12 * dy01 - eu31 * dy20)
    dP2u_dy = 0.5 * (-eu12 * dy01 + eu23 * dy12_)
    dP3u_dy = 0.5 * (-eu23 * dy12_ + eu31 * dy20)

    dP1v_dx = 0.5 * (ev12 * dx01 - ev31 * dx20)
    dP2v_dx = 0.5 * (-ev12 * dx01 + ev23 * dx12_)
    dP3v_dx = 0.5 * (-ev23 * dx12_ + ev31 * dx20)

    dP1v_dy = 0.5 * (ev12 * dy01 - ev31 * dy20)
    dP2v_dy = 0.5 * (-ev12 * dy01 + ev23 * dy12_)
    dP3v_dy = 0.5 * (-ev23 * dy12_ + ev31 * dy20)

    B = np.zeros((3, 9))
    # eps_x = du/dx
    B[0, 0], B[0, 3], B[0, 6] = b1 / a2, b2 / a2, b3 / a2
    B[0, 2], B[0, 5], B[0, 8] = dP1u_dx, dP2u_dx, dP3u_dx
    # eps_y = dv/dy
    B[1, 1], B[1, 4], B[1, 7] = c1 / a2, c2 / a2, c3 / a2
    B[1, 2], B[1, 5], B[1, 8] = dP1v_dy, dP2v_dy, dP3v_dy
    # gamma_xy = du/dy + dv/dx
    B[2, 0], B[2, 3], B[2, 6] = c1 / a2, c2 / a2, c3 / a2
    B[2, 1], B[2, 4], B[2, 7] = b1 / a2, b2 / a2, b3 / a2
    B[2, 2] = dP1u_dy + dP1v_dx
    B[2, 5] = dP2u_dy + dP2v_dx
    B[2, 8] = dP3u_dy + dP3v_dx
    return B


def _stabilization_matrix(coords, E: float, nu: float, t: float,
                          alpha: float) -> np.ndarray:
    """9x9 Cook stabilization term — see the module docstring. Zero for any
    constant-strain state (incl. rigid rotation); nonzero only for the
    "equal drilling rotation, zero translation" spurious mode."""
    if not alpha:
        return np.zeros((9, 9))
    g = _edge_geometry(coords)
    a2, area = g['a2'], g['area']
    b1, b2, b3 = g['b']
    c1, c2, c3 = g['c']
    G = E / (2.0 * (1.0 + nu))

    # omega_bar = w . d, where d = [u1,v1,t1,u2,v2,t2,u3,v3,t3]
    w = np.zeros(9)
    w[0], w[1] = -c1 / a2, b1 / a2
    w[3], w[4] = -c2 / a2, b2 / a2
    w[6], w[7] = -c3 / a2, b3 / a2
    w *= 0.5

    theta_idx = (2, 5, 8)
    k_stab = np.zeros((9, 9))
    for i in theta_idx:
        e_i = np.zeros(9)
        e_i[i] = 1.0
        r = e_i - w                      # r_i = t_i - omega_bar, as a linear form
        k_stab += np.outer(r, r)
    return (alpha * G * t * area) * k_stab


def allman_stiffness(coords, E: float, nu: float, t: float,
                     plane_strain: bool = False,
                     stabilization_alpha: float = _COOK_STABILIZATION_ALPHA):
    """Return (k, B0, D, area): 9x9 element stiffness (drilling DOF at every
    node) plus the pieces needed for stress recovery.

    *B0* is B evaluated at the centroid — used to report a single,
    representative (constant) stress per element, matching the convention
    the rest of the app expects from a triangle (see
    ``tri_elements.tri_stresses``); the Allman element's true strain field
    is linear, not constant, but the app-wide results shape stays
    unchanged (dev/allman_triangle_plan.md §3/§6).
    """
    g = _edge_geometry(coords)
    area = g['area']
    D = plane_D(E, nu, plane_strain)

    k = np.zeros((9, 9))
    for (L1, L2, L3), w in zip(_GAUSS_L, _GAUSS_W):
        B = allman_B(coords, L1, L2, L3)
        k += (w * area) * (B.T @ D @ B)
    k *= t

    k += _stabilization_matrix(coords, E, nu, t, stabilization_alpha)

    B0 = allman_B(coords, 1 / 3, 1 / 3, 1 / 3)
    return k, B0, D, area


def allman_stresses(struc, u_vec, thermal=None) -> dict:
    """Constant (centroid) stress per Allman triangle from a solved
    displacement vector. Mirrors ``tri_elements.tri_stresses``.

    Contract with assembly (see ``assembly._tri_element_matrix``): every
    ``struc._tri_cache[tid]`` entry carries a ``'formulation'`` marker
    (``'CST'`` or ``'Allman'``) alongside the usual ``dofs``/``B``/``D``, so
    this function (and ``tri_elements.tri_stresses``) can each pick out only
    the elements they know how to handle from a cache that holds both kinds
    side by side.

    *thermal* ({tri_id: ε₀}) is the free thermal strain removed before the
    constitutive step, so a freely expanding triangle reports zero stress.
    """
    thermal = thermal or {}
    cache = getattr(struc, "_tri_cache", {}) or {}
    out = {}
    for tid, c in cache.items():
        if c.get('formulation') != 'Allman':
            continue
        ue = np.asarray(u_vec)[c['dofs']]
        strain = c['B'] @ ue
        e0 = thermal.get(tid)
        if e0 is not None:
            strain = strain - e0
        sig = c['D'] @ strain
        sx, sy, txy = float(sig[0]), float(sig[1]), float(sig[2])
        s1, s2, th, vm = principal_stresses(sx, sy, txy)
        out[tid] = {'sx': sx, 'sy': sy, 'txy': txy,
                    's1': s1, 's2': s2, 'theta': th, 'vm': vm,
                    'formulation': 'Allman'}
    return out


def allman_stresses_from_disp(struc, disp: dict, thermal=None) -> dict:
    """Constant (centroid) stress per Allman triangle from a displacement
    dict ``{node_id: [ux, uy, tz]}``. Mirrors
    ``tri_elements.tri_stresses_from_disp`` — but, unlike the CST version,
    this *does* use each node's tz, since it is a real DOF of this element.
    Recomputes B/D per element from geometry, independent of the assembly
    cache (needed for analysis-case/combination displacement sets, which
    are not necessarily backed by a live assembly).

    *thermal* ({tri_id: ε₀}) is the free thermal strain removed before the
    constitutive step (same role as in the CST recovery)."""
    thermal = thermal or {}
    out = {}
    for tri in getattr(struc, "tri_elements", []):
        sec = struc.tri_sections.get(tri.section_name)
        if sec is None or getattr(sec, 'formulation', 'CST') != 'Allman':
            continue
        try:
            ni = struc.nodes[tri.node_i]; nj = struc.nodes[tri.node_j]
            nk = struc.nodes[tri.node_k]
        except KeyError:
            continue
        mat = struc.materials.get(sec.material_name)
        if mat is None:
            continue
        coords = [(ni.x, ni.y), (nj.x, nj.y), (nk.x, nk.y)]
        B0 = allman_B(coords, 1 / 3, 1 / 3, 1 / 3)
        D = plane_D(mat.elastic_modulus, getattr(mat, "poisson", 0.2),
                    sec.plane_strain)
        di = disp.get(tri.node_i, (0.0, 0.0, 0.0))
        dj = disp.get(tri.node_j, (0.0, 0.0, 0.0))
        dk = disp.get(tri.node_k, (0.0, 0.0, 0.0))
        ue = np.array([
            di[0], di[1], di[2] if len(di) > 2 else 0.0,
            dj[0], dj[1], dj[2] if len(dj) > 2 else 0.0,
            dk[0], dk[1], dk[2] if len(dk) > 2 else 0.0,
        ])
        strain = B0 @ ue
        e0 = thermal.get(tri.id)
        if e0 is not None:
            strain = strain - e0
        sig = D @ strain
        sx, sy, txy = float(sig[0]), float(sig[1]), float(sig[2])
        s1, s2, th, vm = principal_stresses(sx, sy, txy)
        out[tri.id] = {'sx': sx, 'sy': sy, 'txy': txy,
                       's1': s1, 's2': s2, 'theta': th, 'vm': vm,
                       'formulation': 'Allman'}
    return out
