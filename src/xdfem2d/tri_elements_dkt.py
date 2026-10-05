"""DKT plate-bending triangle — the plate-domain counterpart of the CST.

The Discrete Kirchhoff Triangle of Batoz, Bathe & Ho (IJNME 1980): a 3-node,
9-DOF thin-plate element with nodal DOFs (w, θx, θy) — transverse deflection
and rotations about the global X and Y axes, right-hand rule, i.e. exactly
the plate domain's (w, tx, ty) and the same convention the grillage bar uses
(θx = ∂w/∂y, θy = −∂w/∂x for the Kirchhoff field). Kirchhoff's constraint is
imposed discretely along the sides, which is what makes a 3-node triangle
both thin-plate accurate and free of shear locking; it passes the constant-
curvature patch test.

Bending constitutive matrix: D_b = (t³/12)·plane_D(E, ν) — the plane-stress
D the membrane elements already use, scaled by the plate rigidity.

Moment sign convention: sagging positive under a downward (negative fz)
load, matching the grillage bar's M. Moments vary linearly inside a DKT;
the per-element value reported is the centroid one, and the (constant)
shear forces come from the exact linear moment field:
vx = ∂mx/∂x + ∂mxy/∂y, vy = ∂mxy/∂x + ∂my/∂y.

Wood–Armer (W-A) design moments (orthogonal reinforcement along X/Y):
  bottom  m*x = mx + |mxy|, m*y = my + |mxy|;
  top     m*x = mx − |mxy|, m*y = my − |mxy|.
When a value crosses zero the wrong-sign term is not kept: that direction is
set to zero and the other corrected with the mxy²/m term —
  bottom, if m*x < 0:  m*x = 0, m*y = my + |mxy²/mx|  (symmetrically for m*y);
  top,    if m*x > 0:  m*x = 0, m*y = my − |mxy²/mx|  (symmetrically for m*y).
"""
from __future__ import annotations

import numpy as np

# Shared plate machinery. ``plate_D`` keeps its historical name here (the DKT's
# bending constitutive) by re-exporting the common ``bending_D``; ``wood_armer``
# is re-exported too, since ``tri_elements_dkt.wood_armer`` is a public import
# path (used by tests and rc_design).
from .plate_common import bending_D as plate_D
from .plate_common import plate_moment_result, wood_armer  # noqa: F401


def _side_coefficients(coords):
    """Batoz's a..e coefficients for the three mid-sides k = 4, 5, 6,
    corresponding to edges ij = 23, 31, 12."""
    (x1, y1), (x2, y2), (x3, y3) = coords
    out = {}
    for k, (xi_, yi_, xj_, yj_) in ((4, (x2, y2, x3, y3)),
                                    (5, (x3, y3, x1, y1)),
                                    (6, (x1, y1, x2, y2))):
        xij = xi_ - xj_
        yij = yi_ - yj_
        l2 = xij * xij + yij * yij
        out[k] = dict(
            a=-xij / l2,
            b=0.75 * xij * yij / l2,
            c=(0.25 * xij * xij - 0.5 * yij * yij) / l2,
            d=-yij / l2,
            e=(0.25 * yij * yij - 0.5 * xij * xij) / l2,
        )
    return out


def _shape_and_derivs(xi: float, eta: float):
    """Quadratic shape functions N1..N6 on the unit triangle and their
    (ξ, η) derivatives. λ = 1 − ξ − η."""
    lam = 1.0 - xi - eta
    N = np.array([2.0 * lam * lam - lam,
                  2.0 * xi * xi - xi,
                  2.0 * eta * eta - eta,
                  4.0 * xi * eta,
                  4.0 * eta * lam,
                  4.0 * xi * lam])
    dNdxi = np.array([1.0 - 4.0 * lam,
                      4.0 * xi - 1.0,
                      0.0,
                      4.0 * eta,
                      -4.0 * eta,
                      4.0 * (lam - xi)])
    dNdeta = np.array([1.0 - 4.0 * lam,
                       0.0,
                       4.0 * eta - 1.0,
                       4.0 * xi,
                       4.0 * (lam - eta),
                       -4.0 * xi])
    return N, dNdxi, dNdeta


def _H_vectors(C, N):
    """Hx, Hy (9,) — Batoz eq. (21): β = H·U with U = [w1, tx1, ty1, …]."""
    c4, c5, c6 = C[4], C[5], C[6]
    Hx = np.array([
        1.5 * (c6['a'] * N[5] - c5['a'] * N[4]),
        c5['b'] * N[4] + c6['b'] * N[5],
        N[0] - c5['c'] * N[4] - c6['c'] * N[5],
        1.5 * (c4['a'] * N[3] - c6['a'] * N[5]),
        c6['b'] * N[5] + c4['b'] * N[3],
        N[1] - c6['c'] * N[5] - c4['c'] * N[3],
        1.5 * (c5['a'] * N[4] - c4['a'] * N[3]),
        c4['b'] * N[3] + c5['b'] * N[4],
        N[2] - c4['c'] * N[3] - c5['c'] * N[4],
    ])
    Hy = np.array([
        1.5 * (c6['d'] * N[5] - c5['d'] * N[4]),
        -N[0] + c5['e'] * N[4] + c6['e'] * N[5],
        -c5['b'] * N[4] - c6['b'] * N[5],
        1.5 * (c4['d'] * N[3] - c6['d'] * N[5]),
        -N[1] + c6['e'] * N[5] + c4['e'] * N[3],
        -c6['b'] * N[5] - c4['b'] * N[3],
        1.5 * (c5['d'] * N[4] - c4['d'] * N[3]),
        -N[2] + c4['e'] * N[3] + c5['e'] * N[4],
        -c4['b'] * N[3] - c5['b'] * N[4],
    ])
    return Hx, Hy


def dkt_B(coords, xi: float, eta: float) -> np.ndarray:
    """The 3×9 curvature-displacement matrix at (ξ, η): κ = B·U in Batoz's
    β-orientation (κ ≈ [+w,xx, +w,yy, +2w,xy]); the reported moments flip the
    sign once, in _moment_entry, to be sagging-positive."""
    (x1, y1), (x2, y2), (x3, y3) = coords
    x21, y21 = x2 - x1, y2 - y1
    x31, y31 = x3 - x1, y3 - y1
    two_A = x21 * y31 - x31 * y21
    C = _side_coefficients(coords)

    _, dxi, deta = _shape_and_derivs(xi, eta)
    Hx_xi, Hy_xi = _H_vectors(C, dxi)
    Hx_eta, Hy_eta = _H_vectors(C, deta)

    # Chain rule: ∂/∂x = (y31·∂ξ − y21·∂η)/2A, ∂/∂y = (−x31·∂ξ + x21·∂η)/2A.
    Hx_x = (y31 * Hx_xi - y21 * Hx_eta) / two_A
    Hx_y = (-x31 * Hx_xi + x21 * Hx_eta) / two_A
    Hy_x = (y31 * Hy_xi - y21 * Hy_eta) / two_A
    Hy_y = (-x31 * Hy_xi + x21 * Hy_eta) / two_A

    return np.vstack([Hx_x, Hy_y, Hx_y + Hy_x])


def dkt_stiffness(coords, E: float, nu: float, t: float):
    """Return (k, B0, D_b, area) for a DKT triangle.

    k is the 9×9 stiffness (exact 3-point Gauss integration — B is linear in
    ξ, η so the quadratic integrand is captured exactly); B0 is the centroid
    B used for the reported per-element moments; D_b the bending matrix."""
    (x1, y1), (x2, y2), (x3, y3) = coords
    two_A = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    area = abs(two_A) / 2.0
    Db = plate_D(E, nu, t)

    k = np.zeros((9, 9))
    for (xi, eta) in ((0.5, 0.0), (0.5, 0.5), (0.0, 0.5)):
        B = dkt_B(coords, xi, eta)
        k += (area / 3.0) * (B.T @ Db @ B)
    B0 = dkt_B(coords, 1.0 / 3.0, 1.0 / 3.0)
    return k, B0, Db, area


# ---------------------------------------------------------------------------
# Thermal (through-thickness gradient) loads
# ---------------------------------------------------------------------------

def dkt_thermal_curvature(alpha: float, dt_grad: float, t: float) -> np.ndarray:
    """Free thermal curvature κ₀ of a DKT under a through-thickness gradient
    ΔT = T_top − T_bottom, in the β-orientation ``dkt_B`` produces (κ = [w,xx,
    w,yy, 2w,xy]).

    Isotropic: equal in x and y, no twist, magnitude α·ΔT/t. Sign: a positive
    (top-hotter) gradient bows the free plate concave-down (κ = w,xx < 0), the
    2-D analogue of the grillage bar's gradient curvature. Returns [0,0,0] for
    a zero thickness."""
    k0 = -alpha * dt_grad / t if t > 0.0 else 0.0
    return np.array([k0, k0, 0.0])


def dkt_thermal_load(coords, E: float, nu: float, t: float,
                     kappa0: np.ndarray) -> np.ndarray:
    """Equivalent nodal load (9,) for a free thermal curvature κ₀:
    f_th = ∫ Bᵀ·D_b·κ₀ dA, by the same 3-point Gauss rule as the stiffness (B
    is linear in ξ, η so the integral is exact). Applied to the (w, tx, ty)
    DOFs; the matching subtraction is done in the moment recovery so a plate
    free to curve reports zero moment."""
    (x1, y1), (x2, y2), (x3, y3) = coords
    two_A = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    area = abs(two_A) / 2.0
    Db = plate_D(E, nu, t)
    f = np.zeros(9)
    for (xi, eta) in ((0.5, 0.0), (0.5, 0.5), (0.0, 0.5)):
        B = dkt_B(coords, xi, eta)
        f += (area / 3.0) * (B.T @ Db @ kappa0)
    return f


# ---------------------------------------------------------------------------
# Moment / shear recovery
# ---------------------------------------------------------------------------
# ``wood_armer`` lives in ``plate_common`` and is re-exported at the top of this
# module (the ``tri_elements_dkt.wood_armer`` import path is kept for tests and
# rc_design). ``plate_moment_result`` assembles the shared reported dict.


def _moment_entry(coords, Db, u9, kappa0=None) -> dict:
    """Everything reported for one DKT triangle from its 9 local DOFs:
    centroid moments, principal values, constant shear from the exact linear
    moment field, and the Wood–Armer design moments.

    The moments (and their derived shears) are DKT-specific — a thin element
    with the transverse shear recovered from the moment gradient — but the
    principal values, Wood–Armer moments and the shape of the result dict are
    shared with the shear-deformable elements through
    :func:`plate_common.plate_moment_result`."""
    # The −1 sets the reported sign convention: m = D_b·(w,xx, w,yy, 2w,xy),
    # i.e. positive where the plate is concave up — sagging positive at the
    # midspan of a downward-loaded slab, matching the grillage bar's M.
    # (B·U carries Batoz's β = −∇w orientation; the stiffness ∫BᵀD B is
    # indifferent to this overall sign, the reported moment is not.)
    # A thermal gradient contributes a free curvature κ₀ that is removed here
    # (m = D_b·(κ − κ₀)); without it a plate free to curve would report the
    # moment of a restrained one, just as the CST subtracts its free ε₀.
    k0 = np.zeros(3) if kappa0 is None else np.asarray(kappa0, dtype=float)
    kappa_c = dkt_B(coords, 1.0 / 3.0, 1.0 / 3.0) @ u9
    m0 = -(Db @ (kappa_c - k0))
    mx, my, mxy = float(m0[0]), float(m0[1]), float(m0[2])

    # Moments are linear: their exact gradient follows from the corner values.
    # The transverse shear is derived from that field (vx = ∂mx/∂x + ∂mxy/∂y,
    # vy = ∂mxy/∂x + ∂my/∂y) — a thin element has no shear DOF to read it from.
    (x1, y1), (x2, y2), (x3, y3) = coords
    two_A = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    mc = [-(Db @ (dkt_B(coords, xi, eta) @ u9))
          for (xi, eta) in ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))]
    b1, b2, b3 = y2 - y3, y3 - y1, y1 - y2
    c1, c2, c3 = x3 - x2, x1 - x3, x2 - x1

    def _grad(comp):
        v = [m[comp] for m in mc]
        ddx = (b1 * v[0] + b2 * v[1] + b3 * v[2]) / two_A
        ddy = (c1 * v[0] + c2 * v[1] + c3 * v[2]) / two_A
        return ddx, ddy

    dmx_dx, _ = _grad(0)
    _, dmy_dy = _grad(1)
    dmxy_dx, dmxy_dy = _grad(2)
    vx = dmx_dx + dmxy_dy
    vy = dmxy_dx + dmy_dy

    return plate_moment_result(mx, my, mxy, vx, vy, 'DKT')


def dkt_moments(struc, u_vec, thermal=None) -> dict:
    """Per-triangle moment results from a solved global displacement vector,
    using the geometry/D cached during assembly. Mirrors ``tri_stresses``
    (the CST recovery). *thermal* ({tri_id: κ₀}) is the free thermal curvature
    to subtract, so a plate free to curve under a gradient reports zero moment;
    absent/empty it is ignored (a mean ΔT does not bend a plate)."""
    cache = getattr(struc, '_tri_cache', {}) or {}
    thermal = thermal or {}
    out = {}
    u = np.asarray(u_vec)
    for tid, c in cache.items():
        if c.get('formulation') != 'DKT':
            continue
        out[tid] = _moment_entry(c['coords'], c['D'], u[c['dofs']],
                                 thermal.get(tid))
    return out


def dkt_moments_from_disp(struc, disp: dict, thermal=None) -> dict:
    """Same as :func:`dkt_moments` but from a displacement dict
    ``{node_id: [w, tx, ty]}`` — used for analysis cases and combinations.
    Recomputes geometry from the model so it does not depend on the assembly
    cache being aligned with this displacement set. *thermal* ({tri_id: κ₀}) is
    the free thermal curvature to subtract (see :func:`dkt_moments`)."""
    out = {}
    thermal = thermal or {}
    z3 = (0.0, 0.0, 0.0)
    for tri in getattr(struc, 'tri_elements', []):
        sec = struc.tri_sections.get(tri.section_name)
        if sec is None or getattr(sec, 'formulation', 'CST') != 'DKT':
            continue
        mat = struc.materials.get(sec.material_name)
        if mat is None:
            continue
        try:
            ni = struc.nodes[tri.node_i]
            nj = struc.nodes[tri.node_j]
            nk = struc.nodes[tri.node_k]
        except KeyError:
            continue
        coords = [(ni.x, ni.y), (nj.x, nj.y), (nk.x, nk.y)]
        Db = plate_D(mat.elastic_modulus, getattr(mat, 'poisson', 0.2),
                     sec.thickness)
        u9 = np.array(list(disp.get(tri.node_i, z3))
                      + list(disp.get(tri.node_j, z3))
                      + list(disp.get(tri.node_k, z3)), dtype=float)
        out[tri.id] = _moment_entry(coords, Db, u9, thermal.get(tri.id))
    return out
