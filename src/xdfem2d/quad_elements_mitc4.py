"""MITC4 plate-bending quadrilateral — the shear-deformable (Mindlin-Reissner)
counterpart of DKT4, and the quad analogue of ``tri_elements_mitc3``'s MITC3.

The MITC4 element of Dvorkin & Bathe (1984) / Bathe & Dvorkin (1985): a 4-node,
12-DOF plate element with nodal DOFs (w, θx, θy) — the same convention DKT,
MITC3 and the grillage bar use (θx = ∂w/∂y, θy = −∂w/∂x in the Kirchhoff
limit). Shear locking is avoided by the MITC (Mixed Interpolation of Tensorial
Components) scheme: the *covariant* transverse-shear strains e_ξ, e_η are
sampled at the 4 edge mid-points (tying points A, B, C, D) and re-interpolated,
instead of being taken directly from the bilinear w/rotation field — the exact
same idea ``tri_elements_mitc3`` uses, generalised from 3 tying points on a
triangle to 4 on a quad.

Bending part: standard bilinear isoparametric B_b (no locking there — only the
shear needs the mixed interpolation), full 2x2 Gauss.

Constitutive: bending D_b = (t³/12)·plane_D (``plate_common.bending_D``); shear
D_s = (5/6)·G·t·I (``plate_common.shear_D``) — shared, unchanged, with the
plate-bending triangles per dev/IMPLEMENT_QUAD.md §2.

Moment sign convention: sagging positive under a downward load, matching DKT/
MITC3/the grillage bar.
"""
from __future__ import annotations

import numpy as np

from .plate_common import bending_D, plate_moment_result, shear_D
from .quad_elements import q4_shape_functions, _jacobian

# DOF layout per node: [w, θx, θy]; u12 = [w1,θx1,θy1, w2,θx2,θy2, ...].

# Tying points (natural coords), Dvorkin & Bathe (1984): A = mid edge 1-2,
# B = mid edge 2-3, C = mid edge 3-4, D = mid edge 4-1.
_TIE_A = (0.0, -1.0)
_TIE_B = (1.0, 0.0)
_TIE_C = (0.0, 1.0)
_TIE_D = (-1.0, 0.0)

_GAUSS2 = (-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0))
_GAUSS2_POINTS = [(xi, eta) for xi in _GAUSS2 for eta in _GAUSS2]


def _bending_B(coords, xi: float, eta: float):
    """3x12 curvature-displacement matrix at (ξ,η): κ = [−θy,x, θx,y,
    θx,x − θy,y], the same orientation ``tri_elements_mitc3._bending_B``
    reports (already sagging-positive, no extra sign flip needed later)."""
    _, dN_dxi = q4_shape_functions(xi, eta)
    _, _, detJ, dN_dxy = _jacobian(coords, dN_dxi)
    Bb = np.zeros((3, 12))
    for a in range(4):
        dNdx, dNdy = dN_dxy[a]
        Bb[0, 3 * a + 2] = -dNdx          # −θy,x
        Bb[1, 3 * a + 1] = dNdy           #  θx,y
        Bb[2, 3 * a + 1] = dNdx           #  θx,x
        Bb[2, 3 * a + 2] = -dNdy          # −θy,y
    return Bb, detJ


def _tying_rows(coords, xi0: float, eta0: float):
    """(g_xi, g_eta) — each a (12,) row such that the natural covariant shear
    strain at (xi0, eta0) is e_ξ = g_xi . U, e_η = g_eta . U, U the 12 nodal
    DOFs. Built directly from the chain-rule identity ∂w/∂ξ = w,x·x,ξ + w,y·y,ξ
    (Bathe & Dvorkin 1985):

        e_ξ = ∂w/∂ξ + θy·x,ξ − θx·y,ξ
        e_η = ∂w/∂η + θy·x,η − θx·y,η

    with γx = w,x + θy, γy = w,y − θx (the same shear-strain convention
    ``tri_elements_mitc3`` uses) and x,ξ etc. the local Jacobian entries at
    (xi0, eta0)."""
    N, dN_dxi = q4_shape_functions(xi0, eta0)
    coords_arr = np.asarray(coords, dtype=float)
    J = dN_dxi.T @ coords_arr            # row0=[x,ξ,y,ξ], row1=[x,η,y,η]
    x_xi, y_xi = J[0]
    x_eta, y_eta = J[1]
    g_xi = np.zeros(12)
    g_eta = np.zeros(12)
    for a in range(4):
        g_xi[3 * a] = dN_dxi[a, 0]
        g_xi[3 * a + 1] = -N[a] * y_xi
        g_xi[3 * a + 2] = N[a] * x_xi
        g_eta[3 * a] = dN_dxi[a, 1]
        g_eta[3 * a + 1] = -N[a] * y_eta
        g_eta[3 * a + 2] = N[a] * x_eta
    return g_xi, g_eta


def _shear_B_builder(coords):
    """Return Bs(ξ,η) -> 2x12, the assumed Cartesian shear strain
    [γx; γy] = J(ξ,η)⁻¹ · [ê_ξ; ê_η] with the MITC4 tying:

        ê_ξ(ξ,η) = ½(1−η)·e_ξ(A) + ½(1+η)·e_ξ(C)     (constant along ξ)
        ê_η(ξ,η) = ½(1−ξ)·e_η(D) + ½(1+ξ)·e_η(B)     (constant along η)
    """
    g_xi_A, _ = _tying_rows(coords, *_TIE_A)
    g_xi_C, _ = _tying_rows(coords, *_TIE_C)
    _, g_eta_B = _tying_rows(coords, *_TIE_B)
    _, g_eta_D = _tying_rows(coords, *_TIE_D)

    def Bs(xi, eta):
        e_xi_row = 0.5 * (1.0 - eta) * g_xi_A + 0.5 * (1.0 + eta) * g_xi_C
        e_eta_row = 0.5 * (1.0 - xi) * g_eta_D + 0.5 * (1.0 + xi) * g_eta_B
        _, dN_dxi = q4_shape_functions(xi, eta)
        _, Jinv, _detJ, _ = _jacobian(coords, dN_dxi)
        return Jinv @ np.vstack([e_xi_row, e_eta_row])
    return Bs


def mitc4_matrices(coords, E: float, nu: float, t: float):
    """Return (k, Bb_c, Bs_c, Db, Ds, area) for a MITC4 quad.

    k is the 12x12 stiffness = bending (∫Bbᵀ Db Bb) + shear (∫Bsᵀ Ds Bs), both
    by 2x2 Gauss. Bb_c/Bs_c are the centroid B's, used for the reported
    moment/shear."""
    Db = bending_D(E, nu, t)
    Ds = shear_D(E, nu, t)
    Bs = _shear_B_builder(coords)

    k = np.zeros((12, 12))
    area = 0.0
    for xi, eta in _GAUSS2_POINTS:
        Bb, detJ = _bending_B(coords, xi, eta)
        Bsp = Bs(xi, eta)
        k += detJ * (Bb.T @ Db @ Bb)
        k += detJ * (Bsp.T @ Ds @ Bsp)
        area += detJ

    Bb_c, _ = _bending_B(coords, 0.0, 0.0)
    Bs_c = Bs(0.0, 0.0)
    return k, Bb_c, Bs_c, Db, Ds, area


def mitc4_stiffness(coords, E: float, nu: float, t: float):
    """(k, Bb, Db, area) — the signature the assembly dispatch will expect
    (mirrors ``mitc3_stiffness``); the shear part is recomputed at recovery."""
    k, Bb, _Bs_c, Db, _Ds, area = mitc4_matrices(coords, E, nu, t)
    return k, Bb, Db, area


# ---------------------------------------------------------------------------
# Thermal (through-thickness gradient) load — bending only (dev/
# IMPLEMENT_QUAD.md Phase 6, added once a quad thermal load became a real
# feature — see ``models.QuadTemperatureLoad``). Mirrors
# ``tri_elements_mitc3.mitc3_thermal_load``: no internal DOFs to condense
# here (unlike DKT4's fan construction), so a plain 2x2 Gauss sum of
# Bbᵀ·Db·κ₀ — the bending part of the stiffness integrand — is exact and
# sufficient. A gradient induces no transverse shear, so there is no shear
# term, same reasoning as the triangle.
# ---------------------------------------------------------------------------

def mitc4_thermal_load(coords, E: float, nu: float, t: float,
                       kappa0) -> np.ndarray:
    """Equivalent nodal load (12,) for a free thermal curvature ``kappa0``."""
    Db = bending_D(E, nu, t)
    kappa0 = np.asarray(kappa0, dtype=float)
    f = np.zeros(12)
    for xi, eta in _GAUSS2_POINTS:
        Bb, detJ = _bending_B(coords, xi, eta)
        f += detJ * (Bb.T @ Db @ kappa0)
    return f


# ---------------------------------------------------------------------------
# Moment / shear recovery — local coords + local u12 (Phase 2). The
# struc-based ``mitc4_moments(struc, u_vec)`` wrapper is below (Phase 5).
# ---------------------------------------------------------------------------

def mitc4_moment_entry(coords, E: float, nu: float, t: float, u12,
                        kappa0=None) -> dict:
    """One MITC4 quad's reported result from its 12 local DOFs: centroid
    moments (m = D_b·(κ − κ₀)) and the shear field q = D_s·γ at the centroid,
    assembled through :func:`plate_common.plate_moment_result`. The shear sign
    flip mirrors ``tri_elements_mitc3._entry`` (empirically the opposite sign
    of the DKT/DKT4 equilibrium shear for the same sagging-positive moments)."""
    _, Bb, Bs_c, Db, Ds, _ = mitc4_matrices(coords, E, nu, t)
    k0 = np.zeros(3) if kappa0 is None else np.asarray(kappa0, dtype=float)
    u12 = np.asarray(u12, dtype=float)
    m0 = Db @ (Bb @ u12 - k0)
    mx, my, mxy = float(m0[0]), float(m0[1]), float(m0[2])
    q = -(Ds @ (Bs_c @ u12))
    return plate_moment_result(mx, my, mxy, float(q[0]), float(q[1]), 'MITC4')


# ---------------------------------------------------------------------------
# struc-based recovery (dev/IMPLEMENT_QUAD.md Phase 5). Mirrors
# tri_elements_mitc3.mitc3_moments exactly: recomputes fresh from
# struc.node_dof_index and the model's nodes/sections/materials each call
# (recovery is not hot).
# ---------------------------------------------------------------------------

def mitc4_moments(struc, u_vec, thermal=None) -> dict:
    """Per-quad results from a solved global displacement vector, for MITC4
    sections, keyed by quad id."""
    thermal = thermal or {}
    out = {}
    u = np.asarray(u_vec)
    idx = getattr(struc, 'node_dof_index', {})
    for quad in getattr(struc, 'quad_elements', []):
        sec = struc.quad_sections.get(quad.section_name)
        if sec is None or getattr(sec, 'formulation', 'MITC4') != 'MITC4':
            continue
        mat = struc.materials.get(sec.material_name)
        if mat is None:
            continue
        try:
            ni, nj, nk, nl = (struc.nodes[quad.node_i], struc.nodes[quad.node_j],
                              struc.nodes[quad.node_k], struc.nodes[quad.node_l])
        except KeyError:
            continue
        coords = [(ni.x, ni.y), (nj.x, nj.y), (nk.x, nk.y), (nl.x, nl.y)]
        dofs = []
        for nid in (quad.node_i, quad.node_j, quad.node_k, quad.node_l):
            b = idx[nid]
            dofs += [b, b + 1, b + 2]
        u12 = u[dofs]
        out[quad.id] = mitc4_moment_entry(coords, mat.elastic_modulus,
                                          getattr(mat, 'poisson', 0.2),
                                          sec.thickness, u12, thermal.get(quad.id))
    return out


def mitc4_moments_from_disp(struc, disp: dict, thermal=None) -> dict:
    """Same as :func:`mitc4_moments` but from a displacement dict
    ``{node_id: [w, tx, ty]}`` — used for analysis cases/combinations."""
    thermal = thermal or {}
    out = {}
    z3 = (0.0, 0.0, 0.0)
    for quad in getattr(struc, 'quad_elements', []):
        sec = struc.quad_sections.get(quad.section_name)
        if sec is None or getattr(sec, 'formulation', 'MITC4') != 'MITC4':
            continue
        mat = struc.materials.get(sec.material_name)
        if mat is None:
            continue
        try:
            ni, nj, nk, nl = (struc.nodes[quad.node_i], struc.nodes[quad.node_j],
                              struc.nodes[quad.node_k], struc.nodes[quad.node_l])
        except KeyError:
            continue
        coords = [(ni.x, ni.y), (nj.x, nj.y), (nk.x, nk.y), (nl.x, nl.y)]
        u12 = np.array(
            list(disp.get(quad.node_i, z3))
            + list(disp.get(quad.node_j, z3))
            + list(disp.get(quad.node_k, z3))
            + list(disp.get(quad.node_l, z3)), dtype=float)
        out[quad.id] = mitc4_moment_entry(coords, mat.elastic_modulus,
                                          getattr(mat, 'poisson', 0.2),
                                          sec.thickness, u12, thermal.get(quad.id))
    return out
