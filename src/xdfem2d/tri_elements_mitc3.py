"""MITC3 plate-bending triangle — the shear-deformable (Mindlin-Reissner)
counterpart of the DKT.

The MITC3 element of Lee & Bathe (2004): a 3-node, 9-DOF plate element with the
same nodal DOFs (w, θx, θy) as the DKT, but including transverse shear
deformation, so it is valid for thick slabs and reports a genuine shear field
(not one derived from the moment gradient). Shear locking is avoided by the
MITC (Mixed Interpolation of Tensorial Components) scheme: the covariant
transverse-shear strains are sampled at the three edge mid-points (tying points)
and re-interpolated, instead of being taken directly from the linearly
interpolated w and rotations.

Kinematics (shared with the DKT convention, so the two agree in the thin limit
and with the grillage bar): θx = ∂w/∂y, θy = −∂w/∂x in the Kirchhoff limit. The
bending curvature (in the same orientation dkt_B produces, κ ≈ [w,xx, w,yy,
2w,xy]) is κ = [−θy,x, θx,y, θx,x − θy,y]; the transverse shear strains are
γxz = w,x + θy, γyz = w,y − θx.

Constitutive: bending D_b = (t³/12)·plane_D (``plate_common.bending_D``); shear
D_s = (5/6)·G·t·I (``plate_common.shear_D``).

Moment sign convention: sagging positive under a downward load, matching the DKT
and the grillage bar (m = −D_b·κ). The shear forces vx, vy are the shear field
D_s·γ at the centroid, sign-aligned with the DKT's equilibrium shear.

When to use which: for a **thick** slab (span/thickness ≲ 20) MITC3 captures the
transverse-shear deflection the thin DKT cannot, and is the right element. For a
**thin** slab the DKT is both more accurate per element and the natural default;
plain MITC3 converges there too but needs a finer mesh as t/L → 0 (the known
mild sensitivity of the 3-node element, which the bubble-enhanced MITC3+ was
later designed to remove). So DKT stays the plate default and MITC3 is the
opt-in thick-plate element.
"""
from __future__ import annotations

import numpy as np

from .plate_common import bending_D, plate_moment_result, shear_D

# DOF layout per node: [w, θx, θy]; u9 = [w1,θx1,θy1, w2,θx2,θy2, w3,θx3,θy3].

# Tying points (natural coords): A on edge 1-2, B on edge 1-3, C on edge 2-3.
_A = (0.5, 0.0)
_B = (0.0, 0.5)
_C = (0.5, 0.5)
# 3-point (mid-edge) Gauss rule, exact for the quadratic shear integrand.
_GAUSS = ((0.5, 0.0), (0.5, 0.5), (0.0, 0.5))


def _geometry(coords):
    (x1, y1), (x2, y2), (x3, y3) = coords
    x21, y21 = x2 - x1, y2 - y1
    x31, y31 = x3 - x1, y3 - y1
    detJ = x21 * y31 - x31 * y21           # = 2·(signed area)
    area = abs(detJ) / 2.0
    # Constant shape-function derivatives (linear triangle).
    Nx = np.array([(y21 - y31), y31, -y21]) / detJ
    Ny = np.array([(x31 - x21), -x31, x21]) / detJ
    return (x21, y21, x31, y31, detJ, area, Nx, Ny)


def _bending_B(Nx, Ny) -> np.ndarray:
    """3×9 curvature-displacement matrix (constant): κ = [−θy,x, θx,y,
    θx,x − θy,y], the same orientation the DKT reports."""
    Bb = np.zeros((3, 9))
    for i in range(3):
        Bb[0, 3 * i + 2] = -Nx[i]          # −θy,x
        Bb[1, 3 * i + 1] = Ny[i]           #  θx,y
        Bb[2, 3 * i + 1] = Nx[i]           #  θx,x
        Bb[2, 3 * i + 2] = -Ny[i]          # −θy,y
    return Bb


def _covariant_vectors(geom):
    """The two covariant transverse-shear strains e_r, e_s as 9-vectors of the
    nodal DOFs, each a function of (r, s). e_r = γ·t_r, e_s = γ·t_s with the
    edge tangents t_r = (x21,y21), t_s = (x31,y31); γ = [w,x+θy, w,y−θx]."""
    x21, y21, x31, y31, detJ, area, Nx, Ny = geom

    def _N(r, s):
        return np.array([1.0 - r - s, r, s])

    def er(r, s):
        Ni = _N(r, s)
        g = np.zeros(9)
        for i in range(3):
            g[3 * i] = x21 * Nx[i] + y21 * Ny[i]     # w  (constant)
            g[3 * i + 1] = -y21 * Ni[i]              # θx (linear)
            g[3 * i + 2] = x21 * Ni[i]               # θy (linear)
        return g

    def es(r, s):
        Ni = _N(r, s)
        g = np.zeros(9)
        for i in range(3):
            g[3 * i] = x31 * Nx[i] + y31 * Ny[i]
            g[3 * i + 1] = -y31 * Ni[i]
            g[3 * i + 2] = x31 * Ni[i]
        return g

    return er, es


def _shear_B_builder(geom):
    """Return a function Bs(r, s) → 2×9 giving the *assumed* Cartesian shear
    strain [γxz; γyz] = J⁻¹·[ê_r; ê_s] at a point, with the MITC3 tying:

        ê_r = e_r^A + s·c,  ê_s = e_s^B − r·c,
        c   = (e_r^C − e_r^A) − (e_s^C − e_s^B).
    """
    x21, y21, x31, y31, detJ, area, Nx, Ny = geom
    er, es = _covariant_vectors(geom)
    erA, erC = er(*_A), er(*_C)
    esB, esC = es(*_B), es(*_C)
    c = (erC - erA) - (esC - esB)
    Jinv = np.array([[y31, -y21], [-x31, x21]]) / detJ

    def Bs(r, s):
        e_r = erA + s * c
        e_s = esB - r * c
        return Jinv @ np.vstack([e_r, e_s])          # 2×9
    return Bs


def mitc3_matrices(coords, E: float, nu: float, t: float):
    """Return (k, Bb, Bs_c, Db, Ds, area) for a MITC3 triangle.

    k is the 9×9 stiffness = bending (∫Bbᵀ Db Bb, Bb constant) + shear
    (∫Bsᵀ Ds Bs, 3-point rule). Bb is the constant bending B; Bs_c the shear B at
    the centroid, used for the reported shear."""
    geom = _geometry(coords)
    area = geom[5]
    Nx, Ny = geom[6], geom[7]
    Db = bending_D(E, nu, t)
    Ds = shear_D(E, nu, t)

    Bb = _bending_B(Nx, Ny)
    Bs = _shear_B_builder(geom)

    k = area * (Bb.T @ Db @ Bb)
    for (r, s) in _GAUSS:
        B = Bs(r, s)
        k += (area / 3.0) * (B.T @ Ds @ B)

    Bs_c = Bs(1.0 / 3.0, 1.0 / 3.0)
    return k, Bb, Bs_c, Db, Ds, area


def mitc3_stiffness(coords, E: float, nu: float, t: float):
    """(k, Bb, Db, area) — the signature the assembly dispatch expects. The
    bending B and Db are cached; the shear part is recomputed at recovery."""
    k, Bb, Bs_c, Db, Ds, area = mitc3_matrices(coords, E, nu, t)
    return k, Bb, Db, area


# ---------------------------------------------------------------------------
# Thermal (through-thickness gradient) loads — bending only
# ---------------------------------------------------------------------------

def mitc3_thermal_load(coords, E: float, nu: float, t: float,
                       kappa0: np.ndarray) -> np.ndarray:
    """Equivalent nodal load (9,) for a free thermal curvature κ₀ (bending
    only): f_th = ∫ Bbᵀ·D_b·κ₀ dA = area·Bbᵀ·D_b·κ₀ (Bb constant). The matching
    subtraction is done in the moment recovery, so a plate free to curve reports
    zero moment. A gradient induces no transverse shear, so there is no shear
    term here."""
    geom = _geometry(coords)
    area = geom[5]
    Bb = _bending_B(geom[6], geom[7])
    Db = bending_D(E, nu, t)
    return area * (Bb.T @ Db @ np.asarray(kappa0, dtype=float))


# ---------------------------------------------------------------------------
# Moment / shear recovery
# ---------------------------------------------------------------------------

def _entry(coords, E, nu, t, u9, kappa0=None) -> dict:
    """One MITC3 triangle's reported result from its 9 local DOFs: centroid
    moments (m = D_b·(κ − κ₀)) and the shear field q = D_s·γ at the centroid,
    assembled through the shared :func:`plate_common.plate_moment_result`.

    Unlike the DKT's ``dkt_B`` (built in Batoz's β = −∇w orientation, so its
    curvature needs an extra sign flip to become the real, sagging-positive
    one — see ``tri_elements_dkt._moment_entry``), ``_bending_B`` above is
    built directly in the real curvature orientation (κ = [−θy,x, θx,y,
    θx,x − θy,y] with θx = ∂w/∂y, θy = −∂w/∂x already gives [+w,xx, +w,yy,
    +2w,xy]). So no extra sign flip belongs here: m = +D_b·(κ − κ₀), matching
    the DKT's final (already-flipped) sagging-positive convention."""
    _, Bb, Bs_c, Db, Ds, _ = mitc3_matrices(coords, E, nu, t)
    k0 = np.zeros(3) if kappa0 is None else np.asarray(kappa0, dtype=float)
    m0 = Db @ (Bb @ u9 - k0)
    mx, my, mxy = float(m0[0]), float(m0[1]), float(m0[2])
    # q = D_s·γ is the transverse shear force per unit width. γ = [w,x+θy,
    # w,y−θx] is built (like Bb) directly, not in Batoz's negated orientation,
    # but empirically it comes out with the *opposite* sign of the DKT's
    # equilibrium shear (vx = ∂mx/∂x + ∂mxy/∂y) for the same sagging-positive
    # moments — so, unlike the moment, the shear does need the extra flip.
    q = -(Ds @ (Bs_c @ u9))
    return plate_moment_result(mx, my, mxy, float(q[0]), float(q[1]), 'MITC3')


def mitc3_moments(struc, u_vec, thermal=None) -> dict:
    """Per-triangle results from a solved global displacement vector, for the
    MITC3 sections. Mirrors ``dkt_moments``; recomputes the element matrices
    from the geometry and material (recovery is not hot)."""
    thermal = thermal or {}
    out = {}
    u = np.asarray(u_vec)
    idx = getattr(struc, 'node_dof_index', {})
    for tri in getattr(struc, 'tri_elements', []):
        sec = struc.tri_sections.get(tri.section_name)
        if sec is None or getattr(sec, 'formulation', 'CST') != 'MITC3':
            continue
        mat = struc.materials.get(sec.material_name)
        try:
            ni, nj, nk = (struc.nodes[tri.node_i], struc.nodes[tri.node_j],
                          struc.nodes[tri.node_k])
        except KeyError:
            continue
        if mat is None:
            continue
        coords = [(ni.x, ni.y), (nj.x, nj.y), (nk.x, nk.y)]
        dofs = []
        for nid in (tri.node_i, tri.node_j, tri.node_k):
            b = idx[nid]
            dofs += [b, b + 1, b + 2]
        u9 = u[dofs]
        out[tri.id] = _entry(coords, mat.elastic_modulus,
                             getattr(mat, 'poisson', 0.2), sec.thickness, u9,
                             thermal.get(tri.id))
    return out


def mitc3_moments_from_disp(struc, disp: dict, thermal=None) -> dict:
    """Same as :func:`mitc3_moments` but from a displacement dict
    ``{node_id: [w, tx, ty]}`` — used for analysis cases and combinations."""
    out = {}
    thermal = thermal or {}
    z3 = (0.0, 0.0, 0.0)
    for tri in getattr(struc, 'tri_elements', []):
        sec = struc.tri_sections.get(tri.section_name)
        if sec is None or getattr(sec, 'formulation', 'CST') != 'MITC3':
            continue
        mat = struc.materials.get(sec.material_name)
        if mat is None:
            continue
        try:
            ni, nj, nk = (struc.nodes[tri.node_i], struc.nodes[tri.node_j],
                          struc.nodes[tri.node_k])
        except KeyError:
            continue
        coords = [(ni.x, ni.y), (nj.x, nj.y), (nk.x, nk.y)]
        u9 = np.array(list(disp.get(tri.node_i, z3))
                      + list(disp.get(tri.node_j, z3))
                      + list(disp.get(tri.node_k, z3)), dtype=float)
        out[tri.id] = _entry(coords, mat.elastic_modulus,
                             getattr(mat, 'poisson', 0.2), sec.thickness, u9,
                             thermal.get(tri.id))
    return out
