"""4-node quadrilateral membrane elements (Q4, QM6) — element matrices and
stresses.

Plane stress (default) or plane strain, isoparametric bilinear quadrilateral.
Two translational DOFs per node (ux, uy); nodal DOF order is
``[u1, v1, u2, v2, u3, v3, u4, v4]`` with nodes wound counter-clockwise in
natural coordinates ``(ξ,η) = (-1,-1), (1,-1), (1,1), (-1,1)``.

Two formulations, mirroring dev/IMPLEMENT_QUAD.md §1:

* :func:`q4_stiffness` — plain bilinear Q4. Kept for patch-test reference and
  comparison; known to lock under bending (parasitic shear), so it is not the
  recommended default.
* :func:`qm6_stiffness` — Q4 enhanced with Wilson's incompatible (bubble)
  displacement modes (Wilson, Taylor & Doherty 1973; consistency-corrected by
  Taylor, Beresford & Wilson 1976 — the "QM6" element found in most FE
  textbooks/codes). Adds 4 internal generalised DOFs (2 bubble modes ×
  2 displacement components), condensed out via
  :func:`elements.condense_local_stiffness` before the 8×8 nodal stiffness is
  returned — reuses the same static-condensation machinery
  ``elements.condense_local_stiffness`` already implements for beam end
  releases, per the Phase 1 plan.

Scope note (Phase 1 of dev/IMPLEMENT_QUAD.md): these are pure functions,
independent of ``Structure2D`` — no ``QuadElement``/``QuadSection`` data model
and no assembly wiring exist yet (that is Phases 3-4). The stress-recovery
helpers here therefore take local coordinates and a local displacement vector
directly, rather than ``(struc, u_vec)`` like :func:`tri_elements.tri_stresses`
does — the ``struc``-based cache-driven wrappers land once the assembly cache
exists for quads.
"""
from __future__ import annotations

import numpy as np

from .elements import condense_local_stiffness
from .tri_elements import plane_D, principal_stresses

#: 2x2 Gauss-Legendre quadrature points for the compatible (bilinear) part.
#: Both directions use 2 points; each point carries unit weight (the
#: reference square [-1,1]x[-1,1] has weight 1x1 per point for 2-pt GL).
_GAUSS2 = (-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0))
_GAUSS2_POINTS = [(xi, eta) for xi in _GAUSS2 for eta in _GAUSS2]


def q4_shape_functions(xi: float, eta: float):
    """Bilinear shape functions N(ξ,η) (4,) and natural derivatives
    dN/d(ξ,η) (4,2) at one point. Node order 1..4 counter-clockwise:
    (ξ,η) = (-1,-1), (1,-1), (1,1), (-1,1)."""
    N = 0.25 * np.array([
        (1.0 - xi) * (1.0 - eta),
        (1.0 + xi) * (1.0 - eta),
        (1.0 + xi) * (1.0 + eta),
        (1.0 - xi) * (1.0 + eta),
    ])
    dN_dxi = 0.25 * np.array([
        [-(1.0 - eta), -(1.0 - xi)],
        [(1.0 - eta), -(1.0 + xi)],
        [(1.0 + eta), (1.0 + xi)],
        [-(1.0 + eta), (1.0 - xi)],
    ])
    return N, dN_dxi


def _inv2x2(J):
    """Inverse and determinant of a 2x2 matrix, raising on a degenerate quad."""
    detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
    if detJ <= 0.0:
        raise ValueError(
            "Non-positive Jacobian determinant — quad is inverted, "
            "degenerate, or its nodes are not wound counter-clockwise"
        )
    Jinv = np.array([[J[1, 1], -J[0, 1]], [-J[1, 0], J[0, 0]]]) / detJ
    return Jinv, detJ


def _jacobian(coords, dN_dxi):
    """coords: (4,2) array-like. Returns (J 2x2, Jinv 2x2, detJ, dN_dxy (4,2))."""
    coords = np.asarray(coords, dtype=float)
    J = dN_dxi.T @ coords
    Jinv, detJ = _inv2x2(J)
    dN_dxy = dN_dxi @ Jinv.T
    return J, Jinv, detJ, dN_dxy


def q4_B_matrix(coords, xi: float, eta: float):
    """Return (B 3x8, detJ) at one (ξ,η) point. B is the standard bilinear
    strain-displacement matrix (varies over the element, unlike the CST's
    constant B), DOF order [u1,v1,u2,v2,u3,v3,u4,v4]."""
    _, dN_dxi = q4_shape_functions(xi, eta)
    _, _, detJ, dN_dxy = _jacobian(coords, dN_dxi)
    B = np.zeros((3, 8))
    for a in range(4):
        dNdx, dNdy = dN_dxy[a]
        B[0, 2 * a] = dNdx
        B[1, 2 * a + 1] = dNdy
        B[2, 2 * a] = dNdy
        B[2, 2 * a + 1] = dNdx
    return B, detJ


def q4_stiffness(coords, E: float, nu: float, t: float, plane_strain: bool = False):
    """8x8 element stiffness for the plain bilinear Q4, 2x2 Gauss quadrature.

    Returns (k, gauss, D, area). ``gauss`` is a list of
    ``(xi, eta, B, detJ)`` at each of the 4 integration points, kept for
    stress recovery (the B matrix is not constant over a quad, unlike a
    triangle's)."""
    D = plane_D(E, nu, plane_strain)
    k = np.zeros((8, 8))
    area = 0.0
    gauss = []
    for xi, eta in _GAUSS2_POINTS:
        B, detJ = q4_B_matrix(coords, xi, eta)
        k += t * (B.T @ D @ B) * detJ
        area += detJ
        gauss.append((xi, eta, B, detJ))
    return k, gauss, D, area


def _qm6_bubble_B(coords, xi: float, eta: float, J0inv, detJ0: float):
    """3x4 strain-displacement matrix for the 2 incompatible (bubble) modes
    λ1=(1-ξ²), λ2=(1-η²), each applied to both displacement components —
    4 generalised internal DOFs ``[a_u1, a_u2, a_v1, a_v2]``.

    Uses the Taylor–Beresford–Wilson (1976) consistency correction: natural
    derivatives of the bubble modes are mapped to x,y with the *centroid*
    Jacobian inverse ``J0inv``, scaled by ``detJ0/detJ(ξ,η)``, rather than the
    local J(ξ,η) inverse — without this correction the element fails the
    constant-strain patch test on non-rectangular (non-parallelogram) quads.
    """
    _, _, detJ, _ = _jacobian(coords, q4_shape_functions(xi, eta)[1])
    dPhi_dxi = np.array([[-2.0 * xi, 0.0],
                          [0.0, -2.0 * eta]])          # (mode, d/dξ or d/dη)
    scale = detJ0 / detJ
    dPhi_dxy = scale * (dPhi_dxi @ J0inv.T)             # (mode, d/dx or d/dy)

    Bq = np.zeros((3, 4))
    for m in range(2):
        dphidx, dphidy = dPhi_dxy[m]
        # column 2*m   -> a_u{m+1} (affects u only)
        # column 2*m+1 -> a_v{m+1} (affects v only)
        Bq[0, m] = dphidx
        Bq[2, m] = dphidy
        Bq[1, 2 + m] = dphidy
        Bq[2, 2 + m] = dphidx
    return Bq, detJ


def _qm6_full_system(coords, E: float, nu: float, t: float,
                     plane_strain: bool = False):
    """The 12x12 (8 nodal + 4 internal bubble-mode) system before
    condensation, plus ``D``/``area`` and a per-Gauss-point list carrying
    BOTH the nodal ``B`` and the bubble ``Bq`` — shared by
    :func:`qm6_stiffness` (needs only ``k_full``) and
    :func:`qm6_thermal_load` (needs ``k_full`` again, to condense the thermal
    load vector by the identical static-condensation step, plus every
    ``Bq`` to build that vector's internal-DOF rows in the first place).
    Extracted so the two can never quietly drift apart on the Gauss loop or
    the consistency-corrected bubble strain (dev/IMPLEMENT_QUAD.md Phase 6,
    added once a quad thermal load became a real feature — see
    ``models.QuadTemperatureLoad``)."""
    D = plane_D(E, nu, plane_strain)
    coords_arr = np.asarray(coords, dtype=float)

    _, dN0_dxi = q4_shape_functions(0.0, 0.0)
    J0 = dN0_dxi.T @ coords_arr
    J0inv, detJ0 = _inv2x2(J0)

    k_full = np.zeros((12, 12))
    area = 0.0
    gauss = []
    for xi, eta in _GAUSS2_POINTS:
        B, detJ = q4_B_matrix(coords, xi, eta)
        Bq, _ = _qm6_bubble_B(coords, xi, eta, J0inv, detJ0)
        tdetJ = t * detJ
        k_full[:8, :8] += tdetJ * (B.T @ D @ B)
        k_full[:8, 8:] += tdetJ * (B.T @ D @ Bq)
        k_full[8:, :8] += tdetJ * (Bq.T @ D @ B)
        k_full[8:, 8:] += tdetJ * (Bq.T @ D @ Bq)
        area += detJ
        gauss.append((xi, eta, B, detJ, Bq, tdetJ))
    return k_full, D, area, gauss


def qm6_stiffness(coords, E: float, nu: float, t: float, plane_strain: bool = False):
    """8x8 QM6 (Wilson incompatible-mode) element stiffness.

    Builds the full 12x12 system (8 nodal + 4 internal bubble-mode DOFs),
    then statically condenses the 4 internal DOFs via
    :func:`elements.condense_local_stiffness` — the same condensation helper
    used for beam-element end releases. 2x2 Gauss quadrature for every term
    (compatible-compatible, compatible-incompatible, incompatible-
    incompatible); this is exact for all three since the integrands are at
    most quadratic in ξ,η.

    Returns (k, gauss, D, area) — same shape/signature as
    :func:`q4_stiffness` so both formulations plug into the same dispatch
    point once Phase 4 wires up assembly.
    """
    k_full, D, area, gauss = _qm6_full_system(coords, E, nu, t, plane_strain)
    condensed = condense_local_stiffness(k_full, [8, 9, 10, 11])
    k = condensed[:8, :8]
    # Public gauss shape is (xi, eta, B, detJ) — unchanged from before this
    # function shared its loop with _qm6_full_system; the bubble Bq/tdetJ
    # _qm6_full_system also computed are QM6-thermal-load-only and dropped
    # here so every existing caller (stress recovery, kernel tests) keeps
    # seeing exactly the tuple shape it always has.
    gauss_out = [(xi, eta, B, detJ) for (xi, eta, B, detJ, _Bq, _tdetJ) in gauss]
    return k, gauss_out, D, area


def _quad_centroid_B(coords):
    """B matrix at the element centroid (ξ=η=0) — used for the single
    reported stress per element, mirroring the constant-stress CST report
    (dev/IMPLEMENT_QUAD.md Phase 1 uses one representative point; per-Gauss-
    point recovery can be added later without changing this signature)."""
    return q4_B_matrix(coords, 0.0, 0.0)


def q4_stress_at_centroid(coords, E: float, nu: float, t: float, ue,
                           plane_strain: bool = False, thermal=None):
    """Centroid stress for a plain Q4 from its 8 nodal displacements ``ue``
    (order ``[u1,v1,...,u4,v4]``). Returns
    ``{sx, sy, txy, s1, s2, theta, vm, formulation}``."""
    B, _ = _quad_centroid_B(coords)
    D = plane_D(E, nu, plane_strain)
    eps = B @ np.asarray(ue, dtype=float)
    if thermal is not None:
        eps = eps - np.asarray(thermal, dtype=float)
    sig = D @ eps
    sx, sy, txy = float(sig[0]), float(sig[1]), float(sig[2])
    s1, s2, th, vm = principal_stresses(sx, sy, txy)
    return {'sx': sx, 'sy': sy, 'txy': txy,
            's1': s1, 's2': s2, 'theta': th, 'vm': vm,
            'formulation': 'Q4'}


def qm6_stress_at_centroid(coords, E: float, nu: float, t: float, ue,
                            plane_strain: bool = False, thermal=None):
    """Centroid stress for a QM6 element from its 8 *nodal* displacements
    ``ue`` — the compatible part only (the internal bubble-mode amplitudes
    were condensed out of the stiffness and are not recovered here; they
    contribute to the element's stiffness but, by construction of static
    condensation, the nodal B matrix alone gives the centroid stress
    consistent with the condensed k). Returns the same dict shape as
    :func:`q4_stress_at_centroid`, with ``formulation='QM6'``."""
    result = q4_stress_at_centroid(coords, E, nu, t, ue, plane_strain, thermal)
    result['formulation'] = 'QM6'
    return result


# ---------------------------------------------------------------------------
# Thermal (in-plane, uniform-over-the-element) loads (dev/IMPLEMENT_QUAD.md
# Phase 6, added once a quad thermal load became a real feature). Mirrors
# ``loads.tri_thermal_strain`` + ``loads._apply_tri_temperature_loads``'s CST
# branch: f_th = t·∫Bᵀ·D·ε₀ dA, ε₀ the free thermal strain — except a Q4/QM6's
# B is not constant over the element (unlike a CST's), so the integral is
# Gauss-summed rather than a single area×B product.
# ---------------------------------------------------------------------------

def q4_thermal_load(coords, E: float, nu: float, t: float, eps0,
                    plane_strain: bool = False):
    """Equivalent nodal load (8,) for a plain Q4 under a uniform free thermal
    strain ``eps0`` = [εx, εy, γxy] (see ``loads.tri_thermal_strain`` for how
    ``eps0`` itself is built from ΔT). 2x2 Gauss, exact — B is linear in ξ,η
    so Bᵀ·D·ε₀ is at most quadratic, same order the stiffness integrand is."""
    D = plane_D(E, nu, plane_strain)
    eps0 = np.asarray(eps0, dtype=float)
    f = np.zeros(8)
    for xi, eta in _GAUSS2_POINTS:
        B, detJ = q4_B_matrix(coords, xi, eta)
        f += t * detJ * (B.T @ D @ eps0)
    return f


def qm6_thermal_load(coords, E: float, nu: float, t: float, eps0,
                     plane_strain: bool = False):
    """Equivalent nodal load (8,) for a QM6 element under a uniform free
    thermal strain ``eps0``.

    Builds the full 12-entry load (8 nodal + 4 internal bubble-mode rows,
    the bubble rows from the same ``eps0`` acting through ``Bq`` — a free
    strain does virtual work against the bubble modes exactly as it does
    against the nodal ones), then statically condenses the 4 internal rows
    via :func:`elements.condense_local_load`, using the SAME (uncondensed)
    stiffness :func:`qm6_stiffness` condenses — required for the condensed
    load to be consistent with the condensed k, the load-vector analogue of
    :func:`elements.condense_local_stiffness`. This is what makes a QM6
    element free to expand under a uniform ΔT report zero stress after
    solving, the same check a CST already passes."""
    k_full, D, _area, gauss = _qm6_full_system(coords, E, nu, t, plane_strain)
    eps0 = np.asarray(eps0, dtype=float)
    f_full = np.zeros(12)
    for (_xi, _eta, B, _detJ, Bq, tdetJ) in gauss:
        f_full[:8] += tdetJ * (B.T @ D @ eps0)
        f_full[8:] += tdetJ * (Bq.T @ D @ eps0)
    from .elements import condense_local_load
    f_cond = condense_local_load(k_full, f_full, [8, 9, 10, 11])
    return f_cond[:8]


# ---------------------------------------------------------------------------
# struc-based recovery (dev/IMPLEMENT_QUAD.md Phase 5). Mirrors
# tri_elements_mitc3.mitc3_moments: recomputes (E, nu, t) fresh from the
# model each call rather than reusing struc._quad_cache's D, since recovery
# is not hot and this keeps the Q4/QM6/DKT4/MITC4 recovery paths uniform —
# see quad_elements_dkt4.dkt4_moments/quad_elements_mitc4.mitc4_moments,
# which need (E, nu, t) rather than a cached D for the same reason.
# ---------------------------------------------------------------------------

def quad_stresses(struc, u_vec, thermal=None) -> dict:
    """Q4/QM6 membrane centroid stress from a solved global displacement
    vector, keyed by quad id. Mirrors ``tri_elements.tri_stresses``.

    *thermal* ({quad_id: eps0}) is the free thermal strain to subtract, built
    by ``loads.thermal_strain_field`` from any ``QuadTemperatureLoad``
    (dev/IMPLEMENT_QUAD.md Phase 6) — the matching subtraction to
    ``q4_thermal_load``/``qm6_thermal_load``'s equivalent nodal load, so a
    quad free to expand reports zero stress after solving."""
    thermal = thermal or {}
    out = {}
    u = np.asarray(u_vec)
    idx = getattr(struc, 'node_dof_index', {})
    for quad in getattr(struc, 'quad_elements', []):
        sec = struc.quad_sections.get(quad.section_name)
        if sec is None or getattr(sec, 'formulation', 'MITC4') not in ('Q4', 'QM6'):
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
            dofs += [b, b + 1]
        ue = u[dofs]
        fn = qm6_stress_at_centroid if sec.formulation == 'QM6' else q4_stress_at_centroid
        out[quad.id] = fn(coords, mat.elastic_modulus, getattr(mat, 'poisson', 0.2),
                          sec.thickness, ue, sec.plane_strain, thermal.get(quad.id))
    return out


def quad_stresses_from_disp(struc, disp: dict, thermal=None) -> dict:
    """Same as :func:`quad_stresses` but from a displacement dict
    ``{node_id: [ux, uy, rz]}`` — used for analysis cases/combinations."""
    thermal = thermal or {}
    out = {}
    z3 = (0.0, 0.0, 0.0)
    for quad in getattr(struc, 'quad_elements', []):
        sec = struc.quad_sections.get(quad.section_name)
        if sec is None or getattr(sec, 'formulation', 'MITC4') not in ('Q4', 'QM6'):
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
        ue = np.array(
            list(disp.get(quad.node_i, z3))[:2]
            + list(disp.get(quad.node_j, z3))[:2]
            + list(disp.get(quad.node_k, z3))[:2]
            + list(disp.get(quad.node_l, z3))[:2], dtype=float)
        fn = qm6_stress_at_centroid if sec.formulation == 'QM6' else q4_stress_at_centroid
        out[quad.id] = fn(coords, mat.elastic_modulus, getattr(mat, 'poisson', 0.2),
                          sec.thickness, ue, sec.plane_strain, thermal.get(quad.id))
    return out


def quad_stresses_dispatch(struc, u_vec, thermal=None) -> dict:
    """All quad stresses/moments (Q4/QM6 + DKT4 + MITC4 combined), keyed by
    element id — the quad-side merge point, analogous to
    ``tri_elements.tri_stresses_dispatch`` on the triangle side. This is what
    ``tri_stresses_dispatch`` itself pulls in for a model that has quads, so
    the merged results still land under the single existing
    ``results['tri_stress']`` key (dev/IMPLEMENT_QUAD.md §5 point 5: keep the
    key, widen its contents — no new results key for quads)."""
    out = dict(quad_stresses(struc, u_vec, thermal=thermal))
    if getattr(struc, 'quad_elements', None):
        from .quad_elements_dkt4 import dkt4_moments
        out.update(dkt4_moments(struc, u_vec, thermal=thermal))
        from .quad_elements_mitc4 import mitc4_moments
        out.update(mitc4_moments(struc, u_vec, thermal=thermal))
    return out


def quad_stresses_from_disp_dispatch(struc, disp: dict, thermal=None) -> dict:
    """Same as :func:`quad_stresses_dispatch` but from a displacement dict —
    used for analysis cases / combinations."""
    out = dict(quad_stresses_from_disp(struc, disp, thermal=thermal))
    if getattr(struc, 'quad_elements', None):
        from .quad_elements_dkt4 import dkt4_moments_from_disp
        out.update(dkt4_moments_from_disp(struc, disp, thermal=thermal))
        from .quad_elements_mitc4 import mitc4_moments_from_disp
        out.update(mitc4_moments_from_disp(struc, disp, thermal=thermal))
    return out
