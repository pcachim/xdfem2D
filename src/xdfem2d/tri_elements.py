"""Constant-strain triangle (CST) plane element — element matrices and stresses.

Plane stress (default) or plane strain. Two translational DOFs per node
(ux, uy); the element DOF order is [u1, v1, u2, v2, u3, v3].
"""
from __future__ import annotations

import numpy as np


def plane_D(E: float, nu: float, plane_strain: bool = False) -> np.ndarray:
    """3×3 constitutive matrix (σ = D·ε) for plane stress or plane strain."""
    if plane_strain:
        f = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
        return f * np.array([[1.0 - nu, nu, 0.0],
                             [nu, 1.0 - nu, 0.0],
                             [0.0, 0.0, (1.0 - 2.0 * nu) / 2.0]])
    f = E / (1.0 - nu * nu)
    return f * np.array([[1.0, nu, 0.0],
                         [nu, 1.0, 0.0],
                         [0.0, 0.0, (1.0 - nu) / 2.0]])


def cst_B_area(coords):
    """Return (B, area) for a CST. *coords* = [(x1,y1),(x2,y2),(x3,y3)].

    B is the 3×6 strain-displacement matrix (constant over the element); *area*
    is the (positive) triangle area."""
    (x1, y1), (x2, y2), (x3, y3) = coords
    a2 = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)   # signed 2·Area
    area = abs(a2) / 2.0
    b1, b2, b3 = y2 - y3, y3 - y1, y1 - y2
    c1, c2, c3 = x3 - x2, x1 - x3, x2 - x1
    B = (1.0 / a2) * np.array([
        [b1, 0.0, b2, 0.0, b3, 0.0],
        [0.0, c1, 0.0, c2, 0.0, c3],
        [c1, b1, c2, b2, c3, b3],
    ])
    return B, area


def cst_stiffness(coords, E: float, nu: float, t: float,
                  plane_strain: bool = False):
    """Return (k, B, D, area): 6×6 element stiffness plus the pieces needed for
    stress recovery (σ = D·B·uₑ)."""
    B, area = cst_B_area(coords)
    D = plane_D(E, nu, plane_strain)
    k = t * area * (B.T @ D @ B)
    return k, B, D, area


def tri_stresses(struc, u_vec, thermal=None) -> dict:
    """Constant stress per triangle from a solved displacement vector.

    *u_vec* is the global displacement vector for one case (shape (ndof,)); uses
    the (B, D, dofs) cached during assembly. Returns
    ``{tri_id: {sx, sy, txy, s1, s2, theta, vm}}``.

    *thermal* is an optional ``{tri_id: ε₀}`` of free thermal strains to remove:
    σ = D·(B·u − ε₀). Without it a triangle that expanded freely — zero real
    stress — would report the stress of a fully restrained one, because B·u
    already holds the thermal part of the strain."""
    import numpy as _np
    cache = getattr(struc, "_tri_cache", {}) or {}
    thermal = thermal or {}
    out = {}
    for tid, c in cache.items():
        if c.get('formulation', 'CST') != 'CST':
            continue   # Allman elements are handled by tri_elements_allman.allman_stresses
        ue = _np.asarray(u_vec)[c['dofs']]
        eps = c['B'] @ ue
        e0 = thermal.get(tid)
        if e0 is not None:
            eps = eps - e0
        sig = c['D'] @ eps
        sx, sy, txy = float(sig[0]), float(sig[1]), float(sig[2])
        s1, s2, th, vm = principal_stresses(sx, sy, txy)
        out[tid] = {'sx': sx, 'sy': sy, 'txy': txy,
                    's1': s1, 's2': s2, 'theta': th, 'vm': vm,
                    'formulation': 'CST'}
    return out


def tri_stresses_from_disp(struc, disp: dict, thermal=None) -> dict:
    """Constant stress per triangle from a displacement dict ``{node_id:
    [ux, uy, rz]}`` (the shape stored in results). Returns the same structure as
    :func:`tri_stresses`. Recomputes B, D per element from geometry so it does
    not depend on the assembly cache being populated for this displacement set."""
    from .tri_elements import cst_B_area, plane_D  # local import: self module
    out = {}
    for tri in getattr(struc, "tri_elements", []):
        try:
            ni = struc.nodes[tri.node_i]; nj = struc.nodes[tri.node_j]
            nk = struc.nodes[tri.node_k]
        except KeyError:
            continue
        sec = struc.tri_sections.get(tri.section_name)
        if sec is not None and getattr(sec, 'formulation', 'CST') != 'CST':
            continue   # Allman elements are handled by tri_elements_allman.allman_stresses_from_disp
        B, _A = cst_B_area([(ni.x, ni.y), (nj.x, nj.y), (nk.x, nk.y)])
        mat = struc.materials.get(sec.material_name) if sec else None
        if mat is None:
            continue
        D = plane_D(mat.elastic_modulus, getattr(mat, "poisson", 0.2),
                    sec.plane_strain)
        ue = np.array([
            disp.get(tri.node_i, (0.0, 0.0, 0.0))[0],
            disp.get(tri.node_i, (0.0, 0.0, 0.0))[1],
            disp.get(tri.node_j, (0.0, 0.0, 0.0))[0],
            disp.get(tri.node_j, (0.0, 0.0, 0.0))[1],
            disp.get(tri.node_k, (0.0, 0.0, 0.0))[0],
            disp.get(tri.node_k, (0.0, 0.0, 0.0))[1],
        ])
        eps = B @ ue
        e0 = (thermal or {}).get(tri.id)
        if e0 is not None:
            eps = eps - e0
        sig = D @ eps
        sx, sy, txy = float(sig[0]), float(sig[1]), float(sig[2])
        s1, s2, th, vm = principal_stresses(sx, sy, txy)
        out[tri.id] = {'sx': sx, 'sy': sy, 'txy': txy,
                       's1': s1, 's2': s2, 'theta': th, 'vm': vm,
                       'formulation': 'CST'}
    return out


def tri_stresses_dispatch(struc, u_vec, thermal=None) -> dict:
    """All surface-element stresses/moments, keyed by element id.

    Single entry point for solver.py: it merges :func:`tri_stresses` (CST
    sections) with ``tri_elements_allman.allman_stresses`` (Allman sections)
    so callers don't need to know which formulation module owns which
    element — see dev/allman_triangle_plan.md §6. Since dev/IMPLEMENT_QUAD.md
    Phase 5 it also merges in every quad formulation (Q4/QM6/DKT4/MITC4) via
    :func:`quad_elements.quad_stresses_dispatch` — a deliberate decision
    (dev/IMPLEMENT_QUAD.md §5 point 5) to widen this merge point rather than
    add a second, parallel one: the function keeps its ``tri_`` name and the
    result keeps landing under ``results['tri_stress']`` (see solver.py),
    both left alone on purpose — that key is written into every saved
    ``.x2d``'s results, and this codebase's convention for the save format is
    additive/backward-compatible, never a rename-with-migration. Quad and
    triangle element ids never collide through the public API
    (``Structure2D.add_quad_element``/``add_tri_element`` each check only
    their own id space), so the two ``.update()`` calls below never overwrite
    each other's entries.

    *thermal* ({tri_id: ε₀}) is subtracted from CST/Allman/ES-FEM/DKT/MITC3
    strains/curvatures; no quad thermal load exists yet, so quads never see a
    nonzero entry here regardless."""
    out = dict(tri_stresses(struc, u_vec, thermal=thermal))
    if getattr(struc, "tri_elements", None):
        from .tri_elements_allman import allman_stresses
        out.update(allman_stresses(struc, u_vec, thermal=thermal))
        from .tri_elements_esfem import esfem_stresses
        out.update(esfem_stresses(struc, u_vec, thermal=thermal))
        from .tri_elements_dkt import dkt_moments
        out.update(dkt_moments(struc, u_vec, thermal=thermal))
        from .tri_elements_mitc3 import mitc3_moments
        out.update(mitc3_moments(struc, u_vec, thermal=thermal))
    if getattr(struc, "quad_elements", None):
        from .quad_elements import quad_stresses_dispatch
        out.update(quad_stresses_dispatch(struc, u_vec, thermal=thermal))
    return out


def tri_stresses_from_disp_dispatch(struc, disp: dict, thermal=None) -> dict:
    """Same as :func:`tri_stresses_dispatch` but from a displacement dict
    (``{node_id: [ux, uy, rz]}``) instead of a global vector — used for
    analysis cases / combinations. Merges CST, Allman, ES-FEM, DKT, MITC3 and
    (Phase 5) every quad formulation."""
    out = dict(tri_stresses_from_disp(struc, disp, thermal=thermal))
    if getattr(struc, "tri_elements", None):
        from .tri_elements_allman import allman_stresses_from_disp
        out.update(allman_stresses_from_disp(struc, disp, thermal=thermal))
        from .tri_elements_esfem import esfem_stresses_from_disp
        out.update(esfem_stresses_from_disp(struc, disp, thermal=thermal))
        from .tri_elements_dkt import dkt_moments_from_disp
        out.update(dkt_moments_from_disp(struc, disp, thermal=thermal))
        from .tri_elements_mitc3 import mitc3_moments_from_disp
        out.update(mitc3_moments_from_disp(struc, disp, thermal=thermal))
    if getattr(struc, "quad_elements", None):
        from .quad_elements import quad_stresses_from_disp_dispatch
        out.update(quad_stresses_from_disp_dispatch(struc, disp, thermal=thermal))
    return out


def principal_stresses(sx: float, sy: float, txy: float):
    """(σ1, σ2, θ_deg, von Mises) from a plane stress state."""
    import math
    c = (sx + sy) / 2.0
    r = math.hypot((sx - sy) / 2.0, txy)
    s1, s2 = c + r, c - r
    theta = 0.5 * math.degrees(math.atan2(2.0 * txy, sx - sy)) if (sx - sy or txy) else 0.0
    vm = math.sqrt(max(sx * sx - sx * sy + sy * sy + 3.0 * txy * txy, 0.0))
    return s1, s2, theta, vm
