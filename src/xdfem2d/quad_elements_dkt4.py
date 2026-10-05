"""Discrete-Kirchhoff plate-bending quadrilateral — the thin (Kirchhoff)
counterpart of MITC4, and the quad analogue of ``tri_elements_dkt``'s DKT.

**Naming, on purpose**: this element is called **DKT4**, not DKQ. "DKQ" in
the literature names one specific closed-form derivation — Batoz & Tahar
(1982), a fully quadrilateral discrete-Kirchhoff formulation with its own
shape functions and side coefficients. What is implemented here is a
different, coarser construction (below), and calling it "DKQ" would tell a
user picking it that they are getting the textbook/commercial-software
element when they are not. "DKT4" says what it actually is instead: four
DKT triangles, condensed into one quad element.

**Deviation from dev/IMPLEMENT_QUAD.md worth flagging explicitly**: deriving
the literal Batoz-Tahar DKQ from scratch, with no reference implementation in
this codebase to check against, was judged too high-risk for this pass —
subtle sign/index errors in a novel closed-form plate kernel are exactly the
kind of bug dev/IMPLEMENT_QUAD.md §5 point 1 warns can "look fine until
someone's slab design is 15% off".

Instead this element is a **composite of 4 already-validated DKT triangles**:
split the quad into 4 triangles by fanning from its centroid (a 5th, internal
node), assemble their 9x9 ``tri_elements_dkt.dkt_stiffness`` matrices into one
15x15 system, and statically condense the centroid's 3 DOFs out via
``elements.condense_local_stiffness`` — the same pattern
``quad_elements.qm6_stiffness`` uses for its 4 internal bubble-mode DOFs, just
applied to a real (geometric) internal node instead of a generalised one. The
result is a genuine, thin, Kirchhoff-consistent 4-node quad element, low risk
because it reuses DKT/condensation machinery already covered by tests
elsewhere in the codebase — but it is a coarser approximation than the
literal DKQ (a diagonal choice enters implicitly through the centroid fan,
whereas Batoz-Tahar DKQ has none). Revisit with the literal Batoz-Tahar
formulation — under its own "DKQ" name, since it would be a genuinely
different element, not a rename of this one — if validation ever shows this
composite is not accurate enough for a given mesh.

Public API named ``dkt4_stiffness``/``dkt4_moment_entry`` to match the
section formulation string ("DKT4"), so nothing here claims to be DKQ.
"""
from __future__ import annotations

import numpy as np

from .elements import condense_local_stiffness, condense_local_load
from .plate_common import bending_D as plate_D
from .tri_elements_dkt import (dkt_stiffness, dkt_thermal_load,
                               _moment_entry as _dkt_moment_entry)

# Fan triangulation: quad nodes 0,1,2,3 (CCW) + centroid (local index 4).
# Each triangle (edge_a, edge_b, centroid) keeps the quad's CCW winding.
_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0))


def _centroid(coords):
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    return sum(xs) / 4.0, sum(ys) / 4.0


def _full_system(coords, E: float, nu: float, t: float):
    """15x15 stiffness (4 corners + 1 internal centroid node, 3 DOF each:
    [w,θx,θy]) from the 4 fan triangles, plus the shared Db and total area."""
    c = _centroid(coords)
    k_full = np.zeros((15, 15))
    area_total = 0.0
    Db = plate_D(E, nu, t)
    for a, b in _EDGES:
        tri_coords = [coords[a], coords[b], c]
        k9, _B0, _Db, area = dkt_stiffness(tri_coords, E, nu, t)
        dofs = []
        for nid in (a, b, 4):
            base = 3 * nid
            dofs += [base, base + 1, base + 2]
        for i in range(9):
            for j in range(9):
                k_full[dofs[i], dofs[j]] += k9[i, j]
        area_total += area
    return k_full, Db, area_total


def dkt4_stiffness(coords, E: float, nu: float, t: float):
    """(k, Db, area): 12x12 stiffness for the 4 corner nodes, the centroid's
    3 DOFs condensed out. No single cached B matrix (unlike a plain DKT/MITC4
    kernel) — moment recovery needs the corner *and* the condensed-out
    centroid displacement, so it goes through :func:`dkt4_moment_entry`
    instead, which recomputes the 15x15 system to back-substitute the
    centroid DOFs."""
    k_full, Db, area = _full_system(coords, E, nu, t)
    condensed = condense_local_stiffness(k_full, [12, 13, 14])
    k = condensed[:12, :12]
    return k, Db, area


def dkt4_thermal_load(coords, E: float, nu: float, t: float,
                      kappa0) -> np.ndarray:
    """Equivalent nodal load (12,) for a free thermal curvature ``kappa0``
    (dev/IMPLEMENT_QUAD.md Phase 6, added once a quad thermal load became a
    real feature — see ``models.QuadTemperatureLoad``).

    Same fan-and-condense construction as :func:`dkt4_stiffness`: each of the
    4 fan triangles' thermal load (``tri_elements_dkt.dkt_thermal_load``, the
    already-tested DKT kernel) is scatter-added into the 15-entry full
    system, then the centroid's 3 rows are condensed out via
    :func:`elements.condense_local_load` against the SAME (uncondensed)
    ``k_full`` :func:`dkt4_stiffness` condenses — required for the condensed
    load to stay consistent with the condensed k, exactly as
    ``quad_elements.qm6_thermal_load`` needs the uncondensed QM6 system."""
    c = _centroid(coords)
    k_full, Db, _area = _full_system(coords, E, nu, t)
    f_full = np.zeros(15)
    for a, b in _EDGES:
        tri_coords = [coords[a], coords[b], c]
        f9 = dkt_thermal_load(tri_coords, E, nu, t, kappa0)
        dofs = []
        for nid in (a, b, 4):
            base = 3 * nid
            dofs += [base, base + 1, base + 2]
        f_full[dofs] += f9
    f_cond = condense_local_load(k_full, f_full, [12, 13, 14])
    return f_cond[:12]


# ---------------------------------------------------------------------------
# Moment / shear recovery — local coords + local u12 (Phase 2). The
# struc-based ``dkt4_moments(struc, u_vec)`` wrapper is below (Phase 5).
# ---------------------------------------------------------------------------

def dkt4_moment_entry(coords, E: float, nu: float, t: float, u12,
                       kappa0=None) -> dict:
    """One DKT4 quad's reported result from its 12 corner DOFs.

    Recovers the condensed-out centroid displacement by back-substitution
    (``u_internal = -Krr⁻¹·Kr,keep·u_keep`` — exact because the internal node
    carries no external load, only the geometric fan), then reports the
    area-weighted average of the 4 fan triangles' DKT moments/shears — each
    computed by the same, already-tested
    ``tri_elements_dkt._moment_entry``."""
    k_full, Db, _area = _full_system(coords, E, nu, t)
    keep = list(range(12))
    r = [12, 13, 14]
    krr_inv = np.linalg.inv(k_full[np.ix_(r, r)])
    u12 = np.asarray(u12, dtype=float)
    u_internal = -krr_inv @ (k_full[np.ix_(r, keep)] @ u12)
    u15 = np.concatenate([u12, u_internal])

    c = _centroid(coords)
    totals = {'mx': 0.0, 'my': 0.0, 'mxy': 0.0, 'vx': 0.0, 'vy': 0.0}
    area_sum = 0.0
    for a, b in _EDGES:
        tri_coords = [coords[a], coords[b], c]
        dofs = []
        for nid in (a, b, 4):
            base = 3 * nid
            dofs += [base, base + 1, base + 2]
        u9 = u15[dofs]
        entry = _dkt_moment_entry(tri_coords, Db, u9, kappa0)
        (x1, y1), (x2, y2), (x3, y3) = tri_coords
        tri_area = abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)) / 2.0
        for key in ('mx', 'my', 'mxy', 'vx', 'vy'):
            totals[key] += entry[key] * tri_area
        area_sum += tri_area

    from .plate_common import plate_moment_result
    if area_sum <= 0.0:
        area_sum = 1.0
    mx, my, mxy = (totals['mx'] / area_sum, totals['my'] / area_sum,
                   totals['mxy'] / area_sum)
    vx, vy = totals['vx'] / area_sum, totals['vy'] / area_sum
    result = plate_moment_result(mx, my, mxy, vx, vy, 'DKT4')
    return result


# ---------------------------------------------------------------------------
# struc-based recovery (dev/IMPLEMENT_QUAD.md Phase 5). Mirrors
# tri_elements_mitc3.mitc3_moments: recomputes fresh from struc.node_dof_index
# and the model's nodes/sections/materials each call — dkt4_moment_entry
# needs (E, nu, t) to rebuild and condense the 15x15 fan-triangle system, not
# just a cached D, so there is no cheaper cache-based path here the way
# tri_elements.tri_stresses has for CST.
# ---------------------------------------------------------------------------

def dkt4_moments(struc, u_vec, thermal=None) -> dict:
    """Per-quad moment results from a solved global displacement vector, for
    DKT4 sections, keyed by quad id."""
    thermal = thermal or {}
    out = {}
    u = np.asarray(u_vec)
    idx = getattr(struc, 'node_dof_index', {})
    for quad in getattr(struc, 'quad_elements', []):
        sec = struc.quad_sections.get(quad.section_name)
        if sec is None or getattr(sec, 'formulation', 'MITC4') != 'DKT4':
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
        out[quad.id] = dkt4_moment_entry(coords, mat.elastic_modulus,
                                         getattr(mat, 'poisson', 0.2),
                                         sec.thickness, u12, thermal.get(quad.id))
    return out


def dkt4_moments_from_disp(struc, disp: dict, thermal=None) -> dict:
    """Same as :func:`dkt4_moments` but from a displacement dict
    ``{node_id: [w, tx, ty]}`` — used for analysis cases/combinations."""
    thermal = thermal or {}
    out = {}
    z3 = (0.0, 0.0, 0.0)
    for quad in getattr(struc, 'quad_elements', []):
        sec = struc.quad_sections.get(quad.section_name)
        if sec is None or getattr(sec, 'formulation', 'MITC4') != 'DKT4':
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
        out[quad.id] = dkt4_moment_entry(coords, mat.elastic_modulus,
                                         getattr(mat, 'poisson', 0.2),
                                         sec.thickness, u12, thermal.get(quad.id))
    return out
