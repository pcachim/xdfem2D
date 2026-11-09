"""
Load vector assembly for xdfem2D.
Point loads, distributed/trapezoidal loads, self-weight.

Plate domain: bar loads act out of plane — the transverse intensity is along
global Z (stored in the fy* slots), the consistent nodal moments land on the
rotational DOFs (tx, ty) through the grillage transformation, and the local
fixed-end vectors are stored in the same (w, θ̃) algebra the plane beam uses
(θ̃ = dw/dx'), so the shared force-recovery reads them identically.
"""
from __future__ import annotations
import numpy as np
from .structure import Structure2D
from .elements import transformation_matrix, condense_local_load


def _is_plate(struc) -> bool:
    return getattr(struc, 'domain', 'plane') == 'plate'


def _elem_T(struc, cache):
    """The 6×6 global→local transformation for a bar — grillage T in the
    plate domain, frame T in the plane domain (both orthogonal)."""
    if _is_plate(struc):
        from .elements_grid import grid_transformation_matrix
        return grid_transformation_matrix(cache['cos'], cache['sin'])
    return transformation_matrix(cache['cos'], cache['sin'])


def _grid_transverse_trapezoid(struc, elem, lc_id, ic, F, q0, q1):
    """Consistent nodal loads + fixed-end forces for a trapezoidal transverse
    load q(x') [kN/m, along global Z] on a grillage bar (q0 at i, q1 at j).

    The Hermitian formulas are the plane beam's, written in (w, θ̃) with
    θ̃ = dw/dx'; the nodal moment components are carried to the (tx, ty) DOFs
    through ψ = −θ̃ and the grillage transformation. The fixed-end vector is
    stored in the same section convention the plane beam uses, so
    the shared recovery algebra applies unchanged.
    """
    cache = struc._elem_cache[elem.id]
    L = cache['L']
    Py_i = (7*q0 + 3*q1) * L / 20.0
    Py_j = (3*q0 + 7*q1) * L / 20.0
    Mz_i = (3*q0 + 2*q1) * L*L / 60.0     # consistent moment on θ̃ at i
    Mz_j = (2*q0 + 3*q1) * L*L / 60.0     # (θ̃-value at j is −Mz_j)

    # (w, φ, ψ) local vector: ψ = −θ̃ flips the moment components.
    s_local = np.array([Py_i, 0.0, -Mz_i, Py_j, 0.0, +Mz_j])
    T = _elem_T(struc, cache)
    g6 = T.T @ s_local

    itotv = struc.node_dof_index[elem.node_i]
    jtotv = struc.node_dof_index[elem.node_j]
    for n, d in enumerate((itotv, itotv+1, itotv+2, jtotv, jtotv+1, jtotv+2)):
        F[d, ic] += g6[n]
    struc._elem_eqload[lc_id][elem.id] += g6

    fef = struc._fixed_end_forces[lc_id][elem.id]
    fef[1] -= Py_i;  fef[2] += Mz_i
    fef[4] += Py_j;  fef[5] += Mz_j


def assemble_loads(struc: Structure2D) -> np.ndarray:
    """
    Build and return the global load matrix F of shape (ndof, n_cases).
    Column index = load case index (0-based).
    Also populates struc._fixed_end_forces[case_id][elem_id] = [NN1, VV1, MM1, NN2, VV2, MM2].
    """
    ndof = struc.num_dofs
    n_cases = len(struc.load_cases)
    F = np.zeros((ndof, n_cases))

    case_index = {lc.id: i for i, lc in enumerate(struc.load_cases)}

    # Initialise fixed-end force store
    struc._fixed_end_forces = {
        lc.id: {elem.id: np.zeros(6) for elem in struc.bar_elements}
        for lc in struc.load_cases
    }
    # Per-element GLOBAL equivalent nodal-load vector (6 dofs, i-end then j-end),
    # accumulated alongside F so released (hinged) elements can be condensed.
    struc._elem_eqload = {
        lc.id: {elem.id: np.zeros(6) for elem in struc.bar_elements}
        for lc in struc.load_cases
    }

    _apply_point_loads(struc, F, case_index)
    _apply_distributed_loads(struc, F, case_index)
    _apply_element_point_loads(struc, F, case_index)
    _apply_self_weight(struc, F, case_index)
    _apply_tri_self_weight(struc, F, case_index)
    _apply_tri_area_loads(struc, F, case_index)
    _apply_quad_self_weight(struc, F, case_index)
    _apply_quad_area_loads(struc, F, case_index)
    _apply_tri_edge_loads(struc, F, case_index)
    _apply_quad_edge_loads(struc, F, case_index)
    _apply_temperature_loads(struc, F, case_index)
    _apply_tri_temperature_loads(struc, F, case_index)
    _apply_quad_temperature_loads(struc, F, case_index)
    _finalize_end_releases(struc, F, case_index)

    return F


# ---------------------------------------------------------------------------
# End releases — condense element equivalent loads for hinged ends
# ---------------------------------------------------------------------------

def _finalize_end_releases(struc: Structure2D, F: np.ndarray, case_index: dict):
    """
    For every element with a moment release, replace its full equivalent nodal
    loads in F with the statically condensed equivalent loads (the released
    moment is redistributed). For elements without releases this is a no-op,
    so results are bit-for-bit identical to the non-hinged formulation.
    """
    for elem in struc.bar_elements:
        released = struc._elem_cache[elem.id].get('released') or []
        if not released:
            continue
        cache = struc._elem_cache[elem.id]
        k_local = cache['k_local']
        T = _elem_T(struc, cache)
        itotv = struc.node_dof_index[elem.node_i]
        jtotv = struc.node_dof_index[elem.node_j]
        dofs = [itotv, itotv+1, itotv+2, jtotv, jtotv+1, jtotv+2]
        for lc in struc.load_cases:
            ic = case_index[lc.id]
            g6 = struc._elem_eqload[lc.id][elem.id]
            if not np.any(g6):
                continue
            q_local = T @ g6                       # global -> local (T orthogonal)
            q_cond  = condense_local_load(k_local, q_local, released)
            g6_cond = T.T @ q_cond                 # local -> global
            for n, d in enumerate(dofs):
                F[d, ic] += g6_cond[n] - g6[n]     # replace full with condensed


# ---------------------------------------------------------------------------
# Point loads
# ---------------------------------------------------------------------------

def _apply_point_loads(struc: Structure2D, F: np.ndarray, case_index: dict):
    for pl in struc.point_loads:
        if pl.load_case_id not in case_index:
            continue
        ic = case_index[pl.load_case_id]
        itotv = struc.node_dof_index[pl.node_id]
        F[itotv,   ic] += pl.fx
        F[itotv+1, ic] += pl.fy
        F[itotv+2, ic] += pl.mz


# ---------------------------------------------------------------------------
# Distributed (trapezoidal) loads – local element axes
# ---------------------------------------------------------------------------

def _apply_distributed_loads(struc: Structure2D, F: np.ndarray, case_index: dict):
    plate = _is_plate(struc)
    for dl in struc.distributed_loads:
        if dl.load_case_id not in case_index:
            continue
        ic = case_index[dl.load_case_id]
        elem = struc.bar_elements_by_id[dl.element_id]
        cache = struc._elem_cache[elem.id]
        L   = cache['L']
        cos = cache['cos']
        sin = cache['sin']

        if plate:
            # Grillage: the transverse intensity is along global Z whatever
            # the coord_sys says (there is no in-plane load direction to
            # rotate); the axial slots (fxe/fxd) have no out-of-plane meaning
            # and are ignored.
            _grid_transverse_trapezoid(struc, elem, dl.load_case_id, ic, F,
                                       dl.fye, dl.fyd)
            continue
        fef = struc._fixed_end_forces[dl.load_case_id][elem.id]
        eq  = struc._elem_eqload[dl.load_case_id][elem.id]

        itotv = struc.node_dof_index[elem.node_i]
        jtotv = struc.node_dof_index[elem.node_j]

        # Resolve the trapezoidal intensities into the element LOCAL axes first
        # (x' = element axis, y' = perpendicular). The consistent nodal loads are
        # then formed in the local frame — axial via linear shape functions,
        # transverse via Hermitian ones — and only afterwards rotated to global.
        # Doing the axial/transverse split in local coordinates is what makes
        # the result correct for inclined and vertical members (e.g. wind on a
        # column); for a horizontal element local == global and the values are
        # identical to before.
        if dl.coord_sys == 'local':
            nxe, nye = dl.fxe, dl.fye
            nxd, nyd = dl.fxd, dl.fyd
        else:
            # global → local:  x' =  cos,sin   y' = -sin,cos
            nxe =  dl.fxe * cos + dl.fye * sin
            nye = -dl.fxe * sin + dl.fye * cos
            nxd =  dl.fxd * cos + dl.fyd * sin
            nyd = -dl.fxd * sin + dl.fyd * cos

        # Local consistent nodal forces (trapezoidal, i-end → j-end)
        Nx_i = (7*nxe + 3*nxd) * L / 20.0      # axial
        Nx_j = (3*nxe + 7*nxd) * L / 20.0
        Py_i = (7*nye + 3*nyd) * L / 20.0      # transverse shear
        Py_j = (3*nye + 7*nyd) * L / 20.0
        Mz_i = (3*nye + 2*nyd) * L*L / 60.0    # transverse moment
        Mz_j = (2*nye + 3*nyd) * L*L / 60.0

        # Local nodal-load vector, then rotate local → global (T is global→local)
        s_local = np.array([Nx_i, Py_i, Mz_i, Nx_j, Py_j, -Mz_j])
        T   = transformation_matrix(cos, sin)
        g6  = T.T @ s_local
        for n, d in enumerate((itotv, itotv+1, itotv+2, jtotv, jtotv+1, jtotv+2)):
            F[d, ic] += g6[n]
        eq += g6

        # Fixed-end member forces (stored in LOCAL coords for force recovery)
        fef[0] += Nx_i;  fef[1] -= Py_i;  fef[2] += Mz_i
        fef[3] -= Nx_j;  fef[4] += Py_j;  fef[5] += Mz_j


# ---------------------------------------------------------------------------
# Element point loads — concentrated force/moment at distance a along the bar
# ---------------------------------------------------------------------------

def _apply_element_point_loads(struc: Structure2D, F: np.ndarray, case_index: dict):
    """Consistent nodal loads for a point load at local distance ``a``.

    Uses the Hermitian (cubic) shape functions for the transverse force and the
    linear shape functions for the axial force, so the equivalent nodal loads
    are the exact fixed-end forces of a beam-column. The assembly mirrors the
    distributed-load path (same s_local / fef sign convention).
    """
    plate = _is_plate(struc)
    for pl in struc.element_point_loads:
        if pl.load_case_id not in case_index:
            continue
        elem = struc.bar_elements_by_id.get(pl.element_id)
        if elem is None:
            continue
        ic = case_index[pl.load_case_id]
        cache = struc._elem_cache[elem.id]
        L   = cache['L']
        cos = cache['cos']
        sin = cache['sin']
        a = min(max(pl.a, 0.0), L)
        xi = a / L if L > 0 else 0.0

        # Resolve the applied force into element LOCAL axes. Plate domain:
        # the transverse force (fy slot / fz alias) is along global Z and mz
        # is the concentrated bending moment — no rotation needed, no axial.
        if plate:
            px, py, m0 = 0.0, pl.fy, pl.mz
        elif pl.coord_sys == 'local':
            px, py, m0 = pl.fx, pl.fy, pl.mz
        else:
            px =  pl.fx * cos + pl.fy * sin
            py = -pl.fx * sin + pl.fy * cos
            m0 =  pl.mz

        # Hermitian shape functions and derivatives at xi.
        N1 = 1 - 3*xi**2 + 2*xi**3
        N2 = L * (xi - 2*xi**2 + xi**3)
        N3 = 3*xi**2 - 2*xi**3
        N4 = L * (-xi**2 + xi**3)
        dN1 = (-6*xi + 6*xi**2)
        dN2 = (1 - 4*xi + 3*xi**2)
        dN3 = (6*xi - 6*xi**2)
        dN4 = (-2*xi + 3*xi**2)

        # Axial consistent loads (linear shape functions).
        Nx_i = px * (1.0 - xi)
        Nx_j = px * xi

        # Transverse consistent loads = force·N + moment·(dN/dx) [dN/dx = dN/L].
        Py_i = py * N1 + m0 * dN1 / L
        Mz_i = py * N2 + m0 * dN2
        Py_j = py * N3 + m0 * dN3 / L
        Mz_j_h = py * N4 + m0 * dN4     # Hermitian j-moment (already signed)

        # Match the distributed-load convention: s_local stores -Mz_j, and the
        # member fixed-end vector stores +Mz_j (= -s_local[5]).
        if plate:
            # (w, φ, ψ) components: ψ = −θ̃ flips the two moment entries.
            s_local = np.array([Py_i, 0.0, -Mz_i, Py_j, 0.0, -Mz_j_h])
        else:
            s_local = np.array([Nx_i, Py_i, Mz_i, Nx_j, Py_j, Mz_j_h])
        T  = _elem_T(struc, cache)
        g6 = T.T @ s_local
        for n, d in enumerate((struc.node_dof_index[elem.node_i] + 0,
                               struc.node_dof_index[elem.node_i] + 1,
                               struc.node_dof_index[elem.node_i] + 2,
                               struc.node_dof_index[elem.node_j] + 0,
                               struc.node_dof_index[elem.node_j] + 1,
                               struc.node_dof_index[elem.node_j] + 2)):
            F[d, ic] += g6[n]
        struc._elem_eqload[pl.load_case_id][elem.id] += g6

        # The fixed-end vector lives in the shared (v, θ) / (w, θ̃) algebra,
        # so it is the same in both domains (the axial slots stay zero for a
        # grillage bar: Nx_i = Nx_j = 0 when px = 0).
        fef = struc._fixed_end_forces[pl.load_case_id][elem.id]
        fef[0] += Nx_i;  fef[1] -= Py_i;  fef[2] += Mz_i
        fef[3] -= Nx_j;  fef[4] += Py_j;  fef[5] -= Mz_j_h


# ---------------------------------------------------------------------------
# Self-weight
# ---------------------------------------------------------------------------

def _apply_self_weight(struc: Structure2D, F: np.ndarray, case_index: dict):
    plate = _is_plate(struc)
    for lc in struc.load_cases:
        if lc.self_weight_factor == 0.0:
            continue
        ic = case_index[lc.id]
        for elem in struc.bar_elements:
            if plate:
                # Gravity acts along −Z, i.e. straight onto the w DOF: a
                # uniform transverse load −γ·A per metre on every bar.
                q = -struc._elem_cache[elem.id]['weight_per_m'] \
                    * lc.self_weight_factor
                _grid_transverse_trapezoid(struc, elem, lc.id, ic, F, q, q)
                continue
            cache = struc._elem_cache[elem.id]
            L   = cache['L']
            cos = cache['cos']
            sin = cache['sin']
            w   = cache['weight_per_m']  # kN/m  (area × unit_weight)

            # Downward (negative y) distributed load scaled by factor
            force = -w * lc.self_weight_factor   # force per unit length

            itotv = struc.node_dof_index[elem.node_i]
            jtotv = struc.node_dof_index[elem.node_j]

            fy = force * L / 2.0
            mz = cos * force * L*L / 12.0

            F[itotv+1, ic] += fy
            F[itotv+2, ic] += mz
            F[jtotv+1, ic] += fy
            F[jtotv+2, ic] -= mz

            eq = struc._elem_eqload[lc.id][elem.id]
            eq[1] += fy;  eq[2] += mz
            eq[4] += fy;  eq[5] -= mz

            fef = struc._fixed_end_forces[lc.id][elem.id]
            fef[0] +=  fy * sin;  fef[1] -=  fy * cos;  fef[2] += mz
            fef[3] -= fy * sin;  fef[4] +=  fy * cos;  fef[5] += mz


# ---------------------------------------------------------------------------
# Triangle (CST plane element) loads
# ---------------------------------------------------------------------------

def _tri_area(struc, tri) -> float:
    ni = struc.nodes[tri.node_i]; nj = struc.nodes[tri.node_j]
    nk = struc.nodes[tri.node_k]
    return abs((nj.x - ni.x) * (nk.y - ni.y)
               - (nk.x - ni.x) * (nj.y - ni.y)) / 2.0


def _apply_tri_self_weight(struc: Structure2D, F: np.ndarray, case_index: dict):
    """Self-weight of CST triangles: total weight γ·t·A is lumped equally to the
    three vertices (W/3 each), acting downward (−y), scaled by the load case's
    self_weight_factor."""
    tris = getattr(struc, "tri_elements", [])
    if not tris:
        return
    for lc in struc.load_cases:
        if lc.self_weight_factor == 0.0:
            continue
        ic = case_index[lc.id]
        for tri in tris:
            sec = struc.tri_sections.get(tri.section_name)
            if sec is None:
                continue
            mat = struc.materials.get(sec.material_name)
            if mat is None:
                continue
            gamma = getattr(mat, "unit_weight", 0.0)
            if gamma == 0.0:
                continue
            A = _tri_area(struc, tri)
            f = -gamma * sec.thickness * A / 3.0 * lc.self_weight_factor
            # Plane: gravity is −Y (component 1). Plate: gravity is −Z,
            # straight onto the w DOF (component 0).
            off = 0 if _is_plate(struc) else 1
            for nid in (tri.node_i, tri.node_j, tri.node_k):
                F[struc.node_dof_index[nid] + off, ic] += f


def _apply_tri_area_loads(struc: Structure2D, F: np.ndarray, case_index: dict):
    """Uniform transverse pressure pz on plate triangles: pz·A/3 lumped to
    each vertex's w DOF (plate domain only; surface area loads were expanded
    to per-triangle loads when the surface was meshed)."""
    area_loads = getattr(struc, "tri_area_loads", [])
    if not area_loads or not _is_plate(struc):
        return
    tris = getattr(struc, "tri_elements_by_id", {})
    for al in area_loads:
        ic = case_index.get(al.load_case_id)
        tri = tris.get(al.tri_id)
        if ic is None or tri is None or al.pz == 0.0:
            continue
        f = al.pz * _tri_area(struc, tri) / 3.0
        for nid in (tri.node_i, tri.node_j, tri.node_k):
            F[struc.node_dof_index[nid], ic] += f


# ---------------------------------------------------------------------------
# Quadrilateral self-weight / area loads (dev/IMPLEMENT_QUAD.md Phase 4) —
# mirrors the triangle path above exactly, lumped 1/4 to each vertex instead
# of 1/3.
# ---------------------------------------------------------------------------

def _quad_area(struc, quad) -> float:
    """Plan area of a quad (shoelace formula) from its four node coordinates
    — already guaranteed convex/CCW at ``add_quad_element`` time (Phase 3),
    so the signed shoelace sum is the true area, no ``abs`` sign ambiguity."""
    pts = [struc.nodes[nid] for nid in
           (quad.node_i, quad.node_j, quad.node_k, quad.node_l)]
    area2 = 0.0
    for a in range(4):
        p, q = pts[a], pts[(a + 1) % 4]
        area2 += p.x * q.y - q.x * p.y
    return abs(area2) / 2.0


def _apply_quad_self_weight(struc: Structure2D, F: np.ndarray, case_index: dict):
    """Self-weight of quad elements: total weight γ·t·A is lumped equally to
    the four vertices (W/4 each), acting downward (−y in the plane domain,
    −z / straight onto w in the plate domain), scaled by the load case's
    self_weight_factor. Mirrors :func:`_apply_tri_self_weight`."""
    quads = getattr(struc, "quad_elements", [])
    if not quads:
        return
    for lc in struc.load_cases:
        if lc.self_weight_factor == 0.0:
            continue
        ic = case_index[lc.id]
        for quad in quads:
            sec = struc.quad_sections.get(quad.section_name)
            if sec is None:
                continue
            mat = struc.materials.get(sec.material_name)
            if mat is None:
                continue
            gamma = getattr(mat, "unit_weight", 0.0)
            if gamma == 0.0:
                continue
            A = _quad_area(struc, quad)
            f = -gamma * sec.thickness * A / 4.0 * lc.self_weight_factor
            off = 0 if _is_plate(struc) else 1
            for nid in (quad.node_i, quad.node_j, quad.node_k, quad.node_l):
                F[struc.node_dof_index[nid] + off, ic] += f


def _apply_quad_area_loads(struc: Structure2D, F: np.ndarray, case_index: dict):
    """Uniform transverse pressure pz on plate quads: pz·A/4 lumped to each
    vertex's w DOF (plate domain only). Mirrors :func:`_apply_tri_area_loads`."""
    area_loads = getattr(struc, "quad_area_loads", [])
    if not area_loads or not _is_plate(struc):
        return
    quads = getattr(struc, "quad_elements_by_id", {})
    for al in area_loads:
        ic = case_index.get(al.load_case_id)
        quad = quads.get(al.quad_id)
        if ic is None or quad is None or al.pz == 0.0:
            continue
        f = al.pz * _quad_area(struc, quad) / 4.0
        for nid in (quad.node_i, quad.node_j, quad.node_k, quad.node_l):
            F[struc.node_dof_index[nid], ic] += f


def tri_thermal_strain(sec, mat, dt_mean):
    """The free thermal strain ε₀ of a CST, as [εx, εy, γxy].

    α·ΔT in plane stress; (1+ν)·α·ΔT in plane strain, because the out-of-plane
    restraint of plane strain raises the effective in-plane expansion. The
    factor has to match the D it is multiplied by — using α·ΔT with a plane
    strain D is the quiet inconsistency this names on purpose.
    """
    nu = getattr(mat, "poisson", 0.2)
    alpha = getattr(mat, "alpha", 1e-5)
    factor = (1.0 + nu) if getattr(sec, "plane_strain", False) else 1.0
    e = factor * alpha * dt_mean
    return np.array([e, e, 0.0])


def thermal_strain_field(struc, coeffs: dict) -> dict:
    """{element_id: ε₀} — each triangle's AND each Q4/QM6 quad's free thermal
    strain under the load cases in *coeffs* ({load_case_id: factor}), summed
    with those factors, in one dict keyed by element id (a triangle id and a
    quad id can never collide — see dev/IMPLEMENT_QUAD.md Phase 5).

    The field the stress recovery must subtract. Empty when nothing is thermal
    or nothing is combined, so the caller can skip the correction cheaply.
    Covers both CST and Allman sections — the free thermal strain ε₀ = α·ΔT is a
    property of the material/section, not of the element formulation, and the
    load vector applies the temperature to both, so both must remove it again.
    A Q4/QM6 quad uses the exact same ε₀ (dev/IMPLEMENT_QUAD.md Phase 6 —
    :func:`tri_thermal_strain` is reused unchanged, since the physics is a
    material/section property, not a shape one; only the *load vector* it
    feeds into differs — Gauss-integrated for a quad instead of a single
    constant-B product, see ``quad_elements.q4_thermal_load``/
    ``qm6_thermal_load``).

    In the plate domain the recovered quantity is a curvature, not a strain, so
    this returns the free thermal *curvature* field κ₀ instead (see
    :func:`plate_thermal_curvature_field`) — the DKT/DKT4/MITC3/MITC4 moment
    recovery subtracts it the same way the CST subtracts ε₀.
    """
    if _is_plate(struc):
        return plate_thermal_curvature_field(struc, coeffs)
    out: dict = {}
    temps = getattr(struc, "tri_temperature_loads", [])
    if temps and coeffs:
        tris = getattr(struc, "tri_elements_by_id", {})
        for tl in temps:
            f = coeffs.get(tl.load_case_id)
            if not f:
                continue
            tri = tris.get(tl.tri_id)
            if tri is None:
                continue
            sec = struc.tri_sections.get(tri.section_name)
            if sec is None:
                continue
            mat = struc.materials.get(sec.material_name)
            if mat is None:
                continue
            e = f * tri_thermal_strain(sec, mat, tl.dt_mean)
            out[tl.tri_id] = out.get(tl.tri_id, 0.0) + e
    qtemps = getattr(struc, "quad_temperature_loads", [])
    if qtemps and coeffs:
        quads = getattr(struc, "quad_elements_by_id", {})
        for tl in qtemps:
            f = coeffs.get(tl.load_case_id)
            if not f:
                continue
            quad = quads.get(tl.quad_id)
            if quad is None:
                continue
            sec = struc.quad_sections.get(quad.section_name)
            if sec is None or getattr(sec, "formulation", "MITC4") not in (
                    "Q4", "QM6"):
                continue
            mat = struc.materials.get(sec.material_name)
            if mat is None:
                continue
            e = f * tri_thermal_strain(sec, mat, tl.dt_mean)
            out[tl.quad_id] = out.get(tl.quad_id, 0.0) + e
    return out


def resolve_to_load_cases(struc, coeffs: dict) -> dict:
    """Fold {case_id: factor} down to {load_case_id: factor}.

    A key may already be a load case, or a Linear/NonLinear analysis case whose
    own coefficients are load cases — those are expanded and multiplied through.
    Used to find the effective load-case weights of an analysis case or a
    combination, so the thermal field of a combined result can be built.
    """
    lc_ids = set(getattr(struc, "load_cases_by_id", {}))
    acs = getattr(struc, "analysis_cases_by_id", {})
    out: dict = {}
    for cid, f in coeffs.items():
        if cid in lc_ids:
            out[cid] = out.get(cid, 0.0) + f
            continue
        ac = acs.get(cid)
        if ac is not None and getattr(ac, "analysis_type", "Linear") in (
                "Linear", "NonLinear"):
            for lc_id, c2 in (getattr(ac, "coefficients", {}) or {}).items():
                out[lc_id] = out.get(lc_id, 0.0) + f * c2
    return out


def plate_thermal_curvature(sec, mat, dt_grad):
    """Free thermal curvature κ₀ (β-orientation) of a DKT plate triangle under a
    through-thickness gradient ΔT = T_top − T_bottom. The plate analogue of
    :func:`tri_thermal_strain`: isotropic, magnitude α·ΔT/t. The curvature is the
    same for either plate element (DKT, MITC3) — both report in the same
    orientation. Zero for a section that is not a plate element (a membrane does
    not bend)."""
    from .tri_elements_dkt import dkt_thermal_curvature
    if getattr(sec, "formulation", "CST") not in ("DKT", "MITC3"):
        return np.zeros(3)
    alpha = getattr(mat, "alpha", 1e-5)
    return dkt_thermal_curvature(alpha, dt_grad, sec.thickness)


def quad_plate_thermal_curvature(sec, mat, dt_grad):
    """Free thermal curvature κ₀ of a DKT4/MITC4 plate quad under a
    through-thickness gradient ΔT = T_top − T_bottom (dev/IMPLEMENT_QUAD.md
    Phase 6, added once a quad thermal load became a real feature).

    Identical physics to :func:`plate_thermal_curvature` — isotropic,
    magnitude α·ΔT/t, the same ``dkt_thermal_curvature`` formula reused
    unchanged (it was never DKT-specific despite living in
    ``tri_elements_dkt``: a free thermal curvature is a material/section
    property, not an element-formulation one) — only gated on the quad
    formulations instead of the triangle ones. Zero for a Q4/QM6 section (a
    membrane does not bend)."""
    from .tri_elements_dkt import dkt_thermal_curvature
    if getattr(sec, "formulation", "MITC4") not in ("DKT4", "MITC4"):
        return np.zeros(3)
    alpha = getattr(mat, "alpha", 1e-5)
    return dkt_thermal_curvature(alpha, dt_grad, sec.thickness)


def _apply_plate_temperature_gradients(struc: Structure2D, F: np.ndarray,
                                       case_index: dict):
    """Equivalent nodal loads for a through-thickness gradient on plate (DKT)
    triangles: f_th = ∫Bᵀ·D_b·κ₀ dA on each element's (w, tx, ty) DOFs, with
    κ₀ = α·ΔT/t the free thermal curvature. The matching subtraction happens in
    the moment recovery (see :func:`plate_thermal_curvature_field`)."""
    from .tri_elements_dkt import dkt_thermal_load
    from .tri_elements_mitc3 import mitc3_thermal_load
    tris = getattr(struc, "tri_elements_by_id", {})
    for tl in getattr(struc, "tri_temperature_loads", []):
        if tl.load_case_id not in case_index:
            continue
        grad = getattr(tl, "dt_gradient", 0.0)
        if grad == 0.0:
            continue
        tri = tris.get(tl.tri_id)
        if tri is None:
            continue
        sec = struc.tri_sections.get(tri.section_name)
        form = getattr(sec, "formulation", "CST") if sec is not None else "CST"
        if form not in ("DKT", "MITC3"):
            continue
        mat = struc.materials.get(sec.material_name)
        if mat is None:
            continue
        ni = struc.nodes[tri.node_i]
        nj = struc.nodes[tri.node_j]
        nk = struc.nodes[tri.node_k]
        coords = [(ni.x, ni.y), (nj.x, nj.y), (nk.x, nk.y)]
        kappa0 = plate_thermal_curvature(sec, mat, grad)
        _thermal_load = (mitc3_thermal_load if form == "MITC3"
                         else dkt_thermal_load)
        f_th = _thermal_load(coords, mat.elastic_modulus,
                             getattr(mat, "poisson", 0.2),
                             sec.thickness, kappa0)
        bi = struc.node_dof_index[tri.node_i]
        bj = struc.node_dof_index[tri.node_j]
        bk = struc.node_dof_index[tri.node_k]
        dofs = (bi, bi + 1, bi + 2, bj, bj + 1, bj + 2, bk, bk + 1, bk + 2)
        ic = case_index[tl.load_case_id]
        for dof, val in zip(dofs, f_th):
            F[dof, ic] += val


def _apply_quad_plate_temperature_gradients(struc: Structure2D, F: np.ndarray,
                                            case_index: dict):
    """Equivalent nodal loads for a through-thickness gradient on plate
    (DKT4/MITC4) quads — the quad analogue of
    :func:`_apply_plate_temperature_gradients` (dev/IMPLEMENT_QUAD.md
    Phase 6, added once a quad thermal load became a real feature)."""
    from .quad_elements_dkt4 import dkt4_thermal_load
    from .quad_elements_mitc4 import mitc4_thermal_load
    quads = getattr(struc, "quad_elements_by_id", {})
    for tl in getattr(struc, "quad_temperature_loads", []):
        if tl.load_case_id not in case_index:
            continue
        grad = getattr(tl, "dt_gradient", 0.0)
        if grad == 0.0:
            continue
        quad = quads.get(tl.quad_id)
        if quad is None:
            continue
        sec = struc.quad_sections.get(quad.section_name)
        form = getattr(sec, "formulation", "MITC4") if sec is not None else "MITC4"
        if form not in ("DKT4", "MITC4"):
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
        kappa0 = quad_plate_thermal_curvature(sec, mat, grad)
        _thermal_load = (mitc4_thermal_load if form == "MITC4"
                         else dkt4_thermal_load)
        f_th = _thermal_load(coords, mat.elastic_modulus,
                             getattr(mat, "poisson", 0.2), sec.thickness,
                             kappa0)
        dofs = []
        for nid in (quad.node_i, quad.node_j, quad.node_k, quad.node_l):
            b = struc.node_dof_index[nid]
            dofs += [b, b + 1, b + 2]
        ic = case_index[tl.load_case_id]
        for dof, val in zip(dofs, f_th):
            F[dof, ic] += val


def plate_thermal_curvature_field(struc, coeffs: dict) -> dict:
    """{element_id: κ₀} — each DKT triangle's AND each DKT4/MITC4 quad's free
    thermal curvature under the load cases in *coeffs*
    ({load_case_id: factor}), summed with those factors, in one dict keyed by
    element id (see :func:`thermal_strain_field` — a triangle and a quad id
    never collide).

    The plate counterpart of :func:`thermal_strain_field`: the field the moment
    recovery subtracts so a plate free to curve reports zero moment. Empty when
    nothing carries a gradient, so the caller can skip the correction cheaply."""
    out: dict = {}
    temps = getattr(struc, "tri_temperature_loads", [])
    if temps and coeffs:
        tris = getattr(struc, "tri_elements_by_id", {})
        for tl in temps:
            f = coeffs.get(tl.load_case_id)
            grad = getattr(tl, "dt_gradient", 0.0)
            if not f or grad == 0.0:
                continue
            tri = tris.get(tl.tri_id)
            if tri is None:
                continue
            sec = struc.tri_sections.get(tri.section_name)
            if sec is None or getattr(sec, "formulation", "CST") not in (
                    "DKT", "MITC3"):
                continue
            mat = struc.materials.get(sec.material_name)
            if mat is None:
                continue
            k0 = f * plate_thermal_curvature(sec, mat, grad)
            prev = out.get(tl.tri_id)
            out[tl.tri_id] = k0 if prev is None else prev + k0
    qtemps = getattr(struc, "quad_temperature_loads", [])
    if qtemps and coeffs:
        quads = getattr(struc, "quad_elements_by_id", {})
        for tl in qtemps:
            f = coeffs.get(tl.load_case_id)
            grad = getattr(tl, "dt_gradient", 0.0)
            if not f or grad == 0.0:
                continue
            quad = quads.get(tl.quad_id)
            if quad is None:
                continue
            sec = struc.quad_sections.get(quad.section_name)
            if sec is None or getattr(sec, "formulation", "MITC4") not in (
                    "DKT4", "MITC4"):
                continue
            mat = struc.materials.get(sec.material_name)
            if mat is None:
                continue
            k0 = f * quad_plate_thermal_curvature(sec, mat, grad)
            prev = out.get(tl.quad_id)
            out[tl.quad_id] = k0 if prev is None else prev + k0
    return out


def _apply_tri_temperature_loads(struc: Structure2D, F: np.ndarray,
                                 case_index: dict):
    """Equivalent nodal loads for a triangle temperature change (CST + Allman).

    f_th = t·A·Bᵀ·D·ε₀ with ε₀ the free thermal strain and B, D, A the same
    pieces the stiffness uses. Only the mean of the three nodal ΔT enters — a
    constant-strain triangle cannot feel more (see TriTemperatureLoad) — so a
    per-node field only takes effect once the mesh is fine enough to give
    neighbouring triangles different means.

    For CST, B is constant (6 membrane dofs). For the Allman element, B is
    linear and B0 is its centroid value (9 dofs, drilling included); because B
    is linear, ∫Bᵀ dA = B0ᵀ·A exactly, so B0 gives the exact consistent load —
    which naturally carries components onto the drilling dofs.

    The matching subtraction happens at stress recovery: without it a free
    triangle would report the stress of a restrained one.
    """
    temps = getattr(struc, "tri_temperature_loads", [])
    if not temps:
        return
    if _is_plate(struc):
        # A mean ΔT stretches a plate's mid-surface, which the plate domain
        # does not model, and bends nothing. A through-thickness gradient
        # ΔT = T_top − T_bottom *does* bend the DKT: apply its free-curvature
        # load f_th = ∫Bᵀ D_b κ₀ dA, the direct analogue of the CST's ε₀ load.
        _apply_plate_temperature_gradients(struc, F, case_index)
        return
    from .tri_elements import cst_B_area, plane_D
    tris = getattr(struc, "tri_elements_by_id", {})
    for tl in temps:
        if tl.load_case_id not in case_index:
            continue
        tri = tris.get(tl.tri_id)
        if tri is None:
            continue
        sec = struc.tri_sections.get(tri.section_name)
        if sec is None:
            continue
        if getattr(sec, "formulation", "CST") == "ES-FEM":
            continue    # assembled over edge domains (see esfem_thermal_loads)
        mat = struc.materials.get(sec.material_name)
        if mat is None:
            continue
        ni = struc.nodes[tri.node_i]
        nj = struc.nodes[tri.node_j]
        nk = struc.nodes[tri.node_k]
        coords = [(ni.x, ni.y), (nj.x, nj.y), (nk.x, nk.y)]
        D = plane_D(mat.elastic_modulus, getattr(mat, "poisson", 0.2),
                    getattr(sec, "plane_strain", False))
        eps0 = tri_thermal_strain(sec, mat, tl.dt_mean)
        bi = struc.node_dof_index[tri.node_i]
        bj = struc.node_dof_index[tri.node_j]
        bk = struc.node_dof_index[tri.node_k]
        ic = case_index[tl.load_case_id]
        if getattr(sec, "formulation", "CST") == "Allman":
            from .tri_elements_allman import allman_B, _edge_geometry
            area = _edge_geometry(coords)["area"]
            B0 = allman_B(coords, 1 / 3, 1 / 3, 1 / 3)     # (3, 9)
            f_th = sec.thickness * area * (B0.T @ D @ eps0)  # (9,)
            dofs = (bi, bi + 1, bi + 2, bj, bj + 1, bj + 2, bk, bk + 1, bk + 2)
        else:
            B, area = cst_B_area(coords)
            f_th = sec.thickness * area * (B.T @ D @ eps0)   # (6,)
            dofs = (bi, bi + 1, bj, bj + 1, bk, bk + 1)
        for dof, val in zip(dofs, f_th):
            F[dof, ic] += val

    # ES-FEM sections: consistent thermal load assembled over edge domains.
    from .tri_elements_esfem import esfem_thermal_loads
    esfem_thermal_loads(struc, F, case_index)


def _apply_quad_temperature_loads(struc: Structure2D, F: np.ndarray,
                                  case_index: dict):
    """Equivalent nodal loads for a quad temperature change (Q4/QM6 in-plane,
    DKT4/MITC4 gradient) — the quad analogue of
    :func:`_apply_tri_temperature_loads` (dev/IMPLEMENT_QUAD.md Phase 6,
    added once a quad thermal load became a real feature — see
    ``models.QuadTemperatureLoad``).

    f_th = t·∫Bᵀ·D·ε₀ dA, Gauss-integrated (see
    ``quad_elements.q4_thermal_load``/``qm6_thermal_load`` — unlike a CST's
    constant B, a Q4/QM6's B varies over the element, so this is not a single
    area×B product the way the triangle branch is). Only the mean of the four
    nodal ΔT enters, mirroring the triangle's constant-strain limitation (see
    ``models.QuadTemperatureLoad``).

    The matching subtraction happens at stress recovery (``quad_elements.
    quad_stresses``'s ``thermal`` argument): without it a free quad would
    report the stress of a restrained one.
    """
    temps = getattr(struc, "quad_temperature_loads", [])
    if not temps:
        return
    if _is_plate(struc):
        # A mean ΔT stretches a plate's mid-surface, which the plate domain
        # does not model, and bends nothing — same reasoning as the triangle
        # branch. A through-thickness gradient DOES bend DKT4/MITC4.
        _apply_quad_plate_temperature_gradients(struc, F, case_index)
        return
    from .quad_elements import q4_thermal_load, qm6_thermal_load
    quads = getattr(struc, "quad_elements_by_id", {})
    for tl in temps:
        if tl.load_case_id not in case_index:
            continue
        quad = quads.get(tl.quad_id)
        if quad is None:
            continue
        sec = struc.quad_sections.get(quad.section_name)
        if sec is None or getattr(sec, "formulation", "MITC4") not in (
                "Q4", "QM6"):
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
        eps0 = tri_thermal_strain(sec, mat, tl.dt_mean)
        fn = qm6_thermal_load if sec.formulation == "QM6" else q4_thermal_load
        f_th = fn(coords, mat.elastic_modulus, getattr(mat, "poisson", 0.2),
                 sec.thickness, eps0, sec.plane_strain)
        dofs = []
        for nid in (quad.node_i, quad.node_j, quad.node_k, quad.node_l):
            b = struc.node_dof_index[nid]
            dofs += [b, b + 1]
        ic = case_index[tl.load_case_id]
        for dof, val in zip(dofs, f_th):
            F[dof, ic] += val


def _edge_load_fxy(el, na, nb, thickness):
    """Force per unit length (fx, fy) [kN/m] for an edge load on segment a→b.

    ``coord_sys='global'`` → the stored (fx, fy) directly. ``'local'`` → the
    normal/tangential pressures (pn, pt) [kN/m²] rotated to global and scaled by
    the thickness (right-hand normal of a→b)."""
    if getattr(el, "coord_sys", "global") != "local":
        return el.fx, el.fy
    dx, dy = nb.x - na.x, nb.y - na.y
    L = (dx * dx + dy * dy) ** 0.5 or 1.0
    tx, ty = dx / L, dy / L
    nx, ny = ty, -tx                          # right-hand normal
    return ((el.pn * nx + el.pt * tx) * thickness,
            (el.pn * ny + el.pt * ty) * thickness)


def _apply_tri_edge_loads(struc: Structure2D, F: np.ndarray, case_index: dict):
    """Consistent nodal loads from a uniform edge load on a triangle edge. The
    resultant per unit length (fx, fy) is split equally (½·L each) to the two
    edge nodes."""
    edge_loads = getattr(struc, "tri_edge_loads", [])
    if not edge_loads:
        return
    for el in edge_loads:
        ic = case_index.get(el.load_case_id)
        if ic is None:
            continue
        na = struc.nodes.get(el.node_a); nb = struc.nodes.get(el.node_b)
        if na is None or nb is None:
            continue
        tri = struc.tri_elements_by_id.get(el.tri_id)
        sec = struc.tri_sections.get(tri.section_name) if tri is not None else None
        thk = sec.thickness if sec is not None else 1.0
        dx, dy = nb.x - na.x, nb.y - na.y
        L = (dx * dx + dy * dy) ** 0.5
        if L == 0.0:
            continue
        half = L / 2.0
        if _is_plate(struc):
            # Plate: the fx slot is the transverse line load fz [kN/m]
            # (a wall standing on the slab edge); halves onto the two w DOFs.
            for nid in (el.node_a, el.node_b):
                F[struc.node_dof_index[nid], ic] += el.fx * half
            continue
        fx, fy = _edge_load_fxy(el, na, nb, thk)
        for nid in (el.node_a, el.node_b):
            b = struc.node_dof_index[nid]
            F[b,     ic] += fx * half
            F[b + 1, ic] += fy * half


def _apply_quad_edge_loads(struc: Structure2D, F: np.ndarray, case_index: dict):
    """Consistent nodal loads from a uniform edge load on a quad edge — the
    4-node analogue of :func:`_apply_tri_edge_loads` (½·L each to the two edge
    nodes; the quad's thickness comes from its quad section)."""
    edge_loads = getattr(struc, "quad_edge_loads", [])
    if not edge_loads:
        return
    for el in edge_loads:
        ic = case_index.get(el.load_case_id)
        if ic is None:
            continue
        na = struc.nodes.get(el.node_a); nb = struc.nodes.get(el.node_b)
        if na is None or nb is None:
            continue
        quad = struc.quad_elements_by_id.get(el.quad_id)
        sec = (struc.quad_sections.get(quad.section_name)
               if quad is not None else None)
        thk = sec.thickness if sec is not None else 1.0
        dx, dy = nb.x - na.x, nb.y - na.y
        L = (dx * dx + dy * dy) ** 0.5
        if L == 0.0:
            continue
        half = L / 2.0
        if _is_plate(struc):
            for nid in (el.node_a, el.node_b):
                F[struc.node_dof_index[nid], ic] += el.fx * half
            continue
        fx, fy = _edge_load_fxy(el, na, nb, thk)
        for nid in (el.node_a, el.node_b):
            b = struc.node_dof_index[nid]
            F[b,     ic] += fx * half
            F[b + 1, ic] += fy * half


# ---------------------------------------------------------------------------
# Temperature loads
# ---------------------------------------------------------------------------

def _apply_temperature_loads(struc: Structure2D, F: np.ndarray, case_index: dict):
    plate = _is_plate(struc)
    for tl in struc.temperature_loads:
        if tl.load_case_id not in case_index:
            continue
        ic = case_index[tl.load_case_id]
        elem = struc.bar_elements_by_id.get(tl.element_id)
        if elem is None:
            continue
        cache = struc._elem_cache[elem.id]
        L   = cache['L']
        cos = cache['cos']
        sin = cache['sin']

        sec  = struc.sections[elem.section_name]
        mat  = struc.materials[sec.material_name]
        EA   = mat.elastic_modulus * sec.area
        EI   = mat.elastic_modulus * sec.inertia
        h    = sec.h   # section height for gradient

        itotv = struc.node_dof_index[elem.node_i]
        jtotv = struc.node_dof_index[elem.node_j]
        fef   = struc._fixed_end_forces[tl.load_case_id][elem.id]
        eq    = struc._elem_eqload[tl.load_case_id][elem.id]

        alpha = mat.alpha   # thermal expansion coeff from material

        # ── Uniform temperature ΔT ──────────────────────────────
        # Equivalent nodal loads in local coords: [-EA·α·ΔT, 0, 0, +EA·α·ΔT, 0, 0]
        # Plate domain: a grillage bar has no axial DOF, so a uniform ΔT is
        # stress-free by construction — nothing to apply.
        if tl.delta_t_uniform != 0.0 and not plate:
            P = EA * alpha * tl.delta_t_uniform
            # Transform local axial → global and add to F
            F[itotv,   ic] -= P * cos
            F[itotv+1, ic] -= P * sin
            F[jtotv,   ic] += P * cos
            F[jtotv+1, ic] += P * sin
            eq[0] -= P * cos;  eq[1] -= P * sin
            eq[3] += P * cos;  eq[4] += P * sin
            # Fixed-end forces are stored in LOCAL member axes with no
            # transverse (V/M) component — a uniform ΔT is a pure axial
            # pre-strain, never a bending effect. Do NOT reuse the cos/sin
            # global-projection factors here (that was the bug: they leaked
            # a spurious V/M term into every inclined member's fixed-end
            # force, corrupting shear/moment recovery and the diagrams
            # derived from it — most visibly as a nonzero moment appearing
            # at a pinned/hinged end, which the FE solve itself always gets
            # right). fef[0]/fef[3] use opposite signs (matching the elastic
            # nn2 = -nn1 relation used in force recovery, see
            # solver._member_end_forces) so that an unrestrained bar reports
            # N = 0 at BOTH ends, and a fully restrained bar reports
            # N = -EA·α·ΔT (compression) at both ends too.
            fef[0] -= P
            fef[3] += P

        # ── Temperature gradient (T_top - T_bottom) ─────────────
        # Curvature κ = α·ΔTgrad/h
        # Equivalent nodal moments: M_i = +EI·κ, M_j = -EI·κ
        if tl.delta_t_gradient != 0.0 and h > 0.0:
            kappa = alpha * tl.delta_t_gradient / h
            M = EI * kappa
            if plate:
                # Same fixed-end moment pair, but on the (w, θ̃) algebra: the
                # θ̃ components (+M, −M) become ψ components (−M, +M) and are
                # rotated onto the (tx, ty) DOFs by the grillage T.
                from .elements_grid import grid_transformation_matrix
                Tg = grid_transformation_matrix(cos, sin)
                g6 = Tg.T @ np.array([0.0, 0.0, -M, 0.0, 0.0, +M])
                for n, d in enumerate((itotv, itotv+1, itotv+2,
                                       jtotv, jtotv+1, jtotv+2)):
                    F[d, ic] += g6[n]
                eq += g6
            else:
                F[itotv+2, ic] += M
                F[jtotv+2, ic] -= M
                eq[2] += M
                eq[5] -= M
            # i-end matches F[i] (+M), j-end is opposite to F[j] (−(−M)=+M).
            fef[2] += M
            fef[5] += M
