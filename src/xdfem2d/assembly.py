"""
Global stiffness matrix assembly for xdfem2D.
"""
from __future__ import annotations
import numpy as np
from .elements import (
    bar_stiffness_local, bar_element_geometry, transformation_matrix, is_beam,
    released_local_dofs, condense_local_stiffness,
)
from .structure import Structure2D


def _bar_element_matrix(struc, elem):
    """Element stiffness (global 6×6) + cache entry for one bar, dispatching on
    the model's domain — plane frame bar in "plane", grillage bar in "plate".
    Shared by the dense and sparse assembly paths so they can never disagree.

    The cache reuses the plane names for the recovery coefficients: in the
    plate domain 'EAL' holds G·J/L (torsion plays the axial role — constant
    along the bar, decoupled from bending) and EIL/EIL2/EIL3 keep their
    meaning, so the shared release/recovery algebra reads the same fields.
    """
    ni = struc.nodes[elem.node_i]
    nj = struc.nodes[elem.node_j]
    sec = struc.sections[elem.section_name]
    mat = struc.materials[sec.material_name]

    L, cos, sin, _ = bar_element_geometry(ni, nj)
    E = mat.elastic_modulus
    A = sec.area
    I = sec.inertia

    if getattr(struc, 'domain', 'plane') == 'plate':
        from .elements_grid import (grid_stiffness_local,
                                    grid_transformation_matrix)
        G = mat.shear_modulus
        J = sec.torsion
        k_local, C0, EIL, EIL2, EIL3 = grid_stiffness_local(E, I, G, J, L)
        T = grid_transformation_matrix(cos, sin)
    else:
        k_local, C0, EIL, EIL2, EIL3 = bar_stiffness_local(E, A, I, L)
        T = transformation_matrix(cos, sin)

    # Bending releases sit at local indices 2 and 5 in both domains (θ for the
    # plane beam, ψ for the grillage bar), so the condensation is shared.
    released = released_local_dofs(elem.hinge_i, elem.hinge_j)
    k_local_c = condense_local_stiffness(k_local, released)
    k_global = T.T @ k_local_c @ T

    cache = dict(
        L=L, cos=cos, sin=sin, EAL=C0, EIL=EIL, EIL2=EIL2, EIL3=EIL3,
        area=A, weight_per_m=A * mat.unit_weight,
        mass_per_m=A * (mat.unit_mass if mat.unit_mass is not None
                        else mat.unit_weight / 9.81),
        k_local=k_local, released=released,
        grid=(getattr(struc, 'domain', 'plane') == 'plate'),
    )
    return k_global, cache


def _elem_spring_global_block(esp, ux, uy):
    """Return the (gxx, gxy, gyy) 2×2 global translational stiffness block for an
    element foundation spring. For a 'local' spring kx is axial (along the
    element) and ky is transverse; it is rotated into global X/Y (which couples
    them for inclined elements). For a 'global' spring kx→X, ky→Y directly."""
    if getattr(esp, 'coord_sys', 'global') == 'local':
        c, s = ux, uy
        ka, kt = esp.kx, esp.ky          # axial, transverse
        gxx = ka * c * c + kt * s * s
        gxy = (ka - kt) * c * s
        gyy = ka * s * s + kt * c * c
        return gxx, gxy, gyy
    return esp.kx, 0.0, esp.ky


def _plate_elem_spring_kw(esp) -> float:
    """Transverse (w) Winkler stiffness per unit length for a plate-domain
    element spring. The transverse slot is ``ky`` (the same slot that is
    transverse for a 'local' spring in the plane domain); ``kx`` has no
    out-of-plane meaning and is ignored."""
    return esp.ky


def _add_element_springs(struc, K):
    """Add all element foundation springs to K (lumped k·L/2 at each end node).

    Plate domain: the spring resists the transverse deflection w only
    (beam on elastic foundation) — a single diagonal term per end node."""
    plate = getattr(struc, 'domain', 'plane') == 'plate'
    for esp in struc.element_springs.values():
        elem = struc.bar_elements_by_id[esp.element_id]
        L = struc._elem_cache[elem.id]['L']
        f = L / 2.0
        if plate:
            kw = _plate_elem_spring_kw(esp)
            for tv in (struc.node_dof_index[elem.node_i],
                       struc.node_dof_index[elem.node_j]):
                K[tv, tv] += kw * f
            continue
        ni = struc.nodes[elem.node_i]; nj = struc.nodes[elem.node_j]
        dx, dy = nj.x - ni.x, nj.y - ni.y
        Ln = (dx * dx + dy * dy) ** 0.5 or 1.0
        gxx, gxy, gyy = _elem_spring_global_block(esp, dx / Ln, dy / Ln)
        for tv in (struc.node_dof_index[elem.node_i],
                   struc.node_dof_index[elem.node_j]):
            K[tv,     tv]     += gxx * f
            K[tv,     tv + 1] += gxy * f
            K[tv + 1, tv]     += gxy * f
            K[tv + 1, tv + 1] += gyy * f


def _is_esfem(struc, tri) -> bool:
    """True when *tri*'s section uses the edge-based ES-FEM formulation, which
    is assembled over edge smoothing domains rather than per element."""
    sec = struc.tri_sections.get(tri.section_name)
    return sec is not None and getattr(sec, "formulation", "CST") == "ES-FEM"


def _tri_element_matrix(struc, tri):
    """Build the element stiffness + DOF map + stress-recovery cache entry
    for one triangle, dispatching on its section's ``formulation`` — 6×6 /
    (ux,uy) DOFs for "CST", 9×9 / (ux,uy,tz) DOFs for "Allman". Shared by
    both the dense and sparse assembly paths so they can never disagree on
    which formulation a given section uses.
    """
    from .tri_elements import cst_stiffness
    from .tri_elements_allman import allman_stiffness

    ni = struc.nodes[tri.node_i]
    nj = struc.nodes[tri.node_j]
    nk = struc.nodes[tri.node_k]
    sec = struc.tri_sections[tri.section_name]
    mat = struc.materials[sec.material_name]
    coords = [(ni.x, ni.y), (nj.x, nj.y), (nk.x, nk.y)]
    nu = getattr(mat, "poisson", 0.2)

    bi = struc.node_dof_index[tri.node_i]
    bj = struc.node_dof_index[tri.node_j]
    bk = struc.node_dof_index[tri.node_k]

    formulation = getattr(sec, "formulation", "CST")
    if formulation == "DKT":
        # Plate bending (plate domain): 9×9 on (w, tx, ty) at each node.
        from .tri_elements_dkt import dkt_stiffness
        k, B, D, area = dkt_stiffness(coords, mat.elastic_modulus, nu,
                                      sec.thickness)
        dofs = [bi, bi + 1, bi + 2, bj, bj + 1, bj + 2, bk, bk + 1, bk + 2]
    elif formulation == "MITC3":
        # Shear-deformable plate bending (plate domain): same 9×9 / (w, tx, ty)
        # layout as the DKT, so everything downstream is shared.
        from .tri_elements_mitc3 import mitc3_stiffness
        k, B, D, area = mitc3_stiffness(coords, mat.elastic_modulus, nu,
                                        sec.thickness)
        dofs = [bi, bi + 1, bi + 2, bj, bj + 1, bj + 2, bk, bk + 1, bk + 2]
    elif formulation == "Allman":
        k, B, D, area = allman_stiffness(coords, mat.elastic_modulus, nu,
                                         sec.thickness, sec.plane_strain)
        dofs = [bi, bi + 1, bi + 2, bj, bj + 1, bj + 2, bk, bk + 1, bk + 2]
    else:
        k, B, D, area = cst_stiffness(coords, mat.elastic_modulus, nu,
                                      sec.thickness, sec.plane_strain)
        dofs = [bi, bi + 1, bj, bj + 1, bk, bk + 1]

    cache_entry = {'B': B, 'D': D, 'area': area, 'dofs': dofs,
                   'coords': coords,
                   'formulation': formulation,
                   'nodes': (tri.node_i, tri.node_j, tri.node_k)}
    return k, dofs, cache_entry


def _quad_element_matrix(struc, quad):
    """Build the element stiffness + DOF map + stress-recovery cache entry
    for one quadrilateral, dispatching on its section's ``formulation`` —
    8×8 / (ux,uy) DOFs for "Q4"/"QM6" (plane domain), 12×12 / (w,tx,ty) DOFs
    for "DKT4"/"MITC4" (plate domain). Mirrors :func:`_tri_element_matrix`
    exactly, dev/IMPLEMENT_QUAD.md Phase 4: same shared machinery, same
    dispatch-on-formulation-not-node-count pattern, so the quad path is one
    more element loop in ``assemble_stiffness``, not a restructure.

    Node order is (node_i, node_j, node_k, node_l) — the CCW winding already
    validated at ``Structure2D.add_quad_element`` time (Phase 3), so no
    Jacobian-sign surprises reach assembly.
    """
    ni = struc.nodes[quad.node_i]
    nj = struc.nodes[quad.node_j]
    nk = struc.nodes[quad.node_k]
    nl = struc.nodes[quad.node_l]
    sec = struc.quad_sections[quad.section_name]
    mat = struc.materials[sec.material_name]
    coords = [(ni.x, ni.y), (nj.x, nj.y), (nk.x, nk.y), (nl.x, nl.y)]
    nu = getattr(mat, "poisson", 0.2)

    bi = struc.node_dof_index[quad.node_i]
    bj = struc.node_dof_index[quad.node_j]
    bk = struc.node_dof_index[quad.node_k]
    bl = struc.node_dof_index[quad.node_l]

    formulation = getattr(sec, "formulation", "MITC4")
    if formulation == "DKT4":
        from .quad_elements_dkt4 import dkt4_stiffness
        k, D, area = dkt4_stiffness(coords, mat.elastic_modulus, nu,
                                    sec.thickness)
        B = None   # no single cached B — dkt4_moment_entry recomputes (Phase 5)
        dofs = [bi, bi+1, bi+2, bj, bj+1, bj+2,
                bk, bk+1, bk+2, bl, bl+1, bl+2]
    elif formulation == "MITC4":
        from .quad_elements_mitc4 import mitc4_stiffness
        k, B, D, area = mitc4_stiffness(coords, mat.elastic_modulus, nu,
                                        sec.thickness)
        dofs = [bi, bi+1, bi+2, bj, bj+1, bj+2,
                bk, bk+1, bk+2, bl, bl+1, bl+2]
    elif formulation == "QM6":
        from .quad_elements import qm6_stiffness
        k, B, D, area = qm6_stiffness(coords, mat.elastic_modulus, nu,
                                      sec.thickness, sec.plane_strain)
        dofs = [bi, bi+1, bj, bj+1, bk, bk+1, bl, bl+1]
    else:   # "Q4" — patch-test/comparison reference, not the default
        from .quad_elements import q4_stiffness
        k, B, D, area = q4_stiffness(coords, mat.elastic_modulus, nu,
                                     sec.thickness, sec.plane_strain)
        dofs = [bi, bi+1, bj, bj+1, bk, bk+1, bl, bl+1]

    cache_entry = {'B': B, 'D': D, 'area': area, 'dofs': dofs,
                   'coords': coords,
                   'formulation': formulation,
                   'nodes': (quad.node_i, quad.node_j, quad.node_k, quad.node_l)}
    return k, dofs, cache_entry


def _quad_elements_triplets(struc, R, C, V):
    """Sparse (COO-triplet) assembly of quadrilateral elements — the path
    :func:`assemble_stiffness` actually uses."""
    cache = struc._quad_cache = {}
    for quad in getattr(struc, "quad_elements", []):
        k, dofs, cache_entry = _quad_element_matrix(struc, quad)
        n = len(dofs)
        for i in range(n):
            for j in range(n):
                R.append(dofs[i]); C.append(dofs[j]); V.append(k[i, j])
        cache[quad.id] = cache_entry


def _tri_elements_triplets(struc, R, C, V):
    """Sparse (COO-triplet) assembly of plane (triangle) elements — the path
    :func:`assemble_stiffness` actually uses."""
    cache = struc._tri_cache = {}
    for tri in getattr(struc, "tri_elements", []):
        if _is_esfem(struc, tri):
            continue                       # assembled by edge domains below
        k, dofs, cache_entry = _tri_element_matrix(struc, tri)
        n = len(dofs)
        for i in range(n):
            for j in range(n):
                R.append(dofs[i]); C.append(dofs[j]); V.append(k[i, j])
        cache[tri.id] = cache_entry
    from .tri_elements_esfem import esfem_domain_matrices
    for k, dofs, key, entry in esfem_domain_matrices(struc):
        n = len(dofs)
        for i in range(n):
            for j in range(n):
                R.append(dofs[i]); C.append(dofs[j]); V.append(k[i, j])
        cache[key] = entry


def _element_springs_triplets(struc, R, C, V):
    """Sparse (COO-triplet) assembly of element foundation springs (lumped
    k·L/2 at each end node). Same physics as :func:`_add_element_springs`."""
    plate = getattr(struc, 'domain', 'plane') == 'plate'
    for esp in struc.element_springs.values():
        elem = struc.bar_elements_by_id[esp.element_id]
        L = struc._elem_cache[elem.id]['L']
        f = L / 2.0
        if plate:
            kw = _plate_elem_spring_kw(esp)
            for tv in (struc.node_dof_index[elem.node_i],
                       struc.node_dof_index[elem.node_j]):
                R.append(tv); C.append(tv); V.append(kw * f)
            continue
        ni = struc.nodes[elem.node_i]; nj = struc.nodes[elem.node_j]
        dx, dy = nj.x - ni.x, nj.y - ni.y
        Ln = (dx * dx + dy * dy) ** 0.5 or 1.0
        gxx, gxy, gyy = _elem_spring_global_block(esp, dx / Ln, dy / Ln)
        for tv in (struc.node_dof_index[elem.node_i],
                   struc.node_dof_index[elem.node_j]):
            R += [tv, tv, tv + 1, tv + 1]
            C += [tv, tv + 1, tv, tv + 1]
            V += [gxx * f, gxy * f, gxy * f, gyy * f]


def _tri_area(struc, tri) -> float:
    """Plan area of a triangle element from its three node coordinates."""
    ni = struc.nodes[tri.node_i]; nj = struc.nodes[tri.node_j]
    nk = struc.nodes[tri.node_k]
    return abs((nj.x - ni.x) * (nk.y - ni.y)
               - (nk.x - ni.x) * (nj.y - ni.y)) / 2.0


def _add_area_springs(struc, K):
    """Add Winkler area (slab-on-grade) springs to a dense K: kz·A/3 lumped onto
    the w DOF of each vertex of every plate triangle carrying one, and kz·A/4
    for every plate quad (dev/IMPLEMENT_QUAD.md Phase 6 — the same A/4
    tributary rule ``QuadAreaLoad`` uses). Plate domain only."""
    if getattr(struc, 'domain', 'plane') != 'plate':
        return
    tris = getattr(struc, 'tri_elements_by_id', {})
    for asp in getattr(struc, 'tri_area_springs', []):
        tri = tris.get(asp.tri_id)
        if tri is None or asp.kz == 0.0:
            continue
        kw = asp.kz * _tri_area(struc, tri) / 3.0
        for nid in (tri.node_i, tri.node_j, tri.node_k):
            tv = struc.node_dof_index[nid]
            K[tv, tv] += kw
    from .loads import _quad_area
    quads = getattr(struc, 'quad_elements_by_id', {})
    for asp in getattr(struc, 'quad_area_springs', []):
        quad = quads.get(asp.quad_id)
        if quad is None or asp.kz == 0.0:
            continue
        kw = asp.kz * _quad_area(struc, quad) / 4.0
        for nid in (quad.node_i, quad.node_j, quad.node_k, quad.node_l):
            tv = struc.node_dof_index[nid]
            K[tv, tv] += kw


def _area_springs_triplets(struc, R, C, V):
    """Sparse (COO-triplet) assembly of Winkler area springs. Same physics as
    :func:`_add_area_springs`."""
    if getattr(struc, 'domain', 'plane') != 'plate':
        return
    tris = getattr(struc, 'tri_elements_by_id', {})
    for asp in getattr(struc, 'tri_area_springs', []):
        tri = tris.get(asp.tri_id)
        if tri is None or asp.kz == 0.0:
            continue
        kw = asp.kz * _tri_area(struc, tri) / 3.0
        for nid in (tri.node_i, tri.node_j, tri.node_k):
            tv = struc.node_dof_index[nid]
            R.append(tv); C.append(tv); V.append(kw)
    from .loads import _quad_area
    quads = getattr(struc, 'quad_elements_by_id', {})
    for asp in getattr(struc, 'quad_area_springs', []):
        quad = quads.get(asp.quad_id)
        if quad is None or asp.kz == 0.0:
            continue
        kw = asp.kz * _quad_area(struc, quad) / 4.0
        for nid in (quad.node_i, quad.node_j, quad.node_k, quad.node_l):
            tv = struc.node_dof_index[nid]
            R.append(tv); C.append(tv); V.append(kw)


#: Penalty stiffness for a constraint equation, expressed as a multiple of the
#: largest diagonal term of the assembled matrix. Large enough that the
#: constraint is satisfied to working precision, small enough to keep the system
#: solvable in double precision. Phase 2 (master-slave reduction) will make the
#: constraint exact and retire this factor.
CONSTRAINT_PENALTY_SCALE = 1.0e8


def _constraint_penalty_alpha(K) -> float:
    """A penalty magnitude scaled to the model: ``SCALE * max|diag(K)|``."""
    if hasattr(K, "diagonal"):
        d = np.abs(K.diagonal())
    else:
        d = np.abs(np.diag(K))
    dmax = float(d.max()) if d.size else 1.0
    return CONSTRAINT_PENALTY_SCALE * (dmax if dmax > 0.0 else 1.0)


def apply_constraint_penalty(struc: Structure2D, K, F, alpha: float | None = None):
    """Impose the model's multi-point constraints on ``(K, F)`` by penalty.

    Each canonical equation ``sum(coef * u[dof]) = value`` adds ``alpha * c cᵀ``
    to K and ``alpha * value * c`` to F, where ``c`` is the sparse coefficient
    vector. Works for a dense ``ndarray`` or a SciPy sparse ``K`` and for a 1-D
    or 2-D (multi-case) ``F``. Returns possibly-new ``(K, F)`` — callers must
    rebind. A model with no (enabled) constraints is returned untouched.

    This is the single entry point used by every solve path so they can never
    disagree on what a constraint does.
    """
    from .constraints import expand_constraints
    eqs = expand_constraints(struc)
    if not eqs:
        return K, F
    if alpha is None:
        alpha = _constraint_penalty_alpha(K)

    F = np.array(F, dtype=float)          # writable copy
    two_d = (F.ndim == 2)

    def _add_load(di, amount):
        if two_d:
            F[di, :] += amount
        else:
            F[di] += amount

    if hasattr(K, "tocsr") and hasattr(K, "diagonal"):
        from scipy import sparse
        R: list[int] = []; C: list[int] = []; V: list[float] = []
        for terms, value in eqs:
            for di, ci in terms:
                _add_load(di, alpha * value * ci)
                for dj, cj in terms:
                    R.append(di); C.append(dj); V.append(alpha * ci * cj)
        P = sparse.coo_matrix((V, (R, C)), shape=K.shape).tocsr()
        K = (K + P).tocsr()
    else:
        K = np.array(K, dtype=float)      # copy: never mutate a cached matrix
        for terms, value in eqs:
            for di, ci in terms:
                _add_load(di, alpha * value * ci)
                for dj, cj in terms:
                    K[di, dj] += alpha * ci * cj
    return K, F


def assemble_stiffness(struc: Structure2D):
    """Build and return the global stiffness matrix as a SciPy sparse (CSR)
    matrix, assembled from COO triplets — O(nnz) memory, so CST meshes with many
    thousands of DOFs stay tractable. The returned matrix behaves like a matrix
    for ``K @ u`` and ``K.diagonal()``; the solver slices and factorises it
    sparsely (see :func:`xdfem2d.solver.solve`)."""
    from scipy import sparse
    ndof = struc.num_dofs
    R: list[int] = []; C: list[int] = []; V: list[float] = []

    # --- Bar elements (plane frame bar or, in the plate domain, grillage) ---
    for elem in struc.bar_elements:
        k_global, cache = _bar_element_matrix(struc, elem)
        struc._elem_cache[elem.id] = cache

        itotv = struc.node_dof_index[elem.node_i]
        jtotv = struc.node_dof_index[elem.node_j]
        dofs = [itotv, itotv+1, itotv+2, jtotv, jtotv+1, jtotv+2]
        for i in range(6):
            for j in range(6):
                R.append(dofs[i]); C.append(dofs[j]); V.append(k_global[i, j])

    # --- Triangular (CST plane) elements: contribute to ux, uy only ---
    _tri_elements_triplets(struc, R, C, V)

    # --- Quadrilateral elements (dev/IMPLEMENT_QUAD.md Phase 4) ---
    _quad_elements_triplets(struc, R, C, V)

    # --- Node springs (diagonal) ---
    for node_id, sp in struc.node_springs.items():
        itotv = struc.node_dof_index[node_id]
        R += [itotv, itotv + 1, itotv + 2]
        C += [itotv, itotv + 1, itotv + 2]
        V += [sp.kx, sp.ky, sp.kt]

    # --- Element foundation springs ---
    _element_springs_triplets(struc, R, C, V)

    # --- Winkler area springs (slab on grade, plate domain) ---
    _area_springs_triplets(struc, R, C, V)

    K = sparse.coo_matrix((V, (R, C)), shape=(ndof, ndof)).tocsr()
    return K


def assemble_stiffness_reduced(struc: Structure2D,
                                beam_factor: float = 1.0,
                                column_factor: float = 1.0) -> np.ndarray:
    """
    Build global stiffness matrix with per-element EI reduction for ANLG.

    beam_factor   — multiplier applied to EI of beam elements   (inclination < 45°)
    column_factor — multiplier applied to EI of column elements (inclination ≥ 45°)

    EA is not reduced (axial stiffness unchanged).
    Node springs and element foundation springs are added unchanged.
    The _elem_cache is updated with reduced EIL/EIL2/EIL3 values so that
    subsequent force-recovery uses the same reduced stiffness.
    """
    ndof = struc.num_dofs
    K = np.zeros((ndof, ndof))

    for elem in struc.bar_elements:
        ni = struc.nodes[elem.node_i]
        nj = struc.nodes[elem.node_j]
        sec = struc.sections[elem.section_name]
        mat = struc.materials[sec.material_name]

        L, cos, sin, _ = bar_element_geometry(ni, nj)
        A = sec.area
        I = sec.inertia

        # Choose reduction factor based on element orientation
        factor = beam_factor if is_beam(cos, sin) else column_factor
        E_eff = mat.elastic_modulus * factor

        k_local, EAL, EIL, EIL2, EIL3 = bar_stiffness_local(E_eff, A, I, L)
        released = released_local_dofs(elem.hinge_i, elem.hinge_j)
        k_local_c = condense_local_stiffness(k_local, released)
        T = transformation_matrix(cos, sin)
        k_global = T.T @ k_local_c @ T

        # Store reduced cache values (used by force recovery in solver)
        struc._elem_cache[elem.id] = dict(
            L=L, cos=cos, sin=sin,
            EAL=EAL, EIL=EIL, EIL2=EIL2, EIL3=EIL3,
            area=A, weight_per_m=A * mat.unit_weight,
            mass_per_m=A * (mat.unit_mass if mat.unit_mass is not None else mat.unit_weight / 9.81),
            k_local=k_local, released=released,
        )

        itotv = struc.node_dof_index[elem.node_i]
        jtotv = struc.node_dof_index[elem.node_j]
        dofs = [itotv, itotv+1, itotv+2, jtotv, jtotv+1, jtotv+2]
        for i in range(6):
            for j in range(6):
                K[dofs[i], dofs[j]] += k_global[i, j]

    # Node springs (unchanged)
    for node_id, sp in struc.node_springs.items():
        itotv = struc.node_dof_index[node_id]
        K[itotv,   itotv  ] += sp.kx
        K[itotv+1, itotv+1] += sp.ky
        K[itotv+2, itotv+2] += sp.kt

    # Element foundation springs (global or local axes)
    _add_element_springs(struc, K)

    # Winkler area springs (slab on grade, plate domain)
    _add_area_springs(struc, K)

    return K
