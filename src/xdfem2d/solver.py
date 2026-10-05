"""
Solver and results post-processing for xdfem2D.
"""
from __future__ import annotations
import numpy as np
from .structure import Structure2D


def _is_sparse(K) -> bool:
    return hasattr(K, "tocsr") and hasattr(K, "diagonal")


def _dense(K) -> np.ndarray:
    """Return *K* as a dense ndarray (identity for arrays, densify for sparse).
    Used by the small-frame advanced paths (modal, spectrum, P-Delta,
    unilateral springs) that operate on dense matrices."""
    return K.toarray() if hasattr(K, "toarray") else np.asarray(K)


def solve(struc: Structure2D, K, F: np.ndarray) -> np.ndarray:
    """
    Solve K·U = F with fixed DOF boundary conditions and support settlements.
    *K* may be dense or a SciPy sparse matrix; the reduced system is factorised
    and solved sparsely when *K* is sparse. Returns U of shape (ndof, n_cases).
    """
    from .assembly import apply_constraint_penalty
    ndof = struc.num_dofs
    n_cases = F.shape[1]
    U = np.zeros((ndof, n_cases))

    # Multi-point constraints (penalty). Applied to local copies of K/F; the
    # caller's matrices and the cached stiffness used for reaction recovery are
    # left untouched.
    K, F = apply_constraint_penalty(struc, K, F)

    fixed = _augment_fixed_zero_stiffness(K, _build_fixed_dofs(struc))
    free  = np.where(~fixed)[0]
    fixed_idx = np.where(fixed)[0]

    # Build prescribed displacement matrix (ndof × n_cases); zero = fully fixed
    U_p = _build_prescribed_displacements(struc, ndof, n_cases)
    # Set prescribed values at fixed DOFs
    U[fixed_idx, :] = U_p[fixed_idx, :]
    if free.size == 0:
        return U

    if _is_sparse(K):
        from scipy.sparse.linalg import splu
        Kc = K.tocsr()
        K_ff = Kc[free][:, free].tocsc()
        K_fp = Kc[free][:, fixed_idx]
        F_mod = np.asarray(F[free, :] - K_fp @ U_p[fixed_idx, :])
        U[free, :] = splu(K_ff).solve(F_mod)
    else:
        K_fp = K[np.ix_(free, fixed_idx)]
        F_mod = F[free, :] - K_fp @ U_p[fixed_idx, :]
        K_ff = K[np.ix_(free, free)]
        U[free, :] = np.linalg.solve(K_ff, F_mod)

    return U


def _constraint_force_node_ids(struc: Structure2D) -> list:
    """Node ids that any enabled constraint couples — the DOFs where the
    constraint force is recovered (method 2: residual of the physical stiffness,
    the same recovery used for support reactions, read at constrained DOFs)."""
    ids: set = set()
    for c in (getattr(struc, 'constraints', {}) or {}).values():
        if not getattr(c, 'enabled', True):
            continue
        if c.kind == 'rigid_link':
            ids.add(c.master); ids.update(c.slaves)
        elif c.kind == 'equal_dof':
            ids.update(c.nodes)
        else:
            ids.update(t[0] for t in c.terms)
    nodes = getattr(struc, 'nodes', {}) or {}
    return [nid for nid in nodes if nid in ids]


def _constraint_forces_from_residual(struc, residual, node_ids) -> dict:
    """{node_id: [fx, fy, mz]} constraint force = residual K·u − F at the
    coupled nodes (near-zero triples dropped)."""
    cf = {}
    for nid in node_ids:
        itotv = struc.node_dof_index[nid]
        v = [float(residual[itotv]), float(residual[itotv + 1]),
             float(residual[itotv + 2])]
        if any(abs(x) > 1e-9 for x in v):
            cf[nid] = v
    return cf


def _build_prescribed_displacements(struc: Structure2D, ndof: int, n_cases: int) -> np.ndarray:
    """Return (ndof × n_cases) matrix of prescribed displacements (zero where not settled)."""
    case_index = {lc.id: i for i, lc in enumerate(struc.load_cases)}
    U_p = np.zeros((ndof, n_cases))
    for ss in struc.support_settlements:
        if ss.load_case_id not in case_index:
            continue
        ic = case_index[ss.load_case_id]
        if ss.node_id not in struc.node_dof_index:
            continue
        itotv = struc.node_dof_index[ss.node_id]
        U_p[itotv,   ic] = ss.ux
        U_p[itotv+1, ic] = ss.uy
        U_p[itotv+2, ic] = ss.tz
    return U_p


# Fraction of the model mass sitting on restrained DOFs above which the modal
# results carry a warning: the effective-mass percentages are relative to the
# free mass, so beyond a few percent they stop being a usable stand-in for the
# EC8 "90 % of the total mass" criterion.
_RESTRAINED_MASS_WARN = 0.02


def _build_fixed_dofs(struc: Structure2D) -> np.ndarray:
    """Return boolean array of shape (ndof,) — True where DOF is fixed."""
    fixed = np.zeros(struc.num_dofs, dtype=bool)
    for assign in struc.support_assignments:
        supp = struc.supports[assign.support_name]
        itotv = struc.node_dof_index[assign.node_id]
        if supp.ux: fixed[itotv]   = True
        if supp.uy: fixed[itotv+1] = True
        if supp.tz: fixed[itotv+2] = True
    return fixed


def _augment_fixed_zero_stiffness(K: np.ndarray, fixed: np.ndarray) -> np.ndarray:
    """
    Return a copy of ``fixed`` with any free DOF that has no stiffness added.

    End releases (hinges) can leave a node rotational DOF with zero stiffness
    (e.g. all members meeting at the node are hinged there and the node is not
    rotationally supported). Such DOFs are physically undetermined; we pin them
    to keep the reduced system non-singular. This never affects well-posed
    DOFs, so non-hinged models are unchanged.
    """
    diag = np.abs(K.diagonal())
    scale = diag.max() if diag.size else 1.0
    tol = max(scale, 1.0) * 1e-12
    dead = (diag <= tol) & (~fixed)
    if not dead.any():
        return fixed
    out = fixed.copy()
    out[dead] = True
    return out


# ---------------------------------------------------------------------------
# Member end-force recovery (hinge-aware)
# ---------------------------------------------------------------------------

def _solve_release_rotations(EIL, EIL2, vl, rz1, rz2, fef, released):
    """
    Recover a hinged member's own end rotations so that the *section* bending
    moment is zero at each released end. ``rz1``/``rz2`` are the node rotations
    (used for any non-released end). Returns (theta_i, theta_j).
    """
    th1, th2 = rz1, rz2
    if released == [2]:               # hinge at i: M_i = 0
        th1 = (EIL2 * vl + fef[2]) / EIL - 0.5 * rz2
    elif released == [5]:             # hinge at j: M_j = 0
        th2 = (EIL2 * vl - fef[5]) / EIL - 0.5 * rz1
    elif released == [2, 5]:          # both ends hinged: M_i = M_j = 0
        a = EIL
        A = np.array([[-a, -0.5 * a], [0.5 * a, a]])
        b = np.array([-(EIL2 * vl + fef[2]), (EIL2 * vl - fef[5])])
        th1, th2 = np.linalg.solve(A, b)
    return th1, th2


def _member_end_forces(cache, ui, uj, fef):
    """
    Local member end forces in the program's section convention, hinge-aware.

    ui = (ux_i, uy_i, rz_i) and uj = (ux_j, uy_j, rz_j) are the *node*
    displacements; fef is the 6-vector of (un-condensed) fixed-end forces.
    For released ends the member end rotation is recovered so the moment is zero.
    Returns ([N_i, V_i, M_i], [N_j, V_j, M_j]).
    """
    cos, sin = cache['cos'], cache['sin']
    EAL, EIL, EIL2, EIL3 = cache['EAL'], cache['EIL'], cache['EIL2'], cache['EIL3']
    released = cache.get('released') or []
    ux1, uy1, rz1 = ui
    ux2, uy2, rz2 = uj
    dx = ux2 - ux1
    dy = uy2 - uy1
    ul =  dx * cos + dy * sin
    vl = -dx * sin + dy * cos
    th1, th2 = rz1, rz2
    if released:
        th1, th2 = _solve_release_rotations(EIL, EIL2, vl, rz1, rz2, fef, released)
    nn1 =  EAL * ul;                       nn2 = -nn1
    vv1 =  EIL2 * (th1 + th2) - EIL3 * vl; vv2 = -vv1
    mm1 =  EIL2 * vl - EIL * (th1 + 0.5 * th2)
    mm2 = -EIL2 * vl + EIL * (th2 + 0.5 * th1)
    return ([nn1 + fef[0], vv1 + fef[1], mm1 + fef[2]],
            [nn2 + fef[3], vv2 + fef[4], mm2 + fef[5]])


def _grid_member_end_forces(cache, ui, uj, fef):
    """Local member end forces for a grillage bar (plate domain), hinge-aware.

    ui = (w_i, tx_i, ty_i) and uj = (w_j, tx_j, ty_j) are the *node*
    displacements in global components. Internally the recovery runs in the
    plane beam's (v, θ) algebra with v ↔ w and θ ↔ θ̃ = dw/dx' = −ψ, so the
    coefficients cached by assembly (EAL = G·J/L, EIL…) and the hinge
    recovery (:func:`_solve_release_rotations`) are shared unchanged.
    Returns ([T_i, V_i, M_i], [T_j, V_j, M_j]) — torsion in the slot the
    plane bar uses for the axial force, so every consumer that superposes,
    envelopes or tabulates positionally keeps working.
    """
    cos, sin = cache['cos'], cache['sin']
    GJL, EIL, EIL2, EIL3 = cache['EAL'], cache['EIL'], cache['EIL2'], cache['EIL3']
    released = cache.get('released') or []
    w1, tx1, ty1 = ui
    w2, tx2, ty2 = uj
    # Global rotations → local: φ about x' = (c, s), ψ about y' = (−s, c).
    phi1 =  tx1 * cos + ty1 * sin
    phi2 =  tx2 * cos + ty2 * sin
    th1  = -(-tx1 * sin + ty1 * cos)      # θ̃ = −ψ
    th2  = -(-tx2 * sin + ty2 * cos)
    vl = w2 - w1                          # relative transverse deflection
    tw = phi2 - phi1                      # relative twist
    if released:
        th1, th2 = _solve_release_rotations(EIL, EIL2, vl, th1, th2, fef,
                                            released)
    nn1 =  GJL * tw;                       nn2 = -nn1
    vv1 =  EIL2 * (th1 + th2) - EIL3 * vl; vv2 = -vv1
    mm1 =  EIL2 * vl - EIL * (th1 + 0.5 * th2)
    mm2 = -EIL2 * vl + EIL * (th2 + 0.5 * th1)
    return ([nn1 + fef[0], vv1 + fef[1], mm1 + fef[2]],
            [nn2 + fef[3], vv2 + fef[4], mm2 + fef[5]])


def _end_forces_dispatch(struc, cache, ui, uj, fef):
    """Member end forces through the domain's recovery — grillage (T, V, M)
    in the plate domain, frame (N, V, M) otherwise."""
    if cache.get('grid'):
        return _grid_member_end_forces(cache, ui, uj, fef)
    return _member_end_forces(cache, ui, uj, fef)


# ---------------------------------------------------------------------------
# Results post-processing
# ---------------------------------------------------------------------------

def compute_results(struc: Structure2D, K: np.ndarray, F: np.ndarray, U: np.ndarray) -> dict:
    """
    Compute and return a results dict:
      {
        'displacements': {case_id: {node_id: [ux, uy, rz]}},
        'reactions':     {case_id: {node_id: [rx, ry, mz]}},
        'element_forces':{case_id: {elem_id: {'i': [N,V,M], 'j': [N,V,M]}}},
        'spring_forces': {case_id: {node_spring_name: [rx,ry,mz]}},
        'combinations':  { same structure as above for each combo id }
      }
    """
    case_index = {lc.id: i for i, lc in enumerate(struc.load_cases)}
    results = {
        'displacements':  {},
        'reactions':      {},
        'element_forces': {},
        'spring_forces':  {},
        'constraint_forces': {},
        'tri_stress':     {},
        'combinations':   {},
    }

    fixed = _build_fixed_dofs(struc)
    con_nodes = _constraint_force_node_ids(struc)

    # ---- Per load case ----
    for lc in struc.load_cases:
        ic = case_index[lc.id]
        u = U[:, ic]

        # Displacements
        disp = {}
        for node in struc.nodes.values():
            itotv = struc.node_dof_index[node.id]
            disp[node.id] = [u[itotv], u[itotv+1], u[itotv+2]]
        results['displacements'][lc.id] = disp

        # Reactions = K·U - F_applied at fixed DOFs
        # (equilibrium: K·U = F_ext + R  →  R = K·U - F_ext at constrained DOFs)
        residual = K @ u - F[:, ic]
        reac = {}
        for node in struc.nodes.values():
            itotv = struc.node_dof_index[node.id]
            rx = residual[itotv]   if fixed[itotv]   else 0.0
            ry = residual[itotv+1] if fixed[itotv+1] else 0.0
            mz = residual[itotv+2] if fixed[itotv+2] else 0.0
            if abs(rx) > 1e-9 or abs(ry) > 1e-9 or abs(mz) > 1e-9:
                reac[node.id] = [rx, ry, mz]
        results['reactions'][lc.id] = reac

        # Constraint forces (method 2): the same residual read at the DOFs a
        # constraint couples is the force the constraint applies there.
        results['constraint_forces'][lc.id] = _constraint_forces_from_residual(
            struc, residual, con_nodes)

        # Element end forces
        elem_forces = {}
        for elem in struc.bar_elements:
            cache = struc._elem_cache[elem.id]
            fef   = struc._fixed_end_forces[lc.id][elem.id]

            itotv = struc.node_dof_index[elem.node_i]
            jtotv = struc.node_dof_index[elem.node_j]

            si, sj = _end_forces_dispatch(
                struc, cache,
                (u[itotv], u[itotv+1], u[itotv+2]),
                (u[jtotv], u[jtotv+1], u[jtotv+2]),
                fef,
            )
            elem_forces[elem.id] = {'i': si, 'j': sj}

        results['element_forces'][lc.id] = elem_forces

        # Node spring forces
        spring_f = {}
        for node_id, sp in struc.node_springs.items():
            itotv = struc.node_dof_index[node_id]
            spring_f[node_id] = [
                u[itotv]   * sp.kx,
                u[itotv+1] * sp.ky,
                u[itotv+2] * sp.kt,
            ]
        results['spring_forces'][lc.id] = spring_f

        # Surface-element stresses/moments (triangle: CST/Allman/ES-FEM/DKT/
        # MITC3; quad since dev/IMPLEMENT_QUAD.md Phase 5: Q4/QM6/DKT4/MITC4
        # — tri_stresses_dispatch merges both, still under 'tri_stress', see
        # its docstring). The thermal strain of this case is removed so a
        # freely expanding triangle reports zero, not the stress of a
        # restrained one (quads have no thermal-load feature yet, so this is
        # a no-op for them).
        if getattr(struc, "tri_elements", []) or getattr(struc, "quad_elements", []):
            from .tri_elements import tri_stresses_dispatch
            from .loads import thermal_strain_field
            th = thermal_strain_field(struc, {lc.id: 1.0})
            results['tri_stress'][lc.id] = tri_stresses_dispatch(
                struc, u, thermal=th)

    # ---- Diagram distributions for load cases (N/V/M along span) ----
    # Must run BEFORE analysis cases so _linear_superpose can superpose them.
    compute_distributions(struc, results)

    # ---- Analysis case results (Linear, Mass, Modal, Spectrum) ----
    # Must run BEFORE combinations so combos can reference analysis case results.
    compute_analysis_case_results(struc, results, K, F, case_index)

    # ---- Analysis-case / combination diagrams ----
    # Runs after the analysis-case solve so NonLinear cases use their converged
    # combined element forces (see compute_combo_distributions).
    compute_combo_distributions(struc, results)

    # ---- Load combinations (may reference analysis cases and other
    # combinations) ----
    # Process in dependency order so a combination used as an input unit is
    # already computed when an envelope (or other parent) references it.
    for combo in _combinations_in_dependency_order(struc):
        results['combinations'][combo.id] = _apply_combination(
            combo, struc, results)

    # ---- Plane-element stresses for analysis cases and combinations ----
    _recover_object_stresses(struc, results)

    return results


def _recover_object_stresses(struc, results):
    """Fill in the surface-element stresses/moments for every analysis case
    and combination.

    Stress is linear in the displacement field, so it is recovered from each
    case's stored displacements (skipping envelope max/min shapes). Split out so
    it can be re-run after an incremental solve adds a case or a combination.
    Covers CST, Allman, ES-FEM, DKT, MITC3 and (dev/IMPLEMENT_QUAD.md Phase 5)
    every quad formulation.
    """
    if not (getattr(struc, "tri_elements", [])
            or getattr(struc, "quad_elements", [])):
        return
    from .tri_elements import tri_stresses_from_disp_dispatch
    from .loads import thermal_strain_field, resolve_to_load_cases
    # Stress is linear, so the thermal term of a combined field is the same
    # combination of the per-case thermal strains. A Linear analysis case
    # carries load-case coefficients directly; a combination is expanded to
    # analysis cases and then folded down to load cases. Envelope shapes (those
    # with 'max') are skipped here as they are for displacement.
    for ac_id, ac in results.get('analysis_cases', {}).items():
        disp = ac.get('displacements')
        if isinstance(disp, dict) and 'max' not in disp:
            src = struc.analysis_cases_by_id.get(ac_id)
            coeffs = resolve_to_load_cases(
                struc, getattr(src, 'coefficients', {}) or {}) if src else {}
            th = thermal_strain_field(struc, coeffs)
            ac['tri_stress'] = tri_stresses_from_disp_dispatch(
                struc, disp, thermal=th)
    combos_by_id = {c.id: c for c in struc.load_combinations}
    for cid, cr in results.get('combinations', {}).items():
        disp = cr.get('displacements')
        if isinstance(disp, dict) and 'max' not in disp:
            th = {}
            combo = combos_by_id.get(cid)
            if combo is not None:
                try:
                    eff = struc.expand_combination_coefficients(combo)
                except ValueError:
                    eff = {}
                th = thermal_strain_field(
                    struc, resolve_to_load_cases(struc, eff))
            cr['tri_stress'] = tri_stresses_from_disp_dispatch(
                struc, disp, thermal=th)


def _combinations_in_dependency_order(struc):
    """Return the load combinations ordered so that any combination referenced
    by another (as an input unit) comes first. Cycles — already rejected at
    definition time — are guarded against here too."""
    by_id = {c.id: c for c in struc.load_combinations}
    ordered, state = [], {}   # state: 1 = visiting, 2 = done

    def _visit(c):
        if state.get(c.id) == 2:
            return
        if state.get(c.id) == 1:      # cycle guard (should not happen)
            return
        state[c.id] = 1
        for key in c.coefficients:
            sub = by_id.get(key)
            if sub is not None:
                _visit(sub)
        state[c.id] = 2
        ordered.append(c)

    for c in struc.load_combinations:
        _visit(c)
    return ordered


def compute_distributions(struc: Structure2D, results: dict, n_points: int = 51):
    """
    For each load case, compute N(x), V(x), M(x) along every element using
    equilibrium from the i-end, incorporating the distributed loads.
    Adds 'element_distribution' to results in-place. Analysis-case and
    combination diagrams are built afterwards by ``compute_combo_distributions``.
    """
    case_dists: dict = {}

    # Per-element sampling grid: a uniform grid plus the positions of any
    # element point loads (and points just before them) so the M/V steps and
    # peaks are captured exactly. The same grid is reused for every load case,
    # analysis case and combination of that element, keeping superposition
    # array-aligned.
    elem_xs: dict = {}
    for elem in struc.bar_elements:
        L = struc._elem_cache[elem.id]['L']
        pts = list(np.linspace(0.0, L, n_points))
        for epl in getattr(struc, 'element_point_loads', []):
            if epl.element_id != elem.id:
                continue
            a = min(max(epl.a, 0.0), L)
            pts.append(a)
            pts.append(max(0.0, a - 1e-6))   # capture the jump on both sides
        elem_xs[elem.id] = np.unique(np.array(pts, dtype=float))

    for lc in struc.load_cases:
        lc_dist: dict = {}
        for elem in struc.bar_elements:
            ef = results['element_forces'][lc.id][elem.id]
            cache = struc._elem_cache[elem.id]
            L   = cache['L']
            cos = cache['cos']
            sin = cache['sin']

            # Accumulate local distributed loads (trapezoidal)
            qx0 = qx1 = qy0 = qy1 = 0.0
            grid = bool(cache.get('grid'))

            for dl in struc.distributed_loads:
                if dl.element_id != elem.id or dl.load_case_id != lc.id:
                    continue
                if grid:
                    # Plate domain: the transverse intensity (fy* slots) is
                    # along global Z whatever coord_sys says; the axial slots
                    # have no out-of-plane meaning (matches the load vector).
                    qy0 += dl.fye;  qy1 += dl.fyd
                elif dl.coord_sys == 'local':
                    qx0 += dl.fxe;  qx1 += dl.fxd
                    qy0 += dl.fye;  qy1 += dl.fyd
                else:
                    # Global → local: x'= cos,sin  y'= -sin,cos
                    qx0 += dl.fxe * cos + dl.fye * sin
                    qx1 += dl.fxd * cos + dl.fyd * sin
                    qy0 += -dl.fxe * sin + dl.fye * cos
                    qy1 += -dl.fxd * sin + dl.fyd * cos

            # Self-weight: global qy = -w, global qx = 0 (plate: straight
            # onto the transverse direction, no projection — gravity is −Z)
            if lc.self_weight_factor != 0.0:
                w = cache['weight_per_m'] * lc.self_weight_factor
                if grid:
                    qy0 += -w;  qy1 += -w
                else:
                    qx0 += -w * sin;  qx1 += -w * sin
                    qy0 += -w * cos;  qy1 += -w * cos

            N_i, V_i, M_i = ef['i']
            xs = elem_xs[elem.id]

            dqy = qy1 - qy0
            dqx = qx1 - qx0

            # Axial: N(x) = N_i − ∫qx (the i-end member axial sign convention is
            # opposite to the transverse one, so the distributed-axial term is
            # subtracted — otherwise N grows away from a free end instead of
            # decaying to zero, most visibly on vertical members under gravity).
            Ns = N_i - qx0 * xs - dqx * xs**2 / (2.0 * L)
            Vs = V_i + qy0 * xs + dqy * xs**2 / (2.0 * L)
            Ms = M_i + V_i * xs + qy0 * xs**2 / 2.0 + dqy * xs**3 / (6.0 * L)

            # Element point loads: add the step discontinuities for x >= a.
            for epl in getattr(struc, 'element_point_loads', []):
                if epl.element_id != elem.id or epl.load_case_id != lc.id:
                    continue
                a = min(max(epl.a, 0.0), L)
                if grid:
                    px, py, m0 = 0.0, epl.fy, epl.mz
                elif epl.coord_sys == 'local':
                    px, py, m0 = epl.fx, epl.fy, epl.mz
                else:
                    px =  epl.fx * cos + epl.fy * sin
                    py = -epl.fx * sin + epl.fy * cos
                    m0 =  epl.mz
                mask = xs >= a
                Ns[mask] -= px          # same axial sign convention as above
                Vs[mask] += py
                Ms[mask] += py * (xs[mask] - a) + m0

            lc_dist[elem.id] = {'x': xs, 'N': Ns, 'V': Vs, 'M': Ms}
        case_dists[lc.id] = lc_dist

    results['element_distribution'] = case_dists
    # Keep the per-element sampling grid for the analysis-case / combination
    # distribution build (which runs after the analysis-case solve).
    results['_elem_grid'] = elem_xs


def compute_combo_distributions(struc: Structure2D, results: dict):
    """Build analysis-case and combination N/V/M diagrams.

    Runs *after* ``compute_analysis_case_results`` so NonLinear cases can use
    their converged combined element forces. Linear (and NonLinear-without-
    conditional-springs) cases are the linear superposition of the load-case
    diagrams; for a NonLinear case the superposed diagram is corrected so its
    boundary (end) forces match the true non-linear combined solution.
    """
    elem_xs = results.get('_elem_grid', {})
    case_dists = results.get('element_distribution', {})
    ac_solved  = results.get('analysis_cases', {})

    # Analysis-case distributions: linear superposition of the load-case
    # diagrams (only Linear / NonLinear cases have simple N/V/M diagrams; modal /
    # spectrum / P-Delta cases are skipped for diagram purposes).
    ac_dists: dict = {}
    for ac in getattr(struc, 'analysis_cases', []):
        if ac.analysis_type not in ('Linear', 'NonLinear'):
            continue
        a_dist: dict = {}
        for elem in struc.bar_elements:
            xs = elem_xs.get(elem.id)
            if xs is None:
                continue
            npts = len(xs)
            Ns = np.zeros(npts); Vs = np.zeros(npts); Ms = np.zeros(npts)
            for lc_id, coeff in ac.coefficients.items():
                d = case_dists.get(lc_id, {}).get(elem.id)
                if d is None:
                    continue
                Ns += coeff * d['N']; Vs += coeff * d['V']; Ms += coeff * d['M']

            # For a non-linear solve, the superposed end forces are not the true
            # combined end forces. Correct the diagram so its boundary values
            # match the converged solution (linear in end forces + member loads,
            # so a per-end shift is exact). No-op for linear cases.
            res = ac_solved.get(ac.id, {})
            if ac.analysis_type == 'NonLinear' and 'element_forces' in res:
                ef_i = res['element_forces'].get(elem.id, {}).get('i')
                if ef_i is not None and npts:
                    dN = ef_i[0] - Ns[0]
                    dV = ef_i[1] - Vs[0]
                    dM = ef_i[2] - Ms[0]
                    Ns = Ns + dN
                    Vs = Vs + dV
                    Ms = Ms + dM + dV * xs
            a_dist[elem.id] = {'x': xs, 'N': Ns, 'V': Vs, 'M': Ms}
        ac_dists[ac.id] = a_dist

    # Combinations — combine the analysis-case diagrams (combos reference
    # analysis cases). Spectrum cases have no simple diagram and are skipped.
    combo_dists: dict = {}
    # Process combinations in dependency order so any referenced sub-combination
    # is already built when an envelope combines it as an input unit.
    for combo in _combinations_in_dependency_order(struc):
        # NonLinearCombo and SequenceCombo are pass-throughs of a single case
        # with factor 1.0 → treat as linear superposition.
        ctype = ('LinearSum' if combo.combo_type in ('NonLinearCombo', 'SequenceCombo')
                 else combo.combo_type)
        # LinearSum combinations flatten any referenced sub-combinations to
        # analysis cases (exact). Non-linear combinations keep each referenced
        # LinearSum combination as a single input unit (looked up from the
        # already-built combo diagrams).
        if ctype == 'LinearSum':
            eff_coeffs = struc.expand_combination_coefficients(combo)
        else:
            eff_coeffs = dict(combo.coefficients)
        c_dist: dict = {}
        for elem in struc.bar_elements:
            xs = elem_xs[elem.id]
            npts = len(xs)
            if ctype == 'Envelope':
                # True component-wise max/min envelope of the input diagrams.
                # Each input contributes a band [lo, hi] (a single curve for a
                # plain input, or its own max/min band for a nested envelope),
                # scaled by coeff.
                # Envelope over the actual inputs only (start unset, not at 0,
                # so a single-signed band does not gain a spurious zero bound).
                hi = {'N': None, 'V': None, 'M': None}
                lo = {'N': None, 'V': None, 'M': None}
                for case_id, coeff in eff_coeffs.items():
                    src = ac_dists.get(case_id) or combo_dists.get(case_id)
                    if src is None:
                        continue
                    d = src.get(elem.id)
                    if d is None:
                        continue
                    for comp in ('N', 'V', 'M'):
                        d_hi = d[comp]
                        d_lo = d.get(comp + '_min', d_hi)  # nested envelope band
                        a, b = coeff * d_hi, coeff * d_lo
                        c_hi = np.maximum(a, b)
                        c_lo = np.minimum(a, b)
                        if hi[comp] is None:
                            hi[comp], lo[comp] = c_hi, c_lo
                        else:
                            hi[comp] = np.maximum(hi[comp], c_hi)
                            lo[comp] = np.minimum(lo[comp], c_lo)
                for comp in ('N', 'V', 'M'):
                    if hi[comp] is None:
                        hi[comp] = np.zeros(npts)
                        lo[comp] = np.zeros(npts)
                c_dist[elem.id] = {
                    'x': xs,
                    'N': hi['N'], 'V': hi['V'], 'M': hi['M'],
                    'N_min': lo['N'], 'V_min': lo['V'], 'M_min': lo['M'],
                    'is_envelope': True,
                }
                continue
            Ns = np.zeros(npts)
            Vs = np.zeros(npts)
            Ms = np.zeros(npts)
            for case_id, coeff in eff_coeffs.items():
                src = ac_dists.get(case_id) or combo_dists.get(case_id)
                if src is None:
                    continue
                d = src.get(elem.id)
                if d is None:
                    continue
                if ctype == 'LinearSum':
                    Ns += coeff * d['N']
                    Vs += coeff * d['V']
                    Ms += coeff * d['M']
                elif ctype == 'AbsSum':
                    Ns += abs(coeff) * np.abs(d['N'])
                    Vs += abs(coeff) * np.abs(d['V'])
                    Ms += abs(coeff) * np.abs(d['M'])
                elif ctype == 'SRSS':
                    Ns += (coeff * d['N']) ** 2
                    Vs += (coeff * d['V']) ** 2
                    Ms += (coeff * d['M']) ** 2
            if ctype == 'SRSS':
                Ns = np.sqrt(Ns)
                Vs = np.sqrt(Vs)
                Ms = np.sqrt(Ms)
            c_dist[elem.id] = {'x': xs, 'N': Ns, 'V': Vs, 'M': Ms}
        combo_dists[combo.id] = c_dist

    results['combo_distribution'] = combo_dists


class CombinationInputError(ValueError):
    """A load combination references an input with no analysis-case (or
    combination) result to resolve it — never silently satisfied by falling
    back to a raw load case."""


def _get_lc_result(results: dict, case_id: str) -> dict | None:
    """Return a unified result dict for an analysis case, or a (non-spectrum)
    combination used as an input. Returns ``None`` if *case_id* resolves to
    neither — the caller must treat that as an error in the combination's
    definition, not as a zero contribution.

    Combinations must always resolve their inputs through analysis cases —
    an analysis case already carries its own coefficient (e.g. gamma_G)
    applied to the underlying load case, which is exactly what
    ``combo.coefficients`` (keyed by analysis-case id, see
    ``expand_combination_coefficients``) expects to consume. A raw load case
    is NEVER a valid combination input on its own, even when no analysis
    case happens to exist for that id — every load case has a "twin"
    analysis case sharing its id (see ``Structure2D.add_load_combination``),
    and referencing that twin is the only way a load case enters a
    combination. Falling back to the raw (unfactored) load-case dict here
    used to be a bug: an analysis case's id commonly coincides with the id
    of the load case it wraps, so an id lookup against
    ``results['displacements']`` would silently succeed and return the
    *unfactored* load case result instead of the factored analysis case
    result whenever the ids matched — combinations quietly dropped every
    analysis-case coefficient (e.g. gamma_G = 1.35 on a permanent action).
    """
    # Analysis case (the only direct-case input a combination may use —
    # already has its own coefficient applied).
    ac = results.get('analysis_cases', {}).get(case_id, {})
    if ac and 'displacements' in ac:
        return ac
    # Combination used as an input unit (only LinearSum combinations are
    # allowed as inputs, so the result has the flat node->[..] shape).
    combo_res = results.get('combinations', {}).get(case_id)
    if combo_res and 'displacements' in combo_res \
            and 'max' not in combo_res['displacements']:
        return {
            'displacements':  combo_res['displacements'],
            'reactions':      combo_res.get('reactions', {}),
            'element_forces': combo_res.get('element_forces', {}),
            'spring_forces':  combo_res.get('spring_forces', {}),
            'is_spectrum':    False,
        }
    return None


def _get_input_result(results: dict, case_id: str):
    """Result dict for an input to an Envelope combination: an analysis
    case, or *any* combination (flat or max/min shaped). Returns ``None`` if
    *case_id* resolves to neither (never falls back to a raw load case —
    see ``_get_lc_result``)."""
    r = _get_lc_result(results, case_id)
    if r is not None:
        return r
    # Envelope / seismic combinations have the max/min shape that _get_lc_result
    # deliberately rejects; pick them up here so envelopes can nest.
    return results.get('combinations', {}).get(case_id)


def _range_at(res: dict, quantity: str, ident: str, k: int, end=None):
    """Return ``(lo, hi)`` for component *k* of *quantity* at *ident* in *res*,
    handling both flat (``lo == hi``) and max/min shaped results."""
    q = res.get(quantity)
    if not q:
        return 0.0, 0.0
    if 'max' in q and 'min' in q:
        hi_map, lo_map = q['max'], q['min']
    else:
        hi_map = lo_map = q

    def _val(m):
        v = m.get(ident)
        if v is None:
            return 0.0
        if end is not None:
            v = v.get(end, [0.0, 0.0, 0.0])
        return v[k]

    return _val(lo_map), _val(hi_map)


def _apply_combination(combo, struc, results: dict) -> dict:
    """
    Apply a load combination and return result dict.

    `combo.coefficients` is keyed by analysis case id. The combo_type rule is
    applied to the non-spectrum cases:
      LinearSum — Σ coeff_i * X_i
      Envelope — max/min envelope over weighted cases
      AbsSum — Σ |coeff_i * X_i|
      SRSS   — √(Σ (coeff_i * X_i)²)

    If any Spectrum analysis case is referenced, the result becomes a seismic
    envelope:  gravity_base ± Σ |coeff_s * spectrum_s|.
    """
    import math

    # NonLinearCombo and SequenceCombo each wrap a single case with factor 1.0;
    # both are pass-throughs handled exactly like a LinearSum superposition.
    ctype = ('LinearSum' if combo.combo_type in ('NonLinearCombo', 'SequenceCombo')
             else combo.combo_type)
    node_ids     = list(struc.nodes.keys())
    elem_ids     = [e.id for e in struc.bar_elements]
    spring_names = list(struc.node_springs.keys())

    ac_results = results.get('analysis_cases', {})

    # Combinations reference analysis cases (combo.coefficients). Split them into
    # spectrum cases (-> seismic envelope) and the rest (-> the combo_type rule).
    def _is_spec(cid):
        return ac_results.get(cid, {}).get('is_spectrum', False)
    # LinearSum combinations flatten any referenced sub-combinations to
    # analysis cases (exact). Non-linear combinations keep each referenced
    # LinearSum combination as a single input unit, resolved from the
    # already-computed combination results (see _get_lc_result).
    if ctype == 'LinearSum':
        eff_coeffs = struc.expand_combination_coefficients(combo)
    else:
        eff_coeffs = dict(combo.coefficients)
    grav = {k: v for k, v in eff_coeffs.items() if not _is_spec(k)}
    spec = {k: v for k, v in eff_coeffs.items() if _is_spec(k)}
    has_spectrum = bool(spec)

    # ── Envelope (pure envelope over the non-spectrum cases) ──────────────
    if ctype == 'Envelope' and not has_spectrum:
        # Initialise to ±inf so the envelope is taken over the *actual* inputs
        # only (no spurious zero baseline); untouched entries are reset to 0.
        _NINF, _PINF = float('-inf'), float('inf')
        env_max_d  = {nid: [_NINF]*3 for nid in node_ids}
        env_min_d  = {nid: [_PINF]*3 for nid in node_ids}
        env_max_r  = {nid: [_NINF]*3 for nid in node_ids}
        env_min_r  = {nid: [_PINF]*3 for nid in node_ids}
        env_max_ef = {eid: {'i':[_NINF]*3,'j':[_NINF]*3} for eid in elem_ids}
        env_min_ef = {eid: {'i':[_PINF]*3,'j':[_PINF]*3} for eid in elem_ids}
        env_max_sp = {sn: [_NINF]*3 for sn in spring_names}
        env_min_sp = {sn: [_PINF]*3 for sn in spring_names}
        env_con_ids = _constraint_force_node_ids(struc)
        env_max_cf = {nid: [_NINF]*3 for nid in env_con_ids}
        env_min_cf = {nid: [_PINF]*3 for nid in env_con_ids}

        for case_id, coeff in grav.items():
            res = _get_input_result(results, case_id)
            if res is None:
                raise CombinationInputError(
                    f"Combination '{combo.id}' references '{case_id}', which "
                    "is not a solved analysis case or combination. A load "
                    "case is never a valid combination input on its own — "
                    "every load case needs its analysis-case 'twin' "
                    "referenced instead.")
            # Each input contributes a range [lo, hi] (a single value for a flat
            # result, a max/min band for an envelope/seismic sub-combination)
            # scaled by coeff; fold the scaled band into the running envelope.
            def _fold(env_max, env_min, quantity, ident, end=None):
                for k in range(3):
                    lo, hi = _range_at(res, quantity, ident, k, end=end)
                    a, b = coeff * lo, coeff * hi
                    cmax, cmin = (a, b) if a >= b else (b, a)
                    tgt_max = env_max[ident] if end is None else env_max[ident][end]
                    tgt_min = env_min[ident] if end is None else env_min[ident][end]
                    tgt_max[k] = max(tgt_max[k], cmax)
                    tgt_min[k] = min(tgt_min[k], cmin)
            for nid in node_ids:
                _fold(env_max_d, env_min_d, 'displacements', nid)
                _fold(env_max_r, env_min_r, 'reactions', nid)
            for eid in elem_ids:
                for end in ('i', 'j'):
                    _fold(env_max_ef, env_min_ef, 'element_forces', eid, end=end)
            for sn in spring_names:
                _fold(env_max_sp, env_min_sp, 'spring_forces', sn)
            for nid in env_con_ids:
                _fold(env_max_cf, env_min_cf, 'constraint_forces', nid)

        # Reset any entry never touched by an input (still ±inf) back to 0.
        def _san_vec(v):
            return [0.0 if (x == _PINF or x == _NINF) else x for x in v]
        for mp in (env_max_d, env_min_d, env_max_r, env_min_r,
                   env_max_sp, env_min_sp, env_max_cf, env_min_cf):
            for ident in mp:
                mp[ident] = _san_vec(mp[ident])
        for mp in (env_max_ef, env_min_ef):
            for ident in mp:
                mp[ident] = {'i': _san_vec(mp[ident]['i']),
                             'j': _san_vec(mp[ident]['j'])}

        return {
            'displacements':  {'max': env_max_d,  'min': env_min_d},
            'reactions':      {'max': env_max_r,  'min': env_min_r},
            'element_forces': {'max': env_max_ef, 'min': env_min_ef},
            'spring_forces':  {'max': env_max_sp, 'min': env_min_sp},
            'constraint_forces': {'max': env_max_cf, 'min': env_min_cf},
        }

    # ── LinearSum / AbsSum / SRSS: accumulate gravity base ──────────
    con_ids  = _constraint_force_node_ids(struc)
    disp_c   = {nid: [0.0]*3 for nid in node_ids}
    reac_c   = {nid: [0.0]*3 for nid in node_ids}
    ef_c     = {eid: {'i':[0.0]*3,'j':[0.0]*3} for eid in elem_ids}
    spring_c = {sn: [0.0]*3 for sn in spring_names}
    conf_c   = {nid: [0.0]*3 for nid in con_ids}

    # Non-spectrum analysis cases (using selected combo_type)
    for case_id, coeff in grav.items():
        res = _get_lc_result(results, case_id)
        if res is None:
            raise CombinationInputError(
                f"Combination '{combo.id}' references '{case_id}', which is "
                "not a solved analysis case or combination. A load case is "
                "never a valid combination input on its own — every load "
                "case needs its analysis-case 'twin' referenced instead.")
        for nid in node_ids:
            d = res['displacements'].get(nid, [0.0]*3)
            r = res['reactions'].get(nid, [0.0]*3)
            for k in range(3):
                cv = coeff * d[k]; rv = coeff * r[k]
                if ctype == 'LinearSum':
                    disp_c[nid][k] += cv;   reac_c[nid][k] += rv
                elif ctype == 'AbsSum':
                    disp_c[nid][k] += abs(cv); reac_c[nid][k] += abs(rv)
                elif ctype == 'SRSS':
                    disp_c[nid][k] += cv*cv;  reac_c[nid][k] += rv*rv
        cf_in = res.get('constraint_forces', {})
        for nid in con_ids:
            fv = cf_in.get(nid, [0.0]*3)
            for k in range(3):
                cv = coeff * fv[k]
                if ctype == 'LinearSum':    conf_c[nid][k] += cv
                elif ctype == 'AbsSum':  conf_c[nid][k] += abs(cv)
                elif ctype == 'SRSS':    conf_c[nid][k] += cv*cv
        for eid in elem_ids:
            ef = res['element_forces'].get(eid, {'i':[0.0]*3,'j':[0.0]*3})
            for end in ('i','j'):
                for k in range(3):
                    cv = coeff * ef[end][k]
                    if ctype == 'LinearSum':    ef_c[eid][end][k] += cv
                    elif ctype == 'AbsSum':  ef_c[eid][end][k] += abs(cv)
                    elif ctype == 'SRSS':    ef_c[eid][end][k] += cv*cv
        for sn in spring_names:
            sf = res.get('spring_forces', {}).get(sn, [0.0]*3)
            for k in range(3):
                cv = coeff * sf[k]
                if ctype == 'LinearSum':    spring_c[sn][k] += cv
                elif ctype == 'AbsSum':  spring_c[sn][k] += abs(cv)
                elif ctype == 'SRSS':    spring_c[sn][k] += cv*cv

    if ctype == 'SRSS':
        for nid in node_ids:
            disp_c[nid] = [math.sqrt(v) for v in disp_c[nid]]
            reac_c[nid] = [math.sqrt(v) for v in reac_c[nid]]
        for eid in elem_ids:
            for end in ('i','j'):
                ef_c[eid][end] = [math.sqrt(v) for v in ef_c[eid][end]]
        for sn in spring_names:
            spring_c[sn] = [math.sqrt(v) for v in spring_c[sn]]
        for nid in con_ids:
            conf_c[nid] = [math.sqrt(v) for v in conf_c[nid]]

    if not has_spectrum:
        return {
            'displacements':  disp_c,
            'reactions':      reac_c,
            'element_forces': ef_c,
            'spring_forces':  spring_c,
            'constraint_forces': conf_c,
        }

    # ── Seismic envelope: gravity_base ± Σ |coeff_s × spectrum_s| ───────────
    # Start from accumulated gravity base, then add/subtract each spectrum.
    env_max_d  = {nid: list(v) for nid, v in disp_c.items()}
    env_min_d  = {nid: list(v) for nid, v in disp_c.items()}
    env_max_r  = {nid: list(v) for nid, v in reac_c.items()}
    env_min_r  = {nid: list(v) for nid, v in reac_c.items()}
    env_max_ef = {eid: {'i': list(ef_c[eid]['i']), 'j': list(ef_c[eid]['j'])}
                  for eid in elem_ids}
    env_min_ef = {eid: {'i': list(ef_c[eid]['i']), 'j': list(ef_c[eid]['j'])}
                  for eid in elem_ids}
    env_max_cf = {nid: list(v) for nid, v in conf_c.items()}
    env_min_cf = {nid: list(v) for nid, v in conf_c.items()}

    for ac_id, coeff in spec.items():
        ac_res = ac_results.get(ac_id, {})
        if not ac_res.get('is_spectrum', False):
            continue
        for nid in node_ids:
            d = ac_res['displacements'].get(nid, [0.0]*3)
            r = ac_res['reactions'].get(nid, [0.0]*3)
            for k in range(3):
                delta_d = coeff * abs(d[k])
                delta_r = coeff * abs(r[k])
                env_max_d[nid][k] += delta_d;  env_min_d[nid][k] -= delta_d
                env_max_r[nid][k] += delta_r;  env_min_r[nid][k] -= delta_r
        cf_s = ac_res.get('constraint_forces', {})
        for nid in conf_c:
            fv = cf_s.get(nid, [0.0]*3)
            for k in range(3):
                delta_c = coeff * abs(fv[k])
                env_max_cf[nid][k] += delta_c;  env_min_cf[nid][k] -= delta_c
        for eid in elem_ids:
            ef = ac_res['element_forces'].get(eid, {'i':[0.0]*3,'j':[0.0]*3})
            for end in ('i','j'):
                for k in range(3):
                    delta = coeff * abs(ef[end][k])
                    env_max_ef[eid][end][k] += delta
                    env_min_ef[eid][end][k] -= delta

    return {
        'displacements':  {'max': env_max_d,  'min': env_min_d},
        'reactions':      {'max': env_max_r,  'min': env_min_r},
        'element_forces': {'max': env_max_ef, 'min': env_min_ef},
        'spring_forces':  {'max': {}, 'min': {}},
        'constraint_forces': {'max': env_max_cf, 'min': env_min_cf},
        'is_seismic_envelope': True,
    }


# ---------------------------------------------------------------------------
# Analysis case post-processing (Linear superposition and Mass)
# ---------------------------------------------------------------------------

def compute_analysis_case_results(struc: Structure2D, results: dict,
                                   K: np.ndarray, F: np.ndarray, case_index: dict,
                                   g: float = 9.81):
    """
    Compute results for each AnalysisCase and store under results['analysis_cases'].

    Linear:              linear superposition of individual load case results.
    Mass:                nodal masses from load-derived forces + concentrated nodal masses.
    Modal:               eigenvalue analysis using a referenced Mass case.
    Spectrum:            response-spectrum analysis.
    GeometricNonlinear:  P-Delta iteration with optional stiffness reduction.
    """
    ac_results: dict = {}
    # stored_stiffness: {anlg_case_id → K_matrix (ndarray)} — populated by ANLG cases
    stored_stiffness: dict = results.setdefault('stored_stiffness', {})

    # Process in order so later cases can reference earlier results
    for ac in struc.analysis_cases:
        ac_results[ac.id] = _compute_one_analysis_case(
            struc, ac, results, K, F, case_index, stored_stiffness,
            ac_results, g)

    results['analysis_cases'] = ac_results


def case_needs_stiffness(struc, ac) -> bool:
    """Whether an analysis case needs K/F assembled, or is pure superposition.

    A Linear (or NonLinear without conditional springs, and not reusing an ANLG
    matrix) case is a superposition of the already-computed load-case results —
    no stiffness needed. Everything else — Mass, Modal, Spectrum, ANLG, and the
    conditional-spring or stored-stiffness paths — needs K, F, or both. Used by
    :func:`solve_pending_cases` to skip assembly when only Linear cases are
    pending.
    """
    if ac.analysis_type in ('Linear', 'NonLinear'):
        if ac.analysis_type == 'NonLinear' and _has_conditional_springs(struc):
            return True
        return bool(ac.stored_stiffness_id)
    return True


def _compute_one_analysis_case(struc, ac, results, K, F, case_index,
                               stored_stiffness, ac_results, g=9.81):
    """The result of a single analysis case.

    Split out of :func:`compute_analysis_case_results` so one case can be
    (re)computed on its own — that is what makes an added case on a locked model
    run without redoing every other. ``ac_results`` holds the cases already
    done this pass, so a Spectrum can find its Modal.
    """
    # Plate domain: dynamics (Mass / Modal / Spectrum) is supported — the mass
    # is placed on the transverse DOF w and the modes are the out-of-plane
    # vibration. Only P-Delta stays blocked (its geometric stiffness needs
    # in-plane axial force). check_references already refuses it at calculate()
    # time; this guard covers the incremental solve path too, answering with a
    # result-shaped error rather than an exception mid-pipeline.
    if (getattr(struc, 'domain', 'plane') == 'plate'
            and ac.analysis_type == 'GeometricNonlinear'):
        return {'error': f"Analysis case '{ac.id}' (P-Delta): geometric "
                         "nonlinearity uses in-plane axial force and is not "
                         "available in the plate domain."}

    if ac.analysis_type == 'NonLinear' and _has_conditional_springs(struc):
        try:
            return _solve_nonlinear_springs(struc, ac, F, case_index, results)
        except Exception as exc:                        # noqa: BLE001
            import traceback
            return {'error': str(exc), 'traceback': traceback.format_exc()}

    if ac.analysis_type in ('Linear', 'NonLinear'):
        if ac.stored_stiffness_id and ac.stored_stiffness_id in stored_stiffness:
            K_use = stored_stiffness[ac.stored_stiffness_id]
            return _linear_superpose_with_K(ac.coefficients, struc, K_use, F,
                                            case_index)
        return _linear_superpose(ac.coefficients, results, struc)

    if ac.analysis_type == 'Mass':
        return _compute_nodal_masses(ac, case_index, F, struc, g)

    if ac.analysis_type == 'Modal':
        K_modal = stored_stiffness.get(ac.stored_stiffness_id, K) \
            if ac.stored_stiffness_id else K
        mass_ac = struc.analysis_cases_by_id.get(ac.modal_case_id)
        if mass_ac is None or mass_ac.analysis_type != 'Mass':
            return {'error': f"Modal case '{ac.id}': mass case "
                             f"'{ac.modal_case_id}' not found or not of type "
                             f"Mass."}
        try:
            return _solve_modal(struc, K_modal, mass_ac, F, case_index,
                                ac.num_modes, g)
        except Exception as exc:                        # noqa: BLE001
            return {'error': str(exc)}

    if ac.analysis_type == 'Spectrum':
        modal_ac = struc.analysis_cases_by_id.get(ac.modal_case_id)
        spf = struc.spectral_functions.get(ac.spectrum_id)
        modal_res = ac_results.get(ac.modal_case_id) if modal_ac else None
        if modal_ac is None or modal_ac.analysis_type != 'Modal':
            return {'error': f"Spectrum '{ac.id}': modal case "
                             f"'{ac.modal_case_id}' not found or not of type "
                             f"Modal."}
        if spf is None:
            return {'error': f"Spectrum '{ac.id}': spectral function "
                             f"'{ac.spectrum_id}' not found."}
        if modal_res is None or 'modal_info' not in modal_res:
            return {'error': f"Spectrum '{ac.id}': modal case "
                             f"'{ac.modal_case_id}' has no results — run modal "
                             f"analysis first."}
        try:
            return _solve_spectrum(struc, K, modal_res, spf,
                                   ac.combination_rule, ac.direction,
                                   ac.damping)
        except Exception as exc:                        # noqa: BLE001
            return {'error': str(exc)}

    if ac.analysis_type == 'GeometricNonlinear':
        try:
            anlg_res, K_final = _solve_geometric_nonlinear(
                struc, ac, F, case_index)
            stored_stiffness[ac.id] = K_final
            return anlg_res
        except Exception as exc:                        # noqa: BLE001
            import traceback
            return {'error': str(exc), 'traceback': traceback.format_exc()}

    return {'error': f"Unknown analysis type '{ac.analysis_type}'"}


# ---------------------------------------------------------------------------
# Geometric nonlinear (P-Delta) solver
# ---------------------------------------------------------------------------

def _axial_force_from_u(struc: Structure2D, u: np.ndarray) -> dict:
    """Return {elem_id: N} axial force at i-end for each element."""
    axial = {}
    for elem in struc.bar_elements:
        c = struc._elem_cache[elem.id]
        cos, sin, EAL = c['cos'], c['sin'], c['EAL']
        itotv = struc.node_dof_index[elem.node_i]
        jtotv = struc.node_dof_index[elem.node_j]
        dx = u[jtotv]   - u[itotv]
        dy = u[jtotv+1] - u[itotv+1]
        ul = dx * cos + dy * sin
        axial[elem.id] = EAL * ul
    return axial


def _assemble_geometric_stiffness(struc: Structure2D, axial: dict) -> np.ndarray:
    """Assemble global geometric stiffness matrix from element axial forces."""
    from .elements import (bar_geometric_stiffness_global,
                           transformation_matrix, condense_local_stiffness)
    ndof = struc.num_dofs
    Kg = np.zeros((ndof, ndof))
    for elem in struc.bar_elements:
        c = struc._elem_cache[elem.id]
        N = axial.get(elem.id, 0.0)
        kg = bar_geometric_stiffness_global(N, c['L'], c['cos'], c['sin'])
        # Condense released (hinged) rotational DOFs to match the elastic matrix
        released = c.get('released') or []
        if released:
            T = transformation_matrix(c['cos'], c['sin'])
            kg_local = T @ kg @ T.T
            kg = T.T @ condense_local_stiffness(kg_local, released) @ T
        itotv = struc.node_dof_index[elem.node_i]
        jtotv = struc.node_dof_index[elem.node_j]
        dofs = [itotv, itotv+1, itotv+2, jtotv, jtotv+1, jtotv+2]
        for i in range(6):
            for j in range(6):
                Kg[dofs[i], dofs[j]] += kg[i, j]
    return Kg


def _solve_geometric_nonlinear(struc: Structure2D, ac, F: np.ndarray,
                                case_index: dict) -> tuple[dict, np.ndarray]:
    """
    P-Delta (geometric nonlinear) analysis.

    Steps:
      1. Assemble K_reduced (with beam/column stiffness reduction factors).
      2. Build combined load vector F_combined = Σ factor_i * F_i.
      3. Iterate: solve (K_red + Kg)*u = F_combined, update Kg from axial forces.
      4. Store K_final = K_red + Kg (tangent stiffness at convergence).
      5. Re-solve K_final * u_lc = F_lc for each individual load case.
      6. Return per-load-case and combined results.
    """
    from .assembly import assemble_stiffness_reduced
    from .loads import assemble_loads

    # ── 1. Reduced elastic stiffness ──────────────────────────────────
    K_red = assemble_stiffness_reduced(
        struc,
        beam_factor=ac.beam_stiffness_factor,
        column_factor=ac.column_stiffness_factor,
    )

    # ── 2. Combined load vector ────────────────────────────────────────
    n_lc = len(struc.load_cases)
    if n_lc == 0:
        raise ValueError("No load cases defined.")

    F_combined = np.zeros(struc.num_dofs)
    for lc_id, factor in ac.coefficients.items():
        if lc_id in case_index:
            F_combined += factor * F[:, case_index[lc_id]]

    if np.allclose(F_combined, 0.0):
        raise ValueError(
            f"ANLG case '{ac.id}': combined load vector is zero. "
            "Add load case coefficients.")

    # Multi-point constraints (penalty) on the elastic stiffness — carried
    # through K_total = K_red + Kg and K_final. One alpha (from K_red) is used
    # for the matrix and both load vectors so they stay consistent.
    from .assembly import apply_constraint_penalty, _constraint_penalty_alpha
    if struc.constraints:
        _alpha = _constraint_penalty_alpha(K_red)
        K_red, F = apply_constraint_penalty(struc, K_red, F, alpha=_alpha)
        _, F_combined = apply_constraint_penalty(struc, K_red, F_combined,
                                                 alpha=_alpha)

    # ── 3. P-Delta iteration ──────────────────────────────────────────
    fixed = _augment_fixed_zero_stiffness(K_red, _build_fixed_dofs(struc))
    free  = np.where(~fixed)[0]
    fixed_idx = np.where(fixed)[0]
    U_p = _build_prescribed_displacements(struc, struc.num_dofs, n_lc)

    def _solve_free(K_total: np.ndarray, f_vec: np.ndarray) -> np.ndarray:
        u = np.zeros(struc.num_dofs)
        u[fixed_idx] = 0.0   # settlements not supported for combined vector
        K_fp = K_total[np.ix_(free, fixed_idx)]
        f_mod = f_vec[free] - K_fp @ u[fixed_idx]
        u[free] = np.linalg.solve(K_total[np.ix_(free, free)], f_mod)
        return u

    Kg = np.zeros_like(K_red)
    u_prev = np.zeros(struc.num_dofs)
    converged = False
    for iteration in range(ac.max_iterations):
        K_total = K_red + Kg
        u_new = _solve_free(K_total, F_combined)
        axial = _axial_force_from_u(struc, u_new)
        Kg = _assemble_geometric_stiffness(struc, axial)

        # Convergence check: relative norm of displacement change
        delta = np.linalg.norm(u_new - u_prev)
        ref   = max(np.linalg.norm(u_new), 1e-12)
        if delta / ref < ac.tolerance:
            converged = True
            u_prev = u_new
            break
        u_prev = u_new

    K_final = K_red + Kg   # tangent stiffness at convergence

    # ── 4. Re-solve each individual load case with K_final ─────────────
    # This allows load-case-level distributions and combination post-processing.
    n_cases = F.shape[1]
    U_lc = np.zeros((struc.num_dofs, n_cases))
    U_lc[fixed_idx, :] = U_p[fixed_idx, :]
    K_ff  = K_final[np.ix_(free, free)]
    K_fp2 = K_final[np.ix_(free, fixed_idx)]
    F_mod_all = F[free, :] - K_fp2 @ U_p[fixed_idx, :]
    U_lc[free, :] = np.linalg.solve(K_ff, F_mod_all)

    # ── 5. Build results dict ─────────────────────────────────────────
    node_ids  = list(struc.nodes.keys())
    elem_ids  = [e.id for e in struc.bar_elements]

    # Combined displacement / reaction
    residual_comb = K_final @ u_prev - F_combined
    disp_comb  = {nid: [float(u_prev[struc.node_dof_index[nid] + k]) for k in range(3)]
                  for nid in node_ids}
    reac_comb  = {}
    for nid in node_ids:
        itotv = struc.node_dof_index[nid]
        rx = float(residual_comb[itotv])   if fixed[itotv]   else 0.0
        ry = float(residual_comb[itotv+1]) if fixed[itotv+1] else 0.0
        mz = float(residual_comb[itotv+2]) if fixed[itotv+2] else 0.0
        if abs(rx) > 1e-9 or abs(ry) > 1e-9 or abs(mz) > 1e-9:
            reac_comb[nid] = [rx, ry, mz]

    ef_comb = _elem_forces_from_u(struc, u_prev)

    # Per-load-case results (using K_final for all)
    disp_lc:  dict = {}
    reac_lc:  dict = {}
    ef_lc:    dict = {}

    for lc in struc.load_cases:
        ic = case_index[lc.id]
        u  = U_lc[:, ic]
        d_lc = {nid: [float(u[struc.node_dof_index[nid]+k]) for k in range(3)]
                for nid in node_ids}
        residual = K_final @ u - F[:, ic]
        r_lc = {}
        for nid in node_ids:
            itotv = struc.node_dof_index[nid]
            rx = float(residual[itotv])   if fixed[itotv]   else 0.0
            ry = float(residual[itotv+1]) if fixed[itotv+1] else 0.0
            mz = float(residual[itotv+2]) if fixed[itotv+2] else 0.0
            if abs(rx) > 1e-9 or abs(ry) > 1e-9 or abs(mz) > 1e-9:
                r_lc[nid] = [rx, ry, mz]
        disp_lc[lc.id] = d_lc
        reac_lc[lc.id] = r_lc
        ef_lc[lc.id]   = _elem_forces_from_u(struc, u)

    return {
        'displacements':     disp_comb,
        'reactions':         reac_comb,
        'element_forces':    ef_comb,
        'load_case_displacements': disp_lc,
        'load_case_reactions':     reac_lc,
        'load_case_element_forces': ef_lc,
        'converged':         converged,
        'iterations':        iteration + 1,
        'beam_factor':       ac.beam_stiffness_factor,
        'column_factor':     ac.column_stiffness_factor,
        'is_anlg':           True,
    }, K_final


# ---------------------------------------------------------------------------
# Non-linear unilateral (tension-/compression-only) node springs
# ---------------------------------------------------------------------------

def _conditional_spring_contributions(struc: Structure2D) -> list:
    """Return the unilateral (tension-/compression-only) spring contributions.

    Each contribution is a dict::

        {'dofs': (d0, d1),     # the node's X / Y global DOF indices
         'block': 2x2 ndarray, # stiffness added to K[d0:d1+1, d0:d1+1] when active
         'proj': (px, py),     # unit direction; activity uses px·u[d0] + py·u[d1]
         'mode': 'tension' | 'compression'}

    Covers node springs (axis-aligned) and element foundation (Winkler) springs.
    Element springs are lumped ``k·L/2`` at each end-node and decomposed into
    their per-axis (X/Y or axial/transverse) rank-1 blocks, so the activity of
    each component is evaluated per end-node. Bilateral ('both') components and
    rotational stiffness are never returned (they stay baked into K).
    """
    contribs = []

    # ── Node springs (axis-aligned) ───────────────────────────────────
    for nid, sp in struc.node_springs.items():
        base = struc.node_dof_index[nid]
        if sp.mode_x != 'both' and sp.kx != 0.0:
            contribs.append({'dofs': (base, base + 1),
                             'block': np.array([[sp.kx, 0.0], [0.0, 0.0]]),
                             'proj': (1.0, 0.0), 'mode': sp.mode_x})
        if sp.mode_y != 'both' and sp.ky != 0.0:
            contribs.append({'dofs': (base, base + 1),
                             'block': np.array([[0.0, 0.0], [0.0, sp.ky]]),
                             'proj': (0.0, 1.0), 'mode': sp.mode_y})

    # ── Element foundation (Winkler) springs, lumped L/2 at each end ───
    for esp in struc.element_springs.values():
        if esp.mode_x == 'both' and esp.mode_y == 'both':
            continue
        elem = struc.bar_elements_by_id[esp.element_id]
        L = struc._elem_cache[elem.id]['L']
        f = L / 2.0
        ni = struc.nodes[elem.node_i]; nj = struc.nodes[elem.node_j]
        dx, dy = nj.x - ni.x, nj.y - ni.y
        Ln = (dx * dx + dy * dy) ** 0.5 or 1.0
        cx, cy = dx / Ln, dy / Ln
        if getattr(esp, 'coord_sys', 'global') == 'local':
            comps = [((cx, cy), esp.kx, esp.mode_x),      # axial
                     ((-cy, cx), esp.ky, esp.mode_y)]     # transverse
        else:
            comps = [((1.0, 0.0), esp.kx, esp.mode_x),    # global X
                     ((0.0, 1.0), esp.ky, esp.mode_y)]    # global Y
        for (dvec, k, mode) in comps:
            if mode == 'both' or k == 0.0:
                continue
            px, py = dvec
            block = k * f * np.array([[px * px, px * py],
                                      [px * py, py * py]])
            for nid2 in (elem.node_i, elem.node_j):
                base = struc.node_dof_index[nid2]
                contribs.append({'dofs': (base, base + 1),
                                 'block': block.copy(),
                                 'proj': (px, py), 'mode': mode})
    return contribs


def _has_conditional_springs(struc: Structure2D) -> bool:
    return bool(_conditional_spring_contributions(struc))


def _spring_active(mode: str, proj_disp: float) -> bool:
    """Active state of a unilateral spring given the displacement projected on
    its component direction.

    'tension'     → active when the projected displacement is positive
                    (the spring is stretched).
    'compression' → active when the projected displacement is negative.
    """
    return proj_disp > 0.0 if mode == 'tension' else proj_disp < 0.0


def _solve_nonlinear_springs(struc: Structure2D, ac, F: np.ndarray,
                             case_index: dict, results: dict) -> dict:
    """
    NonLinear case with unilateral (tension-/compression-only) springs
    (node springs and/or element Winkler springs).

    Solved by an active-set iteration on the *combined* load vector
    (Σ factor_i · F_i). Each unilateral spring component is included in the
    stiffness only while it is active (its end-node displacement projected on
    the component direction has the sign matching its mode). The active set is
    updated from the resulting displacements until it no longer changes (or a
    cycle / iteration cap is reached).

    Support settlements are not applied to the combined vector (consistent with
    the P-Delta combined solve); fixed DOFs are held at zero.
    """
    from .assembly import assemble_stiffness

    contribs = _conditional_spring_contributions(struc)
    n_lc = len(struc.load_cases)
    if n_lc == 0:
        raise ValueError("No load cases defined.")

    # Combined load vector
    F_comb = np.zeros(struc.num_dofs)
    for lc_id, factor in ac.coefficients.items():
        if lc_id in case_index:
            F_comb += factor * F[:, case_index[lc_id]]
    if np.allclose(F_comb, 0.0):
        raise ValueError(
            f"NonLinear case '{ac.id}': combined load vector is zero. "
            "Add load case coefficients.")

    # Full bilateral stiffness, then remove the conditional spring components so
    # they can be added back only while active. The active-set loop mutates
    # individual entries and uses dense fancy-indexing, so densify here (these
    # are small frame models with unilateral supports).
    K_full = _dense(assemble_stiffness(struc))
    K_base = K_full.copy()

    def _add_block(K, c, sign):
        (d0, d1), b = c['dofs'], c['block']
        K[d0, d0] += sign * b[0, 0]; K[d0, d1] += sign * b[0, 1]
        K[d1, d0] += sign * b[1, 0]; K[d1, d1] += sign * b[1, 1]

    for c in contribs:
        _add_block(K_base, c, -1.0)

    # Multi-point constraints (penalty) on the bilateral base stiffness, so they
    # are present in every active-set assembly (which copies K_base).
    from .assembly import apply_constraint_penalty, _constraint_penalty_alpha
    if struc.constraints:
        _alpha = _constraint_penalty_alpha(K_base)
        K_base, F_comb = apply_constraint_penalty(struc, K_base, F_comb,
                                                  alpha=_alpha)

    base_fixed = _build_fixed_dofs(struc)
    ndof = struc.num_dofs

    def _assemble_active(active: list) -> np.ndarray:
        K_act = K_base.copy()
        for c, a in zip(contribs, active):
            if a:
                _add_block(K_act, c, +1.0)
        return K_act

    def _proj_disp(c, u: np.ndarray) -> float:
        (d0, d1), (px, py) = c['dofs'], c['proj']
        return px * u[d0] + py * u[d1]

    def _solve_with(K_act: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        fixed = _augment_fixed_zero_stiffness(K_act, base_fixed)
        free = np.where(~fixed)[0]
        u = np.zeros(ndof)
        u[free] = np.linalg.solve(K_act[np.ix_(free, free)], F_comb[free])
        return u, fixed

    # Start from the bilateral solution (all conditional springs active).
    active = [True] * len(contribs)
    seen = set()
    converged = False
    iteration = 0
    u = np.zeros(ndof)
    fixed = base_fixed
    max_iter = max(ac.max_iterations, 20)
    for iteration in range(max_iter):
        K_act = _assemble_active(active)
        u, fixed = _solve_with(K_act)
        new_active = [_spring_active(c['mode'], _proj_disp(c, u))
                      for c in contribs]
        if new_active == active:
            converged = True
            break
        key = tuple(new_active)
        if key in seen:        # cycling → accept last state, stop
            active = new_active
            break
        seen.add(key)
        active = new_active

    # Final stiffness / solution consistent with the converged active set
    K_final = _assemble_active(active)
    u, fixed = _solve_with(K_final)

    node_ids = list(struc.nodes.keys())
    residual = K_final @ u - F_comb

    disp = {nid: [float(u[struc.node_dof_index[nid] + k]) for k in range(3)]
            for nid in node_ids}
    reac = {}
    for nid in node_ids:
        b = struc.node_dof_index[nid]
        rx = float(residual[b])   if fixed[b]   else 0.0
        ry = float(residual[b+1]) if fixed[b+1] else 0.0
        mz = float(residual[b+2]) if fixed[b+2] else 0.0
        if abs(rx) > 1e-9 or abs(ry) > 1e-9 or abs(mz) > 1e-9:
            reac[nid] = [rx, ry, mz]

    ef = _elem_forces_from_u(struc, u)

    # Node-spring forces (zeroed for inactive unilateral components).
    spring_f = {}
    for nid, sp in struc.node_springs.items():
        b = struc.node_dof_index[nid]
        fx = u[b]   * sp.kx
        fy = u[b+1] * sp.ky
        ft = u[b+2] * sp.kt
        if sp.mode_x != 'both' and sp.kx != 0.0 \
                and not _spring_active(sp.mode_x, u[b]):
            fx = 0.0
        if sp.mode_y != 'both' and sp.ky != 0.0 \
                and not _spring_active(sp.mode_y, u[b+1]):
            fy = 0.0
        spring_f[nid] = [float(fx), float(fy), float(ft)]

    # Per-element N/V/M distribution so diagrams render like a Linear case:
    # superpose the load-case diagrams, then correct the boundary (end) forces
    # to the true non-linear combined values (linear in end forces + member
    # loads, so a per-end shift is exact).
    elem_dist: dict = {}
    src = results.get('element_distribution', {})
    for elem in struc.bar_elements:
        L = struc._elem_cache[elem.id]['L']
        ref = None
        for lc_id in ac.coefficients:
            ref = src.get(lc_id, {}).get(elem.id)
            if ref is not None:
                break
        xs = (np.asarray(ref['x'], dtype=float) if ref is not None
              else np.linspace(0.0, L, 51))
        npts = len(xs)
        Ns = np.zeros(npts); Vs = np.zeros(npts); Ms = np.zeros(npts)
        for lc_id, coeff in ac.coefficients.items():
            d = src.get(lc_id, {}).get(elem.id)
            if d is None:
                continue
            Ns += coeff * np.asarray(d['N'], dtype=float)
            Vs += coeff * np.asarray(d['V'], dtype=float)
            Ms += coeff * np.asarray(d['M'], dtype=float)
        ef_i = ef.get(elem.id, {}).get('i')
        if ef_i is not None and npts:
            dN = ef_i[0] - Ns[0]; dV = ef_i[1] - Vs[0]; dM = ef_i[2] - Ms[0]
            Ns = Ns + dN; Vs = Vs + dV; Ms = Ms + dM + dV * xs
        elem_dist[elem.id] = {'x': xs, 'N': Ns, 'V': Vs, 'M': Ms}

    return {
        'displacements':  disp,
        'reactions':      reac,
        'element_forces': ef,
        'spring_forces':  spring_f,
        'element_distribution': elem_dist,
        'converged':      converged,
        'iterations':     iteration + 1,
        'is_nonlinear_springs': True,
    }


# ---------------------------------------------------------------------------
# Linear superposition using a provided K (for stored-stiffness cases)
# ---------------------------------------------------------------------------

def _linear_superpose_with_K(coefficients: dict, struc: Structure2D,
                               K_stored: np.ndarray,
                               F: np.ndarray, case_index: dict) -> dict:
    """
    Re-solve each load case using K_stored (a previously computed ANLG tangent
    stiffness), then superpose with the given coefficients.
    """
    n_cases = F.shape[1]
    # Solve K_stored * U = F for all load cases
    U = solve(struc, K_stored, F)

    # Build temporary results dict for this K
    node_ids   = list(struc.nodes.keys())
    elem_ids   = [e.id for e in struc.bar_elements]
    spring_ids = list(struc.node_springs.keys())
    fixed      = _build_fixed_dofs(struc)

    con_ids  = _constraint_force_node_ids(struc)

    disp_c   = {nid: [0.0]*3 for nid in node_ids}
    reac_c   = {nid: [0.0]*3 for nid in node_ids}
    ef_c     = {eid: {'i': [0.0]*3, 'j': [0.0]*3} for eid in elem_ids}
    spring_c = {sid: [0.0]*3 for sid in spring_ids}
    conf_c   = {nid: [0.0]*3 for nid in con_ids}

    for lc_id, coeff in coefficients.items():
        if lc_id not in case_index:
            continue
        ic = case_index[lc_id]
        u  = U[:, ic]

        for nid in node_ids:
            itotv = struc.node_dof_index[nid]
            for k in range(3):
                disp_c[nid][k] += coeff * float(u[itotv + k])

        residual = K_stored @ u - F[:, ic]
        for nid in node_ids:
            itotv = struc.node_dof_index[nid]
            rx = float(residual[itotv])   if fixed[itotv]   else 0.0
            ry = float(residual[itotv+1]) if fixed[itotv+1] else 0.0
            mz = float(residual[itotv+2]) if fixed[itotv+2] else 0.0
            reac_c[nid][0] += coeff * rx
            reac_c[nid][1] += coeff * ry
            reac_c[nid][2] += coeff * mz
        for nid in con_ids:
            itotv = struc.node_dof_index[nid]
            for k in range(3):
                conf_c[nid][k] += coeff * float(residual[itotv + k])

        ef_k = _elem_forces_from_u(struc, u)
        for eid in elem_ids:
            ef = ef_k.get(eid, {'i': [0.0]*3, 'j': [0.0]*3})
            for end in ('i', 'j'):
                for k in range(3):
                    ef_c[eid][end][k] += coeff * ef[end][k]

        for sid in spring_ids:
            sp = struc.node_springs[sid]
            itotv = struc.node_dof_index[sid]
            spring_c[sid][0] += coeff * u[itotv]   * sp.kx
            spring_c[sid][1] += coeff * u[itotv+1] * sp.ky
            spring_c[sid][2] += coeff * u[itotv+2] * sp.kt

    return {
        'displacements':  disp_c,
        'reactions':      reac_c,
        'element_forces': ef_c,
        'spring_forces':  spring_c,
        'constraint_forces': conf_c,
        'uses_stored_stiffness': True,
    }


def _linear_superpose(coefficients: dict, results: dict, struc: Structure2D) -> dict:
    """Linear superposition of load case results (equivalent to LinearSum combination)."""
    node_ids  = list(struc.nodes.keys())
    elem_ids  = [e.id for e in struc.bar_elements]
    spring_ids = list(struc.node_springs.keys())

    con_ids = _constraint_force_node_ids(struc)

    disp_c   = {nid: [0.0]*3 for nid in node_ids}
    reac_c   = {nid: [0.0]*3 for nid in node_ids}
    ef_c     = {eid: {'i': [0.0]*3, 'j': [0.0]*3} for eid in elem_ids}
    spring_c = {sid: [0.0]*3 for sid in spring_ids}
    conf_c   = {nid: [0.0]*3 for nid in con_ids}

    for case_id, coeff in coefficients.items():
        if case_id not in results['displacements']:
            continue
        for nid in node_ids:
            d = results['displacements'][case_id].get(nid, [0.0]*3)
            r = results['reactions'][case_id].get(nid, [0.0]*3)
            for k in range(3):
                disp_c[nid][k] += coeff * d[k]
                reac_c[nid][k] += coeff * r[k]
        cf_case = results.get('constraint_forces', {}).get(case_id, {})
        for nid in con_ids:
            cv = cf_case.get(nid, [0.0]*3)
            for k in range(3):
                conf_c[nid][k] += coeff * cv[k]
        for eid in elem_ids:
            ef = results['element_forces'][case_id].get(eid, {'i': [0.0]*3, 'j': [0.0]*3})
            for end in ('i', 'j'):
                for k in range(3):
                    ef_c[eid][end][k] += coeff * ef[end][k]
        for sid in spring_ids:
            sf = results['spring_forces'][case_id].get(sid, [0.0]*3)
            for k in range(3):
                spring_c[sid][k] += coeff * sf[k]

    # Also superpose element distributions. The per-element sampling grid is
    # whatever compute_distributions produced (uniform grid plus any element
    # point-load positions), so derive its length from an existing case rather
    # than assuming a fixed number of points.
    elem_dist: dict = {}
    src = results.get('element_distribution', {})
    for elem in struc.bar_elements:
        L  = struc._elem_cache[elem.id]['L']
        ref = None
        for case_id in coefficients:
            ref = src.get(case_id, {}).get(elem.id)
            if ref is not None:
                break
        if ref is not None:
            xs = np.asarray(ref['x'], dtype=float)
        else:
            xs = np.linspace(0.0, L, 51)
        npts = len(xs)
        Ns = np.zeros(npts); Vs = np.zeros(npts); Ms = np.zeros(npts)
        for case_id, coeff in coefficients.items():
            d = src.get(case_id, {}).get(elem.id)
            if d is None:
                continue
            Ns += coeff * np.asarray(d['N'], dtype=float)
            Vs += coeff * np.asarray(d['V'], dtype=float)
            Ms += coeff * np.asarray(d['M'], dtype=float)
        elem_dist[elem.id] = {'x': xs, 'N': Ns, 'V': Vs, 'M': Ms}

    return {
        'displacements':      disp_c,
        'reactions':          reac_c,
        'element_forces':     ef_c,
        'spring_forces':      spring_c,
        'constraint_forces':  conf_c,
        'element_distribution': elem_dist,
    }


def _compute_nodal_masses(mass_ac, case_index: dict,
                           F: np.ndarray, struc: Structure2D, g: float) -> dict:
    """
    Derive nodal masses for a Mass analysis case:
      - from load-derived vertical forces (Fy / g) per load case coefficient
      - from concentrated NodalMass entries on the structure

    Returns {node_id: {'mx': t, 'my': t, 'mtz': t·m²}}.
    """
    node_masses: dict[str, dict] = {
        nid: {'mx': 0.0, 'my': 0.0, 'mtz': 0.0} for nid in struc.nodes}
    plate = getattr(struc, 'domain', 'plane') == 'plate'
    grav = 0 if plate else 1   # load-vector component carrying gravity

    # ── Loads → mass ──────────────────────────────────────────────────
    for case_id, factor in mass_ac.coefficients.items():
        if case_id not in case_index:
            continue
        ic = case_index[case_id]
        for nid in struc.nodes:
            itotv = struc.node_dof_index[nid]
            fg = F[itotv + grav, ic]
            m = factor * abs(fg) / g   # F/g converts force [kN] to mass [t]
            node_masses[nid]['mx'] += m
            if not plate:
                node_masses[nid]['my'] += m   # plate: mass is vertical only (mx)

    # ── Concentrated nodal masses ─────────────────────────────────────
    for nm in struc.nodal_masses:
        if nm.mass_case_id != mass_ac.id:
            continue
        if nm.node_id not in node_masses:
            continue
        node_masses[nm.node_id]['mx']  += nm.mx
        node_masses[nm.node_id]['my']  += nm.my
        node_masses[nm.node_id]['mtz'] += nm.mtz

    # Remove zero entries for cleaner output
    nodal_masses = {nid: m for nid, m in node_masses.items()
                    if m['mx'] > 1e-14 or m['my'] > 1e-14 or m['mtz'] > 1e-14}
    # The model's total *translational* mass: sum of my (plane) or, in plate,
    # sum of mx — the vertical mass, since my/mtz there are rotational inertias.
    total_mass = sum(m['mx'] if plate else m['my']
                     for m in nodal_masses.values())

    # ── Mass that a support holds down, per direction ─────────────────
    # Mass lumped on a restrained DOF cannot move, so it produces no inertia
    # force and is excluded from the eigenproblem (it goes straight to the
    # foundation). It is reported here so the modal percentages — which are
    # relative to the FREE mass — can be read against the model's total.
    fixed = _build_fixed_dofs(struc)
    restrained_x = restrained_y = 0.0
    for nid, m in nodal_masses.items():
        itotv = struc.node_dof_index[nid]
        if fixed[itotv]:
            restrained_x += m['mx']
        if fixed[itotv + 1]:
            restrained_y += m['my']
    model_x = sum(m['mx'] for m in nodal_masses.values())

    return {
        'nodal_masses': nodal_masses,   # {node_id: {'mx','my','mtz'}} [t]
        'total_mass':   total_mass,     # sum of my [t] — the model's total
        'model_mass_x': model_x,        # all mx, restrained included [t]
        'model_mass_y': total_mass,     # all my, restrained included [t]
        'restrained_mass_x': restrained_x,   # held by supports, cannot vibrate
        'restrained_mass_y': restrained_y,
    }


def _assemble_mass_vector(struc: Structure2D, mass_ac, F: np.ndarray,
                           case_index: dict, g: float) -> np.ndarray:
    """Build the diagonal mass vector (ndof,) for a Mass analysis case.

    Plane domain: a load's mass F/g is placed on both translational DOFs (ux,
    uy), since a gravity load stands in for mass that can sway either way.
    Plate domain: the vibration is out-of-plane, so the mass sits on the single
    transverse DOF w (component 0), taken from the gravity component of the
    load vector (which is w there); the rotation DOFs carry only whatever
    rotational inertia a NodalMass supplies."""
    ndof = struc.num_dofs
    M = np.zeros(ndof)
    plate = getattr(struc, 'domain', 'plane') == 'plate'
    grav = 0 if plate else 1   # load-vector component carrying gravity

    # Loads → mass
    for case_id, factor in mass_ac.coefficients.items():
        if case_id not in case_index:
            continue
        ic = case_index[case_id]
        for nid in struc.nodes:
            itotv = struc.node_dof_index[nid]
            m = factor * abs(F[itotv + grav, ic]) / g
            if plate:
                M[itotv] += m           # w only (out-of-plane)
            else:
                M[itotv]   += m         # ux
                M[itotv+1] += m         # uy

    # Concentrated nodal masses. The three slots follow the domain's DOF
    # meaning: plane (mx, my, mtz) → (ux, uy, θz); plate (mx, my, mtz) →
    # (vertical mass on w, rotational inertia about x on θx, about y on θy).
    for nm in struc.nodal_masses:
        if nm.mass_case_id != mass_ac.id:
            continue
        if nm.node_id not in struc.node_dof_index:
            continue
        itotv = struc.node_dof_index[nm.node_id]
        M[itotv]   += nm.mx
        M[itotv+1] += nm.my
        M[itotv+2] += nm.mtz

    return M


def _solve_modal(struc: Structure2D, K: np.ndarray, mass_ac,
                 F: np.ndarray, case_index: dict,
                 num_modes: int, g: float = 9.81) -> dict:
    """
    Solve the generalised eigenvalue problem  K·φ = ω²·M·φ  using numpy.
    M is assembled from the given Mass analysis case (lumped diagonal).

    Approach: transform to standard form via M^{-½}
      K̃ = M^{-½} K M^{-½}  →  K̃ ψ = ω² ψ  →  φ = M^{-½} ψ
    """
    K = _dense(K)                    # eigen-solve works on the dense reduced K
    # Multi-point constraints (penalty) on the stiffness only — never on the
    # mass. A penalised constraint shows up as a very stiff (high-frequency)
    # coupling, which the positive-eigenvalue filter below leaves out of the
    # low modes of interest. (Phase 2 will impose it exactly.)
    if struc.constraints:
        from .assembly import apply_constraint_penalty
        K, _ = apply_constraint_penalty(struc, K, np.zeros(struc.num_dofs))
    # ── Build diagonal mass vector ────────────────────────────────────
    M_diag = _assemble_mass_vector(struc, mass_ac, F, case_index, g)

    # ── Restrict to free DOFs ─────────────────────────────────────────
    # Pin any DOF left with zero stiffness by end releases (avoids singular M^-½).
    fixed  = _augment_fixed_zero_stiffness(K, _build_fixed_dofs(struc))
    free   = np.where(~fixed)[0]
    n_free = len(free)

    M_free = M_diag[free]
    K_free = K[np.ix_(free, free)]

    # ── Zero-mass DOFs (typically rotations, or ux with only vertical mass) ─
    # They carry no inertia, so K·φ = ω²·M·φ has, on those rows, the static
    # relation  K_zz φ_z + K_zm φ_m = 0.  Eliminating them (static / Guyan
    # condensation) is EXACT here — not an approximation, because M_zz = 0 —
    #     K_r = K_mm − K_mz K_zz⁻¹ K_zm ,   φ_z = −K_zz⁻¹ K_zm φ_m
    # and leaves a well-conditioned problem on the massive DOFs only. Giving
    # them a token mass instead (m_max·1e-8) put a ~10⁸ spread in K̃ and made
    # the frequencies depend on the LAPACK build.
    m_trans_max = M_free.max()
    if m_trans_max < 1e-20:
        raise ValueError(
            "Mass case has zero total mass — add loads or concentrated masses.")
    eps = m_trans_max * 1e-8
    zero = M_free < eps
    idx_z = np.where(zero)[0]
    idx_m = np.where(~zero)[0]
    X = None
    if len(idx_z):
        K_zz = K_free[np.ix_(idx_z, idx_z)]
        K_zm = K_free[np.ix_(idx_z, idx_m)]
        # K_zz is singular if the massless DOFs can move as a mechanism; the
        # condition number catches it, and then the token-mass regularisation
        # is the fallback.
        if np.linalg.cond(K_zz) < 1e12:
            X = np.linalg.solve(K_zz, K_zm)
    if X is not None:
        K_r = K_free[np.ix_(idx_m, idx_m)] - K_zm.T @ X
        M_red = M_free[idx_m]
        M_free = np.where(zero, 0.0, M_free)
    else:
        idx_m = np.arange(n_free)
        K_r = K_free
        M_free = np.where(zero, eps, M_free)       # regularise
        M_red = M_free

    # ── Transform to standard symmetric eigenvalue problem ───────────
    M_sqrt     = np.sqrt(M_red)                  # M^{½}  (diagonal)
    M_inv_sqrt = 1.0 / M_sqrt                    # M^{-½} (diagonal)
    K_tilde    = K_r * M_inv_sqrt[:, None] * M_inv_sqrt[None, :]

    # Symmetrise (guard against floating-point asymmetry)
    K_tilde = 0.5 * (K_tilde + K_tilde.T)

    # ── Solve — numpy returns eigenvalues ascending ───────────────────
    omega_sq_all, psi_all = np.linalg.eigh(K_tilde)

    # Keep only positive eigenvalues (discard numerical noise / rigid-body)
    valid_mask = omega_sq_all > m_trans_max * 1e-6
    omega_sq = omega_sq_all[valid_mask]
    psi      = psi_all[:, valid_mask]

    n_modes = min(num_modes, len(omega_sq))
    omega_sq = omega_sq[:n_modes]
    psi      = psi[:, :n_modes]

    # ── Back-transform mode shapes ────────────────────────────────────
    phi_m = psi * M_inv_sqrt[:, None]      # (massive DOFs, n_modes)
    if X is not None:
        phi_free = np.zeros((n_free, n_modes))
        phi_free[idx_m] = phi_m
        phi_free[idx_z] = -X @ phi_m           # massless DOFs, recovered
    else:
        phi_free = phi_m                       # (n_free, n_modes)

    # ── Frequencies ───────────────────────────────────────────────────
    omega = np.sqrt(omega_sq)              # rad/s
    freq  = omega / (2.0 * np.pi)         # Hz
    periods = 1.0 / freq                  # s

    # ── Modal participation factors and effective masses ──────────────
    # Build direction vectors in free-DOF space. Plane: two translational
    # directions, X on component 0 and Y on component 1. Plate: one
    # out-of-plane direction (vertical), on component 0 (w) — it takes the "x"
    # slot so the whole participation/spectrum pipeline is reused unchanged;
    # there is no second translational direction, so r_y stays zero.
    plate = getattr(struc, 'domain', 'plane') == 'plate'
    free_set     = set(free.tolist())
    free_pos     = {dof: idx for idx, dof in enumerate(free.tolist())}
    r_x = np.zeros(n_free)
    r_y = np.zeros(n_free)
    for nid in struc.nodes:
        itotv = struc.node_dof_index[nid]
        if itotv   in free_set: r_x[free_pos[itotv]]   = 1.0
        if not plate and itotv+1 in free_set:
            r_y[free_pos[itotv+1]] = 1.0

    total_mass_x = float(M_free @ r_x)
    total_mass_y = float(M_free @ r_y)

    modal_info = []
    for k in range(n_modes):
        phi_k    = phi_free[:, k]
        norm_k   = float(phi_k @ (M_free * phi_k))     # φ^T M φ
        gamma_x  = float(phi_k @ (M_free * r_x))
        gamma_y  = float(phi_k @ (M_free * r_y))
        meff_x   = gamma_x**2 / norm_k if norm_k > 0 else 0.0
        meff_y   = gamma_y**2 / norm_k if norm_k > 0 else 0.0
        modal_info.append({
            'mode':         k + 1,
            'omega':        float(omega[k]),
            'frequency':    float(freq[k]),
            'period':       float(periods[k]),
            'gamma_x':      gamma_x,
            'gamma_y':      gamma_y,
            'meff_x':       meff_x,
            'meff_y':       meff_y,
            'meff_x_pct':   100 * meff_x / total_mass_x if total_mass_x > 0 else 0.0,
            'meff_y_pct':   100 * meff_y / total_mass_y if total_mass_y > 0 else 0.0,
        })

    # ── Full-DOF normalised mode shapes ──────────────────────────────
    mode_shapes = []
    for k in range(n_modes):
        phi_full = np.zeros(struc.num_dofs)
        phi_full[free] = phi_free[:, k]
        mx = float(np.max(np.abs(phi_full)))
        if mx > 1e-12:
            phi_full /= mx
        # Store phi_max in modal_info (needed for spectrum analysis)
        modal_info[k]['phi_max'] = mx if mx > 1e-12 else 1.0
        shapes = {}
        for nid in struc.nodes:
            itotv = struc.node_dof_index[nid]
            shapes[nid] = [float(phi_full[itotv]),
                           float(phi_full[itotv+1]),
                           float(phi_full[itotv+2])]
        mode_shapes.append(shapes)

    # ── Model mass vs participating mass ──────────────────────────────
    # ``total_mass_*`` is the mass on the FREE DOFs — the denominator of
    # ``meff_*_pct``, chosen so that the percentages of all modes add up to
    # 100 %. It is NOT the model's total mass: whatever was lumped on a
    # restrained DOF cannot vibrate and is excluded. That difference is
    # reported explicitly here, because EC8 §4.3.3.3.1 asks for 90 % of the
    # TOTAL mass, and because the excluded fraction depends on the mesh (on a
    # simply supported beam with n elements it is exactly 1/n of the mass) —
    # so it is an artefact of the lumped mass matrix, not of the structure.
    model_mass_x = model_mass_y = 0.0
    for nid in struc.nodes:
        itotv = struc.node_dof_index[nid]
        model_mass_x += M_diag[itotv]
        if not plate:
            model_mass_y += M_diag[itotv + 1]   # plate: component 1 is θx inertia
    # (total_mass_* is measured on the regularised M, so it can exceed the raw
    # model mass by a few eps; clamp instead of reporting a negative.)
    restrained_x = max(0.0, model_mass_x - total_mass_x)
    restrained_y = max(0.0, model_mass_y - total_mass_y)

    warnings: list[str] = []
    for lbl, restr, model in (('X', restrained_x, model_mass_x),
                              ('Y', restrained_y, model_mass_y)):
        if model > 1e-14 and restr / model > _RESTRAINED_MASS_WARN:
            warnings.append(
                f"{restr / model:.1%} da massa em {lbl} está em graus de "
                f"liberdade apoiados e não participa nos modos. As "
                f"percentagens de massa efetiva são relativas à massa livre "
                f"({model - restr:.4g} t), não à massa total do modelo "
                f"({model:.4g} t).")

    return {
        'num_modes':   n_modes,
        'modal_info':  modal_info,       # list of per-mode dicts
        'mode_shapes': mode_shapes,      # list[dict[node_id → [ux,uy,rz]]]
        # Mass on the FREE DOFs — the reference for meff_*_pct.
        'total_mass_x': total_mass_x,
        'total_mass_y': total_mass_y,
        # Mass of the whole model, restrained DOFs included.
        'model_mass_x': model_mass_x,
        'model_mass_y': model_mass_y,
        'restrained_mass_x': restrained_x,
        'restrained_mass_y': restrained_y,
        # In the plate domain the "x" slot holds the single vertical
        # (out-of-plane) direction; "y" is unused. Flagged so result readers
        # can label it correctly.
        'plate': plate,
        'warnings': warnings,
    }


# ---------------------------------------------------------------------------
# Response Spectrum Analysis
# ---------------------------------------------------------------------------

def _interpolate_spectrum(spf, T: float) -> float:
    """Linearly interpolate Sa from a SpectralFunction at period T [s]."""
    pts = sorted(spf.points, key=lambda p: p[0])
    if not pts:
        return 0.0
    if T <= pts[0][0]:
        return pts[0][1]
    if T >= pts[-1][0]:
        return pts[-1][1]
    for i in range(len(pts) - 1):
        T1, Sa1 = pts[i];  T2, Sa2 = pts[i + 1]
        if T1 <= T <= T2:
            t = (T - T1) / (T2 - T1)
            return Sa1 + t * (Sa2 - Sa1)
    return 0.0


def _cqc_rho(omega_i: float, omega_j: float, xi: float) -> float:
    """CQC cross-correlation coefficient (equal damping, Der Kiureghian 1980)."""
    if omega_j == 0:
        return 1.0
    r = omega_i / omega_j          # ≤ 1 if omega_i ≤ omega_j
    r2 = r * r
    num = 8.0 * xi**2 * r**1.5 * (1.0 + r)
    den = (1.0 - r2)**2 + 4.0 * xi**2 * r * (1.0 + r)**2
    return num / den if abs(den) > 1e-20 else 1.0


def _elem_forces_from_u(struc: Structure2D, u_vec: np.ndarray) -> dict:
    """Elastic element end-forces (local) for a given global displacement vector."""
    ef = {}
    zero_fef = np.zeros(6)
    for elem in struc.bar_elements:
        c = struc._elem_cache[elem.id]
        itotv = struc.node_dof_index[elem.node_i]
        jtotv = struc.node_dof_index[elem.node_j]
        si, sj = _end_forces_dispatch(
            struc, c,
            (u_vec[itotv], u_vec[itotv+1], u_vec[itotv+2]),
            (u_vec[jtotv], u_vec[jtotv+1], u_vec[jtotv+2]),
            zero_fef,
        )
        ef[elem.id] = {'i': [float(v) for v in si], 'j': [float(v) for v in sj]}
    return ef


def _solve_spectrum(struc: Structure2D, K: np.ndarray, modal_res: dict,
                    spf, combination_rule: str, direction: str,
                    damping: float = 0.05) -> dict:
    """
    Response spectrum analysis.

    modal_res  : results dict for a Modal analysis case (from _solve_modal)
    spf        : SpectralFunction with .points [[T, Sa], ...]
    combination_rule : 'SRSS' | 'CQC'
    direction  : 'X' | 'Y' | 'XY'
    damping    : damping ratio for CQC (ξ)

    Returns peak (absolute) displacements, reactions, element forces.
    """
    # Multi-point constraints enter here through the modal mode shapes, which
    # _solve_modal already computed on the penalised stiffness. K is used below
    # only as K @ u for force recovery, so it is left un-penalised on purpose:
    # the modes approximately satisfy the constraints, and adding alpha*c cᵀ
    # here would inject a spurious penalty force into the recovered reactions.
    modal_info  = modal_res['modal_info']
    mode_shapes = modal_res['mode_shapes']
    n_modes     = len(modal_info)

    fixed = _build_fixed_dofs(struc)
    ndof  = struc.num_dofs
    node_ids = list(struc.nodes.keys())
    elem_ids = [e.id for e in struc.bar_elements]

    # Plate: the only excitation direction is vertical, whose participation the
    # modal solve placed in the "x" slot (see _solve_modal). Any requested
    # direction collapses to that single vertical case.
    plate = getattr(struc, 'domain', 'plane') == 'plate'
    dirs = ['X'] if plate else (['X', 'Y'] if direction == 'XY' else [direction])

    # Per-direction combination result
    res_disp = {nid: [0.0]*3 for nid in node_ids}
    res_reac = {nid: [0.0]*3 for nid in node_ids}
    res_ef   = {eid: {'i': [0.0]*3, 'j': [0.0]*3} for eid in elem_ids}

    for drn in dirs:
        gamma_key = 'gamma_x' if drn == 'X' else 'gamma_y'

        # Build per-mode displacement vectors
        modal_u: list[np.ndarray] = []
        for k, mi in enumerate(modal_info):
            omega_k   = mi['omega']
            gamma_k   = mi[gamma_key]
            phi_max_k = mi.get('phi_max', 1.0)
            T_k       = mi['period']

            Sa_k = _interpolate_spectrum(spf, T_k)
            Sd_k = Sa_k / omega_k**2 if omega_k > 1e-10 else 0.0

            # u_k = gamma_k * Sd_k * phi_k  (phi_k = mass-normalised)
            # stored shape = phi_k / phi_max_k  → phi_k = stored * phi_max_k
            scale = gamma_k * Sd_k * phi_max_k
            phi_hat = mode_shapes[k]       # {node_id: [ux,uy,rz]}

            u_k = np.zeros(ndof)
            for nid, vals in phi_hat.items():
                itotv = struc.node_dof_index[nid]
                for j in range(3):
                    u_k[itotv + j] = scale * vals[j]
            modal_u.append(u_k)

        # Modal element forces and reactions
        modal_ef:   list[dict] = [_elem_forces_from_u(struc, u) for u in modal_u]
        modal_reac: list[dict] = []
        for u_k in modal_u:
            Ku = K @ u_k
            reac_k = {}
            for nid in node_ids:
                itotv = struc.node_dof_index[nid]
                rx = Ku[itotv]   if fixed[itotv]   else 0.0
                ry = Ku[itotv+1] if fixed[itotv+1] else 0.0
                mz = Ku[itotv+2] if fixed[itotv+2] else 0.0
                reac_k[nid] = [rx, ry, mz]
            modal_reac.append(reac_k)

        # ── Combine modes ─────────────────────────────────────────
        if combination_rule == 'SRSS' or n_modes == 1:
            # SRSS: √(Σ R_k²)
            disp_d = {nid: [0.0]*3 for nid in node_ids}
            reac_d = {nid: [0.0]*3 for nid in node_ids}
            ef_d   = {eid: {'i': [0.0]*3, 'j': [0.0]*3} for eid in elem_ids}
            for k in range(n_modes):
                for nid in node_ids:
                    itotv = struc.node_dof_index[nid]
                    for j in range(3):
                        disp_d[nid][j] += modal_u[k][itotv + j]**2
                        reac_d[nid][j] += modal_reac[k][nid][j]**2
                for eid in elem_ids:
                    for end in ('i', 'j'):
                        for j in range(3):
                            ef_d[eid][end][j] += modal_ef[k][eid][end][j]**2
            # sqrt
            for nid in node_ids:
                disp_d[nid] = [np.sqrt(v) for v in disp_d[nid]]
                reac_d[nid] = [np.sqrt(v) for v in reac_d[nid]]
            for eid in elem_ids:
                for end in ('i', 'j'):
                    ef_d[eid][end] = [np.sqrt(v) for v in ef_d[eid][end]]

        else:  # CQC
            omegas = [mi['omega'] for mi in modal_info]
            disp_d = {nid: [0.0]*3 for nid in node_ids}
            reac_d = {nid: [0.0]*3 for nid in node_ids}
            ef_d   = {eid: {'i': [0.0]*3, 'j': [0.0]*3} for eid in elem_ids}
            for i in range(n_modes):
                for j in range(n_modes):
                    rho = _cqc_rho(omegas[i], omegas[j], damping)
                    for nid in node_ids:
                        itotv = struc.node_dof_index[nid]
                        for d in range(3):
                            disp_d[nid][d] += rho * modal_u[i][itotv+d] * modal_u[j][itotv+d]
                            reac_d[nid][d] += rho * modal_reac[i][nid][d] * modal_reac[j][nid][d]
                    for eid in elem_ids:
                        for end in ('i', 'j'):
                            for d in range(3):
                                ef_d[eid][end][d] += rho * modal_ef[i][eid][end][d] * modal_ef[j][eid][end][d]
            # sqrt
            for nid in node_ids:
                disp_d[nid] = [np.sqrt(max(v, 0.0)) for v in disp_d[nid]]
                reac_d[nid] = [np.sqrt(max(v, 0.0)) for v in reac_d[nid]]
            for eid in elem_ids:
                for end in ('i', 'j'):
                    ef_d[eid][end] = [np.sqrt(max(v, 0.0)) for v in ef_d[eid][end]]

        # ── Accumulate across directions (SRSS for XY) ────────────
        for nid in node_ids:
            for d in range(3):
                res_disp[nid][d] += disp_d[nid][d]**2
                res_reac[nid][d] += reac_d[nid][d]**2
        for eid in elem_ids:
            for end in ('i', 'j'):
                for d in range(3):
                    res_ef[eid][end][d] += ef_d[eid][end][d]**2

    # Final sqrt over directions
    n_dirs = len(dirs)
    if n_dirs > 1:
        for nid in node_ids:
            res_disp[nid] = [np.sqrt(v) for v in res_disp[nid]]
            res_reac[nid] = [np.sqrt(v) for v in res_reac[nid]]
        for eid in elem_ids:
            for end in ('i', 'j'):
                res_ef[eid][end] = [np.sqrt(v) for v in res_ef[eid][end]]
    else:
        for nid in node_ids:
            res_disp[nid] = [np.sqrt(v) for v in res_disp[nid]]
            res_reac[nid] = [np.sqrt(v) for v in res_reac[nid]]
        for eid in elem_ids:
            for end in ('i', 'j'):
                res_ef[eid][end] = [np.sqrt(v) for v in res_ef[eid][end]]

    # Filter near-zero reactions
    res_reac = {nid: v for nid, v in res_reac.items()
                if any(abs(x) > 1e-9 for x in v)}

    return {
        'displacements':  res_disp,
        'reactions':      res_reac,
        'element_forces': res_ef,
        'combination_rule': combination_rule,
        'direction':        direction,
        'is_spectrum':      True,         # marker for results panel
    }


# ---------------------------------------------------------------------------
# Incremental solve — run only the analysis cases marked unsolved
# ---------------------------------------------------------------------------

def solve_pending_cases(struc: Structure2D, results: dict) -> dict | None:
    """Compute the analysis cases whose ``solved`` flag is False, in place.

    The base model has not changed (it is locked), so the per-load-case results
    already in *results* are still valid. A pending Linear case is a
    superposition of those — no stiffness, no factorisation. A pending Mass /
    Modal / Spectrum / geometric case needs K or F, which are re-assembled once
    and only when such a case is actually pending; even then only the pending
    cases are computed, so an already-solved eigenproblem is not redone.

    Combinations and object stresses are rebuilt afterwards — cheap, and a new
    case may feed a combination.

    Returns the updated *results*, or **None** when the per-load-case results
    are absent (a model reloaded without them): the caller must then fall back
    to a full :meth:`Structure2D.calculate`. Marks the pending cases solved.
    """
    if not results.get('displacements'):
        return None            # nothing to superpose — caller does a full solve
    import time
    t0 = time.perf_counter()
    pending = [ac for ac in struc.analysis_cases
               if not getattr(ac, 'solved', False)]
    # Even with no pending analysis case there is work: a combination may have
    # been added or removed, and combinations are always rebuilt below.

    # The results are keyed by the *compiled* geometry; compile again (same
    # deterministic mesh) so ids and any re-assembled K/F line up with them.
    if getattr(struc, 'geometry_objects', None):
        from .geo_expand import expand_geometry
        compiled, _ = expand_geometry(struc)
    else:
        compiled = struc

    case_index = {lc.id: i for i, lc in enumerate(compiled.load_cases)}
    pending_ids = {ac.id for ac in pending}

    K = F = None
    if any(case_needs_stiffness(compiled, compiled.analysis_cases_by_id[i])
           for i in pending_ids
           if i in compiled.analysis_cases_by_id):
        from .assembly import assemble_stiffness
        from .loads import assemble_loads
        K = assemble_stiffness(compiled)
        F = assemble_loads(compiled)

    ac_results = results.setdefault('analysis_cases', {})
    stored_stiffness = results.setdefault('stored_stiffness', {})
    # Compute in model order so a pending Spectrum sees its Modal, whether that
    # Modal was already solved (in ac_results) or is pending in this pass.
    for ac in compiled.analysis_cases:
        if ac.id in pending_ids:
            ac_results[ac.id] = _compute_one_analysis_case(
                compiled, ac, results, K, F, case_index, stored_stiffness,
                ac_results)

    # Diagrams, then every combination (a new case may feed one), then stresses.
    compute_combo_distributions(compiled, results)
    results['combinations'] = {}
    for combo in _combinations_in_dependency_order(compiled):
        results['combinations'][combo.id] = _apply_combination(
            combo, compiled, results)
    _recover_object_stresses(compiled, results)

    for ac in struc.analysis_cases:      # the editable cases carry the flag
        ac.solved = True
    # Same stats keys as Structure2D.calculate(), so the GUI's post-Run
    # statistics box works the same whether this was a full or a pending-only
    # solve. Only 'pending' analysis cases were actually (re)computed here —
    # recorded separately from the total case count already in 'analysis_cases'.
    fixed = _build_fixed_dofs(compiled)
    results['system_size'] = {
        'nodes':        len(compiled.nodes),
        'bar_elements': len(compiled.bar_elements),
        'tri_elements': len(getattr(compiled, 'tri_elements', [])),
        'dofs_total':   int(compiled.num_dofs),
        'dofs_free':    int((~fixed).sum()),
        'dofs_fixed':   int(fixed.sum()),
    }
    results['timing'] = {'total': time.perf_counter() - t0}
    results['pending_cases_solved'] = len(pending)
    return results
