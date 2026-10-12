"""EC2 §6.4 punching-shear verification of slabs at columns (plate domain).

A column punches a slab locally; the slab FEM does not know the column's size or
shape, so the check runs over the explicit :class:`~xdfem2d.models.PunchColumn`
list the user defines. For each column and each design combination it computes
the design punching shear stress ``v_Ed`` (with the eccentricity factor β from
the transferred moments) and the concrete resistance ``v_Rd,c``, both delegated
to :mod:`eurocodepy.ec2.uls.punch`, and reports the utilization and whether
punching reinforcement is required.

Design lives here — a separate module — rather than in ``rc_design`` because
punching is its own verification with its own modelling data (columns), matching
how the application keeps modelling and design apart.
"""
from __future__ import annotations

MM = 1000.0        # m → mm


def _slab_section_at(struc, node_id):
    """The plate section of a slab element (triangle DKT/MITC3 or quad
    DKT4/MITC4 — dev/IMPLEMENT_QUAD.md Phase 7) touching *node_id* (or None).

    A node can touch more than one plate element, and in a mixed tri/quad
    mesh that can include both kinds. When every touching element resolves
    to the same section this is unambiguous, as before; when they resolve to
    *different* sections (material and/or thickness), the section used by
    the most touching elements wins, ties broken by the fixed scan order
    below (triangles in element order, then quads in element order) — a
    real conflict at a punching column (e.g. two adjacent slabs of different
    thickness meeting at the column node) is resolved deterministically
    rather than left to dict/iteration order, but is not surfaced to the
    caller as a warning: ``design_punching`` has no warnings channel today,
    and adding one is a broader change than this phase's scope — deferred,
    not fixed."""
    from collections import Counter
    candidates = []
    for tri in getattr(struc, 'tri_elements', []):
        if node_id in (tri.node_i, tri.node_j, tri.node_k):
            sec = struc.tri_sections.get(tri.section_name)
            if sec is not None and getattr(sec, 'formulation', 'CST') in (
                    'DKT', 'MITC3'):
                candidates.append(sec)
    for quad in getattr(struc, 'quad_elements', []):
        if node_id in (quad.node_i, quad.node_j, quad.node_k, quad.node_l):
            sec = struc.quad_sections.get(quad.section_name)
            if sec is not None and getattr(sec, 'formulation', 'MITC4') in (
                    'DKT4', 'MITC4'):
                candidates.append(sec)
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    key = lambda s: (s.material_name, s.thickness)
    counts = Counter(key(s) for s in candidates)
    top_key = counts.most_common(1)[0][0]
    for sec in candidates:
        if key(sec) == top_key:
            return sec
    return candidates[0]   # unreachable, but avoids a bare fall-through


def _rho_l_at(struc, results, node_id, d_m, default=0.005):
    """Flexural reinforcement ratio ρl near a column, from the slab design in
    ``results['tri_reinforcement']`` (bottom bars), or a default when absent.

    ρl = As/(b·d) with b = 1 m; the mean of the bottom-x/bottom-y areas of
    the triangles AND quads (dev/IMPLEMENT_QUAD.md Phase 7 — a quad row's
    ``'triangle'`` key holds the quad's own id, see
    ``rc_design.design_concrete_planes``'s docstring) touching the node is
    used, capped at EC2's 0.02."""
    payload = (results or {}).get('tri_reinforcement') or {}
    rows = payload.get('rows') or []
    if not rows or d_m <= 0:
        return default
    elems = {e.id: e for e in list(getattr(struc, 'tri_elements', []))
             + list(getattr(struc, 'quad_elements', []))}
    areas = []
    for r in rows:
        e = elems.get(r.get('triangle'))
        if e is None:
            continue
        node_ids = ((e.node_i, e.node_j, e.node_k, e.node_l)
                    if hasattr(e, 'node_l') else (e.node_i, e.node_j, e.node_k))
        if node_id not in node_ids:
            continue
        as_bot = 0.5 * ((r.get('Asx_bot') or 0.0) + (r.get('Asy_bot') or 0.0))
        areas.append(as_bot)
    if not areas:
        return default
    rho = (sum(areas) / len(areas)) / (1.0 * d_m)
    return min(max(rho, 1e-4), 0.02)


def design_punching(struc, results: dict,
                    combinations=None, dmax: float = 20.0,
                    gamma_c: float = 1.5, gamma_s: float = 1.15,
                    gamma_v: float = 1.4, alpha_cc: float = 1.0,
                    rho_l_default: float = 0.005,
                    eta_sys: float = 1.5, edition: str = "2004",
                    with_reports: bool = False) -> list:
    """EC2 §6.4 punching check for every :class:`PunchColumn`.

    For each column and each combination the design punching stress ``v_Ed`` and
    the resistance ``v_Rd,c`` (with its minimum) are computed; the row carries
    the utilization ``v_Ed/v_Rd`` and ``needs_reinf``. The punching force is the
    column's ``force`` [kN] or, when None, the support reaction ``Rz`` at its
    node; the transferred moments come from the node's reaction ``Mx``/``My``.

    Envelope combinations carry no single reaction and are skipped when the force
    is taken from the reaction.

    Returns a list of dict, one per (column, combination); ``governing`` is the
    largest-utilization row per column.
    """
    # Pick the EC2 edition: the 2023 punching model lives in ec2.uls2023, the
    # 2004 one in ec2.uls. Both expose the same function names.
    is_2023 = str(edition) == "2023"
    if is_2023:
        from eurocodepy.ec2 import uls2023 as _punch
    else:
        from eurocodepy.ec2 import uls as _punch
    calc_perimeters = _punch.calc_perimeters
    calc_vedp = _punch.calc_vedp
    from .rc_design import _section_strengths

    combos = results.get('combinations', {})
    if combinations is not None:
        wanted = set(combinations)
        combos = {k: v for k, v in combos.items() if k in wanted}

    rows: list = []
    for col in getattr(struc, 'punch_columns', []):
        sec = _slab_section_at(struc, col.node_id)
        if sec is None:
            continue
        st = _section_strengths(struc, sec)
        if st is None:
            continue
        fck, fyk = st
        fyd = fyk / gamma_s
        covers = sec.resolved_covers()
        d_m = sec.thickness - 0.5 * (covers['bot_x'] + covers['bot_y'])
        if d_m <= 0:
            continue
        dv = d_m * MM
        bx = col.bx * MM
        by = None if col.shape == 'circular' else col.by * MM
        dx, dy = col.dx * MM, col.dy * MM

        col_rows = []
        col_row_src = []      # parallel re-run inputs for the report (with_reports)
        for combo_id, cd in combos.items():
            reac = cd.get('reactions')
            if isinstance(reac, dict) and 'max' in reac:
                continue                      # envelope → no single reaction
            r = (reac or {}).get(col.node_id) if reac else None
            if col.force is not None:
                ned = abs(col.force)
                medx = medy = 0.0
                if r is not None:
                    medx, medy = abs(r[1]), abs(r[2])
            elif r is not None:
                ned, medx, medy = abs(r[0]), abs(r[1]), abs(r[2])
            else:
                continue
            if ned <= 0.0:
                continue

            rho_l = _rho_l_at(struc, results, col.node_id, d_m, rho_l_default)
            # Perimeters: (u0 at the loaded-area face, u1 the control perimeter,
            # bb). The control perimeter is at 2·d for 2004, 0.5·d for 2023.
            u0, u1, _bb = calc_perimeters(dv, bx, by, position=col.position,
                                          dx=dx, dy=dy)
            v_ed = float(calc_vedp(ned, medx, medy, dv, bx, by,
                                   position=col.position, dx=dx, dy=dy))
            # Apply the user's minimum β (EC2 §6.4.3(6) simplified values) as a
            # floor on the computed one — supports with no transferred moment
            # otherwise give the minimum β.
            v0 = ned / u1 / dv * 1e3               # v_Ed without β [MPa]
            beta_min = getattr(col, 'beta_min', 1.0) or 1.0
            beta = v_ed / v0 if v0 > 0 else 1.0
            beta_eff = max(beta, beta_min)
            v_ed = beta_eff * v0

            if is_2023:
                v_rdc = float(_punch.calc_vrdcp(
                    dmax, rho_l, fck, dv, bx, by, gamma_v=gamma_v,
                    position=col.position, dx=dx, dy=dy))
                v_rd_min = float(_punch.calc_vrdcminp(fck, fyd, dv, dmax=dmax,
                                                      gamma_v=gamma_v))
                # EN 1992-1-1:2023 §8.4.4: the resistance at the 0.5·d
                # perimeter is capped at η_sys·τRd,c even with reinforcement.
                v_rd_max = eta_sys * v_rdc
                v_ed_max = v_ed
                # Outer perimeter beyond which no punching reinforcement is
                # needed: u_out,ef = β·VEd/(τRd,c·d).
                needs0 = v_ed > max(v_rdc, v_rd_min)
                u_out_eff = (beta_eff * ned * 1e3 / (v_rdc * dv) / MM
                             if (needs0 and v_rdc > 0) else None)
            else:
                v_rdc = float(_punch.calc_vrdcp(rho_l, fck, dv, gamma_c=gamma_c))
                v_rd_min = float(_punch.calc_vrdcminp(fck, dv))
                # EN 1992-1-1:2004 §6.4.5(3): crushing at the loaded-area
                # perimeter u0 — v_Ed(u0) ≤ v_Rd,max = 0.5·ν·fcd.
                nu = 0.6 * (1.0 - fck / 250.0)
                fcd = alpha_cc * fck / gamma_c
                v_rd_max = 0.5 * nu * fcd
                v_ed_max = beta_eff * ned / u0 / dv * 1e3
                u_out_eff = None
            v_rd = max(v_rdc, v_rd_min)
            util = v_ed / v_rd if v_rd > 0 else float('inf')
            needs = bool(v_ed > v_rd)
            crushing = bool(v_ed_max > v_rd_max)
            # Cast to plain Python types: the eurocodepy calls return numpy
            # scalars (float64 / bool_) that the results JSON encoder does not
            # serialise ("bool is not serializable").
            col_rows.append({
                'column': col.id, 'node': col.node_id, 'combination': combo_id,
                'position': col.position, 'N_Ed': float(ned),
                'beta': float(beta_eff),
                'd': float(d_m), 'rho_l': float(rho_l),
                # Control perimeter [m] — at 2·d (2004) or 0.5·d (2023).
                'u1': float(u1 / MM),
                'u_out_eff': (float(u_out_eff)
                              if u_out_eff is not None else None),
                'v_Ed': float(v_ed),
                'v_Rdc': float(v_rdc), 'v_Rd_min': float(v_rd_min),
                'v_Rd_max': float(v_rd_max),
                'utilization': float(util), 'needs_reinf': needs,
                'crushing': crushing,
                'edition': "2023" if is_2023 else "2004",
                'governing': False,
            })
            if with_reports:
                col_row_src.append(dict(
                    d=d_m, bx=col.bx,
                    by=(None if col.shape == 'circular' else col.by),
                    fck=fck, fyk=fyk, position=col.position,
                    dx=col.dx, dy=col.dy, gamma_c=gamma_c, gamma_s=gamma_s,
                    gamma_v=gamma_v, alpha_cc=alpha_cc, dmax=dmax,
                    eta_sys=eta_sys, edition=("2023" if is_2023 else "2004"),
                    n_ed=ned, m_ed_x=medx, m_ed_y=medy, rho_l=rho_l,
                    beta_min=beta_min))
        if col_rows:
            gov = max(col_rows, key=lambda x: x['utilization'])
            gov['governing'] = True
            # On-demand: rebuild the governing column's step-by-step report via
            # the composite (same numbers by construction). Recording only.
            if with_reports and col_row_src:
                from eurocodepy.calc_report import CalcReport
                from eurocodepy.ec2.uls import (PunchInput,
                                                eurocode2_punching_check)
                s = col_row_src[col_rows.index(gov)]
                rep = CalcReport(
                    title=(f"Punching {col.id} — "
                           + ("EN 1992-1-1:2023" if is_2023
                              else "EN 1992-1-1:2004")),
                    meta={'id': col.id, 'column': col.id, 'node': col.node_id,
                          'combination': gov['combination'],
                          'position': col.position,
                          'utilization': gov['utilization'],
                          'ok': gov['utilization'] <= 1.0})
                pinp = PunchInput(
                    d=s['d'], bx=s['bx'], by=s['by'], fck=s['fck'],
                    fyk=s['fyk'], position=s['position'], dx=s['dx'], dy=s['dy'],
                    gamma_c=s['gamma_c'], gamma_s=s['gamma_s'],
                    gamma_v=s['gamma_v'], alpha_cc=s['alpha_cc'],
                    dmax=s['dmax'], eta_sys=s['eta_sys'], edition=s['edition'])
                eurocode2_punching_check(
                    pinp, s['n_ed'], s['m_ed_x'], s['m_ed_y'],
                    rho_l=s['rho_l'], beta_min=s['beta_min'], trace=rep)
                gov['report'] = rep.to_dict()
            rows.extend(col_rows)
    return rows


def store_punching(results: dict, rows, *, prefs: dict | None = None):
    """Embed a punching design into ``results['punching']`` (or remove it when
    *rows* is empty). Returns the stored payload (or None)."""
    if not rows:
        results.pop('punching', None)
        return None
    payload = {'rows': rows, 'prefs': dict(prefs or {})}
    results['punching'] = payload
    return payload


def design_punching_and_store(struc, results: dict, *, combinations=None,
                              prefs: dict | None = None, **kw):
    """Design all punching columns and store the result in one step."""
    rows = design_punching(struc, results, combinations=combinations, **kw)
    return store_punching(results, rows, prefs=prefs)
