"""Expand objects (parametric geometry) into concrete nodes + bar
elements for analysis.

This is the *compile* step of the two-layer design (see
dev/geometry_objects_plan.md): the editable model holds objects (arcs, polylines);
:func:`expand_geometry` turns them into a plain :class:`~xdfem2d.structure.Structure2D`
whose geometry the solver can assemble. The expansion is **deterministic** —
node/element ids derive from the object id and a local index
(``<object>.n<k>`` / ``<object>.e<k>``), and endpoints that coincide (within a
tolerance) with existing nodes or with another object's nodes are **merged**, so
objects tie cleanly into the frame and into each other.

The mesh produced here is transient and never stored in the editable model.
"""
from __future__ import annotations

import copy
import math

from .models import (GeoArc, GeoMultisegment, GeoSegment,
                     GeoRectangle, GeoPolygon)

# Algorithm version — bump when the discretisation changes so that cached
# results (Option B) computed by an older version can be detected as stale.
# 2: a surface object referencing a QuadSection can now mesh into QuadElements
# instead of an all-triangle split (dev/IMPLEMENT_QUAD.md Phase 6).
EXPANSION_VERSION = 2

Point = tuple[float, float]


# ── point generation (node-driven: geometry read from the object's nodes) ───

def circle_from_3pts(p1, p2, p3):
    """Return (cx, cy, r) of the circle through three points, or None if they
    are collinear."""
    ax, ay = p1; bx, by = p2; cx, cy = p3
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-12:
        return None
    a2 = ax * ax + ay * ay; b2 = bx * bx + by * by; c2 = cx * cx + cy * cy
    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
    return (ux, uy, math.hypot(ax - ux, ay - uy))


def _subdivide(a, b, n) -> list[Point]:
    return [(a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n)
            for i in range(n + 1)]


def _line_points(pts, obj) -> list[Point]:
    a, b = pts[0], pts[1]
    L = math.hypot(b[0] - a[0], b[1] - a[1])
    if L < 1e-12:
        return []
    n = (max(1, math.ceil(L / obj.max_chord)) if obj.max_chord and obj.max_chord > 0
         else max(1, int(obj.divisions)))
    return _subdivide(a, b, n)


def _polyline_points(pts, obj) -> list[Point]:
    if len(pts) < 2:
        return []
    spans = list(zip(pts, pts[1:]))
    if obj.closed and len(pts) > 2:
        spans.append((pts[-1], pts[0]))
    # Per-span override (GeoMultisegment.span_divisions): only trusted when it
    # has exactly one entry per current span — a stale list left over from a
    # vertex added/removed since it was set would silently misapply to the
    # wrong span, so a length mismatch falls back to the uniform `divisions`
    # instead (see the field's docstring).
    span_div = getattr(obj, "span_divisions", None)
    if not span_div or len(span_div) != len(spans):
        span_div = None
    out: list[Point] = [pts[0]]
    for i, (a, b) in enumerate(spans):
        L = math.hypot(b[0] - a[0], b[1] - a[1])
        if obj.max_chord and obj.max_chord > 0:
            n = max(1, math.ceil(L / obj.max_chord))
        elif span_div is not None:
            n = max(1, int(span_div[i]))
        else:
            n = max(1, int(obj.divisions))
        out.extend(_subdivide(a, b, n)[1:])
    return out


def _arc_points(pts, obj) -> list[Point]:
    """Points along the arc through pts = [start, mid, end].

    The mid point is not just a construction aid to pick the sweep direction
    — unlike add_geo_arc's own mid (always the exact angular bisector),
    add_geo_arc_3pts's mid may sit anywhere along the arc, and a single
    uniform-angle sampling from start to end then generally skips right over
    it (see dev/ — a 3-point arc through (0,0),(6,4),(10,0) sampled that way
    lands its nearest point near (5.0, 4.1), not (6.0, 4.0)). Sampling the
    two sub-arcs start→mid and mid→end separately, and concatenating without
    duplicating the shared mid sample, guarantees the mid point is always an
    exact mesh vertex — the same guarantee _polyline_points already gives
    every real vertex of a GeoMultisegment."""
    p1, pm, p2 = pts[0], pts[1], pts[2]
    circ = circle_from_3pts(p1, pm, p2)
    if circ is None:
        return [p1, pm, p2]                     # degenerate → straight
    cx, cy, r = circ
    a1 = math.atan2(p1[1] - cy, p1[0] - cx)
    am = math.atan2(pm[1] - cy, pm[0] - cx)
    a2 = math.atan2(p2[1] - cy, p2[0] - cx)
    two_pi = 2.0 * math.pi
    ccw = (a2 - a1) % two_pi                     # CCW sweep a1→a2
    am_ccw = (am - a1) % two_pi
    sweep = ccw if am_ccw <= ccw + 1e-12 else -(two_pi - ccw)
    # Split the total sweep into the two sub-sweeps either side of the mid
    # point, signed the same way as the full sweep (am lies "along the way"
    # by construction, so both sub-sweeps always agree in sign with it).
    #
    # Bug fixed here: the CCW branch used to take the *raw* (am - a1), which
    # is only right when a1 and am sit on the same side of atan2's branch cut
    # (±180°). An arc mostly in the negative-x half-plane routinely has a1
    # near +170° and am near -170° — a real angular gap of 20°, but a raw
    # difference of -340° — so sweep1 came out wildly wrong (and sweep2 with
    # it) for any arc crossing that cut, discretising into a mesh that swept
    # almost the wrong way around the circle. am_ccw (already computed above
    # as (am - a1) % two_pi, i.e. the wrapped CCW distance from a1 to am) is
    # the correct magnitude in the CCW case, mirroring the CW branch below
    # which already wraps its own (a1 - am) % two_pi the same way.
    sweep1 = am_ccw if sweep >= 0 else -((a1 - am) % two_pi)
    sweep2 = sweep - sweep1

    def _sub_n(sub_sweep):
        # divisions is shared out between the two sub-arcs by their angular
        # share of the total sweep (so the common case — mid at the exact
        # bisector, e.g. every add_geo_arc-built arc — reproduces the old
        # single-sweep sampling exactly); max_chord instead sizes each
        # sub-arc independently from its own arc length, same as the
        # whole-arc case did.
        share = max(1, int(obj.divisions))
        if sweep:
            share = max(1, round(share * abs(sub_sweep) / abs(sweep)))
        if obj.max_chord and obj.max_chord > 0:
            share = max(share, math.ceil(abs(r * sub_sweep) / obj.max_chord))
        return share

    n1 = _sub_n(sweep1)
    n2 = _sub_n(sweep2)
    pts1 = [(cx + r * math.cos(a1 + sweep1 * i / n1),
             cy + r * math.sin(a1 + sweep1 * i / n1)) for i in range(n1 + 1)]
    pts2 = [(cx + r * math.cos(am + sweep2 * i / n2),
             cy + r * math.sin(am + sweep2 * i / n2)) for i in range(n2 + 1)]
    return pts1 + pts2[1:]          # pts1's last point IS pts2's first (am)


def _surface_polygon(s, obj) -> list[Point]:
    """Ordered boundary vertices of a surface object, from its node positions.
    Both rectangles and polygon surfaces are stored with a node per vertex, so
    the polygon is simply the ordered defining nodes."""
    if not isinstance(obj, (GeoRectangle, GeoPolygon)):
        return []
    pts = [(s.nodes[nid].x, s.nodes[nid].y)
           for nid in getattr(obj, "node_ids", []) if nid in s.nodes]
    return pts if len(pts) >= 3 else []


# Quad formulation -> the triangle formulation used for a per-cell or
# whole-object fallback (dev/IMPLEMENT_QUAD.md Phase 6): the membrane pair
# (Q4, QM6) both fall back to CST, the two plate formulations fall back to
# their nearest triangle analogue (DKT4 -> DKT, MITC4 -> MITC3) — same
# correspondence QuadSection's own docstring draws between the two families.
_QUAD_TO_TRI_FORMULATION = {'Q4': 'CST', 'QM6': 'CST',
                           'DKT4': 'DKT', 'MITC4': 'MITC3'}


def _tri_fallback_section_for(compiled, qsec):
    """Find (or create) a TriSection matching *qsec*'s material, thickness and
    plane-strain, formulation mapped via :data:`_QUAD_TO_TRI_FORMULATION` —
    mirrors ``xdfem2d.structure_io._tri_section_for``'s find-or-create
    pattern. Used when a surface object references a QuadSection but a
    particular cell (or the whole object) still needs a triangle: a per-cell
    quality fallback in :func:`_expand_surface`, or a non-convex/non-quad
    outline meshed by the Delaunay branch, which never emits quads
    (dev/IMPLEMENT_QUAD.md Phase 6)."""
    formulation = _QUAD_TO_TRI_FORMULATION.get(qsec.formulation, 'CST')
    for cand in compiled.tri_sections.values():
        if (cand.material_name == qsec.material_name
                and abs(cand.thickness - qsec.thickness) < 1e-12
                and cand.plane_strain == qsec.plane_strain
                and cand.formulation == formulation):
            return cand.name
    name = f"{qsec.name}~tri"
    base = name; n = 1
    while name in compiled.tri_sections:
        n += 1; name = f"{base}#{n}"
    compiled.add_tri_section(name, qsec.material_name,
                             thickness=qsec.thickness,
                             plane_strain=qsec.plane_strain,
                             formulation=formulation)
    return name


def _expand_surface(struc, compiled, m, mid, node_key, key_fn,
                    existing_tris, existing_quads):
    """Mesh a surface object into nodes + TriElements/QuadElements in
    *compiled*. Returns the trace entry
    ``{'nodes': [...], 'elems': [], 'tris': [...], 'quads': [...]}`` (all
    empty lists when degenerate).

    *sec* (``m.tri_section_name``) resolves against ``struc.tri_sections``
    and/or ``struc.quad_sections`` (dev/IMPLEMENT_QUAD.md Phase 6 widens the
    field's meaning rather than adding a second one — see
    ``GeoRectangle``/``GeoPolygon``'s docstring). Phase 8's "Panel sections"
    GUI dialog always creates a same-named TriSection AND QuadSection pair
    together, so *sec* alone can no longer signal which kind to mesh into —
    ``m.prefer_quad`` (default False) breaks the tie: False checks
    ``tri_sections`` first (old behaviour, unchanged for files saved before
    this field existed), True checks ``quad_sections`` first. Either way,
    if the preferred dict has no match the other one is still tried — a
    lone unpaired QuadSection (or TriSection) still resolves regardless of
    ``prefer_quad``, same fallback Phase 6 always had. When *sec* resolves
    to a QuadSection and the outline is a convex quad, the structured branch
    tries ``mesh_quad_structured_cells`` and emits a QuadElement for every
    well-shaped cell, a TriElement (via :func:`_tri_fallback_section_for`)
    for the rest — a surface area spring or temperature load on the object
    applies per-cell to whichever kind (quad or triangle) each cell ended up
    as (see :func:`_apply_surface_area_springs` / :func:`_apply_area_temperatures`).
    Every other case — no QuadSection, a non-quad or non-convex outline —
    meshes exactly as before Phase 6, via ``mesh_polygon`` (triangle-only,
    structured-or-Delaunay)."""
    from .meshing import mesh_polygon, mesh_quad_structured_cells, _is_convex_quad
    poly = _surface_polygon(struc, m)
    empty = {'nodes': [], 'elems': [], 'tris': [], 'quads': []}
    if len(poly) < 3:
        return empty
    # Conform to every existing node lying on the boundary or inside --
    # every node materialised by an object processed earlier (see
    # expand_geometry's mesh-order docstring for why "earlier" is enough:
    # objects are ordered finest target_size first, so whichever of two
    # differently-sized panels shares this edge and is coarser is always
    # the one processed later, and the unstructured branch's own boundary
    # refinement (refine_boundary) unconditionally honours every point in
    # `forced`, however awkward the fit -- no in-between accommodation
    # needed here for that side of the pairing).
    forced = [(nd.x, nd.y) for nd in compiled.nodes.values()]
    target = float(getattr(m, "target_size", 0.5) or 0.5)

    sec = getattr(m, "tri_section_name", "") or ""
    prefer_quad = bool(getattr(m, "prefer_quad", False))
    if prefer_quad:
        qsec = struc.quad_sections.get(sec)
        tsec = struc.tri_sections.get(sec) if qsec is None else None
    else:
        tsec = struc.tri_sections.get(sec)
        qsec = struc.quad_sections.get(sec) if tsec is None else None
    quad_eligible = qsec is not None

    # Whether quads never got a chance geometrically (outline not a convex
    # quad) vs. a chance that the structured grid then couldn't honour (a
    # forced point -- typically a neighbouring object's node -- landing off
    # the grid): tracked here, not re-derived later, because _is_convex_quad
    # is exactly the test mesh_quad_structured_cells/_structured_grid already
    # ran; model_check surfaces this via trace[mid]['quad_fallback_reason'].
    quad_outline_is_convex_quad = _is_convex_quad(poly)

    points = quads_cells = tris_cells = None
    quad_fallback_reason = None
    if quad_eligible:
        result = mesh_quad_structured_cells(poly, target, forced=forced)
        if result is not None:
            points, quads_cells, tris_cells = result
        elif not quad_outline_is_convex_quad:
            quad_fallback_reason = "outline is not a convex quadrilateral"
        else:
            quad_fallback_reason = (
                "a point shared with another object (or the model) does not "
                "land on the structured grid at this element size")
    if points is None:
        points, tris_cells = mesh_polygon(poly, target, forced=forced)
        quads_cells = []
    if not tris_cells and not quads_cells:
        return empty

    # Only materialise points that actually end up in a kept cell. mesh_polygon
    # (and, in principle, the structured branch) can return boundary/interior
    # points that no surviving triangle/quad references -- e.g. a thin/sliver
    # outline where boundary refinement adds points along the long edge but
    # the Delaunay pass, after dropping cells whose centroid falls outside the
    # polygon, never uses some of them. Adding a node for every returned point
    # regardless used to leave those as permanently unconnected nodes in the
    # compiled mesh -- exactly the "free node" false positives Model check
    # was reporting for otherwise-clean geometry-only models.
    used_idx: set[int] = set()
    for cell in (quads_cells or ()):
        used_idx.update(cell)
    for cell in (tris_cells or ()):
        used_idx.update(cell)

    n_counter = 0
    local_to_id: list[str | None] = [None] * len(points)
    node_ids: list[str] = []
    for idx in sorted(used_idx):
        x, y = points[idx]
        k = key_fn(x, y)
        nid = node_key.get(k)
        if nid is None:
            nid = f"{mid}.n{n_counter}"; n_counter += 1
            compiled.add_node(nid, x, y)
            node_key[k] = nid
        local_to_id[idx] = nid
        if nid not in node_ids:
            node_ids.append(nid)

    quad_ids: list[str] = []
    q_counter = 0
    for (a, b, c, d) in quads_cells or []:
        q_counter += 1
        qid = f"{mid}.q{q_counter}"
        while qid in existing_quads:
            qid = f"{mid}.q{q_counter}_{len(existing_quads)}"
        existing_quads.add(qid)
        compiled.add_quad_element(qid, local_to_id[a], local_to_id[b],
                                  local_to_id[c], local_to_id[d], sec)
        quad_ids.append(qid)

    # The section for any triangle cell: the object's own section when it is
    # already a TriSection (unchanged pre-Phase-6 behaviour, including a
    # dangling/empty name passed through as-is for reference_problems() to
    # catch) — checked directly against tri_sections here, NOT via the
    # prefer_quad-ordered `tsec` above, so a real same-named TriSection (the
    # Phase 8 "Panel sections" pairing) is reused for fallback cells even
    # when prefer_quad=True deliberately left `tsec` as None for dispatch
    # priority; otherwise, when no real TriSection exists under that name,
    # the derived fallback triangle section.
    real_tsec = struc.tri_sections.get(sec)
    tri_sec = sec if real_tsec is not None else (
        _tri_fallback_section_for(compiled, qsec) if qsec is not None else sec)
    tri_ids: list[str] = []
    t_counter = 0
    for (a, b, c) in tris_cells or []:
        t_counter += 1
        tid = f"{mid}.t{t_counter}"
        while tid in existing_tris:
            tid = f"{mid}.t{t_counter}_{len(existing_tris)}"
        existing_tris.add(tid)
        compiled.add_tri_element(tid, local_to_id[a], local_to_id[b],
                                 local_to_id[c], tri_sec)
        tri_ids.append(tid)

    result = {'nodes': node_ids, 'elems': [], 'tris': tri_ids, 'quads': quad_ids}
    # Reporting-only fields for model_check (#3/#7 -- never consulted by the
    # solver or by anything else that reads this trace): whether this object
    # actually asked for quads, whether it got any, why not when it didn't,
    # and the two formulations in play when both a quad section and its
    # triangle fallback are present in the same object's cells.
    if quad_eligible:
        result['quad_requested'] = True
        result['quad_formulation'] = qsec.formulation
        if not quad_ids:
            result['quad_fallback_reason'] = (
                quad_fallback_reason
                or "every grid cell failed the quad quality thresholds "
                   "(skew/aspect ratio)")
        if quad_ids and tri_ids:
            tri_sec_obj = compiled.tri_sections.get(tri_sec)
            result['tri_fallback_formulation'] = (
                tri_sec_obj.formulation if tri_sec_obj is not None else None)
    return result


def _node_restrained_dofs(struc, node_id, by_node):
    """Union of the restrained (ux, uy, tz) over every support on *node_id*."""
    ux = uy = tz = False
    for sname in by_node.get(node_id, ()):
        sup = struc.supports.get(sname)
        if sup is not None:
            ux |= bool(sup.ux); uy |= bool(sup.uy); tz |= bool(sup.tz)
    return ux, uy, tz


def _support_for_dofs(compiled, ux, uy, tz):
    """Name of a support restraining exactly (ux, uy, tz): reuse an existing one
    with those DOFs (so a user's own support is preferred and no equivalent is
    duplicated), else create the canonical one (``FIXED`` / ``PIN`` / …). None
    when no DOF is restrained."""
    from .models import canonical_support_name
    if not (ux or uy or tz):
        return None
    for name, sup in compiled.supports.items():
        if (bool(sup.ux), bool(sup.uy), bool(sup.tz)) == (ux, uy, tz):
            return name
    name = canonical_support_name(
        ux, uy, tz, domain=getattr(compiled, 'domain', 'plane'))
    compiled.add_support(name, ux=ux, uy=uy, tz=tz)
    return name


def _propagate_edge_supports_explicit(struc, compiled, ids, edge_supports, tol):
    """Direct per-edge restraint: edge i (``ids[i]``-``ids[i+1]``) gets exactly
    ``edge_supports[i]`` (a support name, or ``None``/``"free"`` for no
    restraint), applied to every node the mesher generated along it — corners
    included, each resolved to the more restrictive of its two adjacent
    edges when they differ.

    Bypasses the corner-intersection logic in :func:`_propagate_edge_supports`
    entirely — that mode derives an edge's restraint from its two corners,
    which cannot express an alternating pattern (clamped/simply/clamped/
    simply): the corner shared by a clamped edge and a simply-supported edge
    has to be promoted to clamped for the clamped edge to work, and that
    promotion then also clamps the simply-supported edge through the same
    corner. See :attr:`xdfem2d.models.GeoRectangle.edge_supports`.
    """
    n = len(ids)
    if len(edge_supports) != n:
        raise ValueError(
            f"edge_supports has {len(edge_supports)} entries, need {n} "
            f"(one per edge, in perimeter order).")

    def dofs_of(name):
        if not name or name == 'free':
            return (False, False, False)
        sup = struc.supports.get(name)
        if sup is None:
            raise ValueError(f"edge_supports references unknown support "
                             f"'{name}'")
        return (bool(sup.ux), bool(sup.uy), bool(sup.tz))

    edge_dofs = [dofs_of(s) for s in edge_supports]
    # Corner i sits between edge (i-1) (ending there) and edge i (starting
    # there); its restraint is the union (more restrictive) of the two.
    corner_dofs = [tuple(a or b for a, b in zip(edge_dofs[i - 1], edge_dofs[i]))
                  for i in range(n)]

    already = {(a.node_id, a.support_name) for a in compiled.support_assignments}
    added: list[tuple[str, str]] = []

    def assign(nid, dofs):
        if not any(dofs):
            return
        name = _support_for_dofs(compiled, *dofs)
        if name is None or (nid, name) in already:
            return
        compiled.assign_support(nid, name)
        already.add((nid, name))
        added.append((nid, name))

    for i, cid in enumerate(ids):
        assign(cid, corner_dofs[i])

    seg_tol = max(tol, 1e-6)
    for i in range(n):
        a_id, b_id = ids[i], ids[(i + 1) % n]
        dofs = edge_dofs[i]
        if a_id == b_id or not any(dofs):
            continue
        na, nb = struc.nodes[a_id], struc.nodes[b_id]
        ax, ay = na.x, na.y
        dx, dy = nb.x - ax, nb.y - ay
        L2 = dx * dx + dy * dy
        if L2 < 1e-18:
            continue
        for nid, nd in compiled.nodes.items():
            if nid == a_id or nid == b_id:
                continue        # corners already handled above
            t = ((nd.x - ax) * dx + (nd.y - ay) * dy) / L2
            if not (-1e-9 <= t <= 1.0 + 1e-9):
                continue
            px, py = ax + t * dx, ay + t * dy
            if (nd.x - px) ** 2 + (nd.y - py) ** 2 > seg_tol * seg_tol:
                continue
            assign(nid, dofs)
    return added


def _propagate_edge_supports(struc, compiled, m, tol):
    """Restrain the mesh nodes along an edge from the restraints at its two
    corners, per the surface's ``edge_support_mode``:

      * ``"none"``   — nothing.
      * ``"common"`` — the DOFs restrained at BOTH corners (their intersection),
        compared by DOF (not by support name). An empty intersection — e.g. one
        corner ux, the other uy — restrains nothing. The result is named by the
        canonical supports table (reusing an existing equal support). Default.
      * ``"equal"``  — only when both corners restrain the SAME DOF set.

    A surface is defined by its corner nodes, and the nodes along an edge are
    created by the mesher only after the corner supports are assigned; this is
    what lets "fixed along the base" be written by holding just the two corners.
    Edges come from consecutive defining nodes, the last closing back to the
    first — the same traversal the surface itself is built from.

    If the object carries an explicit ``edge_supports`` list (see
    :attr:`xdfem2d.models.GeoRectangle.edge_supports`), that takes over
    entirely instead — see :func:`_propagate_edge_supports_explicit`.
    """
    ids = [nid for nid in getattr(m, 'node_ids', []) or [] if nid in struc.nodes]
    edge_supports = getattr(m, 'edge_supports', None)
    if edge_supports:
        return _propagate_edge_supports_explicit(struc, compiled, ids,
                                                  edge_supports, tol)
    mode = getattr(m, "edge_support_mode", "common")
    if mode == "none":
        return []
    if len(ids) < 2:
        return []

    by_node: dict[str, set] = {}
    for a in struc.support_assignments:
        by_node.setdefault(a.node_id, set()).add(a.support_name)

    already = {(a.node_id, a.support_name) for a in compiled.support_assignments}
    seg_tol = max(tol, 1e-6)
    added: list[tuple[str, str]] = []
    for i in range(len(ids)):
        a_id, b_id = ids[i], ids[(i + 1) % len(ids)]
        if a_id == b_id:
            continue
        da = _node_restrained_dofs(struc, a_id, by_node)
        db = _node_restrained_dofs(struc, b_id, by_node)
        if mode == "equal":
            if da != db or not any(da):
                continue
            dofs = da
        else:   # "common": DOFs restrained at both corners
            dofs = (da[0] and db[0], da[1] and db[1], da[2] and db[2])
        name = _support_for_dofs(compiled, *dofs)
        if name is None:
            continue
        na, nb = struc.nodes[a_id], struc.nodes[b_id]
        ax, ay = na.x, na.y
        dx, dy = nb.x - ax, nb.y - ay
        L2 = dx * dx + dy * dy
        if L2 < 1e-18:
            continue
        for nid, nd in compiled.nodes.items():
            if nid == a_id or nid == b_id or (nid, name) in already:
                continue        # corners keep their own supports
            t = ((nd.x - ax) * dx + (nd.y - ay) * dy) / L2
            if not (-1e-9 <= t <= 1.0 + 1e-9):
                continue
            px, py = ax + t * dx, ay + t * dy
            if (nd.x - px) ** 2 + (nd.y - py) ** 2 > seg_tol * seg_tol:
                continue
            compiled.assign_support(nid, name)
            already.add((nid, name))
            added.append((nid, name))
    return added


def _apply_area_temperatures(struc, compiled, trace):
    """Turn each area-object temperature into per-cell loads on the mesh.

    The object's triangles AND quads (dev/IMPLEMENT_QUAD.md Phase 6 follow-up
    — ``QuadTemperatureLoad`` closes the gap that used to force these
    surfaces to an all-triangle mesh) are known only now, from the trace. A
    uniform load sets the same ΔT at every node of every generated cell; a
    field load samples the field at each cell's node coordinates — which is
    where a gradient finally resolves, because the mesh is as fine as the
    object asked for. ``dt_gradient`` has no per-node slot of its own (it is
    a single through-thickness curvature per cell, same as the uniform part
    is a single free strain per cell — the CST/Q4/QM6 mean of its dt_i/dt_j/
    dt_k/dt_l corner values), so a gradient field is sampled at the same
    corner nodes and reduced to their mean, rather than read once at an
    arbitrary corner: the two fields are then resolved by the same rule,
    only differing in what the element formulation does with the samples
    afterwards. The compiled cell already carries the section, so only the
    temperature is added here.
    """
    from .models import evaluate_field
    area_temps = getattr(struc, "area_temperature_loads", [])
    if not area_temps:
        return
    for atl in area_temps:
        tr = trace.get(atl.object_id)
        if not tr:
            continue
        fld = struc.fields.get(atl.field_name) if atl.field_name else None
        gfld = struc.fields.get(getattr(atl, 'grad_field_name', '')) \
            if getattr(atl, 'grad_field_name', '') else None
        for tid in tr.get('tris', []):
            tri = compiled.tri_elements_by_id.get(tid)
            if tri is None:
                continue
            if fld is not None:
                samples = [evaluate_field(fld.expression, compiled.nodes[nid].x,
                                          compiled.nodes[nid].y)
                          for nid in (tri.node_i, tri.node_j, tri.node_k)]
                # CST feels only the mean of dt_i/dt_j/dt_k (see class
                # TriTemperatureLoad) — reduced here, not left as distinct
                # per-corner values the element would collapse anyway.
                d = sum(samples) / len(samples)
            else:
                d = atl.dt_uniform
            vals = [d, d, d]
            if gfld is not None:
                gvals = []
                for nid in (tri.node_i, tri.node_j, tri.node_k):
                    nd = compiled.nodes[nid]
                    gvals.append(evaluate_field(gfld.expression, nd.x, nd.y))
                grad = sum(gvals) / len(gvals)
            else:
                grad = getattr(atl, 'dt_gradient', 0.0)
            compiled.add_tri_temperature_load(tid, atl.load_case_id,
                                              *vals, dt_gradient=grad)
        for qid in tr.get('quads', []):
            quad = compiled.quad_elements_by_id.get(qid)
            if quad is None:
                continue
            if fld is not None:
                samples = [evaluate_field(fld.expression, compiled.nodes[nid].x,
                                          compiled.nodes[nid].y)
                          for nid in (quad.node_i, quad.node_j, quad.node_k,
                                     quad.node_l)]
                # Q4/QM6 also only feels the mean of the four corner values
                # (see class QuadTemperatureLoad) — reduced here, not left as
                # distinct per-corner values the element would collapse anyway.
                d = sum(samples) / len(samples)
            else:
                d = atl.dt_uniform
            vals = [d, d, d, d]
            if gfld is not None:
                gvals = []
                for nid in (quad.node_i, quad.node_j, quad.node_k, quad.node_l):
                    nd = compiled.nodes[nid]
                    gvals.append(evaluate_field(gfld.expression, nd.x, nd.y))
                grad = sum(gvals) / len(gvals)
            else:
                grad = getattr(atl, 'dt_gradient', 0.0)
            compiled.add_quad_temperature_load(qid, atl.load_case_id,
                                               *vals, dt_gradient=grad)


def _apply_line_temperatures(struc, compiled, trace):
    """Turn each line-object temperature into a temperature on its bars.

    The object's bars are known only now, from the trace. dt_gradient applies
    to every bar unchanged (it is through the section depth, not along the
    axis). A uniform ΔT is the same on every bar; a field sets the uniform ΔT
    from the field at each bar's midpoint, so a finely divided object resolves
    the axial variation the field describes.
    """
    from .models import evaluate_field
    line_temps = getattr(struc, "line_temperature_loads", [])
    if not line_temps:
        return
    for ltl in line_temps:
        tr = trace.get(ltl.object_id)
        if not tr:
            continue
        fld = struc.fields.get(ltl.field_name) if ltl.field_name else None
        gfld = struc.fields.get(getattr(ltl, 'grad_field_name', '')) \
            if getattr(ltl, 'grad_field_name', '') else None
        for eid in tr.get('elems', []):
            bar = compiled.bar_elements_by_id.get(eid)
            if bar is None:
                continue
            ni = compiled.nodes[bar.node_i]
            nj = compiled.nodes[bar.node_j]
            mx, my = (ni.x + nj.x) / 2.0, (ni.y + nj.y) / 2.0
            dt = (evaluate_field(fld.expression, mx, my) if fld is not None
                  else ltl.dt_uniform)
            grad = (evaluate_field(gfld.expression, mx, my) if gfld is not None
                    else ltl.dt_gradient)
            compiled.add_temperature_load(eid, ltl.load_case_id, dt, grad)


def _apply_line_distributed_loads(struc, compiled, trace):
    """Turn each line-object distributed load into a trapezoid on its bars.

    Each direction is a constant or a field. A constant is uniform on every
    generated bar; a field sets the two end values from the field at the bar's
    own nodes, so the trapezoid follows the field along the object — a finely
    divided object approximates it closely.
    """
    from .models import evaluate_field
    line_dists = getattr(struc, "line_distributed_loads", [])
    if not line_dists:
        return
    for ld in line_dists:
        tr = trace.get(ld.object_id)
        if not tr:
            continue
        fxf = struc.fields.get(ld.fx_field) if ld.fx_field else None
        fyf = struc.fields.get(ld.fy_field) if ld.fy_field else None
        for eid in tr.get('elems', []):
            bar = compiled.bar_elements_by_id.get(eid)
            if bar is None:
                continue
            ni = compiled.nodes[bar.node_i]
            nj = compiled.nodes[bar.node_j]
            fxe = evaluate_field(fxf.expression, ni.x, ni.y) if fxf else ld.fx
            fxd = evaluate_field(fxf.expression, nj.x, nj.y) if fxf else ld.fx
            fye = evaluate_field(fyf.expression, ni.x, ni.y) if fyf else ld.fy
            fyd = evaluate_field(fyf.expression, nj.x, nj.y) if fyf else ld.fy
            compiled.add_distributed_load(eid, ld.load_case_id,
                                          fxe=fxe, fxd=fxd, fye=fye, fyd=fyd,
                                          coord_sys=ld.coord_sys)


def _apply_line_element_springs(struc, compiled, trace):
    """Turn each line-object foundation spring into a spring on its bars.

    kx and ky are per unit length, a constant or a field sampled at each
    generated bar's midpoint — a Winkler modulus that follows the field along
    the object. coord_sys and the unilateral modes carry through.
    """
    from .models import evaluate_field
    line_springs = getattr(struc, "line_element_springs", [])
    if not line_springs:
        return
    for ls in line_springs:
        tr = trace.get(ls.object_id)
        if not tr:
            continue
        kxf = struc.fields.get(ls.kx_field) if ls.kx_field else None
        kyf = struc.fields.get(ls.ky_field) if ls.ky_field else None
        for eid in tr.get('elems', []):
            bar = compiled.bar_elements_by_id.get(eid)
            if bar is None:
                continue
            ni = compiled.nodes[bar.node_i]
            nj = compiled.nodes[bar.node_j]
            mx, my = (ni.x + nj.x) / 2.0, (ni.y + nj.y) / 2.0
            kx = evaluate_field(kxf.expression, mx, my) if kxf else ls.kx
            ky = evaluate_field(kyf.expression, mx, my) if kyf else ls.ky
            compiled.add_element_spring(eid, kx=kx, ky=ky,
                                        coord_sys=ls.coord_sys,
                                        mode_x=ls.mode_x, mode_y=ls.mode_y)


def bake_object_loads(struc, trace, kind, ids):
    """Translate the loads defined on the given about-to-be-baked objects into
    permanent per-element loads, and strip the object-level entries.

    The application's ``_bake_objects`` (main window) copies a geometry object's
    generated elements into the live model, then deletes the object. Its loads
    (pressure/edge/temperature for a surface, distributed/temperature for a
    line) are normally derived from the object's definition by
    ``expand_geometry`` every time it runs, via *trace* — but once the object
    is gone from ``struc.geometry_objects``, ``expand_geometry`` has no trace
    entry for it and the load is silently dropped. Call this — with *trace*
    from the same ``expand_geometry()`` call that produced the baked elements,
    and *ids* the object ids about to be baked — before deleting the objects,
    to bake their loads in too.

    Qt-free and struc-only (no MainWindow needed), so it is usable directly
    from engine-level code and unit tests, and reuses the real
    ``_apply_surface_edge_loads`` / ``_apply_area_temperatures`` /
    ``_apply_line_distributed_loads`` / ``_apply_line_temperatures`` so the
    baked-in loads match exactly what a live (unbaked) object would have
    produced.
    """
    baked = set(ids)

    if kind == 'area':
        from .models import TriAreaLoad, QuadAreaLoad

        kept_al = []
        for al in getattr(struc, "surface_area_loads", []) or []:
            if al.object_id not in baked:
                kept_al.append(al)
                continue
            tr = trace.get(al.object_id) or {}
            for tid in tr.get('tris', []) or []:
                struc.tri_area_loads.append(TriAreaLoad(
                    tri_id=tid, load_case_id=al.load_case_id, pz=al.pz))
            for qid in tr.get('quads', []) or []:
                struc.quad_area_loads.append(QuadAreaLoad(
                    quad_id=qid, load_case_id=al.load_case_id, pz=al.pz))
        struc.surface_area_loads = kept_al

        all_edge = getattr(struc, "surface_edge_loads", []) or []
        baked_edge = [el for el in all_edge if el.object_id in baked]
        if baked_edge:
            struc.surface_edge_loads = baked_edge
            _apply_surface_edge_loads(struc, struc, 1.0e-6)
        struc.surface_edge_loads = [el for el in all_edge
                                    if el.object_id not in baked]

        all_atl = getattr(struc, "area_temperature_loads", []) or []
        baked_atl = [a for a in all_atl if a.object_id in baked]
        if baked_atl:
            struc.area_temperature_loads = baked_atl
            _apply_area_temperatures(struc, struc, trace)
        struc.area_temperature_loads = [a for a in all_atl
                                        if a.object_id not in baked]

    elif kind == 'curve':
        all_ldl = getattr(struc, "line_distributed_loads", []) or []
        baked_ldl = [d for d in all_ldl if d.object_id in baked]
        if baked_ldl:
            struc.line_distributed_loads = baked_ldl
            _apply_line_distributed_loads(struc, struc, trace)
        struc.line_distributed_loads = [d for d in all_ldl
                                        if d.object_id not in baked]

        all_ltl = getattr(struc, "line_temperature_loads", []) or []
        baked_ltl = [t for t in all_ltl if t.object_id in baked]
        if baked_ltl:
            struc.line_temperature_loads = baked_ltl
            _apply_line_temperatures(struc, struc, trace)
        struc.line_temperature_loads = [t for t in all_ltl
                                        if t.object_id not in baked]


def _propagate_edge_springs(struc, compiled, m, tol):
    """Spring the mesh nodes along an edge by interpolating the two corners'
    node springs — the elastic twin of :func:`_propagate_edge_supports`, per the
    surface's ``edge_spring_mode``:

      * ``"none"``   — nothing.
      * ``"linear"`` — kx, ky and kt vary linearly along the edge between the
        corner values (a missing corner spring counts as 0), applied per DOF.
        The unilateral mode is carried when both corners agree, else bilateral.

    A linearly interpolated *nodal* spring is a convenience, not a consistent
    Winkler line spring; for anything more, mesh the surface into explicit
    triangles and place the springs node by node.
    """
    mode = getattr(m, "edge_spring_mode", "linear")
    if mode == "none":
        return []
    ids = [nid for nid in getattr(m, 'node_ids', []) or [] if nid in struc.nodes]
    if len(ids) < 2:
        return []

    springs = getattr(struc, 'node_springs', {}) or {}
    seg_tol = max(tol, 1e-6)
    already = set(getattr(compiled, 'node_springs', {}))
    added: list[str] = []
    for i in range(len(ids)):
        a_id, b_id = ids[i], ids[(i + 1) % len(ids)]
        if a_id == b_id:
            continue
        sa, sb = springs.get(a_id), springs.get(b_id)
        kxa, kya, kta = (sa.kx, sa.ky, sa.kt) if sa else (0.0, 0.0, 0.0)
        kxb, kyb, ktb = (sb.kx, sb.ky, sb.kt) if sb else (0.0, 0.0, 0.0)
        if not any((kxa, kya, kta, kxb, kyb, ktb)):
            continue
        present = [s for s in (sa, sb) if s is not None]
        mx = present[0].mode_x if len({s.mode_x for s in present}) == 1 else "both"
        my = present[0].mode_y if len({s.mode_y for s in present}) == 1 else "both"
        na, nb = struc.nodes[a_id], struc.nodes[b_id]
        ax, ay = na.x, na.y
        dx, dy = nb.x - ax, nb.y - ay
        L2 = dx * dx + dy * dy
        if L2 < 1e-18:
            continue
        for nid, nd in compiled.nodes.items():
            if nid in already or nid == a_id or nid == b_id:
                continue        # corners keep their own springs
            t = ((nd.x - ax) * dx + (nd.y - ay) * dy) / L2
            if not (-1e-9 <= t <= 1.0 + 1e-9):
                continue
            px, py = ax + t * dx, ay + t * dy
            if (nd.x - px) ** 2 + (nd.y - py) ** 2 > seg_tol * seg_tol:
                continue
            kx = (1 - t) * kxa + t * kxb
            ky = (1 - t) * kya + t * kyb
            kt = (1 - t) * kta + t * ktb
            if not any((kx, ky, kt)):
                continue
            compiled.add_node_spring(nid, kx=kx, ky=ky, kt=kt,
                                     mode_x=mx, mode_y=my)
            already.add(nid)
            added.append(nid)
    return added


def _apply_surface_area_loads(struc, compiled, trace):
    """Turn each SurfaceAreaLoad into per-cell TriAreaLoads/QuadAreaLoads on
    the compiled mesh (plate domain): the pressure is uniform, so every
    triangle AND every quad the surface meshed into (dev/IMPLEMENT_QUAD.md
    Phase 6 — a surface can now mesh into either, or a mix of both) carries
    the same pz."""
    from .models import TriAreaLoad, QuadAreaLoad
    for al in getattr(struc, "surface_area_loads", []) or []:
        entry = trace.get(al.object_id) or {}
        for tid in entry.get('tris', []) or []:
            compiled.tri_area_loads.append(TriAreaLoad(
                tri_id=tid, load_case_id=al.load_case_id, pz=al.pz))
        for qid in entry.get('quads', []) or []:
            compiled.quad_area_loads.append(QuadAreaLoad(
                quad_id=qid, load_case_id=al.load_case_id, pz=al.pz))


def _apply_surface_area_springs(struc, compiled, trace):
    """Turn each SurfaceAreaSpring into per-cell TriAreaSprings/QuadAreaSprings
    on the compiled mesh (plate domain): the subgrade modulus is uniform, so
    every triangle AND every quad the surface meshed into
    (dev/IMPLEMENT_QUAD.md Phase 6 follow-up — ``QuadAreaSpring`` closes the
    gap that used to force these surfaces to an all-triangle mesh) carries the
    same kz."""
    from .models import TriAreaSpring, QuadAreaSpring
    for asp in getattr(struc, "surface_area_springs", []) or []:
        entry = trace.get(asp.object_id) or {}
        for tid in entry.get('tris', []) or []:
            compiled.tri_area_springs.append(TriAreaSpring(
                tri_id=tid, kz=asp.kz))
        for qid in entry.get('quads', []) or []:
            compiled.quad_area_springs.append(QuadAreaSpring(
                quad_id=qid, kz=asp.kz))


def _apply_surface_edge_loads(struc, compiled, tol):
    """Distribute each surface edge load (on a boundary side node_a→node_b of a
    surface object) as consistent nodal loads over the boundary MESH nodes that
    fall on that side. The per-unit-length (fx, fy) is split ½·L to each node of
    every sub-segment."""
    edge_loads = getattr(struc, "surface_edge_loads", [])
    if not edge_loads:
        return
    from .loads import _edge_load_fxy
    seg_tol = max(tol, 1e-6)
    for el in edge_loads:
        if el.load_case_id not in compiled.load_cases_by_id:
            continue
        na = struc.nodes.get(el.node_a); nb = struc.nodes.get(el.node_b)
        if na is None or nb is None:
            continue
        obj = struc.geometry_objects.get(el.object_id)
        sec_name = getattr(obj, "tri_section_name", "") if obj is not None else ""
        # tri_section_name resolves against EITHER section table since
        # dev/IMPLEMENT_QUAD.md Phase 6 — a surface meshed into quads still
        # needs its (now QuadSection) thickness here, not the previous
        # tri_sections-only lookup, which silently fell back to 1.0 for one.
        sec = struc.tri_sections.get(sec_name) or struc.quad_sections.get(sec_name)
        thk = sec.thickness if sec is not None else 1.0
        ax, ay, bx, by = na.x, na.y, nb.x, nb.y
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        if L2 < 1e-18:
            continue
        # Compiled mesh nodes lying on the segment a→b, ordered along it.
        on_seg = []
        for nid, nd in compiled.nodes.items():
            t = ((nd.x - ax) * dx + (nd.y - ay) * dy) / L2
            if -1e-9 <= t <= 1.0 + 1e-9:
                px, py = ax + t * dx, ay + t * dy
                if (nd.x - px) ** 2 + (nd.y - py) ** 2 <= seg_tol * seg_tol:
                    on_seg.append((t, nid))
        on_seg.sort()
        if len(on_seg) < 2:
            continue
        fx, fy = _edge_load_fxy(el, na, nb, thk)
        Llen = L2 ** 0.5
        for (t0, id0), (t1, id1) in zip(on_seg, on_seg[1:]):
            half = (t1 - t0) * Llen / 2.0
            compiled.add_point_load(id0, el.load_case_id, fx=fx * half, fy=fy * half)
            compiled.add_point_load(id1, el.load_case_id, fx=fx * half, fy=fy * half)


def _object_points(s, obj) -> list[Point]:
    """Discretised points for an object, using its current node positions."""
    pts = [(s.nodes[nid].x, s.nodes[nid].y)
           for nid in getattr(obj, "node_ids", []) if nid in s.nodes]
    if isinstance(obj, GeoArc):
        return _arc_points(pts, obj) if len(pts) >= 3 else []
    if isinstance(obj, GeoMultisegment):
        return _polyline_points(pts, obj) if len(pts) >= 2 else []
    if isinstance(obj, GeoSegment):
        return _line_points(pts, obj) if len(pts) >= 2 else []
    return []


# ── expansion ─────────────────────────────────────────────────────────────

def expand_geometry(struc, tol: float = 1.0e-6):
    """Return ``(compiled, trace)``.

    *compiled* is a :class:`Structure2D` copy of *struc* with every object turned
    into concrete nodes and bar elements (and its own ``geometry_objects`` cleared).
    *trace* maps ``{object_id: {'nodes': [...], 'elems': [...]}}`` so results on
    the compiled mesh can be mapped back to each object.

    Nodes closer than *tol* (metres) are merged; existing model nodes are seeded
    first so object endpoints tie into the frame. Surface objects (GeoRectangle/
    GeoPolygon) are processed finest ``target_size`` first, ties broken by id
    (dev-suggestion #4): the unstructured (Delaunay) branch's own boundary
    refinement always honours every forced point it is given, however
    awkward the fit, so whichever of two differently-sized neighbours is
    meshed LAST is guaranteed to conform to the other — processing the finer
    one first means that guarantee always falls on the coarser side, instead
    of on whichever id happened to sort first. Other objects (bars, arcs, …)
    keep sorting purely by id, ahead of every surface, exactly as before —
    their own nodes come from fixed endpoints, not a target_size heuristic,
    so reordering them relative to each other has no such effect and would
    only cost reproducibility.

    An earlier attempt at this (dev-suggestion #2: a pre-pass predicting
    every object's own boundary points in isolation and forcing the whole
    pool onto every neighbour) was reverted — it injected a coarser
    neighbour's speculative, never-actually-materialised grid points into
    an otherwise perfectly-fitting finer panel, producing forced points a
    fraction of a target-size away from that panel's own real grid lines
    and the sliver triangles that come with trying to honour both. Ordering
    by density turned out to be sufficient on its own, without that
    downside.
    """
    compiled = copy.deepcopy(struc)
    trace: dict[str, dict] = {}

    # Spatial hash: quantised (x, y) -> node id. Seed with existing nodes so
    # object endpoints snap onto the frame.
    def _key(x, y):
        return (round(x / tol), round(y / tol))

    node_key: dict[tuple[int, int], str] = {}
    for nid, nd in compiled.nodes.items():
        node_key.setdefault(_key(nd.x, nd.y), nid)

    existing_elems = {e.id for e in compiled.bar_elements}
    existing_tris = {t.id for t in compiled.tri_elements}
    existing_quads = {q.id for q in compiled.quad_elements}

    # Mesh order (dev-suggestion #4): surface objects sorted finest-first by
    # target_size (ties by id); every non-surface object sorts ahead of every
    # surface, ordered by id among themselves — see expand_geometry's
    # docstring for why. key=str, not a bare sort, for the reason given below.
    def _mesh_order_key(mid):
        m = struc.geometry_objects[mid]
        if isinstance(m, (GeoRectangle, GeoPolygon)):
            ts = float(getattr(m, "target_size", 0.0) or 0.0)
        else:
            ts = 0.0
        return (ts, str(mid))

    # key=str inside _mesh_order_key, not a bare sort: an id is meant to be a
    # string everywhere else in the model (nodes, elements, sections all key
    # by str), but nothing actually enforced that for an object's own id — a
    # script or an old file that gave one a bare numeric id (e.g.
    # add_geo_rectangle(5, ...)) stores the int 5 as-is, and sorting that
    # together with the usual string ids crashed outright ("'<' not supported
    # between instances of 'int' and 'str'") instead of just reordering it.
    # str() makes the tie-break total again without changing it for the
    # normal all-string case.
    for mid in sorted(struc.geometry_objects, key=_mesh_order_key):
        m = struc.geometry_objects[mid]
        # Surface objects mesh into triangles/quads (not bars) — handled
        # separately.
        if isinstance(m, (GeoRectangle, GeoPolygon)):
            trace[mid] = _expand_surface(struc, compiled, m, mid,
                                         node_key, _key, existing_tris,
                                         existing_quads)
            # After meshing, not before: the nodes to restrain are the ones the
            # mesher just created.
            if getattr(struc, 'propagate_edge_supports', True):
                trace[mid]['edge_supports'] = _propagate_edge_supports(
                    struc, compiled, m, tol)
            if getattr(struc, 'propagate_edge_springs', True):
                trace[mid]['edge_springs'] = _propagate_edge_springs(
                    struc, compiled, m, tol)
            continue
        pts = _object_points(struc, m)
        if len(pts) < 2:
            continue
        n_counter = 0
        e_counter = 0
        node_ids: list[str] = []
        elem_ids: list[str] = []

        def _get_node(x, y):
            nonlocal n_counter
            k = _key(x, y)
            nid = node_key.get(k)
            if nid is None:
                nid = f"{mid}.n{n_counter}"
                n_counter += 1
                compiled.add_node(nid, x, y)
                node_key[k] = nid
            return nid

        prev = _get_node(*pts[0])
        if prev not in node_ids:
            node_ids.append(prev)
        for (x, y) in pts[1:]:
            cur = _get_node(x, y)
            if cur != prev:
                e_counter += 1
                eid = f"{mid}.e{e_counter}"
                while eid in existing_elems:
                    eid = f"{mid}.e{e_counter}_{len(existing_elems)}"
                existing_elems.add(eid)
                compiled.add_bar_element(
                    eid, prev, cur, m.section_name,
                    sd_ky=getattr(m, "sd_ky", None),
                    sd_kz=getattr(m, "sd_kz", None),
                    sd_klt=getattr(m, "sd_klt", None),
                    sd_ltb=getattr(m, "sd_ltb", True),
                    is_column=getattr(m, "is_column", None),
                    beam=getattr(m, "beam", None),
                )
                elem_ids.append(eid)
                if cur not in node_ids:
                    node_ids.append(cur)
                prev = cur
        trace[mid] = {'nodes': node_ids, 'elems': elem_ids}

        # ── Loads defined on the object (Phase 4) ──
        # Supports/springs live on the real endpoint/vertex nodes (materialised
        # by the model), so nothing to do for them here.
        # Uniform loads spread over every generated segment.
        for ld in getattr(m, "loads", []) or []:
            lc = ld.get("load_case")
            if not lc or lc not in compiled.load_cases_by_id:
                continue
            wx = float(ld.get("wx", 0.0)); wy = float(ld.get("wy", 0.0))
            if wx == 0.0 and wy == 0.0:
                continue
            csys = ld.get("coord_sys", "global")
            for eid in elem_ids:
                compiled.add_distributed_load(eid, lc, fxe=wx, fxd=wx,
                                              fye=wy, fyd=wy, coord_sys=csys)

    # ── Surface edge loads → consistent nodal loads on the boundary mesh ──
    _apply_surface_edge_loads(struc, compiled, tol)

    # ── Surface area loads (plate) → one TriAreaLoad per generated triangle ──
    _apply_surface_area_loads(struc, compiled, trace)

    # ── Surface area springs (plate) → one TriAreaSpring per triangle ──
    _apply_surface_area_springs(struc, compiled, trace)

    # ── Area temperatures → a per-node temperature on the object's triangles ──
    _apply_area_temperatures(struc, compiled, trace)

    # ── Line temperatures → a temperature on the object's bars ──
    _apply_line_temperatures(struc, compiled, trace)

    # ── Line distributed loads → a trapezoidal load on the object's bars ──
    _apply_line_distributed_loads(struc, compiled, trace)

    # ── Line element springs → a foundation spring on the object's bars ──
    _apply_line_element_springs(struc, compiled, trace)

    # ── Topology: dedup coincident bars, then split T-junctions ───────
    _dedup_bars(compiled, trace)
    # Only junctions involving object geometry are resolved, so a pure explicit
    # frame is left untouched (its results are unchanged by adding objects).
    object_nodes = set()
    object_bars = set()
    for tr in trace.values():
        object_nodes.update(tr['nodes'])
        object_bars.update(tr['elems'])
    # Bars from objects flagged "intersect_crossings" — used by the OR rule.
    intersect_bars = set()
    for mid, tr in trace.items():
        if getattr(struc.geometry_objects.get(mid), "intersect_crossings", False):
            intersect_bars.update(tr['elems'])

    sm = _split_tee_junctions(compiled, tol, trace, object_nodes, object_bars)
    _propagate(object_bars, intersect_bars, sm)

    # ── Crossings (X): connect where the OR rule asks ─────────────────
    cross_nodes = _intersect_crossings(compiled, tol, intersect_bars)
    if cross_nodes:
        object_nodes |= cross_nodes
        sm2 = _split_tee_junctions(compiled, tol, trace, object_nodes, object_bars)
        _propagate(object_bars, intersect_bars, sm2)

    compiled.geometry_objects = {}
    return compiled, trace


def _propagate(object_bars, intersect_bars, split_map):
    """After a split, sub-segments inherit their parent's set membership."""
    for old, children in split_map.items():
        if old in object_bars:
            object_bars.update(children)
        if old in intersect_bars:
            intersect_bars.update(children)


def _seg_intersection(s, b1, b2, tol):
    """Return the (x, y) crossing point of two bars if their segments intersect
    strictly in the interior of both, else None."""
    p1 = s.nodes.get(b1.node_i); p2 = s.nodes.get(b1.node_j)
    p3 = s.nodes.get(b2.node_i); p4 = s.nodes.get(b2.node_j)
    if None in (p1, p2, p3, p4):
        return None
    x1, y1, x2, y2 = p1.x, p1.y, p2.x, p2.y
    x3, y3, x4, y4 = p3.x, p3.y, p4.x, p4.y
    d = (x2 - x1) * (y4 - y3) - (y2 - y1) * (x4 - x3)
    if abs(d) < 1e-12:
        return None                       # parallel / collinear
    t = ((x3 - x1) * (y4 - y3) - (y3 - y1) * (x4 - x3)) / d
    u = ((x3 - x1) * (y2 - y1) - (y3 - y1) * (x2 - x1)) / d
    L1 = math.hypot(x2 - x1, y2 - y1); L2 = math.hypot(x4 - x3, y4 - y3)
    e1 = tol / L1 if L1 else 0.0; e2 = tol / L2 if L2 else 0.0
    if t <= e1 or t >= 1.0 - e1 or u <= e2 or u >= 1.0 - e2:
        return None                       # meet at (or beyond) an endpoint
    return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))


def _intersect_crossings(s, tol, intersect_bars):
    """Add a node at every crossing between two bars where the OR rule applies
    (at least one bar is in *intersect_bars*) and the bars don't share a node.
    Returns the set of crossing node ids (to be resolved by a T-split pass)."""
    def _key(x, y):
        return (round(x / tol), round(y / tol))
    node_key = {_key(n.x, n.y): nid for nid, n in s.nodes.items()}
    counter = [0]

    def _get_or_add(x, y):
        k = _key(x, y)
        nid = node_key.get(k)
        if nid is None:
            counter[0] += 1
            nid = f"X.n{counter[0]}"
            while nid in s.nodes:
                nid += "_"
            s.add_node(nid, x, y)
            node_key[k] = nid
        return nid

    cross = set()
    bars = list(s.bar_elements)
    for i in range(len(bars)):
        for j in range(i + 1, len(bars)):
            b1, b2 = bars[i], bars[j]
            if not (b1.id in intersect_bars or b2.id in intersect_bars):
                continue
            if {b1.node_i, b1.node_j} & {b2.node_i, b2.node_j}:
                continue
            pt = _seg_intersection(s, b1, b2, tol)
            if pt is not None:
                cross.add(_get_or_add(*pt))
    return cross


def _dedup_bars(s, trace):
    """Drop bars that duplicate another bar (same unordered node pair), keeping
    the first and moving the duplicate's loads/spring onto it. Prevents doubled
    stiffness where an object segment coincides with an existing bar. *trace* is
    updated (dup elem id → kept id)."""
    seen: dict[frozenset, str] = {}
    dup_map: dict[str, str] = {}
    for bar in list(s.bar_elements):
        if bar.node_i == bar.node_j:
            continue
        key = frozenset((bar.node_i, bar.node_j))
        kept = seen.get(key)
        if kept is None:
            seen[key] = bar.id
            continue
        # Move loads onto the kept bar (reassign element_id so remove doesn't
        # take them), then remove the duplicate.
        for dl in s.distributed_loads:
            if dl.element_id == bar.id:
                dl.element_id = kept
        for ep in s.element_point_loads:
            if ep.element_id == bar.id:
                ep.element_id = kept
        for tl in s.temperature_loads:
            if tl.element_id == bar.id:
                tl.element_id = kept
        sp = s.element_springs.pop(bar.id, None)
        if sp is not None and kept not in s.element_springs:
            sp.element_id = kept
            s.element_springs[kept] = sp
        s.bar_elements = [e for e in s.bar_elements if e.id != bar.id]
        s.bar_elements_by_id.pop(bar.id, None)
        dup_map[bar.id] = kept

    if dup_map:
        for tr in trace.values():
            tr['elems'] = [dup_map.get(e, e) for e in tr['elems']]


def _split_tee_junctions(s, tol, trace, object_nodes, object_bars):
    """Split every bar that has a node lying on its interior (a T-junction), so
    the geometry actually connects. Restricted to junctions where the node or
    the bar comes from an object. Distributed loads are apportioned by linear
    interpolation; springs/temperature are copied to the sub-segments; element
    point loads are routed to the segment that contains them; hinges stay at the
    true member ends. *trace* is updated (elem → sub-elems)."""
    nodes = s.nodes
    split_map: dict[str, list[str]] = {}

    for bar in list(s.bar_elements):
        ni = nodes.get(bar.node_i)
        nj = nodes.get(bar.node_j)
        if ni is None or nj is None:
            continue
        dx, dy = nj.x - ni.x, nj.y - ni.y
        L2 = dx * dx + dy * dy
        if L2 < tol * tol:
            continue
        L = math.sqrt(L2)
        bar_is_obj = bar.id in object_bars
        interior = []
        for nid, nd in nodes.items():
            if nid == bar.node_i or nid == bar.node_j:
                continue
            if not (bar_is_obj or nid in object_nodes):
                continue
            t = ((nd.x - ni.x) * dx + (nd.y - ni.y) * dy) / L2
            if t <= tol / L or t >= 1.0 - tol / L:
                continue
            px, py = ni.x + t * dx, ni.y + t * dy
            if math.hypot(nd.x - px, nd.y - py) <= tol:
                interior.append((t, nid))
        if not interior:
            continue
        interior.sort()
        chain = [bar.node_i] + [nid for _, nid in interior] + [bar.node_j]
        ts = [0.0] + [t for t, _ in interior] + [1.0]

        # Capture the bar's loads before removing it.
        dloads = [dl for dl in s.distributed_loads if dl.element_id == bar.id]
        eploads = [ep for ep in s.element_point_loads if ep.element_id == bar.id]
        tloads = [tl for tl in s.temperature_loads if tl.element_id == bar.id]
        espring = s.element_springs.get(bar.id)
        hi, hj, sec = bar.hinge_i, bar.hinge_j, bar.section_name
        rc = (bar.rc_design, bar.rc_cover, bar.rc_cotg_theta, bar.rc_alpha_s)
        # Buckling-length overrides + column flag (dev/
        # BUCKLING_COLUMN_PERSISTENCE.md §4): a T/X split must not silently
        # drop these from the resulting sub-segments.
        buck = (bar.sd_ky, bar.sd_kz, bar.sd_klt, bar.sd_ltb, bar.is_column)
        beam_tag = bar.beam

        s.remove_element(bar.id)   # also drops its distributed loads + spring
        # remove_element doesn't clear these two collections — do it here.
        s.element_point_loads = [ep for ep in s.element_point_loads
                                 if ep.element_id != bar.id]
        s.temperature_loads = [tl for tl in s.temperature_loads
                               if tl.element_id != bar.id]

        new_ids = []
        nseg = len(chain) - 1
        for k in range(nseg):
            a, b = chain[k], chain[k + 1]
            ta, tb = ts[k], ts[k + 1]
            new_id = f"{bar.id}~{k}"
            while new_id in s.bar_elements_by_id:
                new_id += "_"
            e2 = s.add_bar_element(new_id, a, b, sec,
                                   hinge_i=(hi if k == 0 else False),
                                   hinge_j=(hj if k == nseg - 1 else False))
            e2.rc_design, e2.rc_cover, e2.rc_cotg_theta, e2.rc_alpha_s = rc
            (e2.sd_ky, e2.sd_kz, e2.sd_klt, e2.sd_ltb, e2.is_column) = buck
            e2.beam = beam_tag
            new_ids.append(new_id)
            for dl in dloads:
                fxe = dl.fxe + (dl.fxd - dl.fxe) * ta
                fxd = dl.fxe + (dl.fxd - dl.fxe) * tb
                fye = dl.fye + (dl.fyd - dl.fye) * ta
                fyd = dl.fye + (dl.fyd - dl.fye) * tb
                s.add_distributed_load(new_id, dl.load_case_id, fxe=fxe, fxd=fxd,
                                       fye=fye, fyd=fyd, coord_sys=dl.coord_sys)
            for tl in tloads:
                s.add_temperature_load(new_id, tl.load_case_id,
                                       dt_uniform=tl.delta_t_uniform,
                                       dt_gradient=tl.delta_t_gradient)
            if espring is not None:
                s.add_element_spring(new_id, kx=espring.kx, ky=espring.ky,
                                     coord_sys=getattr(espring, 'coord_sys', 'local'),
                                     mode_x=getattr(espring, 'mode_x', 'both'),
                                     mode_y=getattr(espring, 'mode_y', 'both'))
            for ep in eploads:
                af = (ep.a / L)
                in_seg = (ta <= af <= tb or (k == 0 and af < ta)
                          or (k == nseg - 1 and af > tb))
                if in_seg:
                    seg_len = (tb - ta) * L
                    a_local = min(max((af - ta) * L, 0.0), seg_len)
                    s.add_element_point_load(new_id, ep.load_case_id, a=a_local,
                                             fx=ep.fx, fy=ep.fy, mz=ep.mz,
                                             coord_sys=ep.coord_sys)
        split_map[bar.id] = new_ids

    # Update the object trace: replace split elem ids with their sub-segments.
    if split_map:
        for tr in trace.values():
            new_elems = []
            for eid in tr['elems']:
                new_elems.extend(split_map.get(eid, [eid]))
            tr['elems'] = new_elems
    return split_map
