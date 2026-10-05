"""Topology model check.

Runs on the geometry the solver actually sees — the compiled mesh when the model
has geometry objects (so auto-resolved T-junctions / opted-in crossings are
already applied), otherwise the editable model — and reports the issues that are
NOT auto-fixed: free (isolated) nodes, coincident nodes, unconnected T-junctions,
unconnected crossings and duplicate bars.

Each issue is a dict: ``{'type', 'msg', 'at'(optional x,y), 'items'(optional)}``.
"""
from __future__ import annotations

import math


def quad_geometry_problems(coords) -> list[str]:
    """Geometry problems for a 4-node quad, given its corner coordinates
    ``[(x_i,y_i), (x_j,y_j), (x_k,y_k), (x_l,y_l)]`` in the order they will
    be wound. Returns a list of human-readable problems; empty means the
    quad is valid.

    Checked *eagerly*, at ``Structure2D.add_quad_element`` time
    (dev/IMPLEMENT_QUAD.md Phase 3) — rejects a self-intersecting/bowtie/
    non-convex/near-zero-Jacobian quad at load time with a clear message,
    rather than letting it fail later inside assembly with a raw
    "non-positive Jacobian determinant" from ``quad_elements``, or (worse)
    silently producing a wrong stiffness. Decision (§5 point 2 of the plan):
    reject, do not auto-fix — a silent reorder of the nodes could disagree
    with what the caller intended.

    Three checks, cheapest/most-diagnostic first:

    1. counter-clockwise winding (shoelace signed area > 0) — also catches
       a self-intersecting (bowtie) quad, whose signed area is ambiguous/
       near zero;
    2. convexity — the cross product of consecutive edge vectors must have
       the same (positive, given check 1) sign at all four corners;
    3. Jacobian sign at all four corners of the bilinear isoparametric map
       (reuses ``quad_elements``'s own Jacobian) — the check the assembly
       code depends on implicitly; redundant with convexity for a simple
       quad, but catches near-degenerate corners (a corner angle extremely
       close to 0° or 180°) that the coarser convexity sign test can miss
       in floating point.
    """
    problems: list[str] = []
    if len(coords) != 4:
        return [f"expected 4 nodes, got {len(coords)}"]

    # 1. Winding / self-intersection (shoelace signed area).
    area2 = 0.0
    for i in range(4):
        x1, y1 = coords[i]
        x2, y2 = coords[(i + 1) % 4]
        area2 += x1 * y2 - x2 * y1
    if area2 <= 0.0:
        problems.append(
            "nodes are not wound counter-clockwise (node_i -> node_j -> "
            "node_k -> node_l), or the quad is self-intersecting/degenerate "
            "(signed area <= 0)")
        return problems   # further checks are meaningless with bad winding

    # 2. Convexity.
    for i in range(4):
        x0, y0 = coords[(i - 1) % 4]
        x1, y1 = coords[i]
        x2, y2 = coords[(i + 1) % 4]
        cross = (x1 - x0) * (y2 - y1) - (y1 - y0) * (x2 - x1)
        if cross <= 0.0:
            names = ('node_i', 'node_j', 'node_k', 'node_l')
            problems.append(f"quad is non-convex at {names[i]} (interior "
                            "angle >= 180 degrees)")

    # 3. Jacobian sign at the four corners of the bilinear map.
    from .quad_elements import q4_shape_functions, _jacobian
    for xi, eta in ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)):
        _, dN_dxi = q4_shape_functions(xi, eta)
        try:
            _, _, detJ, _ = _jacobian(coords, dN_dxi)
        except ValueError:
            detJ = -1.0
        if detJ <= 0.0:
            problems.append("Jacobian is non-positive at a corner (inverted "
                            "or near-degenerate quad)")
            break

    return problems


def _on_segment(px, py, ax, ay, bx, by, tol):
    """True if (px,py) lies on the interior of segment a→b (within tol, not at
    the endpoints)."""
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    if L2 < tol * tol:
        return False
    t = ((px - ax) * dx + (py - ay) * dy) / L2
    L = math.sqrt(L2)
    if t <= tol / L or t >= 1.0 - tol / L:
        return False
    qx, qy = ax + t * dx, ay + t * dy
    return math.hypot(px - qx, py - qy) <= tol


def _crossing_point(a, b, c, d, tol):
    """Interior crossing point of segments a-b and c-d, or None."""
    x1, y1, x2, y2 = a[0], a[1], b[0], b[1]
    x3, y3, x4, y4 = c[0], c[1], d[0], d[1]
    den = (x2 - x1) * (y4 - y3) - (y2 - y1) * (x4 - x3)
    if abs(den) < 1e-12:
        return None
    t = ((x3 - x1) * (y4 - y3) - (y3 - y1) * (x4 - x3)) / den
    u = ((x3 - x1) * (y2 - y1) - (y3 - y1) * (x2 - x1)) / den
    L1 = math.hypot(x2 - x1, y2 - y1); L2 = math.hypot(x4 - x3, y4 - y3)
    e1 = tol / L1 if L1 else 0.0; e2 = tol / L2 if L2 else 0.0
    if t <= e1 or t >= 1 - e1 or u <= e2 or u >= 1 - e2:
        return None
    return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))


def model_check(struc, tol: float = 1.0e-6) -> list[dict]:
    """Return a list of topology issues for *struc* (see module docstring)."""
    # Analyse the mesh the solver sees.
    trace: dict = {}
    if getattr(struc, "geometry_objects", None):
        from .geo_expand import expand_geometry
        mesh, trace = expand_geometry(struc, tol=tol)
    else:
        mesh = struc

    nodes = mesh.nodes
    bars = list(mesh.bar_elements)
    tris = list(getattr(mesh, 'tri_elements', []) or [])
    quads = list(getattr(mesh, 'quad_elements', []) or [])
    issues: list[dict] = []

    # ── Free / isolated nodes ─────────────────────────
    # Triangles (and, since 2026-08, quads) count. They were missed here, and
    # the consequence was not a near miss: on a meshed wall — a model with
    # triangles and no bars — every single node was reported as isolated. One
    # report had 357 of them, and the reading it invited was that the mesh
    # had failed to generate, which is both alarming and false. The quad
    # case reproduced the exact same symptom for a surface object meshed
    # entirely into QuadElements (prefer_quad=True): every one of its nodes
    # came back "free" because quads were never added to `used`. A node is
    # free when no *element* of any kind uses it.
    used = set()
    for b in bars:
        used.add(b.node_i); used.add(b.node_j)
    for t in tris:
        used.add(t.node_i); used.add(t.node_j); used.add(t.node_k)
    for q in quads:
        used.add(q.node_i); used.add(q.node_j); used.add(q.node_k); used.add(q.node_l)
    for nid in nodes:
        if nid not in used:
            n = nodes[nid]
            issues.append({'type': 'free_node',
                           'msg': f"Node '{nid}' is not connected to any element.",
                           'at': (n.x, n.y)})

    # ── Coincident nodes (distinct ids within tolerance) ──────
    buckets: dict[tuple, list] = {}
    for nid, n in nodes.items():
        buckets.setdefault((round(n.x / tol), round(n.y / tol)), []).append(nid)
    for ids in buckets.values():
        if len(ids) > 1:
            issues.append({'type': 'coincident_nodes',
                           'msg': f"Coincident nodes not merged: {', '.join(ids)}.",
                           'items': list(ids)})

    # ── Unconnected T-junctions (node on an element edge interior) ────
    # Originally bar-only. A node created by one surface object landing,
    # unconnected, on the straight edge of a neighbouring object's coarser
    # triangle or quad (different target_size, shared boundary) is exactly
    # the same defect — a genuine non-conforming mesh (a "hanging" node),
    # not a cosmetic one — so triangle and quad element edges are now
    # checked the same way bar elements always were. expand_geometry's
    # finest-target_size-first mesh order (dev-suggestion #4) prevents most
    # of these (whichever panel is coarser always meshes last, and the
    # unstructured branch's boundary refinement always honours every forced
    # point it is given); this check is what surfaces whatever it still
    # can't (e.g. a non-convex outline where that guarantee doesn't apply).
    def _element_edges():
        for b in bars:
            yield b.id, b.node_i, b.node_j
        for t in tris:
            yield t.id, t.node_i, t.node_j
            yield t.id, t.node_j, t.node_k
            yield t.id, t.node_k, t.node_i
        for q in quads:
            yield q.id, q.node_i, q.node_j
            yield q.id, q.node_j, q.node_k
            yield q.id, q.node_k, q.node_l
            yield q.id, q.node_l, q.node_i

    for eid, na, nb in _element_edges():
        pa, pb = nodes.get(na), nodes.get(nb)
        if pa is None or pb is None:
            continue
        for nid, n in nodes.items():
            if nid in (na, nb):
                continue
            if _on_segment(n.x, n.y, pa.x, pa.y, pb.x, pb.y, tol):
                issues.append({'type': 'tee_junction',
                               'msg': f"Node '{nid}' lies on element '{eid}' but "
                                      f"is not connected to it.",
                               'at': (n.x, n.y), 'items': [nid, eid]})

    # ── Duplicate bars (same unordered node pair) ─────────────
    seen: dict[frozenset, str] = {}
    for b in bars:
        key = frozenset((b.node_i, b.node_j))
        if key in seen:
            issues.append({'type': 'duplicate_bar',
                           'msg': f"Elements '{seen[key]}' and '{b.id}' are "
                                  f"duplicated (same end nodes).",
                           'items': [seen[key], b.id]})
        else:
            seen[key] = b.id

    # ── Unconnected crossings (X) ─────────────────────────────
    for i in range(len(bars)):
        bi = bars[i]
        ai, aj = nodes.get(bi.node_i), nodes.get(bi.node_j)
        if ai is None or aj is None:
            continue
        for j in range(i + 1, len(bars)):
            bj = bars[j]
            if {bi.node_i, bi.node_j} & {bj.node_i, bj.node_j}:
                continue
            ci, cj = nodes.get(bj.node_i), nodes.get(bj.node_j)
            if ci is None or cj is None:
                continue
            pt = _crossing_point((ai.x, ai.y), (aj.x, aj.y),
                                 (ci.x, ci.y), (cj.x, cj.y), tol)
            if pt is not None:
                issues.append({'type': 'crossing',
                               'msg': f"Elements '{bi.id}' and '{bj.id}' cross "
                                      f"without a shared node.",
                               'at': pt, 'items': [bi.id, bj.id]})

    issues.extend(_mass_on_supports(struc))
    issues.extend(_constraint_problems(struc))
    issues.extend(_orphaned_area_temperature_loads(struc))
    issues.extend(_quad_mesh_problems(struc, trace))
    issues.extend(_mesh_quality_problems(nodes, tris, quads))
    return issues


# ── Mesh quality (minimum interior angle) ───────────────────────────────────
# The cheapest, most standard proxy for how far a CST/Q4/MITC-family element
# is from its ideal shape (60° for a triangle, 90° for a quad): interpolation
# error and element-matrix conditioning both degrade with distorted angles
# well before a Jacobian actually goes non-positive (mirrors the reasoning
# already used for quad_cell_quality's skew/aspect gate in meshing.py, just
# expressed as one unit both element kinds share). It's also the exact metric
# this session used, by hand, to catch and fix a sliver-triangle regression
# introduced (then reverted) in expand_geometry's forced-point handling —
# worth surfacing permanently instead of only when someone goes looking.

def _element_min_angle_deg(coords: list[tuple]) -> float:
    """Smallest interior angle [deg] of a convex 3- or 4-vertex polygon given
    as ``[(x, y), ...]`` — the tri/quad twin of meshing._tri_min_angle,
    generalised to work directly on element coordinates rather than indices
    into a shared points array (compiled elements have no such array)."""
    n = len(coords)
    best = 180.0
    for i in range(n):
        ax, ay = coords[(i - 1) % n]
        bx, by = coords[i]
        cx, cy = coords[(i + 1) % n]
        ux, uy = ax - bx, ay - by
        vx, vy = cx - bx, cy - by
        lu, lv = math.hypot(ux, uy), math.hypot(vx, vy)
        if lu < 1e-12 or lv < 1e-12:
            continue
        cosang = max(-1.0, min(1.0, (ux * vx + uy * vy) / (lu * lv)))
        best = min(best, math.degrees(math.acos(cosang)))
    return best


def mesh_quality_summary(nodes, tris, quads) -> dict | None:
    """Cheap, element-by-element mesh-quality readout for the compiled mesh:
    the minimum interior angle of every triangle and quad. Returns the worst
    case and how many elements fall under a couple of common warning
    thresholds — not a per-element dump, since the worst element's id is
    already enough to jump straight to it (canvas 'Find entity…', or the
    context menu's 'Show element info' / 'Go to node in table'). ``None``
    when there are no area elements at all (a bar-only model)."""
    worst = None
    worst_id = None
    worst_pt = None
    below_15 = below_5 = total = 0

    def _consider(eid, coords):
        nonlocal worst, worst_id, worst_pt, below_15, below_5, total
        ang = _element_min_angle_deg(coords)
        total += 1
        if ang < 15.0:
            below_15 += 1
        if ang < 5.0:
            below_5 += 1
        if worst is None or ang < worst:
            worst, worst_id = ang, eid
            worst_pt = (sum(c[0] for c in coords) / len(coords),
                       sum(c[1] for c in coords) / len(coords))

    for t in tris:
        _consider(t.id, [(nodes[t.node_i].x, nodes[t.node_i].y),
                         (nodes[t.node_j].x, nodes[t.node_j].y),
                         (nodes[t.node_k].x, nodes[t.node_k].y)])
    for q in quads:
        _consider(q.id, [(nodes[q.node_i].x, nodes[q.node_i].y),
                         (nodes[q.node_j].x, nodes[q.node_j].y),
                         (nodes[q.node_k].x, nodes[q.node_k].y),
                         (nodes[q.node_l].x, nodes[q.node_l].y)])
    if total == 0:
        return None
    return {'worst_angle_deg': worst, 'worst_element': worst_id,
            'worst_at': worst_pt, 'count_below_15deg': below_15,
            'count_below_5deg': below_5, 'total_elements': total}


def _mesh_quality_problems(nodes, tris, quads) -> list[dict]:
    """One issue when the mesh's worst element falls under the 15° warning
    threshold — silent otherwise, same spirit as every other check here.
    15°/5° are the usual rule-of-thumb bands for linear CST/Q4-family
    elements, not tied to any one section or formulation."""
    summary = mesh_quality_summary(nodes, tris, quads)
    if summary is None or summary['worst_angle_deg'] >= 15.0:
        return []
    worst = summary['worst_angle_deg']
    severity = "poor" if worst < 5.0 else "low"
    below5 = (f", {summary['count_below_5deg']} of them under 5°"
             if summary['count_below_5deg'] else "")
    return [{'type': 'mesh_quality',
             'msg': (f"Mesh quality is {severity}: worst element "
                     f"'{summary['worst_element']}' has a minimum interior "
                     f"angle of {worst:.1f}° "
                     f"({summary['count_below_15deg']} of "
                     f"{summary['total_elements']} elements are under "
                     f"15°{below5})."),
             'at': summary['worst_at'], 'items': [summary['worst_element']]}]


def _quad_mesh_problems(struc, trace: dict) -> list[dict]:
    """Surface-meshing issues from a QuadSection surface object (Rectangle or
    Polygon, ``prefer_quad``/paired-section resolved a QuadSection — see
    ``xdfem2d.geo_expand._expand_surface``), read back from the *trace*
    ``expand_geometry`` already produced (no re-meshing here):

    * ``quad_never_generated`` — the object asked for a QuadSection but ended
      up with zero quad elements. This is the case a user picking "Quad" on a
      non-convex or non-quadrilateral outline hits silently otherwise: every
      cell fell back to the mapped triangle formulation
      (:data:`xdfem2d.geo_expand._QUAD_TO_TRI_FORMULATION`) with no quads at
      all, and nothing else in ``model_check`` would flag it.
    * ``quad_mixed_formulation`` — the object got *some* quads and *some*
      fallback triangles in the same mesh: two different element kernels
      (e.g. MITC4 and MITC3) share edges inside one object, which is
      displacement-compatible but not directly comparable in recovered
      moments/stresses across that boundary.

    Both are informational (a valid, intentional outcome of the per-cell
    quality fallback in ``mesh_quad_structured_cells``), not defects, so they
    are reported rather than blocking anything — same spirit as the other
    checks in this module.
    """
    geometry_objects = getattr(struc, "geometry_objects", None) or {}
    issues: list[dict] = []
    for mid in sorted(geometry_objects, key=str):
        entry = trace.get(mid)
        if not entry or not entry.get('quad_requested'):
            continue
        quad_formulation = entry.get('quad_formulation', '?')
        if not entry.get('quads'):
            reason = entry.get('quad_fallback_reason', 'unknown reason')
            issues.append({
                'type': 'quad_never_generated',
                'msg': (f"Object '{mid}' asks for a Quad section "
                        f"({quad_formulation}) but produced no quad "
                        f"elements — {reason}. Every cell meshed as a "
                        f"triangle instead."),
                'items': [str(mid)],
            })
        elif entry.get('tri_fallback_formulation'):
            issues.append({
                'type': 'quad_mixed_formulation',
                'msg': (f"Object '{mid}' mixes {len(entry['quads'])} "
                        f"{quad_formulation} quad element(s) with "
                        f"{len(entry['tris'])} "
                        f"{entry['tri_fallback_formulation']} fallback "
                        f"triangle(s) — some cells failed the quad "
                        f"quality thresholds (skew/aspect ratio)."),
                'items': [str(mid)],
            })
    return issues


def _restrained_components(struc) -> dict:
    """{node_id: set of restrained components ('ux'/'uy'/'tz')} from supports."""
    supports = getattr(struc, 'supports', {}) or {}
    out: dict[str, set] = {}
    for assign in getattr(struc, 'support_assignments', []) or []:
        supp = supports.get(assign.support_name)
        if supp is None:
            continue
        d = out.setdefault(assign.node_id, set())
        if supp.ux: d.add('ux')
        if supp.uy: d.add('uy')
        if supp.tz: d.add('tz')
    return out


def _rotational_nodes(struc) -> set:
    """Nodes that carry an active rotational (tz) DOF: any bar endpoint, or a
    node of an Allman-formulation triangle. Plain CST triangles do not, so a
    constraint on tz there ties DOFs the solver auto-pins to zero."""
    rot: set = set()
    for b in getattr(struc, 'bar_elements', []) or []:
        rot.add(b.node_i); rot.add(b.node_j)
    tri_sections = getattr(struc, 'tri_sections', {}) or {}
    for t in getattr(struc, 'tri_elements', []) or []:
        sec = tri_sections.get(getattr(t, 'section_name', None))
        if sec is not None and getattr(sec, 'formulation', 'CST') == 'Allman':
            rot.add(t.node_i); rot.add(t.node_j); rot.add(t.node_k)
    return rot


def _constraint_nodes_components(c):
    """The (node_id, component) pairs a constraint directly involves, and the
    list of node ids it references — used for both reference and conflict checks."""
    pairs: list[tuple[str, str]] = []
    refs: list[str] = []
    if c.kind == 'rigid_link':
        refs = [c.master] + list(c.slaves)
        for comp in ('ux', 'uy', 'tz'):
            pairs.append((c.master, comp))
            for s in c.slaves:
                pairs.append((s, comp))
    elif c.kind == 'equal_dof':
        refs = list(c.nodes)
        for n in c.nodes:
            for comp in c.components:
                pairs.append((n, comp))
    else:  # 'equation'
        refs = [t[0] for t in c.terms]
        pairs = [(t[0], t[1]) for t in c.terms]
    return pairs, refs


def _constraint_problems(struc) -> list[dict]:
    """Problems specific to multi-point constraints.

    Reported (never auto-fixed, because each is a modelling decision):
      * a constraint that names a node that does not exist;
      * a rigid link whose master is also one of its slaves;
      * two rigid links that make each other's master a slave (a 2-cycle) — the
        penalty solver tolerates it, but it is almost always a mistake and the
        exact (master-slave) solver cannot resolve it;
      * a tied DOF that a support already restrains — the constraint and the
        support pull against each other;
      * a constraint on tz at a node with no rotational DOF (CST-only), which
        ties a DOF the solver pins to zero.
    """
    constraints = getattr(struc, 'constraints', {}) or {}
    if not constraints:
        return []

    valid = set(getattr(struc, 'nodes', {}) or {})
    restrained = _restrained_components(struc)
    rot_nodes = _rotational_nodes(struc)
    issues: list[dict] = []

    # rigid-link master/slave graph, for cycle detection
    masters_of: dict[str, set] = {}   # node -> set of nodes it is a slave of
    for c in constraints.values():
        if c.kind == 'rigid_link' and getattr(c, 'enabled', True):
            for s in c.slaves:
                masters_of.setdefault(s, set()).add(c.master)

    for cid, c in constraints.items():
        pairs, refs = _constraint_nodes_components(c)

        # 1. dangling node references
        missing = sorted({n for n in refs if n not in valid})
        if missing:
            issues.append({
                'type': 'constraint_bad_ref',
                'msg': f"Constraint '{cid}' references missing node(s): "
                       f"{', '.join(missing)}.",
                'items': [cid] + missing})

        if not getattr(c, 'enabled', True):
            continue   # remaining checks concern the solved system

        # 2. rigid link with master among its slaves
        if c.kind == 'rigid_link' and c.master in c.slaves:
            issues.append({
                'type': 'constraint_self_link',
                'msg': f"Rigid link '{cid}' has its master '{c.master}' also as "
                       f"a slave.",
                'items': [cid, c.master]})

        # 3. mutual rigid links (2-cycle): this link makes master M the master of
        # slave s; a cycle exists if another link makes s the master of M.
        if c.kind == 'rigid_link':
            for s in c.slaves:
                if s != c.master and s in masters_of.get(c.master, set()):
                    issues.append({
                        'type': 'constraint_cycle',
                        'msg': f"Rigid links form a cycle between '{c.master}' "
                               f"and '{s}'.",
                        'items': [cid, c.master, s]})
                    break

        # 4. tied DOF also restrained by a support
        clash = sorted({f"{n}.{comp}" for (n, comp) in pairs
                        if comp in restrained.get(n, set())})
        if clash:
            issues.append({
                'type': 'constraint_vs_support',
                'msg': f"Constraint '{cid}' ties DOF(s) already restrained by a "
                       f"support: {', '.join(clash)}.",
                'items': [cid] + clash})

        # 5. tz on a node with no rotational DOF
        no_rot = sorted({n for (n, comp) in pairs
                         if comp == 'tz' and n in valid and n not in rot_nodes})
        if no_rot:
            issues.append({
                'type': 'constraint_no_rotation',
                'msg': f"Constraint '{cid}' acts on tz at node(s) with no "
                       f"rotational DOF (CST-only): {', '.join(no_rot)}.",
                'items': [cid] + no_rot})

    return issues


def _orphaned_area_temperature_loads(struc) -> list[dict]:
    """Temperature loads whose target no longer resolves to the element kind
    that consumes them.

    ``add_area_temperature_load`` used to accept any ``object_id`` string,
    including a line object's — the record was stored and never rejected, but
    ``_apply_area_temperatures`` (geo_expand.py) only turns an
    ``AreaTemperatureLoad`` into tri/quad thermal loads for an actual
    ``GeoRectangle``/``GeoPolygon``, so the load was a completely silent
    no-op: nothing raised, nothing warned, nothing solved. That gap is closed
    at creation time now (``Structure2D._require_area_object``), but a model
    built or hand-edited before that fix, or restored from an old save, can
    still hold one of these orphans — this is what catches it here. The
    tri/quad/line temperature families get the same treatment for a
    dangling/wrong-kind element or object id, checked on the editable model
    like ``_mass_on_supports``, since these are explicit user records, not
    something the mesher generates.
    """
    from .models import GeoRectangle, GeoPolygon, GeoSegment, GeoMultisegment, GeoArc
    issues: list[dict] = []
    geo = getattr(struc, 'geometry_objects', {}) or {}
    tris = getattr(struc, 'tri_elements_by_id', {}) or {}
    quads = getattr(struc, 'quad_elements_by_id', {}) or {}

    for t in getattr(struc, 'area_temperature_loads', []) or []:
        obj = geo.get(t.object_id)
        if isinstance(obj, (GeoRectangle, GeoPolygon)):
            continue
        why = ("a line object" if isinstance(
            obj, (GeoSegment, GeoMultisegment, GeoArc))
            else "not a known geometry object")
        issues.append({
            'type': 'orphaned_temperature_load',
            'msg': f"Area temperature load ({t.load_case_id}) targets "
                   f"'{t.object_id}', which is {why} — a surface object "
                   "(rectangle/polygon) is required, so this load is never "
                   "applied at solve time.",
            'items': [t.object_id, t.load_case_id]})

    for t in getattr(struc, 'line_temperature_loads', []) or []:
        obj = geo.get(t.object_id)
        if isinstance(obj, (GeoSegment, GeoMultisegment, GeoArc)):
            continue
        why = ("a surface object" if isinstance(obj, (GeoRectangle, GeoPolygon))
               else "not a known geometry object")
        issues.append({
            'type': 'orphaned_temperature_load',
            'msg': f"Line temperature load ({t.load_case_id}) targets "
                   f"'{t.object_id}', which is {why} — a line object "
                   "(line/arc/polyline) is required, so this load is never "
                   "applied at solve time.",
            'items': [t.object_id, t.load_case_id]})

    for t in getattr(struc, 'tri_temperature_loads', []) or []:
        if t.tri_id in tris:
            continue
        issues.append({
            'type': 'orphaned_temperature_load',
            'msg': f"Triangle temperature load ({t.load_case_id}) targets "
                   f"'{t.tri_id}', which is not a triangle element in this "
                   "model — this load is never applied at solve time.",
            'items': [t.tri_id, t.load_case_id]})

    for t in getattr(struc, 'quad_temperature_loads', []) or []:
        if t.quad_id in quads:
            continue
        issues.append({
            'type': 'orphaned_temperature_load',
            'msg': f"Quad temperature load ({t.load_case_id}) targets "
                   f"'{t.quad_id}', which is not a quad element in this "
                   "model — this load is never applied at solve time.",
            'items': [t.quad_id, t.load_case_id]})

    return issues


def _mass_on_supports(struc) -> list[dict]:
    """Concentrated masses sitting on a restrained DOF.

    Such a mass cannot move, so it contributes nothing to the eigenproblem: it
    is silently absent from the modes and from the effective masses (which are
    computed relative to the FREE mass). That is correct — the mass goes
    straight to the foundation — but it is invisible, and a mass typed onto a
    support node is far more often a modelling slip than an intention. Checked
    on the editable model, since ``add_nodal_mass`` is an explicit user action
    and is not generated by the mesher.
    """
    issues: list[dict] = []
    masses = list(getattr(struc, 'nodal_masses', []) or [])
    if not masses:
        return issues

    supports = getattr(struc, 'supports', {}) or {}
    restrained: dict[str, set] = {}
    for assign in getattr(struc, 'support_assignments', []) or []:
        supp = supports.get(assign.support_name)
        if supp is None:
            continue
        d = restrained.setdefault(assign.node_id, set())
        if supp.ux: d.add('x')
        if supp.uy: d.add('y')
        if supp.tz: d.add('tz')

    for nm in masses:
        dirs = restrained.get(nm.node_id)
        if not dirs:
            continue
        blocked = sorted({d for d, v in (('x', nm.mx), ('y', nm.my),
                                         ('tz', nm.mtz))
                          if v > 1e-14 and d in dirs})
        if not blocked:
            continue
        node = getattr(struc, 'nodes', {}).get(nm.node_id)
        issues.append({
            'type': 'mass_on_support',
            'msg': f"Nodal mass on '{nm.node_id}' ({nm.mass_case_id}) acts on "
                   f"restrained DOF(s) {', '.join(blocked)}: it cannot "
                   f"vibrate and is excluded from the modal analysis.",
            'at': (node.x, node.y) if node is not None else None,
            'items': [nm.node_id, nm.mass_case_id]})
    return issues
