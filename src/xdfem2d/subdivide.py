"""Divide selected bars and triangles into finer elements, in place.

Two operations, applied to a set of ids the caller chose:

**Bars.** A bar becomes N collinear sub-bars, by count or by target length.
Cutting the geometry is the easy half; the work is everything attached to the
element id, and there is more of it than there looks:

* a trapezoidal distributed load is re-interpolated onto each sub-bar, so a
  0→10 load on a halved bar becomes 0→5 and 5→10, not 0→10 twice;
* an element point load moves to the sub-bar that contains its distance from
  the i-end, with that distance recomputed locally;
* element springs and thermal loads are per-unit-length or per-element
  intensive, so they copy to every sub-bar unchanged;
* the moment hinges are the trap. hinge_i belongs to the first sub-bar's
  i-end and hinge_j to the last sub-bar's j-end; the interior joints stay
  continuous. Copying the hinges to every sub-bar turns a continuous member
  into a mechanism.

**Triangles.** Each side is divided into n equal parts and the triangle is
tiled into n² similar sub-triangles. Applied to the whole selection with one
n, a shared edge is cut into n equal parts from both sides, so the split
points coincide and the mesh stays conforming — no hanging nodes *within* the
selection. An edge on the selection boundary, shared with a triangle that was
not selected, does gain hanging nodes; that is inherent and the caller is
expected to warn.

The one implementation hazard is node identity: a node created on a shared
edge must be the *same* node for both triangles, or the mesh looks continuous
and has an invisible crack along the seam. Every new node here goes through a
spatial hash keyed on rounded coordinates, so a point made from either side
resolves to one id.

Nothing is executed and nothing is solved; this edits the input model, which
is what has to survive a save and reload.
"""
from __future__ import annotations

import math
from typing import Iterable

from . import references

#: Two nodes closer than this (metres) are the same node. Matches the welding
#: tolerance used elsewhere, so a divided model dedups the way an imported one
#: does.
TOL = 1e-6


# ── shared: get-or-create a node by coordinate ──────────────────────────────

def _key(x: float, y: float) -> tuple[int, int]:
    return (round(x / TOL), round(y / TOL))


def _node_hash(struc) -> dict:
    """Coordinate key -> existing node id, for every node in the model."""
    return {_key(n.x, n.y): nid for nid, n in struc.nodes.items()}


def _unique(stem: str, taken: set) -> str:
    """A free id built from *stem*. '~' never appears in generated node ids and
    '/' never in generated element ids, so a second divide does not collide
    with the first."""
    if stem not in taken:
        return stem
    i = 2
    while f"{stem}#{i}" in taken:
        i += 1
    return f"{stem}#{i}"


def _get_node(struc, hash_, taken, x, y, stem):
    """The id of the node at (x, y): an existing one within TOL, or a new one.

    *hash_* and *taken* are updated in place so repeated calls within one
    operation share the nodes they create along common edges.
    """
    k = _key(x, y)
    nid = hash_.get(k)
    if nid is not None:
        return nid
    nid = _unique(stem, taken)
    struc.add_node(nid, x, y)
    taken.add(nid)
    hash_[k] = nid
    return nid


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


# ── bars ────────────────────────────────────────────────────────────────────

def _bar_counts(struc, ids, count=None, size=None):
    """(bar, N) for each id, where N is the number of pieces to cut it into.

    By count directly, or by a target length: N = round(L / size), at least 1.
    Ids that are not bars, or that would not actually divide (N < 2), are
    dropped — dividing into one piece is a no-op, not an error.
    """
    out = []
    for eid in ids:
        bar = struc.bar_elements_by_id.get(eid)
        if bar is None:
            continue
        if count is not None:
            n = int(count)
        else:
            ni, nj = struc.nodes[bar.node_i], struc.nodes[bar.node_j]
            length = math.hypot(nj.x - ni.x, nj.y - ni.y)
            n = max(1, round(length / float(size))) if size else 1
        if n >= 2:
            out.append((bar, n))
    return out


def preview_bars(struc, ids, count=None, size=None) -> dict:
    """How many elements and new nodes a bar division would make. No mutation.

    Interior nodes of a straight bar are not shared with anything, so the node
    count is the plain sum of (N - 1). (A division landing on an existing node
    would merge, making this a rare over-estimate by one or two — never an
    under-estimate.)
    """
    pairs = _bar_counts(struc, ids, count, size)
    elements = sum(n for _bar, n in pairs)
    new_nodes = sum(n - 1 for _bar, n in pairs)
    return {'bars': len(pairs), 'elements': elements, 'new_nodes': new_nodes}


def divide_bars(struc, ids, count=None, size=None) -> dict:
    """Replace each selected bar by N sub-bars, carrying its loads across.

    Returns a summary: how many bars were divided, and the elements and nodes
    that now exist because of it.
    """
    pairs = _bar_counts(struc, ids, count, size)
    hash_ = _node_hash(struc)
    taken_n = set(struc.nodes)
    made_elems = made_nodes = 0

    for bar, n in pairs:
        ni, nj = struc.nodes[bar.node_i], struc.nodes[bar.node_j]
        before_nodes = len(struc.nodes)

        # The chain of node ids i = p0, p1, ..., pN = j.
        chain = [bar.node_i]
        for k in range(1, n):
            t = k / n
            x = _lerp(ni.x, nj.x, t)
            y = _lerp(ni.y, nj.y, t)
            chain.append(_get_node(struc, hash_, taken_n, x, y,
                                   f"{bar.id}~{k}"))
        chain.append(bar.node_j)

        # Read everything attached to the old element, then drop it.
        dloads = [d for d in struc.distributed_loads if d.element_id == bar.id]
        ploads = [p for p in struc.element_point_loads
                  if p.element_id == bar.id]
        temps = [t for t in struc.temperature_loads if t.element_id == bar.id]
        spring = struc.element_springs.get(bar.id)
        length = math.hypot(nj.x - ni.x, nj.y - ni.y)

        _drop_bar(struc, bar.id)

        # The sub-bars. Hinges only at the two true ends; interior continuous.
        taken_e = set(struc.bar_elements_by_id)
        sub_ids = []
        for k in range(n):
            sid = _unique(f"{bar.id}/{k + 1}", taken_e)
            taken_e.add(sid)
            struc.add_bar_element(
                sid, chain[k], chain[k + 1], bar.section_name,
                hinge_i=(bar.hinge_i if k == 0 else False),
                hinge_j=(bar.hinge_j if k == n - 1 else False),
                rc_design=bar.rc_design, rc_cover=bar.rc_cover,
                rc_cotg_theta=bar.rc_cotg_theta, rc_alpha_s=bar.rc_alpha_s,
                sd_ky=bar.sd_ky, sd_kz=bar.sd_kz, sd_klt=bar.sd_klt,
                sd_ltb=bar.sd_ltb, is_column=bar.is_column,
                beam=bar.beam)
            sub_ids.append(sid)

        _reassign_distributed(struc, dloads, sub_ids, n)
        _reassign_point_loads(struc, ploads, sub_ids, n, length)
        for tl in temps:                       # intensive: same on every piece
            for sid in sub_ids:
                struc.add_temperature_load(sid, tl.load_case_id,
                                           tl.delta_t_uniform,
                                           tl.delta_t_gradient)
        if spring is not None:
            for sid in sub_ids:
                struc.add_element_spring(sid, kx=spring.kx, ky=spring.ky,
                                         coord_sys=spring.coord_sys,
                                         mode_x=spring.mode_x,
                                         mode_y=spring.mode_y)

        made_elems += n
        made_nodes += len(struc.nodes) - before_nodes

    struc._node_dof_index = None
    return {'bars': len(pairs), 'elements': made_elems, 'new_nodes': made_nodes}


def _drop_bar(struc, eid):
    """Remove a bar and every record keyed on its id."""
    struc.bar_elements = [e for e in struc.bar_elements if e.id != eid]
    struc.bar_elements_by_id.pop(eid, None)
    struc.distributed_loads = [d for d in struc.distributed_loads
                               if d.element_id != eid]
    struc.element_point_loads = [p for p in struc.element_point_loads
                                 if p.element_id != eid]
    struc.temperature_loads = [t for t in struc.temperature_loads
                               if t.element_id != eid]
    struc.element_springs.pop(eid, None)


def _reassign_distributed(struc, dloads, sub_ids, n):
    """Re-interpolate each trapezoidal load onto the sub-bars.

    The intensity varies linearly from the i-end (fxe, fye) to the j-end (fxd,
    fyd); sub-bar k spans the fractions [k/n, (k+1)/n], so its own end values
    are the parent's interpolated at those fractions. A uniform load stays
    uniform; a triangular one is cut into the right trapezoids.
    """
    for d in dloads:
        for k, sid in enumerate(sub_ids):
            t0, t1 = k / n, (k + 1) / n
            struc.add_distributed_load(
                sid, d.load_case_id,
                fxe=_lerp(d.fxe, d.fxd, t0), fxd=_lerp(d.fxe, d.fxd, t1),
                fye=_lerp(d.fye, d.fyd, t0), fyd=_lerp(d.fye, d.fyd, t1),
                coord_sys=d.coord_sys)


def _reassign_point_loads(struc, ploads, sub_ids, n, length):
    """Move each element point load to the sub-bar that contains it.

    ``a`` is measured from the i-end. Sub-bar k covers [k·L/n, (k+1)·L/n], so
    the owning piece is floor(a / (L/n)); the load's new distance is measured
    from that piece's own i-end. A load sitting on a joint goes to the piece it
    starts (a' = 0 there), which is where a reader would expect it.
    """
    seg = length / n
    for p in ploads:
        k = int(p.a // seg) if seg > 0 else 0
        k = max(0, min(n - 1, k))
        struc.add_element_point_load(
            sub_ids[k], p.load_case_id, a=p.a - k * seg,
            fx=p.fx, fy=p.fy, mz=p.mz, coord_sys=p.coord_sys)


# ── triangles ────────────────────────────────────────────────────────────────

def _tri_list(struc, ids):
    return [struc.tri_elements_by_id[t] for t in ids
            if t in struc.tri_elements_by_id]


def _unique_edges(tris) -> set:
    """The distinct edges of a set of triangles, each as a frozenset of two
    node ids. A shared edge appears once — which is what makes the node count
    exact rather than double-counted."""
    edges = set()
    for t in tris:
        edges.add(frozenset((t.node_i, t.node_j)))
        edges.add(frozenset((t.node_j, t.node_k)))
        edges.add(frozenset((t.node_k, t.node_i)))
    return edges


def preview_triangles(struc, ids, n: int) -> dict:
    """Elements and new nodes for an n-per-side triangle division. No mutation.

    Elements are exact and simple: one triangle becomes n². Nodes are exact
    but not a per-triangle multiply — the naive product double-counts every
    shared edge. Interior nodes ((n-1)(n-2)/2 each) are never shared; edge
    nodes ((n-1) per distinct edge) are counted once; the corners already
    exist.
    """
    n = int(n)
    tris = _tri_list(struc, ids)
    if n < 2 or not tris:
        return {'triangles': len(tris), 'elements': 0, 'new_nodes': 0}
    interior = len(tris) * (n - 1) * (n - 2) // 2
    edge = len(_unique_edges(tris)) * (n - 1)
    return {'triangles': len(tris), 'elements': len(tris) * n * n,
            'new_nodes': interior + edge}


def divide_triangles(struc, ids, n: int) -> dict:
    """Tile each selected triangle into n² conforming sub-triangles."""
    n = int(n)
    tris = _tri_list(struc, ids)
    if n < 2 or not tris:
        return {'triangles': 0, 'elements': 0, 'new_nodes': 0}

    hash_ = _node_hash(struc)
    taken_n = set(struc.nodes)
    before_nodes = len(struc.nodes)
    made_elems = 0

    for tri in tris:
        A = struc.nodes[tri.node_i]
        B = struc.nodes[tri.node_j]
        C = struc.nodes[tri.node_k]
        edge_loads = [e for e in struc.tri_edge_loads if e.tri_id == tri.id]
        temps = [t for t in getattr(struc, 'tri_temperature_loads', [])
                 if t.tri_id == tri.id]
        # Per-area intensities (pressure pz, Winkler kz) copy unchanged onto
        # every sub-triangle, as divide_quads does for a quad.
        area_loads = [a for a in getattr(struc, 'tri_area_loads', [])
                      if a.tri_id == tri.id]
        springs = [a for a in getattr(struc, 'tri_area_springs', [])
                   if a.tri_id == tri.id]

        # Barycentric lattice P[i][j] = A + (i/n)(B-A) + (j/n)(C-A),
        # for i+j <= n. Corners reuse the existing node ids.
        P: dict[tuple[int, int], str] = {}
        for i in range(n + 1):
            for j in range(n + 1 - i):
                if (i, j) == (0, 0):
                    P[(i, j)] = tri.node_i
                elif (i, j) == (n, 0):
                    P[(i, j)] = tri.node_j
                elif (i, j) == (0, n):
                    P[(i, j)] = tri.node_k
                else:
                    x = A.x + (B.x - A.x) * i / n + (C.x - A.x) * j / n
                    y = A.y + (B.y - A.y) * i / n + (C.y - A.y) * j / n
                    P[(i, j)] = _get_node(struc, hash_, taken_n,
                                          x, y, f"{tri.id}~{i}_{j}")

        _drop_triangle(struc, tri.id)

        taken_t = set(struc.tri_elements_by_id)
        # A boundary sub-segment (frozenset of its two node ids) -> the id of
        # the sub-triangle that owns it, so an edge load can be re-placed.
        # Only the upward triangles touch the boundary; the downward ones are
        # strictly interior.
        owner: dict[frozenset, str] = {}
        # (sub-triangle id, its three lattice indices) — kept so a per-node
        # temperature can be interpolated onto the finer mesh below.
        subs: list[tuple] = []
        c = 0
        for i in range(n):
            for j in range(n - i):
                c += 1
                sid = _unique(f"{tri.id}/{c}", taken_t)
                taken_t.add(sid)
                a, b, d = P[(i, j)], P[(i + 1, j)], P[(i, j + 1)]
                struc.add_tri_element(sid, a, b, d, tri.section_name)
                subs.append((sid, ((i, j), (i + 1, j), (i, j + 1))))
                if j == 0:                       # on the A->B side
                    owner[frozenset((a, b))] = sid
                if i == 0:                       # on the A->C side
                    owner[frozenset((a, d))] = sid
                if i + j == n - 1:               # on the B->C side
                    owner[frozenset((b, d))] = sid
                if i + j < n - 1:                # the downward triangle
                    c += 1
                    sid2 = _unique(f"{tri.id}/{c}", taken_t)
                    taken_t.add(sid2)
                    struc.add_tri_element(sid2, P[(i + 1, j)], P[(i + 1, j + 1)],
                                          P[(i, j + 1)], tri.section_name)
                    subs.append((sid2, ((i + 1, j), (i + 1, j + 1),
                                        (i, j + 1))))
        made_elems += n * n
        _reassign_edge_loads(struc, edge_loads, tri, P, owner, n)
        _reassign_tri_temperatures(struc, temps, subs, n)
        for sid, _lattice in subs:
            for a in area_loads:
                struc.add_area_load(sid, a.load_case_id, pz=a.pz)
            for sp in springs:
                struc.add_area_spring(sid, sp.kz)

    struc._node_dof_index = None
    return {'triangles': len(tris), 'elements': made_elems,
            'new_nodes': len(struc.nodes) - before_nodes}


def _quad_list(struc, ids):
    return [struc.quad_elements_by_id[q] for q in ids
            if q in struc.quad_elements_by_id]


def _unique_quad_edges(quads) -> set:
    """The distinct edges of a set of quads, each a frozenset of two node ids —
    a shared edge appears once, so edge nodes are counted once."""
    edges = set()
    for q in quads:
        ns = (q.node_i, q.node_j, q.node_k, q.node_l)
        for a in range(4):
            edges.add(frozenset((ns[a], ns[(a + 1) % 4])))
    return edges


def preview_quads(struc, ids, n: int) -> dict:
    """Elements and new nodes for an n-per-side quad division. No mutation.

    One quad becomes n². Interior nodes ((n-1)² each) are never shared; edge
    nodes ((n-1) per distinct edge) are counted once; the corners already
    exist."""
    n = int(n)
    quads = _quad_list(struc, ids)
    if n < 2 or not quads:
        return {'quads': len(quads), 'elements': 0, 'new_nodes': 0}
    interior = len(quads) * (n - 1) * (n - 1)
    edge = len(_unique_quad_edges(quads)) * (n - 1)
    return {'quads': len(quads), 'elements': len(quads) * n * n,
            'new_nodes': interior + edge}


def divide_quads(struc, ids, n: int) -> dict:
    """Tile each selected quad into n×n conforming sub-quads.

    The 4-node analogue of :func:`divide_triangles`: each quad is subdivided by
    a bilinear (n+1)×(n+1) lattice of its four corners, reusing the corner ids
    and sharing edge nodes through the same spatial hash, so a shared edge cut
    with one n stays conforming from both sides. The parent's section is carried
    to every sub-quad; its area load (pz), area spring (kz) and per-node
    temperature are carried across too — the pressure/stiffness intensities copy
    unchanged (they are per-area), and the nodal ΔT field is bilinearly sampled
    at each sub-quad's corners, mirroring how the triangle version interpolates.
    """
    n = int(n)
    quads = _quad_list(struc, ids)
    if n < 2 or not quads:
        return {'quads': 0, 'elements': 0, 'new_nodes': 0}

    hash_ = _node_hash(struc)
    taken_n = set(struc.nodes)
    before_nodes = len(struc.nodes)
    made_elems = 0

    for quad in quads:
        A = struc.nodes[quad.node_i]
        B = struc.nodes[quad.node_j]
        C = struc.nodes[quad.node_k]
        D = struc.nodes[quad.node_l]
        section = quad.section_name

        # Loads attached to this quad, captured before it is dropped.
        area_loads = [a for a in getattr(struc, 'quad_area_loads', [])
                      if a.quad_id == quad.id]
        springs = [s for s in getattr(struc, 'quad_area_springs', [])
                   if s.quad_id == quad.id]
        temps = [t for t in getattr(struc, 'quad_temperature_loads', [])
                 if t.quad_id == quad.id]
        edge_loads = [e for e in getattr(struc, 'quad_edge_loads', [])
                      if e.quad_id == quad.id]

        def _xy(u, v):
            x = ((1 - u) * (1 - v) * A.x + u * (1 - v) * B.x
                 + u * v * C.x + (1 - u) * v * D.x)
            y = ((1 - u) * (1 - v) * A.y + u * (1 - v) * B.y
                 + u * v * C.y + (1 - u) * v * D.y)
            return x, y

        # Bilinear lattice P[i][j], (u, v) = (i/n, j/n); corners reuse ids.
        P: dict[tuple[int, int], str] = {}
        for i in range(n + 1):
            for j in range(n + 1):
                if (i, j) == (0, 0):
                    P[(i, j)] = quad.node_i
                elif (i, j) == (n, 0):
                    P[(i, j)] = quad.node_j
                elif (i, j) == (n, n):
                    P[(i, j)] = quad.node_k
                elif (i, j) == (0, n):
                    P[(i, j)] = quad.node_l
                else:
                    x, y = _xy(i / n, j / n)
                    P[(i, j)] = _get_node(struc, hash_, taken_n, x, y,
                                          f"{quad.id}~{i}_{j}")

        parent_id = quad.id
        _drop_quad(struc, parent_id)

        taken_q = set(struc.quad_elements_by_id)
        subs: list[tuple] = []          # (sub id, its four (i, j) lattice pts)
        # A boundary sub-segment (frozenset of its two node ids) -> the sub-quad
        # that owns it, so an edge load can be re-placed on it.
        owner: dict[frozenset, str] = {}
        c = 0
        for i in range(n):
            for j in range(n):
                c += 1
                sid = _unique(f"{parent_id}/{c}", taken_q)
                taken_q.add(sid)
                corners = ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1))
                struc.add_quad_element(sid, P[(i, j)], P[(i + 1, j)],
                                       P[(i + 1, j + 1)], P[(i, j + 1)],
                                       section)
                subs.append((sid, corners))
                for k in range(4):
                    ca, cb = corners[k], corners[(k + 1) % 4]
                    on_edge = ((ca[1] == 0 and cb[1] == 0)
                               or (ca[0] == n and cb[0] == n)
                               or (ca[1] == n and cb[1] == n)
                               or (ca[0] == 0 and cb[0] == 0))
                    if on_edge:
                        owner[frozenset((P[ca], P[cb]))] = sid
        made_elems += n * n

        _reassign_quad_edge_loads(struc, edge_loads, quad, P, owner, n)
        # Re-apply the intensive loads on every sub-quad.
        for sid, _corners in subs:
            for a in area_loads:
                struc.add_area_load(sid, a.load_case_id, pz=a.pz)
            for s in springs:
                struc.add_area_spring(sid, s.kz)
        # Per-node temperature: bilinear-sample the parent field at each corner.
        for tl in temps:
            def _dt(u, v):
                return ((1 - u) * (1 - v) * tl.dt_i + u * (1 - v) * tl.dt_j
                        + u * v * tl.dt_k + (1 - u) * v * tl.dt_l)
            for sid, corners in subs:
                vals = [_dt(i / n, j / n) for (i, j) in corners]
                struc.add_quad_temperature_load(
                    sid, tl.load_case_id, dt_i=vals[0], dt_j=vals[1],
                    dt_k=vals[2], dt_l=vals[3], dt_gradient=tl.dt_gradient)

    struc._node_dof_index = None
    return {'quads': len(quads), 'elements': made_elems,
            'new_nodes': len(struc.nodes) - before_nodes}


def _drop_quad(struc, qid):
    struc.quad_elements = [q for q in struc.quad_elements if q.id != qid]
    struc.quad_elements_by_id.pop(qid, None)
    # every load, spring and temperature that named it (references.SITES)
    references.remove(struc, 'quad', qid)


def _drop_triangle(struc, tid):
    struc.tri_elements = [t for t in struc.tri_elements if t.id != tid]
    struc.tri_elements_by_id.pop(tid, None)
    references.remove(struc, 'tri', tid)


def _reassign_tri_temperatures(struc, temps, subs, n):
    """Interpolate each per-node temperature onto the sub-triangles.

    The parent's ΔT is a linear field, its three nodal values read by the
    barycentric weights of the lattice point: at (i, j) the weight on node_i is
    1 - (i+j)/n, on node_j is i/n, on node_k is j/n. Each sub-triangle then
    carries the field sampled at its own three corners — so the mean a CST
    responds to follows the parent field, and a later, finer division reads
    the same values again.
    """
    if not temps:
        return

    def at(lattice, tl):
        i, j = lattice
        wa = 1.0 - (i + j) / n
        return wa * tl.dt_i + (i / n) * tl.dt_j + (j / n) * tl.dt_k

    for tl in temps:
        for sid, (la, lb, lc) in subs:
            struc.add_tri_temperature_load(
                sid, tl.load_case_id,
                dt_i=at(la, tl), dt_j=at(lb, tl), dt_k=at(lc, tl))


def _reassign_edge_loads(struc, edge_loads, tri, P, owner, n):
    """Spread each uniform edge load over the sub-segments of its edge.

    The load is per unit length and uniform, so every sub-segment along the
    edge carries the same fx, fy — one new edge load per sub-segment, on the
    sub-triangle that owns it. The owning triangle is found from the boundary
    map built while the sub-triangles were created.
    """
    if not edge_loads:
        return
    corner = {tri.node_i: (0, 0), tri.node_j: (n, 0), tri.node_k: (0, n)}
    taken = {e.id for e in struc.tri_edge_loads}
    for el in edge_loads:
        ca, cb = corner.get(el.node_a), corner.get(el.node_b)
        if ca is None or cb is None:
            continue
        for s in range(n):
            la = _lattice_on_edge(ca, cb, s, n)
            lb = _lattice_on_edge(ca, cb, s + 1, n)
            na, nb = P.get(la), P.get(lb)
            sid = owner.get(frozenset((na, nb)))
            if sid is None:
                continue
            nid = _unique(f"{el.id}/{s + 1}", taken)
            taken.add(nid)
            struc.add_tri_edge_load(nid, sid, na, nb, el.load_case_id,
                                    fx=el.fx, fy=el.fy, coord_sys=el.coord_sys,
                                    pn=el.pn, pt=el.pt)


def _reassign_quad_edge_loads(struc, edge_loads, quad, P, owner, n):
    """Spread each uniform edge load of a quad over the sub-segments of its edge
    (one new edge load per sub-segment, on the sub-quad that owns it) — the quad
    counterpart of :func:`_reassign_edge_loads`."""
    if not edge_loads:
        return
    corner = {quad.node_i: (0, 0), quad.node_j: (n, 0),
              quad.node_k: (n, n), quad.node_l: (0, n)}
    taken = {e.id for e in struc.quad_edge_loads}
    for el in edge_loads:
        ca, cb = corner.get(el.node_a), corner.get(el.node_b)
        if ca is None or cb is None:
            continue
        for s in range(n):
            la = _lattice_on_edge(ca, cb, s, n)
            lb = _lattice_on_edge(ca, cb, s + 1, n)
            na, nb = P.get(la), P.get(lb)
            sid = owner.get(frozenset((na, nb)))
            if sid is None:
                continue
            nid = _unique(f"{el.id}/{s + 1}", taken)
            taken.add(nid)
            struc.add_quad_edge_load(nid, sid, na, nb, el.load_case_id,
                                     fx=el.fx, fy=el.fy,
                                     coord_sys=el.coord_sys, pn=el.pn, pt=el.pt)


def _lattice_on_edge(ca, cb, s, n):
    """The lattice index s/n of the way from corner ca to corner cb."""
    return (ca[0] + (cb[0] - ca[0]) * s // n,
            ca[1] + (cb[1] - ca[1]) * s // n)
