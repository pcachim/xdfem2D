"""Meshing of polygons for surface objects (triangles, and — where the
surface's referenced section is a quad formulation — quads).

Two branches, dispatched by :func:`mesh_polygon` (triangle-only) and, for a
quad-eligible surface, :func:`mesh_quad_structured_cells` directly (see
``xdfem2d.geo_expand._expand_surface``, dev/IMPLEMENT_QUAD.md Phase 6):

* **Structured (transfinite)** — for a convex quadrilateral with no forced
  interior/edge node: a mapped grid whose diagonals alternate by quadrant, so
  the mesh is symmetric about both mid-planes. See
  :func:`mesh_quad_structured` (always 2 triangles per cell — the original,
  triangle-only mesher, still used for CST/Allman/DKT/MITC3 surfaces and as
  the per-cell fallback below) and :func:`mesh_quad_structured_cells` (the
  same grid, but keeping a cell as one quad element where
  :func:`quad_cell_quality` says it is well-shaped enough, falling back to
  the identical 2-triangle split cell-by-cell otherwise).
* **Unstructured (Delaunay)** — for every other outline: a Delaunay
  triangulation (SciPy/Qhull when available, else a pure-Python Bowyer–Watson
  fallback) over the polygon boundary refined to the target element size, with
  pre-existing nodes forced in, plus an interior grid of points. Triangles whose
  centroid falls outside the polygon are discarded, so concave polygons are
  handled — approximately: the triangulation is *not* constrained, so a strongly
  re-entrant boundary can still be bridged by a triangle whose centroid lies
  inside. Constrained Delaunay would be needed to close that gap. This branch
  is triangle-only — quads are never emitted here (dev/IMPLEMENT_QUAD.md
  Phase 6 explicitly keeps the unstructured branch untouched).

Public entry points: :func:`mesh_polygon` (triangles) and
:func:`mesh_quad_structured_cells` (mixed quad/triangle, structured only).
"""
from __future__ import annotations

import math

Point = tuple[float, float]


# ── geometry helpers ────────────────────────────────────────────────────────

def point_in_poly(x: float, y: float, poly: list[Point]) -> bool:
    """Ray-casting point-in-polygon (polygon given as ordered vertices, the
    closing edge implied)."""
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and \
                (x < (xj - xi) * (y - yi) / (yj - yi + 1e-30) + xi):
            inside = not inside
        j = i
    return inside


def _dist_point_seg(px, py, ax, ay, bx, by):
    """(distance, t) from point to segment a→b; t is the clamped projection."""
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    if L2 < 1e-30:
        return math.hypot(px - ax, py - ay), 0.0
    t = ((px - ax) * dx + (py - ay) * dy) / L2
    t = max(0.0, min(1.0, t))
    cx, cy = ax + t * dx, ay + t * dy
    return math.hypot(px - cx, py - cy), t


def polygon_area(poly: list[Point]) -> float:
    a = 0.0
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        a += x0 * y1 - x1 * y0
    return a / 2.0


# ── boundary refinement ─────────────────────────────────────────────────────

def refine_boundary(poly: list[Point], target: float,
                    forced: list[Point] = (), tol: float = 1e-7) -> list[Point]:
    """Return the ordered boundary loop, each polygon edge split at any *forced*
    point lying on it and then subdivided so every piece is ≤ *target*. The
    returned list does not repeat the first vertex."""
    out: list[Point] = []
    n = len(poly)
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        # Forced points lying on this edge → split parameters.
        cuts = []
        for fx, fy in forced:
            d, t = _dist_point_seg(fx, fy, a[0], a[1], b[0], b[1])
            if d <= tol and 1e-9 < t < 1 - 1e-9:
                cuts.append(t)
        cuts = sorted(set(cuts))
        # Piecewise: a → cut1 → … → b, each piece subdivided to target.
        stops = [0.0] + cuts + [1.0]
        for k in range(len(stops) - 1):
            t0, t1 = stops[k], stops[k + 1]
            p0 = (a[0] + (b[0] - a[0]) * t0, a[1] + (b[1] - a[1]) * t0)
            p1 = (a[0] + (b[0] - a[0]) * t1, a[1] + (b[1] - a[1]) * t1)
            seg = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
            # The epsilon is not cosmetic. Each cut parameter comes from a
            # projection, so a piece that is exactly one target long can be
            # reconstructed as target*(1 + 1e-15) — and a bare ceil() then asks
            # for two divisions, planting a node at the midpoint. On an edge
            # shared with an already-meshed neighbour that node has no partner
            # on the other side, which leaves a hanging node and a crack in the
            # mesh. Only a piece genuinely longer than the target should split.
            m = max(1, math.ceil(seg / target - 1e-9)) if target > 0 else 1
            for s in range(m):        # include p0, exclude p1 (added next piece)
                out.append((p0[0] + (p1[0] - p0[0]) * s / m,
                            p0[1] + (p1[1] - p0[1]) * s / m))
    return out


def _grid_interior(poly, target, boundary, min_frac=0.55):
    """Interior grid points at ~*target* spacing, kept when strictly inside the
    polygon and not closer than ``min_frac·target`` to any boundary point.

    The lattice is offset by half a cell so its points never land on the
    boundary itself.

    The proximity test uses a spatial hash over the boundary points rather than
    scanning them all: cells of side ``dmin`` mean any point within ``dmin`` of
    a query must lie in the query's own cell or one of the 8 neighbours, since
    ``|px - qx| < dmin`` puts ``floor(px/dmin)`` and ``floor(qx/dmin)`` at most
    one apart. Boundary points are spaced ~``target`` apart and
    ``dmin < target``, so a cell holds ~one point and each query is O(1). That
    turns the filter from O(interior x boundary) into O(interior + boundary) —
    on a wall outline it is ~3x faster at target 0.5 and ~45x at 0.03, where
    the brute-force version costs seconds. The result is bit-identical."""
    if target <= 0:
        return []
    xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    dmin = min_frac * target
    dmin2 = dmin * dmin

    # Spatial hash of the boundary points: cell index -> points in that cell.
    cells: dict[tuple[int, int], list[Point]] = {}
    for bx, by in boundary:
        cells.setdefault((int(math.floor(bx / dmin)),
                          int(math.floor(by / dmin))), []).append((bx, by))

    def _near_boundary(x: float, y: float) -> bool:
        ci = int(math.floor(x / dmin)); cj = int(math.floor(y / dmin))
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for bx, by in cells.get((ci + di, cj + dj), ()):
                    if (x - bx) ** 2 + (y - by) ** 2 < dmin2:
                        return True
        return False

    pts: list[Point] = []
    ny = int(math.floor((maxy - miny) / target))
    nx = int(math.floor((maxx - minx) / target))
    for j in range(ny + 1):
        y = miny + target * (j + 0.5)
        if y >= maxy:
            continue
        for i in range(nx + 1):
            x = minx + target * (i + 0.5)
            if x >= maxx:
                continue
            if point_in_poly(x, y, poly) and not _near_boundary(x, y):
                pts.append((x, y))
    return pts


# ── Bowyer–Watson Delaunay ──────────────────────────────────────────────────

def delaunay(points: list[Point]) -> list[tuple[int, int, int]]:
    """Delaunay triangulation of *points*; returns index triples.

    Uses SciPy's Qhull-backed triangulation (O(n log n)) when available — far
    faster on fine meshes — and falls back to the pure-Python Bowyer–Watson
    implementation otherwise."""
    if len(points) < 3:
        return []
    try:
        from scipy.spatial import Delaunay as _Delaunay
        import numpy as _np
        tri = _Delaunay(_np.asarray(points, dtype=float))
        return [tuple(int(i) for i in s) for s in tri.simplices]
    except Exception:
        return _delaunay_bowyer_watson(points)


def _delaunay_bowyer_watson(points: list[Point]) -> list[tuple[int, int, int]]:
    """Pure-Python Delaunay (incremental Bowyer–Watson). Fallback for when SciPy
    is not installed; O(n²), so noticeably slower on fine meshes."""
    n = len(points)
    if n < 3:
        return []
    verts = [(float(x), float(y)) for x, y in points]
    minx = min(p[0] for p in verts); maxx = max(p[0] for p in verts)
    miny = min(p[1] for p in verts); maxy = max(p[1] for p in verts)
    dmax = max(maxx - minx, maxy - miny) or 1.0
    midx = (minx + maxx) / 2.0; midy = (miny + maxy) / 2.0
    verts += [(midx - 20 * dmax, midy - dmax),
              (midx, midy + 20 * dmax),
              (midx + 20 * dmax, midy - dmax)]
    s0, s1, s2 = n, n + 1, n + 2
    tris: list[tuple[int, int, int]] = [(s0, s1, s2)]

    def circum(a, b, c):
        ax, ay = verts[a]; bx, by = verts[b]; cx, cy = verts[c]
        d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(d) < 1e-18:
            return None
        a2 = ax * ax + ay * ay; b2 = bx * bx + by * by; c2 = cx * cx + cy * cy
        ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
        uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
        return ux, uy, (ax - ux) ** 2 + (ay - uy) ** 2

    for ip in range(n):
        px, py = verts[ip]
        bad = []
        for t in tris:
            c = circum(*t)
            if c is not None and (px - c[0]) ** 2 + (py - c[1]) ** 2 <= c[2] + 1e-12:
                bad.append(t)
        # Boundary edges of the cavity (edges not shared by two bad triangles).
        edge_count: dict[tuple[int, int], int] = {}
        for t in bad:
            for e in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
                key = (min(e), max(e))
                edge_count[key] = edge_count.get(key, 0) + 1
        for t in bad:
            tris.remove(t)
        for t in bad:
            for e in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
                key = (min(e), max(e))
                if edge_count[key] == 1:
                    tris.append((e[0], e[1], ip))
    return [t for t in tris if max(t) < n]


# ── public API ──────────────────────────────────────────────────────────────

def _dedup(points: list[Point], tol: float = 1e-7):
    """Merge near-coincident points; return (unique_points, remap old→new)."""
    keyed: dict[tuple[int, int], int] = {}
    uniq: list[Point] = []
    remap: list[int] = []
    q = 1.0 / tol
    for (x, y) in points:
        k = (round(x * q), round(y * q))
        if k in keyed:
            remap.append(keyed[k])
        else:
            keyed[k] = len(uniq)
            remap.append(len(uniq))
            uniq.append((x, y))
    return uniq, remap


def _is_convex_quad(poly: list[Point], tol: float = 1e-12) -> bool:
    """True when *poly* is a non-degenerate convex quadrilateral (the cross
    products of consecutive edges all share one sign)."""
    if len(poly) != 4:
        return False
    sign = 0
    for i in range(4):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % 4]
        cx, cy = poly[(i + 2) % 4]
        cr = (bx - ax) * (cy - by) - (by - ay) * (cx - bx)
        if abs(cr) <= tol:
            return False              # collinear vertices → degenerate
        s = 1 if cr > 0 else -1
        if sign == 0:
            sign = s
        elif s != sign:
            return False
    return True


def _touches(x, y, poly, tol: float) -> bool:
    """True when (x, y) is strictly inside *poly* or lies on its boundary — i.e.
    a point the mesh of *poly* would have to contain."""
    if point_in_poly(x, y, poly):
        return True
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        d, _ = _dist_point_seg(x, y, a[0], a[1], b[0], b[1])
        if d <= tol:
            return True
    return False


def _divisions(length: float, target: float) -> int:
    """Number of cells along an edge of *length* at element size *target*:
    the nearest integer to ``length/target``, minimum 1. No parity constraint
    -- see :func:`_cell_triangles` for how an odd division still yields an
    exactly symmetric mesh (a plain per-cell diagonal used to require an even
    count; a 4-triangle fan through a new centre node now covers the one
    column/row an odd count adds, which a diagonal alone never could).

    This used to also force the result to be even, costing up to an extra
    half-division of deviation from *target* for no reason once the fan
    alternative existed (e.g. ``x = 4.6`` no longer needs to become 4 or 6 --
    5, the honestly nearest count, is now just as usable)."""
    if target <= 0:
        return 1
    return max(1, round(length / target))


# Relative tolerance used by _structured_grid when the caller does not pin an
# absolute one (see _structured_grid's tol parameter). A fixed 1e-6 m is far
# too tight on a model spanning hundreds of metres and unnecessarily loose on
# one spanning a few centimetres — either way it makes whether a neighbouring
# object's node "coincides" with this grid (and therefore whether the whole
# panel gets a structured mesh at all) depend on the model's absolute scale
# rather than on the geometry. Scaling it to the quad's own extent keeps that
# decision consistent across scales, the same way _improve_mesh already scales
# its own tolerance to the polygon's extent.
_STRUCTURED_GRID_REL_TOL = 1e-6


def _edge_params(edge_a, edge_b, forced, tol):
    """Parametric positions (0, 1) of every *forced* point that lies on edge
    *edge_a* or *edge_b* (each a ``((ax,ay), (bx,by))`` pair) within *tol* --
    excluding the shared corners themselves (``t`` near 0 or 1), which are
    always grid nodes regardless of the division count.

    A structured grid's edges are the straight sides of the quad itself, so
    the bilinear map is exactly linear along each one: a point's parameter
    *t* along an edge (from :func:`_dist_point_seg`) is precisely the ``u``
    (or ``v``) value it would need to land on in the grid. That is what lets
    :func:`_fit_divisions` search for a division count under which every
    forced point on these edges lands on an exact grid line, instead of only
    ever checking the one division count *target* happens to produce."""
    out = []
    for (ax, ay), (bx, by) in (edge_a, edge_b):
        for fx, fy in forced:
            d, t = _dist_point_seg(fx, fy, ax, ay, bx, by)
            if d <= tol and 1e-9 < t < 1 - 1e-9:
                out.append(t)
    return out


def _fit_divisions(base_n: int, side_len: float, params: list[float],
                   tol: float, max_steps: int = 6) -> int:
    """Return the even division count closest to *base_n* under which every
    parameter in *params* (0, 1) lands within *tol* (a distance, converted
    here to a fraction of *side_len*) of an exact grid line ``k/n`` -- or
    *base_n* unchanged if none of the counts searched works.

    Searched counts: *base_n* itself, then ``base_n ± 1``, ``base_n ± 2``, …
    up to *max_steps* steps out (so at most ``2*max_steps`` extra candidates)
    -- close enough to *target* to still be "the same element size, gently
    adjusted", never a division wildly different from what was asked for. No
    longer restricted to even counts (:func:`_divisions` dropped that
    requirement), so every integer neighbour of *base_n* is now a candidate,
    not just every other one -- twice the chance of finding an exact fit
    within the same search window. Finding no match here is not a failure:
    :func:`_structured_grid`'s own full-grid coincidence check afterwards is
    still the actual gate, and returning *base_n* unchanged simply leaves
    that check to fail exactly as it did before this search existed, falling
    back to the unstructured branch (:func:`mesh_polygon` docstring) as
    always.

    This is a local, bounded accommodation -- it changes how many divisions
    THIS object's own structured grid uses so a neighbour's node still lands
    on it, not a general hybrid mesher: a forced point that genuinely cannot
    be reached by any nearby count (a neighbour meshed at a very different
    element size, say) still falls back to the unstructured branch for the
    whole object, same as always."""
    if not params or side_len <= 0:
        return base_n
    tol_frac = tol / side_len

    def _fits(n: int) -> bool:
        for t in params:
            k = round(t * n)
            if k <= 0 or k >= n:
                return False
            if abs(t - k / n) > tol_frac:
                return False
        return True

    if _fits(base_n):
        return base_n
    for step in range(1, max_steps + 1):
        for cand in (base_n + step, base_n - step):
            if cand >= 1 and _fits(cand):
                return cand
    return base_n


def _mirror_mid(n: int) -> float:
    """Index of the self-mirror column/row for *n* cells (0-based, index
    range ``0..n-1``): the value ``i`` for which ``i == n-1-i``, i.e. reflects
    onto itself. An odd ``n`` has an exact integer at that index (the true
    middle cell); an even ``n`` gives a half-integer that no integer ``i``
    ever equals, so nothing is ever "the middle" -- exactly the old
    ``i < n/2`` split, since ``i < (n-1)/2 + 0.5`` is the same test as
    ``i < n/2`` for every integer ``i``. Used by both the per-cell quadrant
    test AND, when it lands exactly on an integer, to flag the one row/column
    :func:`_cell_triangles` must fan instead of split by a single diagonal."""
    return (n - 1) / 2.0


def _cell_triangles(points: list, a: int, b: int, c: int, d: int,
                    on_mirror: bool, diag_ac: bool):
    """Triangles covering one grid cell (CCW corners ``a, b, c, d``).

    *on_mirror* is True only for a cell in the self-mirror middle column
    and/or row an odd ``nu``/``nv`` produces (:func:`_mirror_mid`) -- there,
    no single diagonal can be symmetric with itself (it swaps with the
    OTHER diagonal under exactly the reflection that maps the cell onto
    itself), so this appends a new centre-point to *points* (mutated in
    place) and fans out 4 triangles instead; that fan (and only that fan) is
    invariant under any reflection or rotation that maps the cell onto
    itself, which is what a middle cell -- unlike every other cell, which has
    a distinct mirror partner elsewhere in the grid to be the opposite
    diagonal of -- actually needs.

    Otherwise, the ordinary 2-triangle split on the ``a-c`` diagonal when
    *diag_ac* else the ``b-d`` diagonal -- the same per-quadrant choice
    :func:`mesh_quad_structured` has always made, unchanged for every cell
    that isn't on a mirror line."""
    if on_mirror:
        (ax, ay), (bx, by) = points[a], points[b]
        (cx, cy), (dx, dy) = points[c], points[d]
        e = len(points)
        points.append(((ax + bx + cx + dx) / 4.0, (ay + by + cy + dy) / 4.0))
        return [(a, b, e), (b, c, e), (c, d, e), (d, a, e)]
    if diag_ac:
        return [(a, b, c), (a, c, d)]
    return [(a, b, d), (b, c, d)]


def _structured_grid(quad: list[Point], target: float,
                     forced: list[Point] = (), tol: float | None = None):
    """Shared core of the structured (transfinite) branch: build the bilinear
    grid of *quad* at element size *target* and check every *forced* point can
    be honoured. Returns ``(points, nu, nv)`` — *points* row-major, ``nu+1``
    columns by ``nv+1`` rows (index ``j*(nu+1)+i``) — or ``None`` when the
    outline is not a convex quad, ``target<=0``, or a forced point cannot be
    honoured. Used by both :func:`mesh_quad_structured` (always 2 triangles per
    cell) and :func:`mesh_quad_structured_cells` (quad cells where they are
    good enough, dev/IMPLEMENT_QUAD.md Phase 6) so the two stay on the exact
    same grid — sharing a boundary between them still conforms node-for-node.

    *tol* is the coincidence tolerance used both to snap a *forced* point onto
    a grid node and to test whether it merely touches the outline (in which
    case an unsnapped point fails the whole grid, see below). ``None`` (the
    default) derives it from the quad's own size — ``_STRUCTURED_GRID_REL_TOL``
    times the longer of its two diagonals — rather than using a fixed absolute
    distance, so the decision scales with the model instead of silently
    tightening or loosening as the model's units/extent change.

    Before the grid is built, :func:`_fit_divisions` gets a first, local shot
    at any *forced* point that lies on one of the quad's own four edges: it
    looks for a nearby even division count that lands that point on an exact
    grid line, rather than only ever trying the one count *target* alone
    would produce. This is what usually saves a panel from losing its
    structured mesh entirely just because a neighbouring panel happened to
    use a slightly different element size — see :func:`_edge_params` and
    :func:`_fit_divisions` for the details. It is still only a local
    adjustment of THIS grid's own resolution, not a general hybrid mesher: a
    forced point nothing nearby can reach still fails the check below exactly
    as before, and the caller falls back to the unstructured branch.
    """
    if not _is_convex_quad(quad) or target <= 0:
        return None
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = quad
    if tol is None:
        diag1 = math.hypot(x2 - x0, y2 - y0)
        diag2 = math.hypot(x3 - x1, y3 - y1)
        tol = max(1e-9, _STRUCTURED_GRID_REL_TOL * max(diag1, diag2))
    # Divisions from the mean length of each pair of opposite sides.
    side_u = 0.5 * (math.hypot(x1 - x0, y1 - y0) + math.hypot(x2 - x3, y2 - y3))
    side_v = 0.5 * (math.hypot(x3 - x0, y3 - y0) + math.hypot(x2 - x1, y2 - y1))
    nu = _divisions(side_u, target)
    nv = _divisions(side_v, target)

    # Before committing to the target-only division counts, see whether a
    # nearby even count would let every forced point ALREADY on one of this
    # quad's own edges land on an exact grid line -- typically a neighbouring
    # object's node, meshed at a slightly different target size, that the
    # plain target-derived nu/nv would otherwise miss by a fraction of a
    # division and so discard the whole structured grid for (dev
    # suggestion #1: this quad's own grid is always tried first and adjusted
    # locally; only a point neither this search nor the exact-snap check
    # below can honour still falls back to the unstructured branch entirely).
    if forced:
        u_params = _edge_params(((x0, y0), (x1, y1)), ((x3, y3), (x2, y2)),
                                forced, tol)
        v_params = _edge_params(((x0, y0), (x3, y3)), ((x1, y1), (x2, y2)),
                                forced, tol)
        nu = _fit_divisions(nu, side_u, u_params, tol)
        nv = _fit_divisions(nv, side_v, v_params, tol)

    points: list[Point] = []
    for j in range(nv + 1):
        v = j / nv
        for i in range(nu + 1):
            u = i / nu
            # Bilinear (transfinite) map of the unit square onto the corners.
            x = ((1 - u) * (1 - v) * x0 + u * (1 - v) * x1
                 + u * v * x2 + (1 - u) * v * x3)
            y = ((1 - u) * (1 - v) * y0 + u * (1 - v) * y1
                 + u * v * y2 + (1 - u) * v * y3)
            points.append((x, y))

    # Every forced point that touches this outline must already be a grid node.
    if forced:
        q = lambda v: round(v / tol)                       # noqa: E731
        grid = {(q(x), q(y)) for x, y in points}
        for fx, fy in forced:
            if (q(fx), q(fy)) in grid:
                continue
            if _touches(fx, fy, quad, tol):
                return None

    return points, nu, nv


def mesh_quad_structured(quad: list[Point], target: float,
                         forced: list[Point] = (), tol: float | None = None):
    """Structured (transfinite) mesh of the convex quadrilateral *quad*, or
    ``None`` when the outline is not a convex quad or a forced node cannot be
    honoured.

    Nodes come from the bilinear map of the unit square onto the four corners,
    so a rectangle gives a regular grid and a general quad gives a mapped one.

    Each cell is split into two triangles, with the diagonal chosen **per
    quadrant** of the grid::

        +-------+-------+        left-bottom  and right-top   use one diagonal,
        |   \\   |   /   |        right-bottom and left-top    use the other.
        |    \\  |  /    |
        +-------+-------+        This pattern is invariant under reflection
        |    /  |  \\    |        about BOTH mid-planes and under the 180°
        |   /   |   \\   |        rotation, so a symmetric load on a symmetric
        +-------+-------+        outline gives a symmetric result.

    Choosing every diagonal the same way — what a Delaunay triangulation of a
    regular grid does, since the four corners of each cell are cocircular and
    the tie is broken consistently — breaks that symmetry and shows up as
    asymmetric displacements of a few percent.

    ``nu``/``nv`` need not be even any more (:func:`_divisions`): when one of
    them is odd there is a genuine self-mirror middle column and/or row, and
    a single diagonal can never be symmetric with itself there. Those cells
    (only those — every other cell keeps the plain 2-triangle split above)
    get a 4-triangle fan through a new centre point instead
    (:func:`_cell_triangles`), which is exactly what makes the mirror test
    below pass for an odd division too. This adds points beyond the
    ``(nu+1)*(nv+1)`` transfinite grid — callers that index into *points* by
    the ``j*(nu+1)+i`` formula should only ever do so for the first
    ``(nu+1)*(nv+1)`` of them; the rest are these fan centres, referenced only
    from the triangle list.

    *forced* are pre-existing points the mesh must contain. A forced point that
    already coincides with a grid node costs nothing, so the common case of two
    panels sharing an edge — where the neighbour's nodes fall exactly on this
    grid, provided both use the same element size — still meshes structured.
    Any other forced point touching the outline makes this return ``None`` so
    the caller falls back to the Delaunay branch, which conforms by
    construction.

    Returns ``(points, triangles)`` or ``None``.
    """
    grid = _structured_grid(quad, target, forced, tol)
    if grid is None:
        return None
    points, nu, nv = list(grid[0]), grid[1], grid[2]

    idx = lambda i, j: j * (nu + 1) + i        # noqa: E731
    mid_i, mid_j = _mirror_mid(nu), _mirror_mid(nv)
    tris: list[tuple[int, int, int]] = []
    for j in range(nv):
        for i in range(nu):
            a, b = idx(i, j), idx(i + 1, j)
            c, d = idx(i + 1, j + 1), idx(i, j + 1)
            on_mirror = i == mid_i or j == mid_j
            # XOR of the two half-plane tests → the quadrant pattern above.
            diag_ac = (i < mid_i) != (j < mid_j)
            tris += _cell_triangles(points, a, b, c, d, on_mirror, diag_ac)
    # Deliberately NOT run through _improve_mesh: every cell here is already
    # the bilinear map of the quad, split by the fixed, exactly symmetric
    # per-quadrant diagonal (or, on a mirror line, fan) pattern documented
    # above -- that IS the quality guarantee (docstring: "better-shaped
    # elements than a Delaunay triangulation of the same points"), not a
    # rough starting point to polish. _improve_mesh's Laplacian step assumes
    # a roughly uniform triangle-adjacency neighbourhood; the diagonal flip
    # at each quadrant boundary breaks that even on a perfectly regular grid,
    # and measured smoothing here pulls interior nodes up to ~40% of the
    # target element size out of position -- degrading an already-optimal
    # mesh, not improving it. (A version of this ran briefly and was
    # reverted after that measurement; see :func:`mesh_quad_structured_cells`
    # for where a quality pass legitimately does apply, on cells the
    # quadrant pattern was never meant to police in the first place.)
    return points, tris


def quad_cell_quality(coords: list[Point]) -> tuple[float, float]:
    """``(max_skew_deg, aspect_ratio)`` of the convex quad cell *coords*
    (4 CCW corners) — the two metrics :func:`mesh_quad_structured_cells` uses
    to decide whether a grid cell is good enough to keep as one quad element,
    on top of the hard validity check in
    :func:`xdfem2d.model_check.quad_geometry_problems`.

    ``max_skew_deg`` is the largest deviation of any interior angle from 90°
    (0 for a rectangle; a knife-thin corner approaches 90). ``aspect_ratio``
    is the longest edge divided by the shortest (1 for a square). Both are
    the standard, cheap-to-compute proxies for how far a bilinear-map element
    is from its ideal (square) shape — the shape a Q4/QM6/DKT4/MITC4 kernel
    is most accurate on; distortion in either erodes accuracy well before the
    Jacobian actually goes non-positive."""
    edges = []
    for i in range(4):
        x0, y0 = coords[i]
        x1, y1 = coords[(i + 1) % 4]
        edges.append(math.hypot(x1 - x0, y1 - y0))
    aspect = max(edges) / max(min(edges), 1e-12)
    skew = 0.0
    for i in range(4):
        x0, y0 = coords[(i - 1) % 4]
        x1, y1 = coords[i]
        x2, y2 = coords[(i + 1) % 4]
        v1x, v1y = x0 - x1, y0 - y1
        v2x, v2y = x2 - x1, y2 - y1
        dot = v1x * v2x + v1y * v2y
        cross = v1x * v2y - v1y * v2x
        angle = math.degrees(math.atan2(abs(cross), dot))
        skew = max(skew, abs(angle - 90.0))
    return skew, aspect


def mesh_quad_structured_cells(quad: list[Point], target: float,
                               forced: list[Point] = (), tol: float | None = None,
                               skew_tol_deg: float = 30.0,
                               aspect_tol: float = 4.0):
    """Structured (transfinite) mesh of *quad*, emitting each grid cell as a
    single 4-node quad where it is well-shaped, and falling back to the same
    2-triangle split :func:`mesh_quad_structured` uses otherwise
    (dev/IMPLEMENT_QUAD.md Phase 6).

    A cell keeps its quad shape only when BOTH hold:

    * hard validity — CCW winding, convexity, positive Jacobian at all four
      corners, via :func:`xdfem2d.model_check.quad_geometry_problems` (the
      same check ``Structure2D.add_quad_element`` runs eagerly, so a cell that
      would be rejected there is never even attempted here);
    * quality — :func:`quad_cell_quality` within *skew_tol_deg* /
      *aspect_tol* of square. The defaults (30°, 4.0) are deliberately loose:
      this is a *mesh-quality* gate, not a validity gate — a cell well inside
      "valid" but visibly non-square (a mapped grid's edge cells on a
      strongly tapered quad, say) still degrades a Q4/QM6/DKT4/MITC4 kernel's
      accuracy more than a pair of triangles would, so it is worth splitting;
      a cell only mildly off-square is not.

    A cell that fails either check falls back to the SAME per-quadrant
    diagonal :func:`mesh_quad_structured` would have chosen for it, so a
    surface that mixes quad and triangle cells still has the exact symmetric
    pattern at every triangulated cell, and a neighbour meshed with plain
    :func:`mesh_quad_structured` still conforms node-for-node.

    Returns ``(points, quad_cells, tri_cells)`` — *quad_cells* a list of
    ``(i,j,k,l)`` CCW index quadruples, *tri_cells* a list of ``(i,j,k)``
    index triples — or ``None`` under the same conditions as
    :func:`mesh_quad_structured` (not a convex quad, ``target<=0``, or an
    unhonourable forced point).
    """
    grid = _structured_grid(quad, target, forced, tol)
    if grid is None:
        return None
    points, nu, nv = list(grid[0]), grid[1], grid[2]
    from .model_check import quad_geometry_problems

    idx = lambda i, j: j * (nu + 1) + i        # noqa: E731
    mid_i, mid_j = _mirror_mid(nu), _mirror_mid(nv)
    quads: list[tuple[int, int, int, int]] = []
    tris: list[tuple[int, int, int]] = []
    for j in range(nv):
        for i in range(nu):
            a, b = idx(i, j), idx(i + 1, j)
            c, d = idx(i + 1, j + 1), idx(i, j + 1)
            coords = [points[a], points[b], points[c], points[d]]
            keep_quad = not quad_geometry_problems(coords)
            if keep_quad:
                skew, aspect = quad_cell_quality(coords)
                keep_quad = skew <= skew_tol_deg and aspect <= aspect_tol
            if keep_quad:
                quads.append((a, b, c, d))
                continue
            # A quad-mirror cell that failed the quality gate still needs the
            # fan, same as mesh_quad_structured -- a plain diagonal here would
            # be exactly as unsymmetric as everywhere else this rule applies.
            on_mirror = i == mid_i or j == mid_j
            diag_ac = (i < mid_i) != (j < mid_j)
            tris += _cell_triangles(points, a, b, c, d, on_mirror, diag_ac)
    # NOT run through _improve_mesh, for the same reason mesh_quad_structured
    # no longer is: these triangles use the exact same fixed per-quadrant
    # diagonal pattern (just with some cells kept as quads instead), and
    # Laplacian smoothing distorts that pattern's already-regular geometry
    # rather than improving it -- measured, not assumed; see
    # mesh_quad_structured's docstring for the numbers. A genuinely irregular
    # or skewed cell here still gets its two triangles from the exact
    # quadrant rule; a real per-cell reshaping (rather than a global
    # smoothing pass unaware of which nodes are quad corners) is future work.
    return points, quads, tris


# ── mesh-quality improvement (Laplacian smoothing + min-angle edge flips) ────
# A plain Delaunay triangulation over a boundary + interior grid still leaves
# some slivers, especially where a faceted curved boundary meets the interior
# grid. These two passes lift the minimum element angle for every curved object
# (sectors, arcs, polygons) without changing the point count or the boundary.

def _tri_min_angle(points, i, j, k) -> float:
    """Smallest interior angle [deg] of triangle (i, j, k); 0 if degenerate."""
    ax, ay = points[i]; bx, by = points[j]; cx, cy = points[k]
    sides = [(ax - bx, ay - by, cx - bx, cy - by),   # angle at j
             (bx - cx, by - cy, ax - cx, ay - cy),   # angle at k
             (cx - ax, cy - ay, bx - ax, by - ay)]   # angle at i
    best = 180.0
    for ux, uy, vx, vy in sides:
        lu = math.hypot(ux, uy); lv = math.hypot(vx, vy)
        if lu < 1e-15 or lv < 1e-15:
            return 0.0
        cosv = max(-1.0, min(1.0, (ux * vx + uy * vy) / (lu * lv)))
        best = min(best, math.degrees(math.acos(cosv)))
    return best


def _boundary_mask(points, poly, tol):
    """True for every point lying on the polygon boundary (fixed during
    smoothing); interior grid points are movable."""
    mask = [False] * len(points)
    n = len(poly)
    for idx, (x, y) in enumerate(points):
        for i in range(n):
            a, b = poly[i], poly[(i + 1) % n]
            d, _ = _dist_point_seg(x, y, a[0], a[1], b[0], b[1])
            if d <= tol:
                mask[idx] = True
                break
    return mask


def _signed_area2(points, tri):
    a, b, c = tri
    (ax, ay), (bx, by), (cx, cy) = points[a], points[b], points[c]
    return (bx - ax) * (cy - ay) - (cx - ax) * (by - ay)


def _laplacian_smooth(points, tris, movable, poly):
    """One Laplacian pass: move each movable node to the centroid of its mesh
    neighbours. A move is rejected when it would leave the polygon or invert
    (fold) any incident triangle, so the mesh stays a valid tiling — its total
    area is preserved exactly."""
    adj: dict[int, set] = {}
    inc: dict[int, list] = {}
    for ti, tri in enumerate(tris):
        a, b, c = tri
        for u, v in ((a, b), (b, c), (c, a)):
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        for u in (a, b, c):
            inc.setdefault(u, []).append(ti)
    for i in movable:
        nb = adj.get(i)
        if not nb:
            continue
        nx = sum(points[j][0] for j in nb) / len(nb)
        ny = sum(points[j][1] for j in nb) / len(nb)
        if not point_in_poly(nx, ny, poly):
            continue
        old = points[i]
        before = [_signed_area2(points, tris[t]) for t in inc[i]]
        points[i] = (nx, ny)
        after = [_signed_area2(points, tris[t]) for t in inc[i]]
        # Revert if any incident triangle changed orientation or (near-)collapsed.
        if any((b > 0) != (a > 0) or abs(a) < 1e-12
               for b, a in zip(before, after)):
            points[i] = old


def _third_vertex(tri, u, v):
    for w in tri:
        if w != u and w != v:
            return w
    return None


def _flip_pass(points, tris) -> bool:
    """One edge-flip pass: for every interior edge shared by two triangles that
    form a convex quad, flip the diagonal when it raises the pair's minimum
    angle. Boundary edges border a single triangle and are never touched.
    Returns True if any edge was flipped."""
    edge_map: dict = {}
    for ti, tri in enumerate(tris):
        a, b, c = tri
        for u, v in ((a, b), (b, c), (c, a)):
            edge_map.setdefault(frozenset((u, v)), []).append(ti)
    flipped: set = set()
    for edge, ts in edge_map.items():
        if len(ts) != 2:
            continue
        t1, t2 = ts
        if t1 in flipped or t2 in flipped:
            continue
        u, v = tuple(edge)
        w1 = _third_vertex(tris[t1], u, v)
        w2 = _third_vertex(tris[t2], u, v)
        if w1 is None or w2 is None or w1 == w2:
            continue
        quad = [points[u], points[w1], points[v], points[w2]]
        if not _is_convex_quad(quad):
            continue
        before = min(_tri_min_angle(points, u, v, w1),
                     _tri_min_angle(points, u, v, w2))
        after = min(_tri_min_angle(points, w1, w2, u),
                    _tri_min_angle(points, w1, w2, v))
        if after > before + 1e-6:
            tris[t1] = (w1, w2, u)
            tris[t2] = (w1, w2, v)
            flipped.add(t1); flipped.add(t2)
    return bool(flipped)


def _matched_indices(points, targets, tol):
    """Indices of *points* that coincide (within *tol*) with any of *targets*
    — used to keep a forced point or a quad-cell corner out of
    :func:`_laplacian_smooth`'s movable set, on top of the polygon boundary
    (see :func:`_improve_mesh`)."""
    if not targets:
        return set()
    tol2 = tol * tol
    out = set()
    for idx, (x, y) in enumerate(points):
        for tx, ty in targets:
            if (x - tx) ** 2 + (y - ty) ** 2 <= tol2:
                out.add(idx)
                break
    return out


def _improve_mesh(points, tris, poly, iters: int = 3, extra_fixed=()):
    """Laplacian-smooth the interior nodes and flip edges to raise the minimum
    angle, alternating for a few passes. Point count and boundary are
    unchanged, so callers/conformity are unaffected.

    *extra_fixed* are (x, y) points that must also stay put beyond the
    polygon boundary itself — a *forced* point (typically a neighbouring
    object's node the mesh was built to conform to) or the corner of a quad
    cell kept alongside these triangles (:func:`mesh_quad_structured_cells`'s
    per-cell fallback): moving either would break that conformity, or pull a
    triangle's shared edge out from under an exact-shaped quad it must still
    match node-for-node."""
    pts = [tuple(p) for p in points]
    tr = [tuple(t) for t in tris]
    if not tr:
        return pts, tr
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    extent = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    tol = 1e-6 * extent
    fixed = _boundary_mask(pts, poly, tol)
    for idx in _matched_indices(pts, extra_fixed, tol):
        fixed[idx] = True
    movable = [i for i in range(len(pts)) if not fixed[i]]
    for _ in range(max(1, iters)):
        _laplacian_smooth(pts, tr, movable, poly)
        _flip_pass(pts, tr)
    return pts, tr


def mesh_polygon(poly: list[Point], target: float,
                 forced: list[Point] = ()):
    """Mesh the simple polygon *poly* (ordered vertices) at element size
    *target*. *forced* are pre-existing points to conform to: those on the
    boundary split its edges; those inside become interior mesh nodes.

    Two branches (see :func:`mesh_quad_structured` for why):

    * a **structured** (transfinite) mesh when the polygon is a convex
      quadrilateral and no forced point constrains the interior — the common
      case of a rectangular wall or slab. Symmetric by construction, and with
      better-shaped elements than a Delaunay triangulation of the same points.
    * an **unstructured** Delaunay mesh over a refined boundary plus an
      interior grid, for every other outline.

    Returns ``(points, triangles)`` — ``points`` a list of (x, y) and
    ``triangles`` a list of (i, j, k) index triples, all inside the polygon."""
    if len(poly) < 3 or target <= 0:
        return [], []
    forced = list(forced)
    structured = mesh_quad_structured(poly, target, forced)
    if structured is not None and structured[1]:
        return structured
    boundary = refine_boundary(poly, target, forced)
    interior = _grid_interior(poly, target, boundary)
    # Forced interior points (strictly inside, not already on the boundary).
    for fx, fy in forced:
        if not point_in_poly(fx, fy, poly):
            continue
        on_b = any((fx - bx) ** 2 + (fy - by) ** 2 < (1e-6) ** 2
                   for bx, by in boundary)
        if not on_b:
            interior.append((fx, fy))
    points, remap = _dedup(boundary + interior)
    tris = delaunay(points)
    kept = []
    for (a, b, c) in tris:
        cx = (points[a][0] + points[b][0] + points[c][0]) / 3.0
        cy = (points[a][1] + points[b][1] + points[c][1]) / 3.0
        if point_in_poly(cx, cy, poly):
            kept.append((a, b, c))
    # Quality pass: Laplacian smoothing + min-angle edge flips (interior only;
    # the faceted boundary and the point count are preserved). Forced points
    # are kept fixed too, not just the polygon's own boundary -- a forced
    # point can legitimately be strictly interior to this polygon (e.g. a
    # neighbouring object's node touching it away from the edge), and moving
    # it during smoothing would silently break that conformity.
    points, kept = _improve_mesh(points, kept, poly, extra_fixed=forced)
    return points, kept
