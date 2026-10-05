"""Section cuts — geometry kernel (Phase 2), bar resultants (Phase 3), area
resultants (Phase 4) and the combined public API (Phase 5).

Pure, Qt-free geometry: segment/segment intersection (bar crossings) and
segment/triangle clipping (area crossings). See ``dev/CUT_PLAN.md`` for the
overall feature plan and conventions (a cut is a segment
``(x1,y1) -> (x2,y2)``; ``s`` is always the parameter along the *cut*,
``0`` at its start and ``1`` at its end).

Tolerance policy (decided once here, reused by every caller): all
comparisons against the triangle edges / segment ends allow a slack of
``tol`` (default ``1e-9``, in model length units). A cut that grazes a node
or edge within that tolerance is treated as touching it, not crossing it —
grazing contact contributes no length to the area integral (Phase 4) and is
still reported as a point contact for bars (a hit at ``t`` or ``u`` equal to
0 or 1 is a legitimate end-to-end crossing, not an error).

Two things are deliberately *not* resolved here, by design:

* **Collinear overlap** (cut running along a bar, or along a triangle edge
  for more than a point) has no unique crossing point/length attributable to
  "this side vs that side" and is reported as no crossing. A user drawing a
  cut coincident with a member gets an empty result, not a guess.
* **Tangential triangle contact** (the cut only touches a single vertex or
  edge point of a triangle, i.e. the clipped sub-segment has zero length)
  contributes nothing to the line integral and is dropped by the
  triangle-crossing finder.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

Point = tuple[float, float]

__all__ = [
    "SegmentHit",
    "segment_intersection",
    "clip_segment_to_triangle",
    "find_bar_crossings",
    "find_triangle_crossings",
    "cut_bar_resultant",
    "cut_area_resultant",
    "cut_result",
    "cut_distribution",
    "cuts_report",
]


@dataclass
class SegmentHit:
    """A single-point crossing of two segments A (``a1->a2``) and B
    (``b1->b2``)."""
    t: float   # parameter on A, 0..1
    u: float   # parameter on B, 0..1
    x: float
    y: float


def _sub(p: Point, q: Point) -> Point:
    return (p[0] - q[0], p[1] - q[1])


def _cross2(o: Point, a: Point, b: Point) -> float:
    """Z component of (a-o) x (b-o)."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def segment_intersection(a1: Point, a2: Point, b1: Point, b2: Point,
                          tol: float = 1e-9) -> Optional[SegmentHit]:
    """Single-point intersection of segment ``a1->a2`` with ``b1->b2``.

    Returns ``None`` when the segments are parallel (including the collinear
    overlap case — see module docstring) or when the intersection of the two
    *infinite* lines falls outside either segment by more than ``tol``.
    Otherwise returns the crossing with ``t``/``u`` clamped into ``[0, 1]``.
    """
    dx_a, dy_a = a2[0] - a1[0], a2[1] - a1[1]
    dx_b, dy_b = b2[0] - b1[0], b2[1] - b1[1]
    denom = dx_a * dy_b - dy_a * dx_b
    if abs(denom) <= tol:
        return None  # parallel (or one/both segments degenerate)

    dx_ab, dy_ab = b1[0] - a1[0], b1[1] - a1[1]
    t = (dx_ab * dy_b - dy_ab * dx_b) / denom
    u = (dx_ab * dy_a - dy_ab * dx_a) / denom

    # Allow the crossing to sit within tol of either segment's own length,
    # in either parametric or absolute terms (tol is a length tolerance;
    # scale it to a fraction of each segment for the parametric test).
    len_a = math.hypot(dx_a, dy_a)
    len_b = math.hypot(dx_b, dy_b)
    tol_t = tol / len_a if len_a > tol else tol
    tol_u = tol / len_b if len_b > tol else tol
    if t < -tol_t or t > 1 + tol_t or u < -tol_u or u > 1 + tol_u:
        return None

    t = min(1.0, max(0.0, t))
    u = min(1.0, max(0.0, u))
    x = a1[0] + t * dx_a
    y = a1[1] + t * dy_a
    return SegmentHit(t=t, u=u, x=x, y=y)


def _ccw_triangle(tri: list[Point]) -> list[Point]:
    area2 = _cross2(tri[0], tri[1], tri[2])
    return tri if area2 >= 0 else [tri[0], tri[2], tri[1]]


def clip_segment_to_triangle(p1: Point, p2: Point, tri: list[Point],
                              tol: float = 1e-9) -> Optional[tuple[float, float]]:
    """Clip segment ``p1->p2`` to the (convex) triangle ``tri``.

    Cyrus-Beck clipping against the triangle's three edge half-planes.
    Returns ``(s_enter, s_exit)`` — the parametric sub-range of
    ``p1 + s*(p2-p1)``, ``s in [0, 1]``, that lies inside or on the
    boundary of the triangle — or ``None`` if the segment does not meet the
    triangle at all (within ``tol``).

    A grazing contact (single point: ``s_enter == s_exit`` within ``tol``)
    is still returned; callers that only care about area contributions
    (Phase 4) should drop zero-length results themselves — see
    :func:`find_triangle_crossings`.
    """
    tri = _ccw_triangle(list(tri))
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    seg_len = math.hypot(dx, dy)
    if seg_len <= tol:
        return None  # degenerate segment

    lo, hi = 0.0, 1.0
    for i in range(3):
        A = tri[i]
        B = tri[(i + 1) % 3]
        f0 = _cross2(A, B, p1)
        f1 = _cross2(A, B, p2)
        df = f1 - f0
        if abs(df) <= tol:
            # Segment parallel to this edge: either entirely admissible
            # (inside/on the half-plane) or entirely rejected.
            if f0 < -tol:
                return None
            continue
        s0 = -f0 / df
        if df > 0:
            lo = max(lo, s0)
        else:
            hi = min(hi, s0)
        if lo - hi > tol / seg_len:
            return None

    if lo > hi:
        return None
    return (max(0.0, lo), min(1.0, hi))


# ---------------------------------------------------------------------
# Structure-level finders (compiled model + Cut -> crossings)
# ---------------------------------------------------------------------

def _cut_endpoints(cut) -> tuple[Point, Point]:
    p1 = (cut.x1, cut.y1)
    p2 = (cut.x2, cut.y2)
    if math.hypot(p2[0] - p1[0], p2[1] - p1[1]) <= 0.0:
        raise ValueError(f"Cut '{cut.id}' has zero length.")
    return p1, p2


def find_bar_crossings(struc, cut, tol: float = 1e-9) -> list[dict]:
    """Bar elements of ``struc`` (a compiled model, see ``geo_expand``) whose
    segment crosses ``cut``. Each entry: ``{elem_id, a, s, x, y}`` where
    ``a`` is the station along the bar (length units, from ``node_i``) and
    ``s`` is the parameter along the cut (``0..1``)."""
    p1, p2 = _cut_endpoints(cut)
    out = []
    for e in struc.bar_elements:
        ni = struc.nodes[e.node_i]
        nj = struc.nodes[e.node_j]
        b1, b2 = (ni.x, ni.y), (nj.x, nj.y)
        hit = segment_intersection(p1, p2, b1, b2, tol=tol)
        if hit is None:
            continue
        bar_len = math.hypot(b2[0] - b1[0], b2[1] - b1[1])
        out.append({
            "elem_id": e.id,
            "a": hit.u * bar_len,
            "s": hit.t,
            "x": hit.x,
            "y": hit.y,
        })
    return out


def find_triangle_crossings(struc, cut, tol: float = 1e-9) -> list[dict]:
    """Triangles of ``struc`` (a compiled model) whose interior the cut
    passes through. Each entry: ``{tri_id, s0, s1, p0, p1}``, ``s0 < s1``
    being the cut parameters (``0..1``) bounding the clipped sub-segment
    inside that triangle. Grazing contacts (zero-length clip) are dropped."""
    p1, p2 = _cut_endpoints(cut)
    seg_len = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    out = []
    for e in struc.tri_elements:
        ni = struc.nodes[e.node_i]
        nj = struc.nodes[e.node_j]
        nk = struc.nodes[e.node_k]
        tri = [(ni.x, ni.y), (nj.x, nj.y), (nk.x, nk.y)]
        clip = clip_segment_to_triangle(p1, p2, tri, tol=tol)
        if clip is None:
            continue
        s0, s1 = clip
        if (s1 - s0) * seg_len <= tol:
            continue  # tangential contact only, no length to integrate
        p0 = (p1[0] + s0 * (p2[0] - p1[0]), p1[1] + s0 * (p2[1] - p1[1]))
        p1_ = (p1[0] + s1 * (p2[0] - p1[0]), p1[1] + s1 * (p2[1] - p1[1]))
        out.append({
            "tri_id": e.id,
            "s0": s0,
            "s1": s1,
            "p0": p0,
            "p1": p1_,
        })
    return out


def find_quad_crossings(struc, cut, tol: float = 1e-9) -> list[dict]:
    """Quad elements of ``struc`` whose interior the cut passes through — the
    quad analogue of :func:`find_triangle_crossings`
    (dev/IMPLEMENT_QUAD.md Phase 7).

    ``clip_segment_to_triangle`` only handles a (convex) triangle, so each
    quad is split, for this geometric clipping test *only*, into its two
    ``node_i-node_j-node_k`` / ``node_i-node_k-node_l`` triangles along the
    fixed ``i-k`` diagonal, and each half is clipped independently. This is
    deliberately not the same split ``quad_elements_dkt4`` uses for
    stiffness (a 4-triangle centroid fan) — that choice matters there because
    it feeds a stiffness matrix; here the quad's reported stress/moment is a
    single element-constant value (like a triangle's), so which diagonal
    splits the quad for a purely geometric length integral is arbitrary and
    does not affect the result. The diagonal is still fixed (always ``i-k``,
    never chosen per-cut) so the split is deterministic and reproducible
    across calls, not incidental.

    A cut that crosses the internal ``i-k`` diagonal produces two entries for
    the same quad — one per half — whose clipped lengths sum to the correct
    total length inside the quad when consumed by :func:`cut_area_resultant`
    (which sums contributions by element id). Each entry has the same shape
    as :func:`find_triangle_crossings`'s, keyed ``'tri_id'`` (holding the
    quad's own id — kept unrenamed for the same reason
    ``rc_design.design_concrete_planes``'s row ``'triangle'`` key is: it
    is what :func:`cut_area_resultant` already reads, and this codebase's
    convention is additive, never rename-with-migration)."""
    p1, p2 = _cut_endpoints(cut)
    seg_len = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    out = []
    for e in getattr(struc, 'quad_elements', []):
        ni = struc.nodes[e.node_i]
        nj = struc.nodes[e.node_j]
        nk = struc.nodes[e.node_k]
        nl = struc.nodes[e.node_l]
        for tri in (((ni.x, ni.y), (nj.x, nj.y), (nk.x, nk.y)),
                    ((ni.x, ni.y), (nk.x, nk.y), (nl.x, nl.y))):
            clip = clip_segment_to_triangle(p1, p2, list(tri), tol=tol)
            if clip is None:
                continue
            s0, s1 = clip
            if (s1 - s0) * seg_len <= tol:
                continue  # tangential contact only, no length to integrate
            p0 = (p1[0] + s0 * (p2[0] - p1[0]), p1[1] + s0 * (p2[1] - p1[1]))
            p1_ = (p1[0] + s1 * (p2[0] - p1[0]), p1[1] + s1 * (p2[1] - p1[1]))
            out.append({
                "tri_id": e.id,
                "s0": s0,
                "s1": s1,
                "p0": p0,
                "p1": p1_,
            })
    return out


# ---------------------------------------------------------------------
# Bar resultants (Phase 3)
# ---------------------------------------------------------------------
#
# Sign convention. The stored diagrams (``element_distribution``, built by
# ``solver.compute_distributions`` from the i-end and reused unchanged by
# combinations) give, at a station, the internal force the segment *beyond*
# that station (the node_j side) exerts on the segment *before* it (the
# node_i side) — plain statics sign convention, the same the rest of the
# engine already uses. A cut instead wants "what the positive side (the side
# its normal ``n`` points to) exerts on the negative side" (dev/CUT_PLAN.md
# §1). Those coincide only when a bar's own node_i -> node_j direction points
# into the cut's positive side; otherwise the stored value is exerted in the
# opposite sense and is negated before use. Because a bar that actually
# crosses the cut is never parallel to it (segment_intersection excludes
# that case), the bar's tangent always has a non-zero component along the
# cut's normal, so this sign is always well defined.
#
# Rotation. Force components rotate from bar-local to cut axes like any 2-D
# vector: local -> global with the bar's own (cos, sin), global -> cut axes
# by projecting onto the cut's unit tangent/normal. A plate/grillage bar's
# (N, V, M) slots hold (T, V, M) instead (`` _grid_member_end_forces``: the
# torque reuses the axial slot) and V is the transverse (out-of-plane) shear,
# frame-independent; what rotates the same way as a force vector is the
# in-plane *moment* vector (T along the bar, M across it) — see
# dev/CUT_PLAN.md §1 for why plate cuts report (V, Mb, Mt) instead of
# (N, V, M).
#
# Reference point. All contributions are moment-transferred to a common
# point (the cut midpoint by default) before summing, exactly like moving
# the point a statics moment is taken about: M_ref = M_c + (r_c - r_ref) x F.
# For a plate cut, transferring the out-of-plane shear V across an in-plane
# offset likewise adds an in-plane moment contribution.
#
# A note for anyone tempted to "test" this by reversing a cut's drawing
# direction: N and V are *projections* onto (t, n), and reversing the cut
# flips both the underlying force's sign (via the mechanism above) and the
# (t, n) basis it is measured against — the two flips cancel, so N and V
# come out identical. M does not go through that projection (it is an
# absolute, global-frame moment about the reference point) and does flip.
# Both are correct; see test_cuts_bar_resultants.py for the worked-out case.

def _cut_axes(cut) -> tuple[Point, Point, float]:
    """``(t_hat, n_hat, length)`` for a cut: unit tangent, unit normal
    (``t`` rotated +90°), and the cut's own length."""
    dx, dy = cut.x2 - cut.x1, cut.y2 - cut.y1
    length = math.hypot(dx, dy)
    if length <= 0.0:
        raise ValueError(f"Cut '{cut.id}' has zero length.")
    t_hat = (dx / length, dy / length)
    n_hat = (-t_hat[1], t_hat[0])
    return t_hat, n_hat, length


def _cut_midpoint(cut) -> Point:
    return ((cut.x1 + cut.x2) / 2.0, (cut.y1 + cut.y2) / 2.0)


def _element_distribution_for(struc, results, case):
    """``(element_distribution, reason)`` for a case id — a load/analysis
    case (read straight from ``results``) or a ``LinearSum`` combination
    (rebuilt on demand via :func:`postprocess.combined_distribution`)."""
    from .postprocess import combined_distribution
    ac = (results or {}).get('analysis_cases', {}).get(case)
    if ac and ac.get('element_distribution'):
        return ac['element_distribution'], ''
    dist, _factors, reason = combined_distribution(struc, results, case)
    if dist:
        return dist, ''
    return None, reason or f"case '{case}' not found"


def _linear_interp(a: float, xs: list, ys: list) -> float:
    """Plain linear interpolation of ``ys`` sampled at ``xs`` (assumed
    sorted, as the solver's per-element grid always is), clamped past the
    ends."""
    n = len(xs)
    if n == 0:
        return 0.0
    if a <= xs[0]:
        return float(ys[0])
    if a >= xs[-1]:
        return float(ys[-1])
    for i in range(1, n):
        if xs[i] >= a:
            x0, x1 = xs[i - 1], xs[i]
            y0, y1 = ys[i - 1], ys[i]
            if x1 == x0:
                return float(y1)
            f = (a - x0) / (x1 - x0)
            return float(y0 + f * (y1 - y0))
    return float(ys[-1])


def _interp_local(rec: dict, a: float) -> Optional[dict]:
    xs = rec.get('x')
    if xs is None:
        return None
    xs = list(xs)
    out = {}
    for k in ('N', 'V', 'M'):
        col = rec.get(k)
        if col is None:
            continue
        out[k] = _linear_interp(a, xs, list(col))
    return out


def _project(vx: float, vy: float, t_hat: Point, n_hat: Point) -> tuple[float, float]:
    """``(t-component, n-component)`` of global vector ``(vx, vy)``."""
    return (vx * t_hat[0] + vy * t_hat[1], vx * n_hat[0] + vy * n_hat[1])


def _zero_resultant(domain: str) -> dict:
    if domain == 'plate':
        return {'V': 0.0, 'Mb': 0.0, 'Mt': 0.0}
    return {'N': 0.0, 'V': 0.0, 'M': 0.0}


def _dedupe_collinear_bar_hits(struc, hits: list[dict],
                                point_tol: float = 1e-6) -> list[dict]:
    """Collapse bar crossings that land at the same point in *collinear*
    directions to a single representative.

    A cut passing exactly through a node shared by two collinear bars (the
    common case: the node falls mid-span in the mesh, e.g. two elements of
    the same physical member) is a *legitimate* crossing for each bar
    individually (see the module tolerance policy above — a hit at a bar's
    own endpoint is not an error). But it is one physical cross-section, not
    two: both bars report (numerically) the same internal force at that
    station, and summing both would double the resultant.

    Bars meeting at that point from genuinely different directions (an
    angled joint the cut happens to pass through) are kept separately —
    that is a real multi-member cut, not a duplicate.
    """
    bars_by_id = getattr(struc, 'bar_elements_by_id', None) or {
        e.id: e for e in struc.bar_elements
    }
    kept: list[dict] = []
    kept_dirs: list[tuple[float, float, float, float]] = []
    for hit in hits:
        elem = bars_by_id.get(hit['elem_id'])
        if elem is None:
            kept.append(hit)
            continue
        ni = struc.nodes[elem.node_i]
        nj = struc.nodes[elem.node_j]
        dx, dy = nj.x - ni.x, nj.y - ni.y
        length = math.hypot(dx, dy)
        if length <= point_tol:
            kept.append(hit)
            continue
        ux, uy = dx / length, dy / length
        duplicate = False
        for kx, ky, kux, kuy in kept_dirs:
            if (abs(hit['x'] - kx) <= point_tol
                    and abs(hit['y'] - ky) <= point_tol):
                # Parallel or anti-parallel direction -> same physical
                # section, already counted.
                cross = ux * kuy - uy * kux
                if abs(cross) <= 1e-6:
                    duplicate = True
                    break
        if duplicate:
            continue
        kept.append(hit)
        kept_dirs.append((hit['x'], hit['y'], ux, uy))
    return kept


def cut_bar_resultant(struc, results, cut, case, ref_point: Optional[Point] = None,
                       tol: float = 1e-9) -> dict:
    """Resultant of the bar internal forces crossing ``cut`` for ``case`` (a
    load case, analysis case, or ``LinearSum`` combination id).

    Returns ``{'bars': [...], 'resultant': {...}, 'reason': str}``.
    ``reason`` is non-empty (and ``bars``/``resultant`` empty/zero) when the
    diagram for ``case`` could not be built (see
    :func:`postprocess.combined_distribution`).

    Each entry of ``bars`` is ``{elem_id, a, s, x, y, forces_local,
    forces_cut}``: ``a`` is the station along the bar, ``s`` the parameter
    along the cut, ``forces_local`` the bar's own diagram triple at that
    station (keys ``N, V, M`` — ``N`` holds torque ``T`` for a plate bar, see
    the module notes above), and ``forces_cut`` that contribution rotated
    into the cut's axes and moment-transferred to ``ref_point`` (default:
    the cut midpoint) — keys ``N, V, M`` for a plane cut, ``V, Mb, Mt`` for a
    plate cut. ``resultant`` is the sum of ``forces_cut`` over every bar
    crossing.
    """
    domain = getattr(struc, 'domain', 'plane')
    t_hat, n_hat, _cut_len = _cut_axes(cut)
    ref = ref_point if ref_point is not None else _cut_midpoint(cut)

    dist, reason = _element_distribution_for(struc, results, case)
    if dist is None:
        return {'bars': [], 'resultant': _zero_resultant(domain), 'reason': reason}

    bars_by_id = getattr(struc, 'bar_elements_by_id', None) or {
        e.id: e for e in struc.bar_elements
    }

    bars_out = []
    totals = _zero_resultant(domain)
    hits = _dedupe_collinear_bar_hits(struc, find_bar_crossings(struc, cut, tol=tol))
    for hit in hits:
        elem_id = hit['elem_id']
        rec = dist.get(elem_id)
        if rec is None:
            continue
        loc = _interp_local(rec, hit['a'])
        if loc is None:
            continue
        elem = bars_by_id.get(elem_id)
        if elem is None:
            continue
        ni = struc.nodes[elem.node_i]
        nj = struc.nodes[elem.node_j]
        dxb, dyb = nj.x - ni.x, nj.y - ni.y
        bar_len = math.hypot(dxb, dyb)
        if bar_len <= tol:
            continue
        cos_b, sin_b = dxb / bar_len, dyb / bar_len

        # node_i -> node_j points into the cut's positive (n) side?
        sign = 1.0 if (cos_b * n_hat[0] + sin_b * n_hat[1]) >= 0.0 else -1.0

        x_c, y_c = hit['x'], hit['y']
        rx, ry = x_c - ref[0], y_c - ref[1]

        if domain == 'plate':
            T = loc.get('N', 0.0) * sign
            V = loc.get('V', 0.0) * sign
            M = loc.get('M', 0.0) * sign
            # In-plane moment vector (T along the bar, M across it) -> global.
            mx_g = T * cos_b - M * sin_b
            my_g = T * sin_b + M * cos_b
            # Transferring the out-of-plane shear V to the reference point
            # adds an in-plane moment: (r_c - r_ref) x (0, 0, V).
            mx_g += ry * V
            my_g += -rx * V
            # Mb = bending moment vector along the cut's tangent (t);
            # Mt = twisting moment vector along the cut's normal (n) — the
            # same (mn, mnt) split used for the area contribution (Phase 4).
            mb_comp, mt_comp = _project(mx_g, my_g, t_hat, n_hat)
            forces_cut = {'V': V, 'Mb': mb_comp, 'Mt': mt_comp}
        else:
            N = loc.get('N', 0.0) * sign
            V = loc.get('V', 0.0) * sign
            M = loc.get('M', 0.0) * sign
            fx_g = N * cos_b - V * sin_b
            fy_g = N * sin_b + V * cos_b
            v_comp, n_comp = _project(fx_g, fy_g, t_hat, n_hat)
            d_m = rx * fy_g - ry * fx_g
            forces_cut = {'N': n_comp, 'V': v_comp, 'M': M + d_m}

        for k, v in forces_cut.items():
            totals[k] = totals.get(k, 0.0) + v

        bars_out.append({
            'elem_id': elem_id, 'a': hit['a'], 's': hit['s'],
            'x': x_c, 'y': y_c,
            'forces_local': loc, 'forces_cut': forces_cut,
        })

    return {'bars': bars_out, 'resultant': totals, 'reason': ''}


# ---------------------------------------------------------------------
# Area resultants (Phase 4)
# ---------------------------------------------------------------------
#
# tri_stress fields (from ``tri_elements.tri_stresses_dispatch`` /
# ``tri_stresses_from_disp_dispatch``) are element-constant — one value per
# triangle, whichever formulation computed it (CST/Allman/ES-FEM stresses
# ``sx, sy, txy``; DKT/MITC3 moments ``mx, my, mxy`` and shears ``vx, vy`` —
# always present via ``plate_common.plate_moment_result``, so the "shear
# fields might be missing" risk flagged in dev/CUT_PLAN.md does not apply).
# Being constant per triangle, the line integral over the sub-segment
# ``find_triangle_crossings`` clips inside each triangle reduces to (value
# x sub-segment length) for the direct (stress/moment) part, plus an
# analytic integral of the linearly-varying position for the moment-transfer
# part (the position along the sub-segment is not constant even though the
# field is, so that half needs the actual integral, not just a midpoint
# times length — done exactly below, closed form, no quadrature).
#
# Tensor rotation. sx, sy, txy (or mx, my, mxy) are the standard 2x2
# symmetric tensor in global (x, y); given the cut's unit normal (nx, ny),
# the traction on the n-face is S.n, and its components along n and the
# perpendicular direction (t = (ny, -nx), since n = t rotated +90°) are the
# usual Mohr-circle formulas:
#   sigma_nn = sx*nx^2 + sy*ny^2 + 2*txy*nx*ny
#   tau_nt   = (sx - sy)*nx*ny + txy*(ny^2 - nx^2)
# For a plane cut these give the membrane force per length once multiplied
# by the section thickness (n_n = sigma_nn * t, n_t = tau_nt * t). For a
# plate cut the same rotation of (mx, my, mxy) gives the bending/twisting
# moment per length directly (mn, mnt — already "per unit length", no
# thickness factor, matching the bar Mb/Mt split in the section above); the
# transverse shear per length is just v_n = vx*nx + vy*ny (a vector
# component, not a tensor quantity).
#
# Moment transfer. Unlike a bar's single-point crossing, an area
# contribution is a force (plane: n_n, n_t) or a couple + shear (plate: mn,
# mnt, v_n) *distributed* along the sub-segment. A distributed couple (the
# plate's mn/mnt) needs no transfer — a pure moment is reference-point
# independent (only forces are). A distributed force does: each point along
# the sub-segment contributes (position - ref) x force to the moment about
# ref, and because the field is constant while the position varies linearly
# along the sub-segment, that integral has a closed form —
# ``_segment_position_integrals`` below.

def _segment_position_integrals(p1: Point, p2: Point, s0: float, s1: float,
                                 ref: Point) -> tuple[float, float, float]:
    """For the cut ``p1 -> p2`` and its sub-parameter range ``[s0, s1]``
    (``0..1`` over the *whole* cut): the sub-segment's physical ``length``,
    and ``Ix = integral (x(s) - ref_x) d(length)``, ``Iy`` likewise for y —
    exact, since position is linear in the physical arc length."""
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    cut_len = math.hypot(dx, dy)
    ds = s1 - s0
    length = ds * cut_len
    s_sq = (s1 * s1 - s0 * s0) / 2.0
    ix = cut_len * ((p1[0] - ref[0]) * ds + dx * s_sq)
    iy = cut_len * ((p1[1] - ref[1]) * ds + dy * s_sq)
    return length, ix, iy


def _tri_stress_lookup(results: Optional[dict], case: str) -> Optional[dict]:
    """``{tri_id: {...}}`` for ``case`` if already stored in ``results``,
    trying every location the engine/GUI use (see canvas.py's
    ``_tri_stress_dict``): a solved analysis case, a solved combination, or
    the flat per-load-case dict. ``None`` if nowhere."""
    rd = results or {}
    ts = (rd.get('analysis_cases', {}).get(case) or {}).get('tri_stress')
    if ts:
        return ts
    ts = (rd.get('combinations', {}).get(case) or {}).get('tri_stress')
    if ts:
        return ts
    ts = (rd.get('tri_stress', {}) or {}).get(case)
    if ts:
        return ts
    return None


_PLANE_TRI_COMPONENTS = ('sx', 'sy', 'txy')
_PLATE_TRI_COMPONENTS = ('mx', 'my', 'mxy', 'vx', 'vy')


def _combined_tri_stress(struc, results, case: str):
    """Rebuild a ``LinearSum`` combination's tri_stress by summing the
    linear (tensor) components of the load/analysis cases it references —
    the same on-demand approach as :func:`postprocess.combined_distribution`
    for bar diagrams, and with the same limitation: only a flat, already
    ``LinearSum`` combination (no nested-combination expansion, no
    envelopes/SRSS — those have no single linear tri_stress to derive)."""
    combo = next((c for c in (getattr(struc, 'load_combinations', []) or [])
                  if c.id == case), None)
    if combo is None:
        return None, 'not a load combination'
    kind = getattr(combo, 'combo_type', 'LinearSum')
    if kind != 'LinearSum':
        return None, (f"combination type '{kind}' is not a linear sum, so "
                      "its tri stresses cannot be derived from the load cases")
    coef = dict(getattr(combo, 'coefficients', {}) or {})
    parts = []
    for cid, factor in coef.items():
        ts = _tri_stress_lookup(results, cid)
        if ts is None:
            return None, f"case '{cid}' has no stored tri stresses"
        parts.append((float(factor), ts))
    if not parts:
        return None, 'the combination has no coefficients'

    domain = getattr(struc, 'domain', 'plane')
    comp_keys = _PLATE_TRI_COMPONENTS if domain == 'plate' else _PLANE_TRI_COMPONENTS
    out: dict[str, dict] = {}
    for tid in next(iter(parts))[1]:
        acc = {}
        ok = True
        for factor, ts in parts:
            rec = ts.get(tid)
            if rec is None:
                ok = False
                break
            for k in comp_keys:
                acc[k] = acc.get(k, 0.0) + factor * rec.get(k, 0.0)
        if ok:
            out[tid] = acc
    if not out:
        return None, 'the load cases have no triangles in common'
    return out, ''


def _tri_stress_for(struc, results, case: str) -> tuple[Optional[dict], str]:
    ts = _tri_stress_lookup(results, case)
    if ts is not None:
        return ts, ''
    ts, reason = _combined_tri_stress(struc, results, case)
    if ts is not None:
        return ts, ''
    return None, reason or f"case '{case}' has no stored tri stresses"


def _tri_thickness(struc, elem) -> float:
    # A quad element (dev/IMPLEMENT_QUAD.md Phase 7) has a node_l a triangle
    # never does — cheaper than tracking which id-space *elem* came from.
    if hasattr(elem, 'node_l'):
        sec = struc.quad_sections.get(elem.section_name)
    else:
        sec = struc.tri_sections.get(elem.section_name)
    return float(getattr(sec, 'thickness', 0.1)) if sec is not None else 0.1


def cut_area_resultant(struc, results, cut, case, ref_point: Optional[Point] = None,
                        tol: float = 1e-9) -> dict:
    """Resultant of the area (triangle AND quad — dev/IMPLEMENT_QUAD.md
    Phase 7) internal forces crossing ``cut`` for ``case`` (a load case,
    analysis case, or ``LinearSum`` combination id).

    Returns ``{'areas': [...], 'resultant': {...}, 'reason': str}`` — same
    shape as :func:`cut_bar_resultant`, so Phase 5 can sum the two directly.
    Each entry of ``areas`` is ``{tri_id, s0, s1, length, p0, p1, local,
    contribution}`` (``tri_id`` holds a quad's own id for a quad entry, see
    :func:`find_quad_crossings`): ``local`` is the element's own stress/
    moment record plus the rotated per-length quantities (``n_n, n_t`` for a
    plane cut, ``m_n, m_nt, v_n`` for a plate cut); ``contribution`` is that
    element's share of the resultant (already integrated over its
    sub-segment and moment-transferred to ``ref_point``, default the cut
    midpoint) — keys ``N, V, M`` (plane) or ``V, Mb, Mt`` (plate), matching
    the bar side.
    """
    domain = getattr(struc, 'domain', 'plane')
    t_hat, n_hat, _cut_len = _cut_axes(cut)
    ref = ref_point if ref_point is not None else _cut_midpoint(cut)
    p1, p2 = (cut.x1, cut.y1), (cut.x2, cut.y2)

    ts, reason = _tri_stress_for(struc, results, case)
    if ts is None:
        return {'areas': [], 'resultant': _zero_resultant(domain), 'reason': reason}

    tris_by_id = dict(getattr(struc, 'tri_elements_by_id', None) or {
        e.id: e for e in struc.tri_elements
    })
    tris_by_id.update(getattr(struc, 'quad_elements_by_id', None) or {
        e.id: e for e in getattr(struc, 'quad_elements', [])
    })

    areas_out = []
    totals = _zero_resultant(domain)
    nx, ny = n_hat
    crossings = (find_triangle_crossings(struc, cut, tol=tol)
                 + find_quad_crossings(struc, cut, tol=tol))
    for hit in crossings:
        tid = hit['tri_id']
        rec = ts.get(tid)
        if rec is None:
            continue
        elem = tris_by_id.get(tid)
        if elem is None:
            continue
        length, ix, iy = _segment_position_integrals(p1, p2, hit['s0'], hit['s1'], ref)

        if domain == 'plate':
            mx = rec.get('mx', 0.0)
            my = rec.get('my', 0.0)
            mxy = rec.get('mxy', 0.0)
            vx = rec.get('vx', 0.0)
            vy = rec.get('vy', 0.0)
            m_n = mx * nx * nx + my * ny * ny + 2.0 * mxy * nx * ny
            m_nt = (mx - my) * nx * ny + mxy * (ny * ny - nx * nx)
            v_n = vx * nx + vy * ny
            v_i = v_n * length
            mb_direct = m_n * length
            mt_direct = m_nt * length
            # Shear-transfer term: constant v_n over a linearly-varying
            # position, integrated exactly via ix/iy (see module notes).
            mx_extra = iy * v_n
            my_extra = -ix * v_n
            mb_extra, mt_extra = _project(mx_extra, my_extra, t_hat, n_hat)
            contribution = {'V': v_i, 'Mb': mb_direct + mb_extra,
                            'Mt': mt_direct + mt_extra}
            local = {'mx': mx, 'my': my, 'mxy': mxy, 'vx': vx, 'vy': vy,
                     'm_n': m_n, 'm_nt': m_nt, 'v_n': v_n}
        else:
            sx = rec.get('sx', 0.0)
            sy = rec.get('sy', 0.0)
            txy = rec.get('txy', 0.0)
            thickness = _tri_thickness(struc, elem)
            sigma_nn = sx * nx * nx + sy * ny * ny + 2.0 * txy * nx * ny
            tau_nt = (sx - sy) * nx * ny + txy * (ny * ny - nx * nx)
            n_n = sigma_nn * thickness
            n_t = tau_nt * thickness
            n_i = n_n * length
            v_i = n_t * length
            fx_g = n_n * n_hat[0] + n_t * t_hat[0]
            fy_g = n_n * n_hat[1] + n_t * t_hat[1]
            m_i = ix * fy_g - iy * fx_g
            contribution = {'N': n_i, 'V': v_i, 'M': m_i}
            local = {'sx': sx, 'sy': sy, 'txy': txy,
                     'n_n': n_n, 'n_t': n_t, 'thickness': thickness}

        for k, v in contribution.items():
            totals[k] = totals.get(k, 0.0) + v

        areas_out.append({
            'tri_id': tid, 's0': hit['s0'], 's1': hit['s1'], 'length': length,
            'p0': hit['p0'], 'p1': hit['p1'],
            'local': local, 'contribution': contribution,
        })

    return {'areas': areas_out, 'resultant': totals, 'reason': ''}


# ---------------------------------------------------------------------
# Combined results + public API (Phase 5)
# ---------------------------------------------------------------------

def cut_result(struc, results, cut, case, ref_point: Optional[Point] = None,
                tol: float = 1e-9) -> dict:
    """The full result of a cut for one case: bars, areas and their total,
    ready for the GUI table (Phase 7) — Bars / Areas / Total rows.

    Returns::

        {'domain': 'plane' | 'plate',
         'bars':  [...],   # cut_bar_resultant's per-bar list
         'areas': [...],   # cut_area_resultant's per-triangle list
         'resultant': {'bars': {...}, 'areas': {...}, 'total': {...}},
         'reason': str}    # non-empty if either side could not be built

    A model with no bars (or no triangles/quads) contributes a clean zero on
    that side rather than an error — only an actually unsolved/missing case
    (or a non-``LinearSum`` combination) sets ``reason``.
    """
    domain = getattr(struc, 'domain', 'plane')
    has_bars = bool(getattr(struc, 'bar_elements', None))
    has_tris = bool(getattr(struc, 'tri_elements', None)
                    or getattr(struc, 'quad_elements', None))

    if has_bars:
        bar_r = cut_bar_resultant(struc, results, cut, case, ref_point=ref_point, tol=tol)
    else:
        bar_r = {'bars': [], 'resultant': _zero_resultant(domain), 'reason': ''}

    if has_tris:
        area_r = cut_area_resultant(struc, results, cut, case, ref_point=ref_point, tol=tol)
    else:
        area_r = {'areas': [], 'resultant': _zero_resultant(domain), 'reason': ''}

    total = _zero_resultant(domain)
    for k in total:
        total[k] = bar_r['resultant'].get(k, 0.0) + area_r['resultant'].get(k, 0.0)

    reasons = [r for r in (bar_r.get('reason', ''), area_r.get('reason', '')) if r]
    return {
        'domain': domain,
        'bars': bar_r['bars'],
        'areas': area_r['areas'],
        'resultant': {'bars': bar_r['resultant'], 'areas': area_r['resultant'],
                      'total': total},
        'reason': '; '.join(dict.fromkeys(reasons)),  # de-duplicated, order kept
    }


def cut_distribution(cutres: dict, n: int = 101) -> dict:
    """Sample ``cutres`` (a :func:`cut_result` return value) into arrays fit
    for plotting the per-length distribution along ``s in [0, 1]`` (Phase 7):
    the area fields, piecewise-constant per triangle (zero where the cut
    isn't inside any triangle), plus the bar crossings as point markers.

    Returns ``{'s': [...], 'fields': {key: [...], ...}, 'markers': [...]}``.
    ``fields`` has keys ``n_n, n_t`` for a plane cut or ``m_n, m_nt, v_n``
    for a plate cut (the same keys :func:`cut_area_resultant` puts in each
    area entry's ``local`` dict). Each marker is
    ``{s, elem_id, forces_local, forces_cut}``, one per bar crossing.
    """
    domain = cutres.get('domain', 'plane')
    keys = ('m_n', 'm_nt', 'v_n') if domain == 'plate' else ('n_n', 'n_t')
    n = max(2, int(n))
    ss = [i / (n - 1) for i in range(n)]
    areas = cutres.get('areas', [])

    fields = {k: [0.0] * n for k in keys}
    for idx, s in enumerate(ss):
        for a in areas:
            if a['s0'] - 1e-9 <= s <= a['s1'] + 1e-9:
                loc = a['local']
                for k in keys:
                    fields[k][idx] = loc.get(k, 0.0)
                break

    markers = [
        {'s': b['s'], 'elem_id': b['elem_id'],
         'forces_local': b['forces_local'], 'forces_cut': b['forces_cut']}
        for b in cutres.get('bars', [])
    ]
    return {'s': ss, 'fields': fields, 'markers': markers}


def cuts_report(struc, results, cases, ref_point: Optional[Point] = None,
                 tol: float = 1e-9) -> list:
    """One row per ``(cut, case)`` — every cut of ``struc`` against every
    case id in ``cases`` — with the total resultant, flattened for CSV/report
    export (Phase 7/8). ``cases`` may mix load case, analysis case and
    ``LinearSum`` combination ids.
    """
    rows = []
    for cut in getattr(struc, 'cuts', []):
        for case in cases:
            r = cut_result(struc, results, cut, case, ref_point=ref_point, tol=tol)
            row = {'cut_id': cut.id, 'cut_name': cut.name, 'case': case,
                   'reason': r['reason']}
            row.update({f'total_{k}': v for k, v in r['resultant']['total'].items()})
            rows.append(row)
    return rows
