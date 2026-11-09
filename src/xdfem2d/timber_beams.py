"""Timber beams: the bars sharing a ``beam`` tag, taken as one beam (dev/TIMBER_BEAMS.md).

:func:`timber_design.design_timber_members` used to group bars by
:func:`member_utils.identify_members` (collinear, same section). A pitched,
tapered or curved beam is not that -- its bars are not collinear and may change
section -- so the beam is the *tagged chain* instead. This module resolves, for
every tag, the ordered bar path and its **spans**:

* the *beam* (:class:`TimberBeam.member`) is the whole connected chain; it is
  the unit reported in the design table and report;
* a *span* is the stretch between two physical boundaries (a support, a hinge,
  a node where another bar joins, e.g. a column). Spans give the buckling
  length, exactly as a :class:`member_utils.Member` does for steel, but a change
  of section or direction does **not** end one.

Only bars with a timber section (``is_timber``) count. A tag whose bars do not
form one simple open path (a branch, a closed ring) is not resolved: its bars
are left to the per-member design and a warning is returned.

The beam's own design parameters live in ``Structure2D.beams[tag]["timber"]``;
at present ``lateral_spacing`` [m], the distance between lateral restraints
(replaces ``K_LT·L`` in the lateral-torsional check).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .member_utils import (Member, _bar_length, _hinge_at, _incident_bar_count,
                           _supported_nodes)


@dataclass
class TimberBeam:
    """One tagged timber beam: the whole chain and its spans."""

    id: str                                   # "V1", or "V1.2" if the tag is split
    tag: str
    name: str = ""
    member: Member = None                     # the whole chain
    spans: list = field(default_factory=list)  # list[Member], left → right
    params: dict = field(default_factory=dict)  # beams[tag]["timber"]

    @property
    def bar_ids(self) -> list:
        return self.member.bar_ids

    @property
    def lateral_spacing(self):
        """Distance between lateral restraints [m], or None (use K_LT·L)."""
        try:
            v = float(self.params.get("lateral_spacing") or 0.0)
        except (TypeError, ValueError):
            return None
        return v if v > 0.0 else None


def _components(adj: dict) -> list:
    """Connected components of *adj* ({node: [(bar_id, other_node)]}) as sets
    of node ids."""
    seen, out = set(), []
    for n0 in adj:
        if n0 in seen:
            continue
        comp, stack = set(), [n0]
        while stack:
            n = stack.pop()
            if n in comp:
                continue
            comp.add(n)
            stack.extend(o for _, o in adj[n])
        seen |= comp
        out.append(comp)
    return out


def _start_key(struc, node_id):
    """Left-to-right, then bottom-to-top (BEAM_BARS_DEFINITION.md §5.2)."""
    n = struc.nodes[node_id]
    return (round(n.x, 9), round(n.y, 9))


def _walk(struc, comp, adj):
    """The open path through *comp*, from its left end: ``(node_ids, bar_ids)``,
    or None when the component is not a simple open path."""
    ends = [n for n in comp if len(adj[n]) == 1]
    if len(ends) != 2 or any(len(adj[n]) > 2 for n in comp):
        return None
    start = min(ends, key=lambda n: _start_key(struc, n))
    nodes, bars = [start], []
    prev_bar = None
    cur = start
    while True:
        nxt = [(b, o) for b, o in adj[cur] if b != prev_bar]
        if not nxt:
            break
        b, o = nxt[0]
        bars.append(b)
        nodes.append(o)
        prev_bar, cur = b, o
    return (nodes, bars) if len(bars) == len(comp) - 1 else None


def _make_member(struc, mid, nodes, bars) -> Member:
    by_id = struc.bar_elements_by_id
    return Member(id=mid, bar_ids=list(bars), node_ids=list(nodes),
                  section_name=by_id[bars[0]].section_name,
                  length=sum(_bar_length(struc, by_id[b]) for b in bars))


def _split_spans(struc, beam_id, nodes, bars, incident, supported) -> list:
    """Cut the path at supports, hinges and nodes where a third bar joins."""
    by_id = struc.bar_elements_by_id
    cuts = [0]
    for k in range(1, len(nodes) - 1):
        n = nodes[k]
        if (n in supported or incident.get(n, 0) != 2
                or _hinge_at(by_id[bars[k - 1]], n)
                or _hinge_at(by_id[bars[k]], n)):
            cuts.append(k)
    cuts.append(len(bars))
    spans = []
    for s, (a, b) in enumerate(zip(cuts, cuts[1:]), 1):
        spans.append(_make_member(struc, f"{beam_id}.s{s}",
                                  nodes[a:b + 1], bars[a:b]))
    return spans


def resolve_timber_beams(struc, is_timber) -> tuple:
    """The tagged timber beams of *struc*: ``(beams, warnings)``.

    *is_timber* is a callable ``bar -> bool`` selecting the bars that take part
    (the design passes its "timber section" test). Beams are ordered by tag.
    """
    by_tag: dict[str, list] = {}
    other_material: dict[str, int] = {}
    for b in struc.bar_elements:
        if not b.beam:
            continue
        if is_timber(b):
            by_tag.setdefault(b.beam, []).append(b)
        else:
            other_material[b.beam] = other_material.get(b.beam, 0) + 1

    incident = _incident_bar_count(struc)
    supported = _supported_nodes(struc)
    beams, warnings = [], []

    for tag in sorted(by_tag):
        if other_material.get(tag):
            warnings.append(
                f"Beam {tag}: {other_material[tag]} bar(s) are not timber and "
                "were ignored.")
        adj: dict[str, list] = {}
        for b in by_tag[tag]:
            if b.node_i not in struc.nodes or b.node_j not in struc.nodes:
                continue
            adj.setdefault(b.node_i, []).append((b.id, b.node_j))
            adj.setdefault(b.node_j, []).append((b.id, b.node_i))
        paths = []
        for comp in _components(adj):
            p = _walk(struc, comp, adj)
            if p is None:
                warnings.append(
                    f"Beam {tag}: the bars do not form a single open chain "
                    "(a branch or a closed ring); they are designed as "
                    "separate members.")
                continue
            paths.append(p)
        paths.sort(key=lambda p: _start_key(struc, p[0][0]))
        entry = (struc.beams.get(tag) or {})
        params = dict(entry.get("timber") or {})
        for k, (nodes, bars) in enumerate(paths, 1):
            bid = tag if len(paths) == 1 else f"{tag}.{k}"
            beams.append(TimberBeam(
                id=bid, tag=tag, name=entry.get("name", ""),
                member=_make_member(struc, bid, nodes, bars),
                spans=_split_spans(struc, bid, nodes, bars, incident,
                                   supported),
                params=params))
    return beams, warnings


# ── Geometry read from the model (dev/TIMBER_BEAMS.md §3.3, §5) ─────────────
#
# The user draws the beam with the depths he wants (one section per bar); the
# geometry below is *read*, never generated: the depth h(x), which edge is
# inclined, where the apex is and its radius.

import math

_TURN_MIN = math.radians(0.5)     # a node turns the axis by at least this
_APEX_MIN = math.radians(1.0)     # total turn that makes a ridge / curve
_FLAT_TOL = 1e-6                  # relative depth change taken as zero


def _wrap(a: float) -> float:
    """Angle wrapped to (-π, π]."""
    while a > math.pi:
        a -= 2 * math.pi
    while a <= -math.pi:
        a += 2 * math.pi
    return a


@dataclass
class BarGeom:
    """One bar of the beam, seen along the beam (left → right)."""

    bar_id: str
    fwd: bool                  # bar node_i → node_j runs along the beam
    length: float
    b: float
    h_i: float                 # depth at the bar's node_i / node_j
    h_j: float
    alpha: float = 0.0         # angle between the two edges [rad]
    slope_top: float = 0.0     # slopes of the edges to the horizontal [rad]
    slope_bottom: float = 0.0
    s0: float = 0.0            # distance along the beam of the bar's start

    def s_at(self, x: float) -> float:
        """Distance along the beam of the station *x* from the bar's node_i."""
        return self.s0 + (x if self.fwd else self.length - x)

    def h_at(self, x: float) -> float:
        """Depth at *x* from the bar's node_i."""
        if self.length <= 0.0:
            return self.h_i
        t = min(max(x / self.length, 0.0), 1.0)
        return self.h_i + (self.h_j - self.h_i) * t


@dataclass
class ApexNode:
    """A node of the apex / curved zone."""

    node_id: str
    h: float
    alpha_ap: float            # outer edge to the grain at the node [rad]
    r_in: float | None         # inner radius [m]; None = sharp (straight edges)
    s: float = 0.0             # distance along the beam


@dataclass
class Apex:
    nodes: dict = field(default_factory=dict)   # node id → ApexNode
    inner: str = "bottom"                       # side of the centre of curvature
    curved_bars: set = field(default_factory=set)   # bars inside an arc
    total_turn: float = 0.0
    zone_volume: float | None = None   # volume of the curved part [m³]
    zone_length: float | None = None   # length of the curved part [m]


@dataclass
class BeamGeometry:
    kind: str = "straight"
    depth_kind: str = "constant"        # constant | tapered | step
    taper_edge: str | None = None       # the inclined ("cut") edge: top|bottom
    bars: dict = field(default_factory=dict)    # bar id → BarGeom
    apex: Apex | None = None
    volume: float = 0.0                 # beam volume [m³]
    t_lam: float = 33.0                 # lamination thickness [mm]
    reinforced: bool = False            # apex tension perpendicular carried by
    rf_a1: float | None = None          # reinforcement (EN 1995-1-1:2025 §8.3.4.2)
    rf_plane: dict | None = None        # bonded-on plane reinforcement data (SI)
    rf_dowel: dict | None = None        # screws / rods (mm, MPa; lengths in m)
    rf_type: str = "plane"              # which of the two is used
    warnings: list = field(default_factory=list)


def _bar_depth(struc, bar):
    from .models import SectionShape
    sec = struc.sections.get(bar.section_name)
    if sec is None or not (sec.b > 0 and sec.h > 0) or sec.shape in (
            SectionShape.CIRCULAR, SectionShape.CIRCULAR_HOLLOW):
        return None
    return sec.b, sec.h


def _gradient(h, m, k):
    """Depth gradient of bar *k* from its neighbours' mid-points *m*, *h*."""
    n = len(h)
    back = ((h[k] - h[k - 1]) / (m[k] - m[k - 1])) if k > 0 else None
    fwd = ((h[k + 1] - h[k]) / (m[k + 1] - m[k])) if k < n - 1 else None
    if back is None:
        return fwd or 0.0
    if fwd is None:
        return back
    if back * fwd > 0:                    # same regime: average
        return 0.5 * (back + fwd)
    return back if abs(back) >= abs(fwd) else fwd    # ridge / valley: the side


def beam_geometry(struc, beam, step_warn: float = 0.10):
    """Read the geometry of *beam* (a :class:`TimberBeam`) from the model.

    Returns a :class:`BeamGeometry`, or ``None`` when the bars have no
    rectangular b × h to read (circular / generic sections)."""
    mem = beam.member
    by_id = struc.bar_elements_by_id
    nodes, bars = mem.node_ids, mem.bar_ids
    nb = len(bars)
    pts = [(struc.nodes[n].x, struc.nodes[n].y) for n in nodes]
    dims = [_bar_depth(struc, by_id[bid]) for bid in bars]
    if any(d is None for d in dims):
        return None
    params = beam.params
    g = BeamGeometry()
    try:
        g.t_lam = float(params.get("t_lam") or 33.0)
    except (TypeError, ValueError):
        g.t_lam = 33.0
    g.reinforced = bool(params.get("reinforced"))
    try:
        a1 = float(params.get("rf_spacing") or 0.0)
        g.rf_a1 = a1 if a1 > 0.0 else None
    except (TypeError, ValueError):
        g.rf_a1 = None
    plane = params.get("rf_plane")
    g.rf_plane = dict(plane) if isinstance(plane, dict) and plane else None
    dowel = params.get("rf_dowel")
    g.rf_dowel = dict(dowel) if isinstance(dowel, dict) and dowel else None
    g.rf_type = "dowel" if params.get("rf_type") == "dowel" else "plane"

    L = [math.hypot(pts[k + 1][0] - pts[k][0], pts[k + 1][1] - pts[k][1])
         for k in range(nb)]
    theta = [math.atan2(pts[k + 1][1] - pts[k][1], pts[k + 1][0] - pts[k][0])
             for k in range(nb)]
    b_w = [d[0] for d in dims]
    h = [d[1] for d in dims]
    s0, acc = [], 0.0
    for lk in L:
        s0.append(acc)
        acc += lk
    mid = [s0[k] + L[k] / 2 for k in range(nb)]

    # ── depth: constant / step / tapered ────────────────────────────────
    changes = sum(1 for k in range(nb - 1)
                  if abs(h[k + 1] - h[k]) > _FLAT_TOL * max(h[k], h[k + 1]))
    dk = str(params.get("depth_kind") or "").lower()
    if dk not in ("constant", "step", "tapered"):
        dk = ("constant" if changes == 0
              else "tapered" if changes >= 2 else "step")
    g.depth_kind = dk

    h_end = []                            # per bar: (depth at path start, end)
    if dk == "tapered" and nb > 1:
        for k in range(nb):
            gr = _gradient(h, mid, k)
            h_end.append((h[k] - gr * L[k] / 2, h[k] + gr * L[k] / 2))
        node_h = []
        for n in range(nb + 1):
            vals = ([h_end[n - 1][1]] if n > 0 else []) \
                + ([h_end[n][0]] if n < nb else [])
            node_h.append(sum(vals) / len(vals))
        h_end = [(node_h[k], node_h[k + 1]) for k in range(nb)]
        worst = max((abs(h[k + 1] - h[k]) / max(h[k], h[k + 1])
                     for k in range(nb - 1)), default=0.0)
        if worst > step_warn:
            g.warnings.append(
                f"Beam {beam.id}: the depth changes by up to {worst:.0%} "
                "between neighbouring bars; the taper is discretised too "
                "coarsely for the checks at the bar ends.")
    else:
        h_end = [(h[k], h[k]) for k in range(nb)]

    # ── edges ────────────────────────────────────────────────────────────
    for k in range(nb):
        (x0, y0), (x1, y1) = pts[k], pts[k + 1]
        nx, ny = -math.sin(theta[k]), math.cos(theta[k])
        h0, h1 = h_end[k]
        top = math.atan2(y1 + ny * h1 / 2 - (y0 + ny * h0 / 2),
                         x1 + nx * h1 / 2 - (x0 + nx * h0 / 2))
        bot = math.atan2(y1 - ny * h1 / 2 - (y0 - ny * h0 / 2),
                         x1 - nx * h1 / 2 - (x0 - nx * h0 / 2))
        fwd = by_id[bars[k]].node_i == nodes[k]
        hi, hj = (h0, h1) if fwd else (h1, h0)
        g.bars[bars[k]] = BarGeom(
            bar_id=bars[k], fwd=fwd, length=L[k], b=b_w[k], h_i=hi, h_j=hj,
            alpha=abs(_wrap(top - bot)), slope_top=top, slope_bottom=bot,
            s0=s0[k])
    g.volume = sum(b_w[k] * L[k] * 0.5 * sum(h_end[k]) for k in range(nb))

    # the inclined ("cut") edge is the one that is *not* parallel to the grain,
    # i.e. the steeper of the two; the other follows the laminations
    if dk == "tapered":
        edge = str(params.get("taper_edge") or "").lower()
        if edge not in ("top", "bottom"):
            st = sum(abs(bg.slope_top) for bg in g.bars.values())
            sb = sum(abs(bg.slope_bottom) for bg in g.bars.values())
            edge = "top" if st > sb else "bottom"
        g.taper_edge = edge

    # ── apex / curved zone ───────────────────────────────────────────────
    turn = [0.0] + [_wrap(theta[k] - theta[k - 1]) for k in range(1, nb)] + [0.0]
    runs, cur = [], []
    for n in range(1, nb):
        if abs(turn[n]) >= _TURN_MIN and (not cur or turn[n] * turn[cur[-1]] > 0):
            cur.append(n)
        else:
            if cur:
                runs.append(cur)
            cur = [n] if abs(turn[n]) >= _TURN_MIN else []
    if cur:
        runs.append(cur)
    runs = [r for r in runs if sum(abs(turn[n]) for n in r) >= _APEX_MIN]
    if runs:
        runs.sort(key=lambda r: -sum(abs(turn[n]) for n in r))
        if len(runs) > 1:
            g.warnings.append(
                f"Beam {beam.id}: {len(runs)} ridges / curved zones found; "
                "only the one turning the most is checked.")
        run = runs[0]
        total = sum(turn[n] for n in run)
        inner = "bottom" if total < 0 else "top"      # centre of curvature
        outer = "top" if inner == "bottom" else "bottom"
        apex = Apex(inner=inner, total_turn=abs(total))
        n1, n2 = run[0], run[-1]
        r_axis = None
        if len(run) > 1:
            arc = L[n1 - 1] / 2 + sum(L[k] for k in range(n1, n2)) + L[n2] / 2
            r_axis = arc / abs(total)
            apex.curved_bars = set(bars[n1 - 1:n2 + 1])
            apex.zone_length = arc
            # the apex zone is the curved part (EN 1995-1-1 Fig. 6.9)
            def hmean(k, a, b_):          # mean depth of bar k between a..b_ (0..1)
                h0, h1 = h_end[k]
                return h0 + (h1 - h0) * (a + b_) / 2
            apex.zone_volume = (
                b_w[n1 - 1] * L[n1 - 1] * 0.5 * hmean(n1 - 1, 0.5, 1.0)
                + sum(b_w[k] * L[k] * hmean(k, 0.0, 1.0)
                      for k in range(n1, n2))
                + b_w[n2] * L[n2] * 0.5 * hmean(n2, 0.0, 0.5))
            # M_ap and α_ap are those of the middle of the zone (§6.4.3(4))
            mid_nodes = ([run[len(run) // 2]] if len(run) % 2
                         else [run[len(run) // 2 - 1], run[len(run) // 2]])
        else:
            mid_nodes = [run[0]]
        a_mid = None
        if len(run) > 1:
            a_mid = sum(0.5 * (g.bars[bars[n - 1]].alpha + g.bars[bars[n]].alpha)
                        for n in mid_nodes) / len(mid_nodes)
        for n in mid_nodes:
            hn = h_end[n - 1][1] if n > 0 else h_end[0][0]
            ba, bb = g.bars[bars[n - 1]], g.bars[bars[n]]
            phi = (theta[n - 1] + theta[n]) / 2          # grain at the node
            if a_mid is not None:   # arc: the bars only discretise the curve,
                a_ap = a_mid        # so use the taper between the edges
            else:                   # sharp ridge: outer edge to the grain
                a_ap = 0.5 * (abs(_wrap(getattr(ba, f"slope_{outer}") - phi))
                              + abs(_wrap(getattr(bb, f"slope_{outer}") - phi)))
            apex.nodes[nodes[n]] = ApexNode(
                node_id=nodes[n], h=hn, alpha_ap=a_ap,
                r_in=None if r_axis is None else max(r_axis - hn / 2, 0.0),
                s=s0[n] if n < nb else acc)
        try:
            r_ov = float(params.get("r_in") or 0.0)
        except (TypeError, ValueError):
            r_ov = 0.0
        if r_ov > 0.0:                    # user-given inner radius wins
            for an in apex.nodes.values():
                an.r_in = r_ov
        g.apex = apex
        a_ap = max(an.alpha_ap for an in apex.nodes.values())
        if len(run) > 1:
            g.kind = "pitched_cambered" if a_ap > _APEX_MIN else "curved"
        else:
            g.kind = "double_tapered"
    else:
        g.kind = "tapered" if dk == "tapered" else "straight"
    kind = str(params.get("kind") or "").lower()
    if kind in ("double_tapered", "curved", "pitched_cambered") and g.apex:
        g.kind = kind
    return g


# ── EC5 §6.4 checks at one design point ──────────────────────────────────────

def _label_x(label: str, length: float):
    """Distance from the bar's node_i of a design point label ('i', 'j',
    'x=1.25m'), or None for a label it does not understand."""
    if label == "i":
        return 0.0
    if label == "j":
        return length
    if label.startswith("x="):
        try:
            return float(label[2:].rstrip("m"))
        except ValueError:
            return None
    return None


def local_depth(geom, bar, label):
    """Depth [m] of *bar* at the design point *label*, for a beam whose depth
    varies (``tapered``); None when the bar's own constant section applies."""
    if geom is None or geom.depth_kind != "tapered":
        return None
    bg = geom.bars.get(bar.id)
    x = _label_x(label, bg.length) if bg is not None else None
    return None if x is None else bg.h_at(x)


def _reinforcement_data(uls, geom, h_ap):
    """The 2025 reinforcement of the beam parameters, or None: a
    ``DowelReinforcement`` (``rf_type = "dowel"``: ``d, f_uk, gamma_r, gamma_m2``,
    ``d1`` for screws, optionally ``gamma_r, gamma_m2, kind, thread, eps,
    k_screw, l_r_l, l_r_u, n_r``; mm / MPa, lengths in m) or a
    ``PlaneReinforcement`` (``b_r, t_r, f_td``, optionally ``f_b2d, l_r_l,
    l_r_u, b_r_ef, n_r``, in m and MPa). What is not given comes from the
    standard: ``γ_R = 1.30`` and ``γ_M2 = 1.25`` (Table 4.4); ``f_b2,d =
    k_mod·0.75/1.30``; the plane ``l_r`` is ``h_ap/2`` (full height, symmetric
    about the neutral axis, Figure 8.11) and the dowel one ``h_ap/2 − t_lam``."""
    if geom.rf_type == "dowel":
        w = geom.rf_dowel
        if not w or not hasattr(uls, "DowelReinforcement"):
            return None
        try:
            return uls.DowelReinforcement(
                d=float(w["d"]), f_uk=float(w["f_uk"]),
                gamma_r=float(w["gamma_r"]) if w.get("gamma_r") else None,
                gamma_m2=float(w["gamma_m2"]) if w.get("gamma_m2") else None,
                kind=str(w.get("kind") or "screw"),
                d1=float(w["d1"]) if w.get("d1") else None,
                thread=str(w.get("thread") or "ribbed"),
                eps=float(w.get("eps") or 90.0),
                k_screw=float(w.get("k_screw") or 8.2),
                l_r_l=float(w["l_r_l"]) if w.get("l_r_l") else None,
                l_r_u=float(w["l_r_u"]) if w.get("l_r_u") else None,
                n_r=int(w.get("n_r") or 1))
        except (KeyError, TypeError, ValueError):
            return None
    d = geom.rf_plane
    if not d or not hasattr(uls, "PlaneReinforcement"):
        return None
    try:
        return uls.PlaneReinforcement(
            b_r=float(d["b_r"]), t_r=float(d["t_r"]), f_td=float(d["f_td"]),
            f_b2d=float(d["f_b2d"]) if d.get("f_b2d") else None,
            l_r_l=float(d["l_r_l"]) if d.get("l_r_l") else None,
            l_r_u=float(d["l_r_u"]) if d.get("l_r_u") else None,
            b_r_ef=float(d["b_r_ef"]) if d.get("b_r_ef") else None,
            n_r=int(d.get("n_r") or 1))
    except (KeyError, TypeError, ValueError):
        return None


def _recipe(fn, *args, **kw):
    """``recipe(trace)`` re-running ``fn(*args, trace=trace, **kw)``: lets the
    report trace the governing point without keeping the check's objects."""
    return lambda trace: fn(*args, trace=trace, **kw)


def special_point_checks(uls, geom, bar, label, v, m, timber, service_class,
                         load_duration, calls=None, info=None) -> dict:
    """EN 1995-1-1 §6.4 utilisations at one design point of *bar*.

    *uls* is the eurocodepy ULS namespace (2004 or 2025); *v* [kN] and *m*
    [kNm] are the point's shear and bending moment (sagging positive: tension
    on the right of the bar's own travel, i → j). Returns the non-zero ones of
    ``taper`` (§6.4.2), ``apex_bending``, ``apex_t90``, ``apex_inter``
    (§6.4.3) and ``curve`` (k_r in a curved zone).

    When *calls* is a dict it receives, under ``taper`` / ``apex`` / ``curve``,
    a ``recipe(trace)`` that repeats the check with a calculation trace.
    With the beam parameter ``reinforced`` (2025 only) the apex tension
    perpendicular to the grain is carried by reinforcement: Eqs. 8.67 / 8.70 do
    not apply and *info* receives ``rf_per_m`` [kN/m] and ``rf_force`` [kN].
    When *info* is a dict it receives ``reinforce`` (the reason of the optional reinforcement
    recommended by EN 1995-1-1:2025 §8.3.1(3)/(4)) and ``out_of_scope`` (the product is outside §8.3.4) as the
    edition's apex check reports them.
    """
    bg = geom.bars.get(bar.id)
    out: dict = {}
    if bg is None or m == 0.0:
        return out
    x = _label_x(label, bg.length)
    if x is None:
        return out
    h = bg.h_at(x)
    # tension face in the beam's own frame (left → right): the bar's travel
    # either follows the beam (right of travel = bottom) or opposes it
    face = "bottom" if (m > 0) == bg.fwd else "top"

    apex = geom.apex
    # §6.4.3(2): the single-taper check is for the parts outside the apex zone
    in_apex = False
    if apex is not None:
        if bar.id in apex.curved_bars:
            in_apex = True
        else:
            s_pt = bg.s_at(x)
            in_apex = any(abs(s_pt - an.s) < an.h / 2 - 1e-9
                          for an in apex.nodes.values())
    if geom.taper_edge and bg.alpha > 1e-4 and not in_apex:
        args = (abs(m), bg.b, h, bg.alpha, face == geom.taper_edge, timber,
                service_class, load_duration)
        out["taper"] = uls.check_tapered_edge(*args).util
        if calls is not None:
            calls["taper"] = _recipe(uls.check_tapered_edge, *args)

    if apex is not None:
        node = bar.node_i if label == "i" else bar.node_j if label == "j" else None
        an = apex.nodes.get(node) if node else None
        if an is not None:
            from eurocodepy.ec5.uls.special_beams import K_DIS
            kind = geom.kind if geom.kind in K_DIS else "double_tapered"
            ainp = uls.ApexInput(
                m_ap=abs(m), v_ap=abs(v), b=bg.b, h_ap=an.h,
                alpha_ap=an.alpha_ap, r_in=an.r_in, t_lam=geom.t_lam,
                kind=kind, straightening=(face == apex.inner),
                beam_volume=geom.volume, zone_volume=apex.zone_volume,
                reinforced=geom.reinforced and uls.__name__.endswith("uls2025"),
                a1=geom.rf_a1, zone_length=apex.zone_length,
                reinforcement=_reinforcement_data(uls, geom, an.h))
            r = uls.check_apex_zone(ainp, timber, service_class, load_duration)
            if calls is not None:
                calls["apex"] = _recipe(uls.check_apex_zone, ainp, timber,
                                        service_class, load_duration)
            if info is not None:
                if r.details.get("reinforce_advised"):
                    info["reinforce"] = r.details["reinforce_reason"]
                if r.details.get("reinforced"):
                    info["rf_per_m"] = r.details["F_t90_rf_per_m"]
                    info["rf_force"] = r.details["F_t90_rf_Ed"]
                if r.details.get("in_scope") is False:
                    info["out_of_scope"] = True
            out["apex_bending"] = r.util_bending
            rf_ = r.details.get("reinforcement")
            if rf_ is not None:
                out["apex_rf"] = rf_["util"]
                if info is not None:
                    info["rf_util"] = rf_["util"]
                    info["rf_governing"] = rf_["governing"]
                    info["rf_n"] = len(rf_["elements"])
                    notes = list(rf_["worst"][3].details.get("warnings", []))
                    ok_sp, msg_sp = rf_["spacing"]
                    if geom.rf_type == "dowel" and not ok_sp:
                        notes.append(msg_sp + " (§8.3.4.2(4))")
                    info["rf_notes"] = notes
            out["apex_t90"] = r.util_tension90
            out["apex_inter"] = r.util_interaction
        elif bar.id in apex.curved_bars:
            r_in = next((a.r_in for a in apex.nodes.values()
                         if a.r_in is not None), None)
            k_r = uls.calc_k_r(r_in, geom.t_lam)
            if k_r < 1.0:
                out["curve"] = _curve_check(uls, abs(m), bg.b, h, r_in, k_r,
                                            geom.t_lam, timber, service_class,
                                            load_duration)
                if calls is not None:
                    calls["curve"] = _recipe(
                        _curve_check, uls, abs(m), bg.b, h, r_in, k_r,
                        geom.t_lam, timber, service_class, load_duration)
    return {k: v_ for k, v_ in out.items() if v_ > 0.0}


def is_timber_bar(struc, bar) -> bool:
    """True for a bar with a timber section the beam design can take."""
    from .timber_design import _is_timber
    return _is_timber(struc, struc.sections.get(bar.section_name))


def describe_beam(struc, tag: str, step_warn: float = 0.10) -> list:
    """What the design reads from the bars of beam *tag*, as short text lines
    (one block per resolved chain) -- for the panel and the assistant."""
    beams, warns = resolve_timber_beams(
        struc, lambda b: is_timber_bar(struc, b))
    lines = []
    for tb in [b for b in beams if b.tag == tag]:
        g = beam_geometry(struc, tb, step_warn)
        head = (f"{tb.id}: {len(tb.bar_ids)} bar(s), L = "
                f"{tb.member.length:.2f} m, {len(tb.spans)} span(s)")
        if g is None:
            lines.append(head + " — circular / generic section: only the "
                         "§6.1–§6.3 checks apply.")
            continue
        hs = [bg.h_i for bg in g.bars.values()] + [bg.h_j for bg in g.bars.values()]
        lines.append(f"{head}\n  type: {g.kind}; depth: {g.depth_kind} "
                     f"({min(hs):.3f}–{max(hs):.3f} m)"
                     + (f"; inclined edge: {g.taper_edge}" if g.taper_edge else ""))
        if g.apex is not None:
            for nid, an in g.apex.nodes.items():
                r = "sharp" if an.r_in is None else f"r_in = {an.r_in:.2f} m"
                lines.append(f"  apex {nid}: h = {an.h:.3f} m, "
                             f"α = {math.degrees(an.alpha_ap):.1f}°, {r}")
        if g.reinforced:
            lines.append("  reinforced: tension perpendicular to the grain "
                         "carried by reinforcement (§8.3.4.2(1))"
                         + (f", a1 = {g.rf_a1:.3f} m" if g.rf_a1 else ""))
        data = g.rf_dowel if g.rf_type == "dowel" else g.rf_plane
        if g.reinforced and data:
            lines.append(f"  {g.rf_type}-type reinforcement: " if g.rf_type == "dowel"
                         else "  plane reinforcement: ")
            lines[-1] += ", ".join(f"{k} = {v:g}" if isinstance(v, (int, float))
                                   else f"{k} = {v}" for k, v in data.items())
        lines.extend("  ⚠ " + w for w in g.warnings)
    lines.extend("⚠ " + w for w in warns if f"Beam {tag}:" in w)
    return lines


def _curve_check(uls, m, b, h, r_in, k_r, t_lam, timber, service_class,
                 load_duration, trace=None) -> float:
    """Bending of the curved laminations away from the apex (Eq. 6.49):
    ``σ_m,d ≤ k_r·f_m,d``. Returns the utilisation."""
    from eurocodepy.ec5.materials import TimberForcesType
    timber.design_values(service_class, load_duration)
    k_h = timber.k_h(h, TimberForcesType.Bending)
    f_md = timber.fmd * k_h
    sigma = 6.0 * m / (b * h * h) / 1e3
    util = sigma / (k_r * f_md) if f_md > 0 else 0.0
    if trace is not None:
        trace.section("Curved laminations (§6.4.3)")
        trace.step("M_d", m, "kNm")
        trace.step("r_in", r_in, "m")
        trace.step("k_r", k_r, "—", clause="EN 1995-1-1 Eq. 6.49",
                   expr="0.76+0.001·r_in/t", note=f"t = {t_lam:.4g} mm")
        trace.step("σ_m,d", sigma, "MPa", expr="6·M_d/(b·h²)",
                   subst=f"6·{m:.4g}/({b:.4g}·{h:.4g}²)/1e3")
        trace.step("f_m,d", f_md, "MPa", expr="f_m,d·k_h")
        trace.step("σ_m,d / (k_r·f_m,d)", util, "—",
                   clause="EN 1995-1-1 Eq. 6.49", ok=util <= 1.0)
    return util


def geometry_section(tb, geom) -> dict:
    """A report section (``CalcReport.to_dict()`` shape) stating what was read
    from the bars of beam *tb*, so the §6.4 checks can be audited."""
    def step(symbol, value, unit="", note=""):
        return {"symbol": symbol, "value": value, "unit": unit, "clause": "",
                "expr": "", "latex": "", "subst": "", "note": note, "ok": None}
    steps = [step("Beam", tb.id, note=f"{len(tb.bar_ids)} bar(s), "
                                      f"{len(tb.spans)} span(s)"),
             step("L", round(tb.member.length, 4), "m"),
             step("Type", geom.kind.replace("_", " ")),
             step("Depth change", geom.depth_kind)]
    hs = [bg.h_i for bg in geom.bars.values()] + [bg.h_j for bg in geom.bars.values()]
    steps.append(step("h min / max", (round(min(hs), 4), round(max(hs), 4)), "m"))
    if geom.taper_edge:
        steps.append(step("Inclined edge", geom.taper_edge,
                          note="the other edge follows the laminations"))
    steps.append(step("Volume", round(geom.volume, 4), "m³"))
    if geom.apex is not None:
        steps.append(step("Centre of curvature", geom.apex.inner,
                          note="side of the apex nearer to it"))
        for nid, an in geom.apex.nodes.items():
            steps.append(step(
                f"Apex {nid}",
                (round(an.h, 4), round(math.degrees(an.alpha_ap), 2),
                 "∞" if an.r_in is None else round(an.r_in, 3)),
                "m, °, m", note="h_ap, α_ap, r_in"))
        steps.append(step("t_lam", geom.t_lam, "mm"))
    return {"title": "Beam geometry (read from the model)", "steps": steps}


# ── Outline for the utilisation elevation ────────────────────────────────────

def beam_outline(struc, tb, geom) -> dict:
    """The beam drawn as it is: top and bottom edges through the nodes (the
    axis ± h/2 along the node normal), the distance ``s`` of every vertex along
    the beam, the span boundaries (supports) and the apex nodes. Plain lists,
    so it can be saved with the design results."""
    mem = tb.member
    nodes, bars = mem.node_ids, mem.bar_ids
    pts = [(struc.nodes[n].x, struc.nodes[n].y) for n in nodes]
    s = [0.0]
    for k in range(len(bars)):
        s.append(s[-1] + math.hypot(pts[k + 1][0] - pts[k][0],
                                    pts[k + 1][1] - pts[k][1]))
    hn: dict[str, list] = {}
    for bid in bars:
        bg = geom.bars[bid]
        bar = struc.bar_elements_by_id[bid]
        hn.setdefault(bar.node_i, []).append(bg.h_i)
        hn.setdefault(bar.node_j, []).append(bg.h_j)
    top, bot = [], []
    for k, n in enumerate(nodes):
        dirs = []
        if k > 0:
            dirs.append(math.atan2(pts[k][1] - pts[k - 1][1],
                                   pts[k][0] - pts[k - 1][0]))
        if k < len(bars):
            dirs.append(math.atan2(pts[k + 1][1] - pts[k][1],
                                   pts[k + 1][0] - pts[k][0]))
        nx = sum(-math.sin(a) for a in dirs) / len(dirs)
        ny = sum(math.cos(a) for a in dirs) / len(dirs)
        nl = math.hypot(nx, ny) or 1.0
        nx, ny = nx / nl, ny / nl
        h = sum(hn[n]) / len(hn[n])
        top.append([pts[k][0] + nx * h / 2, pts[k][1] + ny * h / 2])
        bot.append([pts[k][0] - nx * h / 2, pts[k][1] - ny * h / 2])
    index = {n: k for k, n in enumerate(nodes)}
    supports = sorted({s[index[sp.node_ids[0]]] for sp in tb.spans}
                      | {s[index[sp.node_ids[-1]]] for sp in tb.spans})
    apex = ([s[index[n]] for n in geom.apex.nodes if n in index]
            if geom.apex is not None else [])
    return {"s": s, "top": top, "bottom": bot, "supports": supports,
            "apex": apex, "length": s[-1], "kind": geom.kind}


# ── Automatic detection and ordering ─────────────────────────────────────────

#: Automatic detection only considers timber bars sloping at most this [degrees]
#: from the horizontal (columns, posts and steep braces are never proposed).
TIMBER_AUTO_MAX_SLOPE_DEG = 45.0
#: Two bars meeting at a node join into one beam only when the axis turns by at
#: most this [degrees] (a ridge of a 30° roof turns 60°).
TIMBER_AUTO_MAX_TURN_DEG = 60.0


def _is_column_bar(struc, bar) -> bool:
    sec = struc.sections.get(bar.section_name)
    own = getattr(bar, "is_column", None)
    return bool(own) if own is not None else bool(getattr(sec, "is_column", False))


def order_bars_along(struc, bar_ids) -> list:
    """*bar_ids* in the order they run along the beam, from its left end
    (smaller x; smaller y when vertical). Pieces that are not one simple open
    chain (disconnected, branched, a ring) fall back to the bars' mid-point
    (x, y), piece by piece, so the order is always defined and stable."""
    by_id = struc.bar_elements_by_id
    bars = [by_id[b] for b in bar_ids if b in by_id]
    adj: dict[str, list] = {}
    for b in bars:
        if b.node_i in struc.nodes and b.node_j in struc.nodes:
            adj.setdefault(b.node_i, []).append((b.id, b.node_j))
            adj.setdefault(b.node_j, []).append((b.id, b.node_i))

    def mid(bid):
        b = by_id[bid]
        ni, nj = struc.nodes[b.node_i], struc.nodes[b.node_j]
        return (round((ni.x + nj.x) / 2, 9), round((ni.y + nj.y) / 2, 9))

    pieces = []
    for comp in _components(adj):
        w = _walk(struc, comp, adj)
        ids = list(w[1]) if w is not None else sorted(
            {bid for n in comp for bid, _ in adj[n]}, key=mid)
        pieces.append(ids)
    pieces.sort(key=lambda ids: mid(ids[0]) if ids else (0, 0))
    out = [bid for ids in pieces for bid in ids]
    out += sorted((b.id for b in bars if b.id not in out), key=mid)
    return out


def beam_start_key(struc, ordered_bar_ids):
    """``(y, x)`` of the left end of a beam whose bars are given in path order
    (:func:`order_bars_along`): the sort key of "bottom → top, left → right",
    with y within 1 mm counting as one level."""
    by_id = struc.bar_elements_by_id
    ids = [b for b in ordered_bar_ids if b in by_id]
    if not ids:
        return (0.0, 0.0)
    b = by_id[ids[0]]
    ni, nj = struc.nodes[b.node_i], struc.nodes[b.node_j]
    if len(ids) > 1:
        nxt = by_id[ids[1]]
        shared = {b.node_i, b.node_j} & {nxt.node_i, nxt.node_j}
        far = nj if b.node_i in shared else ni
    else:
        far = ni if (ni.x, ni.y) <= (nj.x, nj.y) else nj
    return (round(far.y, 3), round(far.x, 9))


def suggest_timber_beams(struc, is_timber) -> list:
    """Beams the automatic detection would form, as lists of bar ids ordered
    along each beam (left → right), the beams ordered bottom → top and left →
    right (y within 1 mm counts as one level). Takes the timber bars that slope
    at most :data:`TIMBER_AUTO_MAX_SLOPE_DEG` and are not columns, ignoring the
    existing ``beam`` tags -- a *suggestion* for the user to accept.

    Two bars join at a node when the axis turns by at most
    :data:`TIMBER_AUTO_MAX_TURN_DEG`; where more than two bars meet, the pairs
    are taken from the straightest, so a beam running through a joint keeps
    going and the others start their own. A change of section, a support or a
    hinge does not end a beam (spans are cut later, for the buckling length).
    """
    limit = math.radians(TIMBER_AUTO_MAX_SLOPE_DEG)
    max_turn = math.radians(TIMBER_AUTO_MAX_TURN_DEG)
    cand = {}
    for b in struc.bar_elements:
        ni, nj = struc.nodes.get(b.node_i), struc.nodes.get(b.node_j)
        if ni is None or nj is None or not is_timber(b) or _is_column_bar(struc, b):
            continue
        if math.atan2(abs(nj.y - ni.y), abs(nj.x - ni.x)) > limit + 1e-9:
            continue
        cand[b.id] = b
    at_node: dict[str, list] = {}
    for b in cand.values():
        at_node.setdefault(b.node_i, []).append(b.id)
        at_node.setdefault(b.node_j, []).append(b.id)

    def out_angle(node, bar):
        a = struc.nodes[node]
        o = struc.nodes[bar.node_j if bar.node_i == node else bar.node_i]
        return math.atan2(o.y - a.y, o.x - a.x)

    links: dict[str, list] = {b: [] for b in cand}     # bar → [(other, node)]
    for node, ids in at_node.items():
        if len(ids) < 2:
            continue
        pairs = []
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a1 = out_angle(node, cand[ids[i]])
                a2 = out_angle(node, cand[ids[j]])
                turn = abs(math.pi - abs(_wrap(a1 - a2)))
                if turn <= max_turn + 1e-9:
                    pairs.append((turn, ids[i], ids[j]))
        used = set()
        for turn, a, b in sorted(pairs):
            if a in used or b in used:
                continue
            used |= {a, b}
            links[a].append((b, node))
            links[b].append((a, node))

    seen, chains = set(), []
    for start in sorted(cand):
        if start in seen or len(links[start]) > 1:
            continue
        chain, cur, prev = [start], start, None
        seen.add(start)
        while True:
            nxt = [o for o, _n in links[cur] if o != prev and o not in seen]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            chain.append(cur)
            seen.add(cur)
        chains.append(chain)
    # bars left over belong to a closed ring: not a beam, not proposed

    out = []
    for chain in chains:
        ordered = order_bars_along(struc, chain)
        out.append(ordered)

    out.sort(key=lambda ids: beam_start_key(struc, ids))
    return out
