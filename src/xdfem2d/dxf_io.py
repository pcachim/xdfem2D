"""DXF import/export for xdfem2D.

Import reads LINE, LWPOLYLINE, POLYLINE (2-D) and — approximated by straight
segments — ARC, CIRCLE and ELLIPSE entities from a DXF drawing and builds a
:class:`~xdfem2d.structure.Structure2D`: every distinct segment endpoint becomes
a node (coincident endpoints merged within a tolerance) and every segment becomes
a bar element. The caller can restrict the import to a chosen set of layers,
rescale the drawing units (e.g. mm or cm to metres) and control how finely curves
are approximated. Each imported layer gets its own section.

Export writes a Structure2D to a DXF drawing with each item type on its own
layer: one layer per section for the elements, plus separate layers for nodes,
node labels, supports and springs.

Requires the optional third-party package ``ezdxf``.
"""
from __future__ import annotations

import math
from typing import Iterable, Iterator

# A drawing unit → metres factor is applied at import time; keep the common ones
# handy for the GUI's unit picker.
UNIT_SCALES: dict[str, float] = {
    "m": 1.0,
    "cm": 0.01,
    "mm": 0.001,
}

# Default number of straight segments used to approximate a full 360° circle.
DEFAULT_CURVE_SEGMENTS = 24

Point = tuple[float, float]
Segment = tuple[Point, Point]


def _require_ezdxf():
    try:
        import ezdxf  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "Importing DXF files needs the 'ezdxf' package, which is not "
            "installed.\n\nInstall it with:\n    pip install ezdxf") from exc
    return ezdxf


def _polyline_segments(pts: list[Point], closed: bool) -> Iterator[Segment]:
    for a, b in zip(pts, pts[1:]):
        yield (a, b)
    if closed and len(pts) > 2:
        yield (pts[-1], pts[0])


def _arc_points(cx: float, cy: float, r: float, a0: float, a1: float,
                seg_full: int) -> list[Point]:
    """Points along an arc from a0 to a1 (radians, counter-clockwise)."""
    sweep = a1 - a0
    if sweep <= 0:
        sweep += 2.0 * math.pi
    n = max(2, math.ceil(seg_full * sweep / (2.0 * math.pi)))
    return [(cx + r * math.cos(a0 + sweep * i / n),
             cy + r * math.sin(a0 + sweep * i / n)) for i in range(n + 1)]


def _entity_segments(e, seg_full: int) -> Iterator[Segment]:
    """Yield straight ((x0, y0), (x1, y1)) segments for every entity type we
    import. Curves (ARC/CIRCLE/ELLIPSE) are approximated with *seg_full*
    segments per full turn."""
    t = e.dxftype()
    if t == "LINE":
        s = e.dxf.start
        en = e.dxf.end
        yield ((s.x, s.y), (en.x, en.y))
    elif t == "LWPOLYLINE":
        pts = [(p[0], p[1]) for p in e.get_points("xy")]
        yield from _polyline_segments(pts, bool(getattr(e, "closed", False)))
    elif t == "POLYLINE":
        if getattr(e, "is_2d_polyline", True):
            pts = [(v.dxf.location.x, v.dxf.location.y) for v in e.vertices]
            yield from _polyline_segments(
                pts, bool(getattr(e, "is_closed", False)))
    elif t == "ARC":
        c = e.dxf.center
        pts = _arc_points(c.x, c.y, e.dxf.radius,
                          math.radians(e.dxf.start_angle),
                          math.radians(e.dxf.end_angle), seg_full)
        yield from _polyline_segments(pts, False)
    elif t == "CIRCLE":
        c = e.dxf.center
        r = e.dxf.radius
        n = max(3, seg_full)
        pts = [(c.x + r * math.cos(2.0 * math.pi * i / n),
                c.y + r * math.sin(2.0 * math.pi * i / n)) for i in range(n)]
        yield from _polyline_segments(pts, True)
    elif t == "ELLIPSE":
        c = e.dxf.center
        mx, my = e.dxf.major_axis.x, e.dxf.major_axis.y
        ratio = e.dxf.ratio
        # Minor axis = major rotated +90° and scaled by the axis ratio.
        nx, ny = -my * ratio, mx * ratio
        a0 = float(getattr(e.dxf, "start_param", 0.0))
        a1 = float(getattr(e.dxf, "end_param", 2.0 * math.pi))
        sweep = a1 - a0
        if abs(sweep) < 1e-12:
            sweep = 2.0 * math.pi
        n = max(3, math.ceil(seg_full * abs(sweep) / (2.0 * math.pi)))
        closed = abs(abs(sweep) - 2.0 * math.pi) < 1e-6
        cnt = n if closed else n + 1
        pts = []
        for i in range(cnt):
            t_ = a0 + sweep * i / n
            ct, st = math.cos(t_), math.sin(t_)
            pts.append((c.x + mx * ct + nx * st, c.y + my * ct + ny * st))
        yield from _polyline_segments(pts, closed)


def dxf_layers(path: str,
               curve_segments: int = DEFAULT_CURVE_SEGMENTS) -> dict[str, int]:
    """Return ``{layer_name: segment_count}`` for every layer that holds
    importable geometry, sorted by layer name."""
    _require_ezdxf()
    import ezdxf  # noqa: PLC0415

    doc = ezdxf.readfile(path)
    counts: dict[str, int] = {}
    for e in doc.modelspace():
        n = sum(1 for _ in _entity_segments(e, curve_segments))
        if n:
            layer = e.dxf.layer
            counts[layer] = counts.get(layer, 0) + n
    return dict(sorted(counts.items()))


def _section_name_for(layer: str, existing: set[str]) -> str:
    """A unique, readable section name derived from a layer name."""
    base = f"Sec_{layer}".strip() or "Sec"
    name = base
    i = 1
    while name in existing:
        i += 1
        name = f"{base}_{i}"
    return name


def _entity_to_object(e, scale, curve_segments):
    """Return ('arc'|'polyline', params) for a DXF entity, mapping arcs/circles
    to a GeoArc and everything else to a GeoMultisegment (scaled to metres). None if
    the entity type is not importable."""
    t = e.dxftype()
    if t == "ARC":
        c = e.dxf.center
        return ("arc", dict(cx=c.x * scale, cy=c.y * scale,
                            radius=e.dxf.radius * scale,
                            start_angle=e.dxf.start_angle,
                            end_angle=e.dxf.end_angle,
                            divisions=max(2, curve_segments)))
    if t == "CIRCLE":
        c = e.dxf.center
        return ("arc", dict(cx=c.x * scale, cy=c.y * scale,
                            radius=e.dxf.radius * scale,
                            start_angle=0.0, end_angle=360.0,
                            divisions=max(3, curve_segments)))
    # LINE / LWPOLYLINE / POLYLINE / ELLIPSE → a polyline of scaled points.
    segs = list(_entity_segments(e, curve_segments))
    if not segs:
        return None
    pts = [segs[0][0]] + [b for _a, b in segs]
    verts = [[p[0] * scale, p[1] * scale] for p in pts]
    closed = bool(getattr(e, "closed", False) or getattr(e, "is_closed", False))
    if closed and len(verts) > 1 and verts[0] == verts[-1]:
        verts = verts[:-1]                 # drop the repeated closing vertex
    return ("polyline", dict(vertices=verts, closed=closed, divisions=1))


def load_dxf(path: str, layers: Iterable[str] | None = None,
             tol: float = 1.0e-4, scale: float = 1.0,
             curve_segments: int = DEFAULT_CURVE_SEGMENTS,
             as_objects: bool = False):
    """Build a Structure2D from the geometry in *path*.

    Parameters
    ----------
    path : str
        DXF file to read.
    layers : iterable of str, optional
        Only import entities on these layers. ``None`` imports every layer.
    tol : float
        Endpoints closer than this (in *drawing* units) are merged into one
        node — this is what stitches the segments into a connected frame.
    scale : float
        Multiply every coordinate by this factor to convert drawing units to
        metres (e.g. 0.001 for a drawing in millimetres).
    curve_segments : int
        Number of straight segments used to approximate a full 360° curve;
        arcs and partial ellipses use a proportional fraction of this.

    Each imported layer is given its own section (``Sec_<layer>``), and every
    element is assigned the section of the layer it came from.
    """
    _require_ezdxf()
    import ezdxf  # noqa: PLC0415
    from xdfem2d.templates import _default_new_structure  # noqa: PLC0415

    doc = ezdxf.readfile(path)
    wanted = set(layers) if layers is not None else None

    # ── Import as geometry objects (arcs/polylines) ──────────────────
    if as_objects:
        s = _default_new_structure()
        section_of: dict[str, str] = {}
        used_names: set[str] = set(s.sections.keys())
        n = 0
        for e in doc.modelspace():
            layer = e.dxf.layer
            if wanted is not None and layer not in wanted:
                continue
            spec = _entity_to_object(e, scale, curve_segments)
            if spec is None:
                continue
            if layer not in section_of:
                name = _section_name_for(layer, used_names)
                used_names.add(name)
                s.add_section(name, "Mat", b=0.30, h=0.50)
                section_of[layer] = name
            kind, p = spec
            n += 1
            oid = f"O{n}"
            if kind == "arc":
                s.add_geo_arc(oid, p["cx"], p["cy"], p["radius"],
                              p["start_angle"], p["end_angle"],
                              section_name=section_of[layer],
                              divisions=p["divisions"])
            else:
                s.add_geo_polyline(oid, p["vertices"], closed=p["closed"],
                                   section_name=section_of[layer],
                                   divisions=p["divisions"])
        return s

    # Collect (layer, segment) pairs so each element keeps its source layer.
    tagged: list[tuple[str, Segment]] = []
    for e in doc.modelspace():
        layer = e.dxf.layer
        if wanted is not None and layer not in wanted:
            continue
        for seg in _entity_segments(e, curve_segments):
            tagged.append((layer, seg))

    s = _default_new_structure()

    # One section per layer actually present in the import.
    section_of: dict[str, str] = {}
    used_names: set[str] = set(s.sections.keys())
    for layer, _seg in tagged:
        if layer not in section_of:
            name = _section_name_for(layer, used_names)
            used_names.add(name)
            s.add_section(name, "Mat", b=0.30, h=0.50)
            section_of[layer] = name

    node_ids: dict[tuple[int, int], str] = {}
    n_counter = 0
    grid = tol * scale if scale else tol   # quantise on the scaled (metres) grid

    def _get_node(x: float, y: float) -> str:
        nonlocal n_counter
        xm = x * scale
        ym = y * scale
        key = (round(xm / grid), round(ym / grid)) if grid else (xm, ym)
        nid = node_ids.get(key)
        if nid is None:
            n_counter += 1
            nid = f"N{n_counter}"
            s.add_node(nid, xm, ym)
            node_ids[key] = nid
        return nid

    seen_pairs: set[frozenset[str]] = set()
    e_counter = 0
    for layer, (a, b) in tagged:
        ni = _get_node(*a)
        nj = _get_node(*b)
        if ni == nj:
            continue                      # zero-length segment
        pair = frozenset((ni, nj))
        if pair in seen_pairs:
            continue                      # duplicate / overlapping segment
        seen_pairs.add(pair)
        e_counter += 1
        s.add_bar_element(f"E{e_counter}", ni, nj, section_of[layer])

    return s


# ---------------------------------------------------------------------------
# Export (write a Structure2D to a DXF drawing)
# ---------------------------------------------------------------------------

# Fixed layer names for the non-element item types.
LAYER_NODES = "NODES"
LAYER_LABELS = "LABELS"
LAYER_SUPPORTS = "SUPPORTS"
LAYER_SPRINGS = "SPRINGS"

# ACI colour indices cycled across the per-section element layers.
_SECTION_COLORS = [5, 3, 6, 4, 30, 8, 2, 1]


def _sanitize_layer(name: str) -> str:
    """Make a DXF-safe layer name (DXF forbids a few characters)."""
    out = "".join("_" if c in '<>/\\":;?*|=`,' else c for c in str(name))
    return out.strip() or "SECTION"


def save_dxf(struc, path: str, scale: float = 1.0,
             include_nodes: bool = True, include_labels: bool = True,
             include_supports: bool = True, include_springs: bool = True):
    """Write *struc* to a DXF file at *path*.

    Layout — each item type on its own layer:
      * one layer per section (``SEC_<section>``) holding that section's element
        LINEs;
      * ``NODES``    — a small circle at every node;
      * ``LABELS``   — the node id text (optional);
      * ``SUPPORTS`` — a triangle glyph at each supported node;
      * ``SPRINGS``  — a marker at nodes/elements carrying springs.

    *scale* multiplies every coordinate (metres → drawing units, e.g. 1000 for
    a drawing in millimetres).
    """
    # Geometry objects are exported as their compiled mesh (bars).
    if getattr(struc, "geometry_objects", None):
        from .geo_expand import expand_geometry
        struc = expand_geometry(struc)[0]
    ezdxf = _require_ezdxf()
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()

    def _add_layer(name, color=7):
        if name not in doc.layers:
            doc.layers.add(name, color=color)

    # Glyph / text size: a small fraction of the model diagonal (in metres),
    # then scaled to drawing units. Falls back to a sane default for a point.
    xs = [n.x for n in struc.nodes.values()]
    ys = [n.y for n in struc.nodes.values()]
    if xs and ys:
        span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    else:
        span = 1.0
    # Two sizes are needed because coordinates go through P() (which multiplies
    # by scale) but radii / text heights are passed to ezdxf directly (already
    # in drawing units). Keep glyph OFFSETS in metres (fed through P) and glyph
    # RADII / heights in drawing units — mixing the two caused a double-scaling.
    g = span * 0.02            # glyph half-size, in metres (used inside P)
    gd = g * scale             # same, in drawing units (radii / heights)
    txt = gd * 1.25            # text height, in drawing units

    def P(x, y):
        return (x * scale, y * scale)

    # ── Elements, one layer per section ────────────────────────
    sec_color: dict[str, int] = {}
    for i, sname in enumerate(struc.sections):
        layer = _sanitize_layer(f"SEC_{sname}")
        col = _SECTION_COLORS[i % len(_SECTION_COLORS)]
        sec_color[sname] = col
        _add_layer(layer, col)
    for e in struc.bar_elements:
        ni = struc.nodes.get(e.node_i)
        nj = struc.nodes.get(e.node_j)
        if ni is None or nj is None:
            continue
        layer = _sanitize_layer(f"SEC_{e.section_name}")
        _add_layer(layer, sec_color.get(e.section_name, 7))
        msp.add_line(P(ni.x, ni.y), P(nj.x, nj.y),
                     dxfattribs={"layer": layer})

    # ── Nodes + labels ─────────────────────────────────────────
    if include_nodes or include_labels:
        _add_layer(LAYER_NODES, 7)
        if include_labels:
            _add_layer(LAYER_LABELS, 2)
        r = g * 0.35            # circle offset in metres
        for n in struc.nodes.values():
            if include_nodes:
                msp.add_circle(P(n.x, n.y), gd * 0.35,
                               dxfattribs={"layer": LAYER_NODES})
            if include_labels:
                msp.add_text(
                    str(n.id), height=txt,
                    dxfattribs={"layer": LAYER_LABELS}
                ).set_placement(P(n.x + r, n.y + r))

    # ── Supports (triangle glyph under each supported node) ────
    if include_supports and struc.support_assignments:
        _add_layer(LAYER_SUPPORTS, 1)
        supp = {s.name: s for s in struc.supports.values()}
        for a in struc.support_assignments:
            n = struc.nodes.get(a.node_id)
            if n is None:
                continue
            x, y = n.x, n.y
            pts = [P(x, y), P(x - g, y - 1.6 * g), P(x + g, y - 1.6 * g)]
            msp.add_lwpolyline(pts, close=True,
                               dxfattribs={"layer": LAYER_SUPPORTS})
            sup = supp.get(a.support_name)
            if sup is not None:
                tag = "".join(c for c, on in
                              (("x", sup.ux), ("y", sup.uy), ("r", sup.tz)) if on)
                msp.add_text(
                    tag or a.support_name, height=txt * 0.8,
                    dxfattribs={"layer": LAYER_SUPPORTS}
                ).set_placement(P(x, y - 2.0 * g))

    # ── Springs (node springs + element springs) ───────────────
    if include_springs:
        node_springs = list(getattr(struc, "node_springs", {}).values())
        elem_springs = list(getattr(struc, "element_springs", {}).values())
        if node_springs or elem_springs:
            _add_layer(LAYER_SPRINGS, 3)
        for sp in node_springs:
            n = struc.nodes.get(sp.node_id)
            if n is None:
                continue
            msp.add_circle(P(n.x, n.y), gd * 0.7,
                           dxfattribs={"layer": LAYER_SPRINGS})
            msp.add_text("spring", height=txt * 0.8,
                         dxfattribs={"layer": LAYER_SPRINGS}
                         ).set_placement(P(n.x + g, n.y - g))
        for es in elem_springs:
            el = struc.bar_elements_by_id.get(es.element_id)
            if el is None:
                continue
            ni = struc.nodes.get(el.node_i)
            nj = struc.nodes.get(el.node_j)
            if ni is None or nj is None:
                continue
            mx, my = (ni.x + nj.x) / 2.0, (ni.y + nj.y) / 2.0
            msp.add_circle(P(mx, my), gd * 0.7,
                           dxfattribs={"layer": LAYER_SPRINGS})
            msp.add_text("spring", height=txt * 0.8,
                         dxfattribs={"layer": LAYER_SPRINGS}
                         ).set_placement(P(mx + g, my - g))

    doc.saveas(path)
