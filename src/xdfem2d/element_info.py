"""Read-only information about ONE element or geometry object, for the
element dialogs' Properties / Results / Design tabs (no GUI, no Qt).

Three entry points, all taking the model and a *target* ``(kind, id)`` where
kind is ``'bar'`` | ``'area'`` (triangle or quad) | ``'object'`` (a geometry
object — segment, arc, multisegment, rectangle, polygon):

* :func:`property_rows`  — ``[(group, label, text)]`` from the model alone.
* :func:`result_tables`  — ``[(title, header, rows)]`` of the analysis results
  for the target's nodes/elements (needs a results dict).
* :func:`design_tables`  — the same for the design checks: EC2 reinforcement,
  crack width, punching, plus steel/timber utilisation ratios.

Results and design reuse :func:`report_io._results_tables` with its ``scope``
filter, so the numbers are exactly those of the Results/Design panels.
"""
from __future__ import annotations

import math

from .report_io import _results_tables

_RESULT_TITLES = {"Displacements", "Reactions", "Element forces",
                  "Span extremes", "Area forces", "Panel stresses",
                  "Constraint forces", "Spring forces"}


def _fmt(v, unit="", digits=4) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        text = f"{v:.{digits}g}"
    else:
        text = str(v)
    return f"{text} {unit}".strip()


def _poly_area(pts) -> float:
    s = 0.0
    for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]):
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def _poly_centroid(pts):
    a = c = d = 0.0
    for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]):
        cr = x1 * y2 - x2 * y1
        a += cr; c += (x1 + x2) * cr; d += (y1 + y2) * cr
    if abs(a) < 1e-12:
        return (sum(p[0] for p in pts) / len(pts),
                sum(p[1] for p in pts) / len(pts))
    return (c / (3.0 * a), d / (3.0 * a))


def _poly_perimeter(pts) -> float:
    return sum(math.dist(a, b) for a, b in zip(pts, pts[1:] + pts[:1]))


def _xy(struc, nid):
    n = struc.nodes[nid]
    return (n.x, n.y)


def _bar_length(struc, bar) -> float:
    return math.dist(_xy(struc, bar.node_i), _xy(struc, bar.node_j))


def _area_nodes(el) -> list:
    ids = [el.node_i, el.node_j, el.node_k]
    if hasattr(el, "node_l"):
        ids.append(el.node_l)
    return ids


# ── material / section blocks ─────────────────────────────────────────────

def _material_rows(struc, name) -> list:
    m = struc.materials.get(name)
    if m is None:
        return []
    rows = [("Material", "Name", m.name),
            ("Material", "E", _fmt(m.elastic_modulus / 1e6, "GPa")),
            ("Material", "Unit weight γ", _fmt(m.unit_weight, "kN/m³"))]
    if m.poisson is not None:
        rows.append(("Material", "Poisson ν", _fmt(m.poisson)))
    mt = getattr(m.material_type, "value", m.material_type)
    rows.append(("Material", "Type", str(mt)))
    for k, v in (m.design or {}).items():
        if isinstance(v, (int, float, str)) and v != "":
            rows.append(("Material", k, _fmt(v)))
    return rows


def _bar_section_rows(struc, name) -> list:
    s = struc.sections.get(name)
    if s is None:
        return []
    out = [("Section", "Name", s.name),
           ("Section", "Shape", str(getattr(getattr(s, "shape", None),
                                            "value", "") or "—"))]
    props = [("A", "area", "m²"), ("I major (Iy)", "inertia_major", "m⁴"),
             ("I minor (Iz)", "inertia_minor", "m⁴"), ("J (torsion)", "torsion", "m⁴"),
             ("Wel,y", "wel_y", "m³"), ("Wpl,y", "wpl_y", "m³"),
             ("Wel,z", "wel_z", "m³"), ("Wpl,z", "wpl_z", "m³"),
             ("Av,y", "av_y", "m²"), ("Av,z", "av_z", "m²"),
             ("i_y", "radius_gyration_y", "m"), ("i_z", "radius_gyration_z", "m"),
             ("Iw (warping)", "warping", "m⁶"), ("Wt", "torsion_modulus", "m³")]
    for label, attr, unit in props:
        try:
            v = getattr(s, attr)
        except Exception:                      # noqa: BLE001 — not defined for this shape
            continue
        if v is None:
            continue
        out.append(("Section", label, _fmt(float(v), unit)))
    return out


def _area_section(struc, name):
    return struc.tri_sections.get(name) or struc.quad_sections.get(name)


def _area_section_rows(struc, name) -> list:
    s = _area_section(struc, name)
    if s is None:
        return []
    out = [("Section", "Name", s.name),
           ("Section", "Thickness t", _fmt(s.thickness * 1000.0, "mm")),
           ("Section", "Formulation", s.formulation)]
    t = s.thickness
    # Geometric properties of the slab/wall cross-section per metre of width
    # (b = 1 m, depth t) — the analogue of a bar's A, I, W.
    g = "Section (per metre width)"
    out += [(g, "A = t", _fmt(t, "m²/m")),
            (g, "I = t³/12", _fmt(t ** 3 / 12.0, "m⁴/m")),
            (g, "Wel = t²/6", _fmt(t ** 2 / 6.0, "m³/m")),
            (g, "Wpl = t²/4", _fmt(t ** 2 / 4.0, "m³/m")),
            (g, "Av = 5t/6 (shear)", _fmt(5.0 * t / 6.0, "m²/m")),
            (g, "J = t³/3 (torsion)", _fmt(t ** 3 / 3.0, "m⁴/m")),
            (g, "i = t/√12", _fmt(t / math.sqrt(12.0), "m"))]
    m = struc.materials.get(s.material_name)
    if m is not None:
        nu = m.poisson if m.poisson is not None else 0.2
        d = m.elastic_modulus * t ** 3 / (12.0 * (1.0 - nu ** 2))
        out.append(("Section", "Self-weight", _fmt(m.unit_weight * t, "kN/m²")))
        out.append(("Section", "Membrane stiffness E·t",
                    _fmt(m.elastic_modulus * t, "kN/m")))
        out.append(("Section", "Bending stiffness D", _fmt(d, "kNm")))
    return out


_DOF_LABELS = {"plane": ("ux", "uy", "rz"), "plate": ("w", "θx", "θy")}
_FORCE_LABELS = {"plane": ("Fx [kN]", "Fy [kN]", "Mz [kNm]"),
                 "plate": ("Fz [kN]", "Mx [kNm]", "My [kNm]")}


def _spring_force_tables(struc, results, nodes) -> list:
    """Node-spring forces per case, for the springs sitting on *nodes*."""
    if not results or not nodes:
        return []
    rows = []

    def emit(kind, case, sf):
        for nid, v in (sf or {}).items():
            if nid in nodes and isinstance(v, (list, tuple)) and len(v) >= 3:
                rows.append([kind, case, nid, v[0], v[1], v[2]])

    def walk(container, kind):
        for case, cr in (container or {}).items():
            if not isinstance(cr, dict):
                continue
            sf = cr.get("spring_forces")
            if isinstance(sf, dict) and "max" in sf:
                emit(kind, f"{case} (MAX)", sf.get("max"))
                emit(kind, f"{case} (MIN)", sf.get("min"))
            else:
                emit(kind, case, sf)
    walk(results.get("analysis_cases"), "Analysis")
    walk(results.get("combinations"), "Combination")
    if not rows:
        return []
    hdr = ["Case type", "Case", "Node"] + list(
        _FORCE_LABELS.get(getattr(struc, "domain", "plane"),
                          _FORCE_LABELS["plane"]))
    return [("Spring forces", hdr, rows)]


def _support_text(struc, nid) -> str:
    a = next((a for a in struc.support_assignments if a.node_id == nid), None)
    sup = struc.supports.get(a.support_name) if a else None
    if sup is None:
        return "free"
    lab = _DOF_LABELS.get(getattr(struc, "domain", "plane"), _DOF_LABELS["plane"])
    fixed = [l for l, f in zip(lab, (sup.ux, sup.uy, sup.tz)) if f]
    return f"{sup.name} (restrains {', '.join(fixed) or 'nothing'})"


def _node_rows(struc, nid) -> list:
    n = struc.nodes[nid]
    rows = [("Node", "Id", nid), ("Node", "X", _fmt(n.x, "m")),
            ("Node", "Y", _fmt(n.y, "m")),
            ("Support", "Restraint", _support_text(struc, nid))]
    sp = struc.node_springs.get(nid)
    if sp is not None:
        rows += [("Spring", "kx", _fmt(sp.kx, "kN/m")),
                 ("Spring", "ky", _fmt(sp.ky, "kN/m")),
                 ("Spring", "kt", _fmt(sp.kt, "kNm/rad")),
                 ("Spring", "Behaviour (x / y)", f"{sp.mode_x} / {sp.mode_y}")]
    else:
        rows.append(("Spring", "Node spring", "none"))
    bars = [b.id for b in struc.bar_elements if nid in (b.node_i, b.node_j)]
    areas = [e.id for e in list(struc.tri_elements) + list(struc.quad_elements)
             if nid in _area_nodes(e)]
    rows.append(("Connectivity", "Bars", ", ".join(bars) or "none"))
    rows.append(("Connectivity", "Area elements", ", ".join(areas) or "none"))
    cons = [c.id for c in struc.constraints.values()
            if nid in ([t[0] for t in c.terms] + list(c.slaves) + list(c.nodes)
                       + [c.master])]
    if cons:
        rows.append(("Connectivity", "Constraints", ", ".join(cons)))
    masses = [m for m in struc.nodal_masses if m.node_id == nid]
    for m in masses:
        rows.append(("Mass", m.mass_case_id,
                     f"mx={m.mx:g} my={m.my:g} t · Jz={m.mtz:g} t·m²"))
    npl = sum(1 for p in struc.point_loads if p.node_id == nid)
    nst = sum(1 for st in struc.support_settlements
              if st.node_id == nid)
    rows.append(("Loads", "Point loads / settlements", f"{npl} / {nst}"))
    return rows


def _nodes_rows(struc, ids) -> list:
    ids = [i for i in ids if i in struc.nodes]
    if not ids:
        return []
    xs = [struc.nodes[i].x for i in ids]; ys = [struc.nodes[i].y for i in ids]
    by_sup: dict = {}
    for i in ids:
        by_sup[_support_text(struc, i)] = by_sup.get(_support_text(struc, i), 0) + 1
    rows = [("Selection", "Nodes", str(len(ids))),
            ("Selection", "X range", f"{min(xs):.4g} … {max(xs):.4g} m"),
            ("Selection", "Y range", f"{min(ys):.4g} … {max(ys):.4g} m"),
            ("Selection", "With node spring",
             str(sum(1 for i in ids if i in struc.node_springs)))]
    rows += [("By support", k, f"{v} node(s)") for k, v in sorted(by_sup.items())]
    springs = [struc.node_springs[i] for i in ids if i in struc.node_springs]
    if springs:
        rows += [("Springs (sum)", "Σ kx", _fmt(sum(x.kx for x in springs), "kN/m")),
                 ("Springs (sum)", "Σ ky", _fmt(sum(x.ky for x in springs), "kN/m")),
                 ("Springs (sum)", "Σ kt",
                  _fmt(sum(x.kt for x in springs), "kNm/rad"))]
    return rows


# ── properties ────────────────────────────────────────────────────────────

def _bar_rows(struc, bar) -> list:
    rows = [("Element", "Id", bar.id),
            ("Element", "Nodes", f"{bar.node_i} → {bar.node_j}")]
    L = _bar_length(struc, bar)
    rows.append(("Element", "Length", _fmt(L, "m")))
    (x1, y1), (x2, y2) = _xy(struc, bar.node_i), _xy(struc, bar.node_j)
    rows.append(("Element", "Angle",
                 _fmt(math.degrees(math.atan2(y2 - y1, x2 - x1)), "°")))
    s = struc.sections.get(bar.section_name)
    m = struc.materials.get(s.material_name) if s else None
    if s is not None and m is not None:
        try:
            rows.append(("Element", "Self-weight",
                         _fmt(m.unit_weight * s.area * L, "kN")))
            rows.append(("Element", "Self-weight / m",
                         _fmt(m.unit_weight * s.area, "kN/m")))
        except Exception:                      # noqa: BLE001
            pass
    rows += _bar_section_rows(struc, bar.section_name)
    if s is not None:
        rows += _material_rows(struc, s.material_name)
    return rows


def _area_rows(struc, el) -> list:
    pts = [_xy(struc, n) for n in _area_nodes(el)]
    kind = "Quad" if hasattr(el, "node_l") else "Triangle"
    area = _poly_area(pts)
    cx, cy = _poly_centroid(pts)
    sec = _area_section(struc, el.section_name)
    rows = [("Element", "Id", el.id), ("Element", "Type", kind),
            ("Element", "Nodes", " → ".join(_area_nodes(el))),
            ("Element", "Area", _fmt(area, "m²")),
            ("Element", "Perimeter", _fmt(_poly_perimeter(pts), "m")),
            ("Element", "Centroid", f"({cx:.4g}, {cy:.4g}) m")]
    edges = [math.dist(a, b) for a, b in zip(pts, pts[1:] + pts[:1])]
    rows.append(("Element", "Edge lengths (min / max)",
                 f"{min(edges):.4g} / {max(edges):.4g} m"))
    rows.append(("Element", "Aspect ratio (longest/shortest edge)",
                 _fmt(max(edges) / min(edges) if min(edges) > 0 else None)))
    m = struc.materials.get(sec.material_name) if sec else None
    if sec is not None and m is not None:
        rows.append(("Element", "Self-weight",
                     _fmt(m.unit_weight * sec.thickness * area, "kN")))
    rows += _area_section_rows(struc, el.section_name)
    if sec is not None:
        rows += _material_rows(struc, sec.material_name)
    return rows


def _object_rows(struc, obj, compiled, trace) -> list:
    tr = (trace or {}).get(obj.id) or {}
    elems = tr.get("elems", [])
    cls = type(obj).__name__.replace("Geo", "")
    rows = [("Object", "Id", obj.id), ("Object", "Type", cls)]
    pts = struc.object_defining_points(obj)
    surface = hasattr(obj, "tri_section_name")
    if surface:
        poly = (pts if len(pts) != 2 else
                [pts[0], (pts[1][0], pts[0][1]), pts[1], (pts[0][0], pts[1][1])])
        if len(pts) >= 4 and cls == "Rectangle":
            poly = pts[:4]
        if len(poly) >= 3:
            cx, cy = _poly_centroid(poly)
            rows += [("Object", "Area", _fmt(_poly_area(poly), "m²")),
                     ("Object", "Perimeter", _fmt(_poly_perimeter(poly), "m")),
                     ("Object", "Centroid", f"({cx:.4g}, {cy:.4g}) m")]
        rows.append(("Mesh", "Target size", _fmt(obj.target_size, "m")))
        tris = [e for e in elems if e in compiled.tri_elements_by_id]
        quads = [e for e in elems if e in compiled.quad_elements_by_id]
        rows.append(("Mesh", "Triangles / quads", f"{len(tris)} / {len(quads)}"))
        rows.append(("Mesh", "Nodes", str(len(tr.get("nodes", [])))))
        areas = [_poly_area([_xy(compiled, n) for n in _area_nodes(
                 compiled.area_element_by_id(e))]) for e in tris + quads]
        if areas:
            rows.append(("Mesh", "Element area (min / max)",
                         f"{min(areas):.4g} / {max(areas):.4g} m²"))
        rows += _area_section_rows(struc, obj.tri_section_name)
        sec = _area_section(struc, obj.tri_section_name)
        if sec is not None:
            rows += _material_rows(struc, sec.material_name)
            m = struc.materials.get(sec.material_name)
            if m is not None and len(poly) >= 3:
                rows.append(("Object", "Self-weight", _fmt(
                    m.unit_weight * sec.thickness * _poly_area(poly), "kN")))
        return rows
    bars = [compiled.bar_elements_by_id[e] for e in elems
            if e in compiled.bar_elements_by_id]
    total = sum(_bar_length(compiled, b) for b in bars)
    rows += [("Object", "Length", _fmt(total, "m")),
             ("Mesh", "Generated bars", str(len(bars))),
             ("Mesh", "Nodes", str(len(tr.get("nodes", []))))]
    s = struc.sections.get(obj.section_name)
    m = struc.materials.get(s.material_name) if s else None
    if s is not None and m is not None:
        try:
            rows.append(("Object", "Self-weight",
                         _fmt(m.unit_weight * s.area * total, "kN")))
        except Exception:                      # noqa: BLE001
            pass
    rows += _bar_section_rows(struc, obj.section_name)
    if s is not None:
        rows += _material_rows(struc, s.material_name)
    return rows


def _compile(struc):
    from .geo_expand import expand_geometry
    return expand_geometry(struc)


def property_rows(struc, kind: str, eid: str, trace=None, compiled=None) -> list:
    """``[(group, label, text)]`` describing the target; ``[]`` if not found."""
    if kind == "node":
        return _node_rows(struc, eid) if eid in struc.nodes else []
    if kind == "bar":
        b = struc.bar_elements_by_id.get(eid)
        return _bar_rows(struc, b) if b else []
    if kind == "area":
        e = struc.area_element_by_id(eid)
        return _area_rows(struc, e) if e else []
    obj = struc.geometry_objects.get(eid)
    if obj is None:
        return []
    if compiled is None or trace is None:
        compiled, trace = _compile(struc)
    return _object_rows(struc, obj, compiled, trace)


# ── results / design ──────────────────────────────────────────────────────

def scope_for(struc, kind: str, eid: str, trace=None, compiled=None):
    """The ``_results_tables`` scope (nodes/bars/areas id sets) for the target,
    or ``None`` when the target does not exist."""
    if kind == "node":
        return ({"nodes": {eid}, "bars": set(), "areas": set()}
                if eid in struc.nodes else None)
    if kind == "bar":
        b = struc.bar_elements_by_id.get(eid)
        return None if b is None else {
            "nodes": {b.node_i, b.node_j}, "bars": {eid}, "areas": set()}
    if kind == "area":
        e = struc.area_element_by_id(eid)
        return None if e is None else {
            "nodes": set(_area_nodes(e)), "bars": set(), "areas": {eid}}
    if struc.geometry_objects.get(eid) is None:
        return None
    if trace is None:
        _compiled, trace = _compile(struc)
    tr = trace.get(eid) or {}
    ids = set(tr.get("elems", []))
    return {"nodes": set(tr.get("nodes", [])), "bars": set(ids),
            "areas": set(ids)}


def result_tables(struc, results, kind: str, eid: str, trace=None) -> list:
    """Analysis results restricted to the target: ``[(title, header, rows)]``."""
    if not results:
        return []
    scope = scope_for(struc, kind, eid, trace)
    if scope is None:
        return []
    tabs = _results_tables(results, getattr(struc, "domain", "plane"), scope)
    out = [t for t in tabs if t[0] in _RESULT_TITLES]
    return out + _spring_force_tables(struc, results, scope["nodes"])


def _ratio_table(title, block, ids) -> list:
    els = (block or {}).get("elements") or {}
    rows = []
    for eid in sorted(ids):
        r = els.get(eid)
        if not isinstance(r, dict):
            continue
        rows.append([eid] + [r.get(k) for k in
                             ("bending", "shear", "torsion", "buckling",
                              "combined")])
    if not rows:
        return []
    return [(title, ["Element", "Bending", "Shear", "Torsion", "Buckling",
                     "Combined (max)"], rows)]


def design_tables(struc, results, kind: str, eid: str, trace=None) -> list:
    """Design checks restricted to the target: EC2 reinforcement / crack width /
    punching tables plus steel and timber utilisation ratios."""
    if not results:
        return []
    scope = scope_for(struc, kind, eid, trace)
    if scope is None:
        return []
    tabs = _results_tables(results, getattr(struc, "domain", "plane"), scope)
    out = [t for t in tabs
           if t[0] not in _RESULT_TITLES and t[0] != "Design parameters"]
    ids = scope["bars"]
    out += _ratio_table("Steel utilisation (EC3)",
                        results.get("steel_design"), ids)
    out += _ratio_table("Timber utilisation (EC5)",
                        results.get("timber_design"), ids)
    return out


# ── several elements at once (bulk-edit dialogs) ──────────────────────────
#
# A selection is summarised, not listed: totals and "varies" for properties, a
# per-case envelope for results, the worst few for the design checks. The full
# tables (the same ones a single element gets, filtered to the whole selection)
# come from :func:`result_tables_many` / :func:`design_tables_many`.

TOP_N = 20            # rows kept in a design summary
UTIL_LIMIT = 1.0      # utilisation above this counts as failing


def scope_for_many(struc, kind: str, ids) -> dict:
    """Scope covering every bar (``kind='bar'``) or area element of *ids*."""
    ids = list(ids)
    nodes = set()
    if kind == "node":
        return {"nodes": {i for i in ids if i in struc.nodes},
                "bars": set(), "areas": set()}
    if kind == "bar":
        for i in ids:
            b = struc.bar_elements_by_id.get(i)
            if b is not None:
                nodes.update((b.node_i, b.node_j))
        return {"nodes": nodes, "bars": set(ids), "areas": set()}
    for i in ids:
        e = struc.area_element_by_id(i)
        if e is not None:
            nodes.update(_area_nodes(e))
    return {"nodes": nodes, "bars": set(), "areas": set(ids)}


def _group_rows(group, items) -> list:
    return [(group, label, text) for label, text in items]


def property_rows_many(struc, kind: str, ids) -> list:
    """``[(group, label, text)]`` summarising several nodes, bars or areas."""
    ids = list(ids)
    if kind == "node":
        return _nodes_rows(struc, ids)
    if kind == "bar":
        els = [struc.bar_elements_by_id[i] for i in ids
               if i in struc.bar_elements_by_id]
        if not els:
            return []
        by_sec: dict = {}
        total_len = total_w = 0.0
        for b in els:
            L = _bar_length(struc, b)
            s = struc.sections.get(b.section_name)
            m = struc.materials.get(s.material_name) if s else None
            w = m.unit_weight * s.area * L if (s and m) else 0.0
            g = by_sec.setdefault(b.section_name, [0, 0.0, 0.0])
            g[0] += 1; g[1] += L; g[2] += w
            total_len += L; total_w += w
        rows = _group_rows("Selection", [
            ("Bars", str(len(els))), ("Total length", _fmt(total_len, "m")),
            ("Total self-weight", _fmt(total_w, "kN"))])
        rows += _group_rows("By section", [
            (name, f"{n} bars · {L:.4g} m · {w:.4g} kN")
            for name, (n, L, w) in sorted(by_sec.items())])
        if len(by_sec) == 1:
            name = next(iter(by_sec))
            rows += _bar_section_rows(struc, name)
            s = struc.sections.get(name)
            if s is not None:
                rows += _material_rows(struc, s.material_name)
        else:
            rows.append(("Section", "Properties",
                         "— (mixed sections: see 'By section')"))
        return rows
    els = [struc.area_element_by_id(i) for i in ids]
    els = [e for e in els if e is not None]
    if not els:
        return []
    by_sec = {}
    total_a = total_w = 0.0
    aspects = []
    for e in els:
        pts = [_xy(struc, n) for n in _area_nodes(e)]
        a = _poly_area(pts)
        edges = [math.dist(p, q) for p, q in zip(pts, pts[1:] + pts[:1])]
        if min(edges) > 0:
            aspects.append(max(edges) / min(edges))
        sec = _area_section(struc, e.section_name)
        m = struc.materials.get(sec.material_name) if sec else None
        w = m.unit_weight * sec.thickness * a if (sec and m) else 0.0
        g = by_sec.setdefault(e.section_name, [0, 0.0, 0.0])
        g[0] += 1; g[1] += a; g[2] += w
        total_a += a; total_w += w
    ntri = sum(1 for e in els if not hasattr(e, "node_l"))
    rows = _group_rows("Selection", [
        ("Elements", f"{len(els)} ({ntri} triangles, {len(els) - ntri} quads)"),
        ("Total area", _fmt(total_a, "m²")),
        ("Total self-weight", _fmt(total_w, "kN"))])
    if aspects:
        rows.append(("Selection", "Aspect ratio (min / max)",
                     f"{min(aspects):.4g} / {max(aspects):.4g}"))
    rows += _group_rows("By section", [
        (name, f"{n} elements · {a:.4g} m² · {w:.4g} kN")
        for name, (n, a, w) in sorted(by_sec.items())])
    if len(by_sec) == 1:
        name = next(iter(by_sec))
        rows += _area_section_rows(struc, name)
        sec = _area_section(struc, name)
        if sec is not None:
            rows += _material_rows(struc, sec.material_name)
    else:
        rows.append(("Section", "Properties",
                     "— (mixed sections: see 'By section')"))
    return rows


def result_tables_many(struc, results, kind: str, ids) -> list:
    """Full results tables restricted to the whole selection (the detail view)."""
    if not results:
        return []
    scope = scope_for_many(struc, kind, ids)
    tabs = _results_tables(results, getattr(struc, "domain", "plane"), scope)
    out = [t for t in tabs if t[0] in _RESULT_TITLES]
    return out + _spring_force_tables(struc, results, scope["nodes"])


def design_tables_many(struc, results, kind: str, ids) -> list:
    """Full design tables restricted to the whole selection (the detail view)."""
    if not results:
        return []
    scope = scope_for_many(struc, kind, ids)
    tabs = _results_tables(results, getattr(struc, "domain", "plane"), scope)
    out = [t for t in tabs
           if t[0] not in _RESULT_TITLES and t[0] != "Design parameters"]
    out += _ratio_table("Steel utilisation (EC3)",
                        results.get("steel_design"), scope["bars"])
    out += _ratio_table("Timber utilisation (EC5)",
                        results.get("timber_design"), scope["bars"])
    return out


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def result_summary(struc, results, kind: str, ids) -> list:
    """Per-case envelope of the selection: min / max of every numeric column and
    the element where the governing column peaks. ``[(title, header, rows)]``."""
    out = []
    for title, header, rows in result_tables_many(struc, results, kind, ids):
        if title not in ("Element forces", "Area forces", "Panel stresses",
                         "Displacements", "Reactions", "Spring forces"):
            continue
        rows = [r for r in rows if not str(r[2]).startswith("Σ")]
        eid_col = header.index("Element") if "Element" in header else (
            header.index("Area") if "Area" in header else 2)
        num_cols = [c for c in range(3, len(header))
                    if any(_is_num(r[c]) for r in rows)]
        if not num_cols:
            continue
        if title == "Element forces":
            gov = num_cols[-1]
        elif kind == "node":       # the component with the largest magnitude
            gov = max(num_cols, key=lambda c: max(
                (abs(r[c]) for r in rows if _is_num(r[c])), default=0.0))
        else:
            gov = num_cols[0]
        groups: dict = {}
        for r in rows:
            groups.setdefault((r[0], r[1]), []).append(r)
        env_rows = []
        for (ctype, case), rs in groups.items():
            row = [ctype, case]
            for c in num_cols:
                vals = [r[c] for r in rs if _is_num(r[c])]
                row += [min(vals), max(vals)] if vals else [None, None]
            peak = max((r for r in rs if _is_num(r[gov])),
                       key=lambda r: abs(r[gov]), default=None)
            row.append(peak[eid_col] if peak else None)
            env_rows.append(row)
        hdr = ["Case type", "Case"]
        for c in num_cols:
            hdr += [f"{header[c]} min", f"{header[c]} max"]
        hdr.append(f"Peak |{header[gov]}| at")
        out.append((f"{title} — envelope of the selection", hdr, env_rows))
    return out


def design_summary(struc, results, kind: str, ids) -> list:
    """The worst of the selection: utilisation ratios sorted high→low (count of
    those above :data:`UTIL_LIMIT` in the title) and the governing rows of the
    concrete / crack / punching tables, each capped at :data:`TOP_N`."""
    out = []
    for title, header, rows in design_tables_many(struc, results, kind, ids):
        if title.endswith("utilisation (EC3)") or title.endswith("utilisation (EC5)"):
            rows = sorted(rows, key=lambda r: -(r[-1] if _is_num(r[-1]) else -1))
            fail = sum(1 for r in rows if _is_num(r[-1]) and r[-1] > UTIL_LIMIT)
            title = f"{title} — {fail} of {len(rows)} above {UTIL_LIMIT:g}"
        else:
            gov = [r for r in rows if r and r[-1] == "★"]
            rows = gov or rows
        shown = rows[:TOP_N]
        if len(shown) < len(rows):
            title += f" (worst {TOP_N} of {len(rows)})"
        out.append((title, header, shown))
    return out
