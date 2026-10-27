"""Model statistics — a quantity take-off / bill of materials for a structure.

Pure geometry and mass, independent of any analysis results, so it can be shown
before a solve and reused from scripts or the report exporters. Counts, lengths
(bars), areas (triangles, split CST / Allman), and — for every grouping — the
volume, weight and mass, broken down by element type, material and section.

Object-based models (line / surface objects) are expanded to their generated
mesh first, so the numbers reflect the elements that will actually be solved.

Units (the engine's consistent set): length m, area m², volume m³, weight kN
(γ·V), mass t (ρ·V, with ρ = unit_mass or γ/g, g = 9.81).
"""
from __future__ import annotations

import math

G = 9.81  # m/s² — the same value the solver uses for weight ↔ mass

# Placeholder built-in sections that carry artificial properties — a rigid
# link (huge area/inertia) and a massless dummy. Their elements still show in
# the counts and lengths, and now also in weight/mass: the reserved Rigid/
# Dummy materials (the application's reserved definitions) are pinned to unit_weight=unit_mass=0
# and protected against edits (read-only panels, reconciled back on file
# open), so their weight/mass is always genuinely zero — nothing to exclude.
# Area and volume are different: Rigid's section carries a fictitious
# area_override (100 m² — a stiffness hack, not a real cross-section), which
# would pollute those two totals with a number that means nothing physically.
# Those two quantities alone skip the excluded names. The names mirror the
# application's reserved definitions.
PLACEHOLDER_SECTIONS = frozenset({"Rigid", "Dummy"})


def _unit_mass(mat) -> float:
    """ρ [t/m³]: the material's unit_mass, or γ/g when it was left unset."""
    um = getattr(mat, "unit_mass", None)
    if um is not None:
        return um
    return getattr(mat, "unit_weight", 0.0) / G


def _tri_area(ni, nj, nk) -> float:
    return abs((nj.x - ni.x) * (nk.y - ni.y)
               - (nk.x - ni.x) * (nj.y - ni.y)) / 2.0


def _quad_area(ni, nj, nk, nl) -> float:
    """Shoelace-formula area of the (CCW-wound) quad ni-nj-nk-nl."""
    xs = (ni.x, nj.x, nk.x, nl.x)
    ys = (ni.y, nj.y, nk.y, nl.y)
    s = 0.0
    for a in range(4):
        b = (a + 1) % 4
        s += xs[a] * ys[b] - xs[b] * ys[a]
    return abs(s) / 2.0


def _blank_group() -> dict:
    return {"count": 0, "length": 0.0, "area": 0.0,
            "volume": 0.0, "weight": 0.0, "mass": 0.0}


def _compiled(struc):
    """The mesh to measure: objects expanded, otherwise the model itself."""
    if getattr(struc, "geometry_objects", None):
        from .geo_expand import expand_geometry
        compiled, _ = expand_geometry(struc)
        return compiled
    return struc


def compute_model_stats(struc, exclude_amounts=None) -> dict:
    """Structured statistics over the compiled mesh. See module docstring.

    *exclude_amounts* — section/material names whose elements are counted (and
    their bar length measured) but left OUT of the area / volume totals only —
    weight and mass are always computed from the element's real material
    properties, excluded names included (see PLACEHOLDER_SECTIONS for why that
    is safe for the two names this defaults to). Defaults to the placeholder
    built-ins (Rigid, Dummy)."""
    exclude = (PLACEHOLDER_SECTIONS if exclude_amounts is None
               else frozenset(exclude_amounts))
    s = _compiled(struc)

    total = {
        # Nodes of the compiled mesh (object geometry expanded) — every node
        # that will actually be solved, not just struc.nodes' defining points
        # (see _compiled). Model info never showed this at all; it is usually
        # the first thing anyone wants to know about a model's size.
        "nodes": len(s.nodes),
        # "triangles" is triangles ONLY (CST + Allman) — it used to also pick
        # up every quad (the quads loop below incremented it alongside
        # "quads"), so a mixed tri+quad model reported an inflated triangle
        # count anywhere this dict is read directly, most importantly
        # ai_context.model_stats_tool, which hands "total" straight to the
        # assistant. "area_elements" is the deliberate union (tri + quad) for
        # anywhere that means "any area element", replacing the reliance on
        # the old, overloaded meaning of "triangles".
        "elements": 0, "bars": 0, "triangles": 0, "cst": 0, "allman": 0,
        "quads": 0, "area_elements": 0, "area_quad": 0.0,
        # Elements whose section/material is in *exclude* — kept out of area
        # and volume, shown here so that exclusion is visible rather than a
        # silent gap between "elements" and what the Area/Volume totals add
        # up to.
        "placeholder_elements": 0,
        "length": 0.0, "area": 0.0, "area_cst": 0.0, "area_allman": 0.0,
        "volume": 0.0, "volume_bars": 0.0, "volume_triangles": 0.0,
        "weight": 0.0, "weight_bars": 0.0, "weight_triangles": 0.0,
        "mass": 0.0, "mass_bars": 0.0, "mass_triangles": 0.0,
    }
    by_material: dict = {}
    by_bar_section: dict = {}
    by_tri_section: dict = {}

    def _mat(name):
        return by_material.setdefault(name, _blank_group())

    # ── Bars ────────────────────────────────────────────────────────────────
    for bar in getattr(s, "bar_elements", []):
        sec = s.sections.get(bar.section_name)
        if sec is None:
            continue
        mat = s.materials.get(sec.material_name)
        ni, nj = s.nodes.get(bar.node_i), s.nodes.get(bar.node_j)
        if mat is None or ni is None or nj is None:
            continue
        L = math.hypot(nj.x - ni.x, nj.y - ni.y)
        amt = bar.section_name not in exclude and sec.material_name not in exclude

        total["elements"] += 1; total["bars"] += 1
        total["length"] += L
        g = _mat(sec.material_name)
        g["count"] += 1; g["length"] += L
        b = by_bar_section.setdefault(bar.section_name, _blank_group())
        b["count"] += 1; b["length"] += L
        if not amt:
            total["placeholder_elements"] += 1
        # Weight/mass always computed from the real material (unit_weight,
        # unit_mass) — for Rigid/Dummy that is genuinely 0, so this changes
        # nothing for them; area/volume alone stay behind the *amt* gate,
        # since sec.area can be a fictitious override (see PLACEHOLDER_
        # SECTIONS) that only volume/area would be misled by.
        V = sec.area * L
        W = V * getattr(mat, "unit_weight", 0.0)
        M = V * _unit_mass(mat)
        total["weight"] += W; total["weight_bars"] += W
        total["mass"] += M; total["mass_bars"] += M
        g["weight"] += W; g["mass"] += M
        b["weight"] += W; b["mass"] += M
        if amt:
            total["volume"] += V; total["volume_bars"] += V
            g["volume"] += V
            b["volume"] += V

    # ── Triangles ───────────────────────────────────────────────────────────
    for tri in getattr(s, "tri_elements", []):
        sec = s.tri_sections.get(tri.section_name)
        if sec is None:
            continue
        mat = s.materials.get(sec.material_name)
        ni = s.nodes.get(tri.node_i); nj = s.nodes.get(tri.node_j)
        nk = s.nodes.get(tri.node_k)
        if mat is None or ni is None or nj is None or nk is None:
            continue
        allman = getattr(sec, "formulation", "CST") == "Allman"
        amt = tri.section_name not in exclude and sec.material_name not in exclude

        total["elements"] += 1; total["triangles"] += 1
        total["area_elements"] += 1
        if allman:
            total["allman"] += 1
        else:
            total["cst"] += 1
        g = _mat(sec.material_name)
        g["count"] += 1
        t = by_tri_section.setdefault(tri.section_name, _blank_group())
        t["count"] += 1
        if not amt:
            total["placeholder_elements"] += 1
        A = _tri_area(ni, nj, nk)
        V = A * getattr(sec, "thickness", 0.0)
        W = V * getattr(mat, "unit_weight", 0.0)
        M = V * _unit_mass(mat)
        total["weight"] += W; total["weight_triangles"] += W
        total["mass"] += M; total["mass_triangles"] += M
        g["weight"] += W; g["mass"] += M
        t["weight"] += W; t["mass"] += M
        if amt:
            total["area"] += A
            if allman:
                total["area_allman"] += A
            else:
                total["area_cst"] += A
            total["volume"] += V; total["volume_triangles"] += V
            g["area"] += A; g["volume"] += V
            t["area"] += A; t["volume"] += V

    # ── Quads ───────────────────────────────────────────────────────────────
    for quad in getattr(s, "quad_elements", []):
        sec = s.quad_sections.get(quad.section_name)
        if sec is None:
            continue
        mat = s.materials.get(sec.material_name)
        ni = s.nodes.get(quad.node_i); nj = s.nodes.get(quad.node_j)
        nk = s.nodes.get(quad.node_k); nl = s.nodes.get(quad.node_l)
        if mat is None or ni is None or nj is None or nk is None or nl is None:
            continue
        amt = quad.section_name not in exclude and sec.material_name not in exclude

        total["elements"] += 1; total["area_elements"] += 1
        total["quads"] += 1
        g = _mat(sec.material_name)
        g["count"] += 1
        t = by_tri_section.setdefault(quad.section_name, _blank_group())
        t["count"] += 1
        if not amt:
            total["placeholder_elements"] += 1
        A = _quad_area(ni, nj, nk, nl)
        V = A * getattr(sec, "thickness", 0.0)
        W = V * getattr(mat, "unit_weight", 0.0)
        M = V * _unit_mass(mat)
        total["weight"] += W; total["weight_triangles"] += W
        total["mass"] += M; total["mass_triangles"] += M
        g["weight"] += W; g["mass"] += M
        t["weight"] += W; t["mass"] += M
        if amt:
            total["area"] += A; total["area_quad"] += A
            total["volume"] += V; total["volume_triangles"] += V
            g["area"] += A; g["volume"] += V
            t["area"] += A; t["volume"] += V

    return {
        "total": total,
        "by_material": by_material,
        "by_bar_section": by_bar_section,
        "by_tri_section": by_tri_section,
        "excluded": exclude,
    }


# ── Table rows (Label, Value, Units) ─────────────────────────────────────────
def _hdr(label):
    return (label, None, None)          # a group-header row (no value/units)


def model_stats_rows(struc, exclude_amounts=None) -> list[tuple]:
    """The Model-info table as ``(label, value, units)`` rows. Header rows carry
    ``value = units = None``. Empty groupings are skipped. Placeholder sections
    (Rigid, Dummy) are kept in the counts, lengths, weight and mass — only area
    and volume omit them (see *exclude_amounts* / :func:`compute_model_stats`).
    When any are present, a "Counts" row states how many and that they are
    excluded from area/volume, so the gap between element counts and area/
    volume totals is explained rather than silent."""
    st = compute_model_stats(struc, exclude_amounts)
    t = st["total"]
    mats = st["by_material"]
    barsec = st["by_bar_section"]
    trisec = st["by_tri_section"]
    excl = st["excluded"]
    rows: list[tuple] = []

    # Topology vocabulary follows the model's domain: a line element is a Beam
    # (plane) or Grid member (plate); an area element is a Panel (plane) or Slab
    # (plate); plane triangles are CST / Allman, plate triangles are the plate-
    # bending elements (MITC3 / DKT).
    plate = getattr(struc, "domain", "plane") == "plate"
    line_role = "Grid" if plate else "Beam"
    area_role = "Slab" if plate else "Panel"
    line_pl = line_role if line_role == "Grid" else line_role + "s"   # Grid/Beams
    area_pl = area_role + "s"                                          # Slabs/Panels
    lr, ar = line_role.lower(), area_role.lower()

    # Counts
    rows.append(_hdr("Counts"))
    rows.append(("Total nodes", t["nodes"], "—"))
    rows.append(("Total elements", t["elements"], "—"))
    rows.append((f"{line_role} elements", t["bars"], "—"))
    rows.append((f"{area_role} elements", t["area_elements"], "—"))
    if plate:
        if t["quads"]:
            # t["triangles"] is triangles only now — no more subtracting
            # t["quads"] out of a total that used to (wrongly) include them.
            rows.append((f"  plate-bending {ar}s · triangles",
                         t["triangles"], "—"))
            rows.append((f"  plate-bending {ar}s · quads", t["quads"], "—"))
        else:
            rows.append((f"  plate-bending {ar}s", t["triangles"], "—"))
    else:
        rows.append(("  CST triangles", t["cst"], "—"))
        rows.append(("  Allman triangles", t["allman"], "—"))
        if t["quads"]:
            rows.append(("  Quad elements", t["quads"], "—"))
    for name, g in sorted(mats.items()):
        rows.append((f"Elements · material {name}", g["count"], "—"))
    for name, g in sorted(barsec.items()):
        rows.append((f"Elements · {lr} section {name}", g["count"], "—"))
    for name, g in sorted(trisec.items()):
        rows.append((f"Elements · {ar} section {name}", g["count"], "—"))
    if t["placeholder_elements"]:
        rows.append(("Rigid/Dummy elements (excluded from area/volume)",
                     t["placeholder_elements"], "—"))

    # Lengths (line elements)
    if t["bars"]:
        rows.append(_hdr("Length"))
        rows.append(("Total length", t["length"], "m"))
        for name, g in sorted(barsec.items()):
            rows.append((f"  {lr} section {name}", g["length"], "m"))
        for name, g in sorted(mats.items()):
            if g["length"]:
                rows.append((f"  material {name}", g["length"], "m"))

    # Areas (area elements)
    if t["area_elements"]:
        rows.append(_hdr("Area"))
        rows.append(("Total area", t["area"], "m²"))
        if plate:
            rows.append(("  plate bending", t["area"], "m²"))
        else:
            rows.append(("  CST", t["area_cst"], "m²"))
            rows.append(("  Allman", t["area_allman"], "m²"))
            if t["quads"]:
                rows.append(("  Quads", t["area_quad"], "m²"))
        for name, g in sorted(trisec.items()):
            if name in excl:
                continue
            rows.append((f"  {ar} section {name}", g["area"], "m²"))
        for name, g in sorted(mats.items()):
            if name not in excl and g["area"]:
                rows.append((f"  material {name}", g["area"], "m²"))

    # Volume / Weight / Mass share the same breakdown structure.
    for key, title, unit, tb, tt in (
        ("volume", "Volume", "m³", "volume_bars", "volume_triangles"),
        ("weight", "Weight", "kN", "weight_bars", "weight_triangles"),
        ("mass", "Mass", "t", "mass_bars", "mass_triangles"),
    ):
        rows.append(_hdr(title))
        rows.append((f"Total {title.lower()}", t[key], unit))
        if t["bars"]:
            rows.append((f"  {line_pl.lower()}", t[tb], unit))
        if t["area_elements"]:
            rows.append((f"  {area_pl.lower()}", t[tt], unit))
        # Weight/mass are real for every name, excluded ones included (see
        # compute_model_stats) — only volume still hides them, since that is
        # the one quantity a fictitious section override (Rigid) would
        # falsify.
        skip = excl if key == "volume" else frozenset()
        for name, g in sorted(mats.items()):
            if name not in skip:
                rows.append((f"  material {name}", g[key], unit))
        for name, g in sorted(barsec.items()):
            if name not in skip:
                rows.append((f"  {lr} section {name}", g[key], unit))
        for name, g in sorted(trisec.items()):
            if name not in skip:
                rows.append((f"  {ar} section {name}", g[key], unit))

    # ── Mesh quality ──────────────────────────────────────────────────────
    # Same compiled mesh already used for every section above — the minimum
    # interior angle of every triangle/quad (see model_check.mesh_quality_
    # summary), shown here unconditionally (unlike model_check, which only
    # reports it as an issue below the 15° warning threshold) so Model info
    # always answers "how good is this mesh", not just "is it bad enough to
    # flag". Skipped entirely for a bar-only model (no area elements).
    from .model_check import mesh_quality_summary
    mesh = _compiled(struc)
    mq = mesh_quality_summary(mesh.nodes, getattr(mesh, "tri_elements", []),
                              getattr(mesh, "quad_elements", []))
    if mq is not None:
        rows.append(_hdr("Mesh quality"))
        rows.append(("Worst element", mq["worst_element"], "—"))
        rows.append(("Worst minimum angle", mq["worst_angle_deg"], "°"))
        rows.append(("Elements below 15°", mq["count_below_15deg"], "—"))
        rows.append(("Elements below 5°", mq["count_below_5deg"], "—"))

    return rows
