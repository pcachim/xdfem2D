"""
Import / reconciliation of an independent model into a shared entity space
(Phase 2 — the common foundation for both variants and construction phases).

Two saved models have independent id namespaces. ``import_model`` folds a second
model ``other`` into the id space of ``base``, welding geometrically coincident
nodes (so the same physical point gets one shared id) and reconciling section /
material definitions. The imported model is returned as an overlay (a
:class:`~xdfem2d.models.Variant` in this phase) over the unified structure, so it
can be combined with the base through the Phase-1 machinery.

All functions here are **pure**: ``base`` and ``other`` are never mutated; a new
unified :class:`~xdfem2d.structure.Structure2D` is returned.

Identity policy (the welding is by *coordinate*, not by id — geometry is the true
physical identity):
  - nodes      : welded when within ``tol`` of a base node; else added (renamed
                 on id collision with a distinct base node);
  - sections / : shared when name+properties match; the imported one is renamed
    materials    on a name clash with different properties; new names imported;
  - elements   : the *same* element when it joins the same welded node pair with
                 the same section; otherwise added as a distinct element.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class WeldReport:
    """Audit of an import: what was welded, added, renamed, or left ambiguous."""
    welded: list = field(default_factory=list)        # (other_id, base_id)
    added_nodes: list = field(default_factory=list)
    renamed_defs: dict = field(default_factory=dict)  # other_name -> new_name
    new_elements: list = field(default_factory=list)
    ambiguities: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.ambiguities


# ---------------------------------------------------------------------------
# Node welding
# ---------------------------------------------------------------------------

def weld_nodes(base, other, tol: float = 1e-6) -> tuple:
    """Map ``other``'s node ids onto ``base``'s id space by coordinate proximity.

    Returns ``(node_map, report_bits)`` where node_map is
    ``{other_node_id -> shared_node_id}`` and report_bits carries welded/added/
    ambiguity lists.
    """
    node_map: dict = {}
    welded: list = []
    added: list = []
    ambiguities: list = []

    base_items = list(base.nodes.items())

    def _near(x, y):
        hits = [bid for bid, bn in base_items
                if math.hypot(bn.x - x, bn.y - y) <= tol]
        return hits

    for oid, on in other.nodes.items():
        hits = _near(on.x, on.y)
        if len(hits) > 1:
            ambiguities.append(
                f"node '{oid}' matches {len(hits)} base nodes within tol={tol}")
            node_map[oid] = hits[0]      # provisional; caller must resolve
            welded.append((oid, hits[0]))
        elif len(hits) == 1:
            node_map[oid] = hits[0]
            welded.append((oid, hits[0]))
        else:
            # New node: keep its id unless it clashes with a *different* base node.
            new_id = oid
            if new_id in base.nodes:
                new_id = _unique_id(oid, set(base.nodes) | set(node_map.values()))
            node_map[oid] = new_id
            added.append(new_id)

    return node_map, {'welded': welded, 'added_nodes': added,
                      'ambiguities': ambiguities}


def _unique_id(stem: str, taken: set) -> str:
    i = 2
    while f"{stem}@{i}" in taken:
        i += 1
    return f"{stem}@{i}"


# ---------------------------------------------------------------------------
# Definition reconciliation (materials, sections)
# ---------------------------------------------------------------------------

def _material_props(m) -> tuple:
    return (round(getattr(m, 'elastic_modulus', 0.0), 9),
            round(getattr(m, 'unit_weight', 0.0), 9),
            round(getattr(m, 'alpha', 0.0), 12))


def _section_props(s) -> tuple:
    return (s.material_name, getattr(s, 'b', None), getattr(s, 'h', None),
            getattr(s, 'area_override', None), getattr(s, 'inertia_override', None))


def reconcile_definitions(base, other) -> tuple:
    """Return ``(mat_map, sec_map, renamed)`` mapping ``other``'s material/section
    names to the names to use in the unified structure."""
    mat_map: dict = {}
    sec_map: dict = {}
    renamed: dict = {}

    for name, m in other.materials.items():
        bm = base.materials.get(name)
        if bm is None:
            mat_map[name] = name
        elif _material_props(bm) == _material_props(m):
            mat_map[name] = name
        else:
            new = _unique_id(name, set(base.materials) | set(mat_map.values()))
            mat_map[name] = new
            renamed[name] = new

    for name, s in other.sections.items():
        bs = base.sections.get(name)
        # Compare section props but account for a possibly-renamed material.
        s_props = (mat_map.get(s.material_name, s.material_name),) + _section_props(s)[1:]
        if bs is None:
            sec_map[name] = name
        elif (bs.material_name,) + _section_props(bs)[1:] == s_props:
            sec_map[name] = name
        else:
            new = _unique_id(name, set(base.sections) | set(sec_map.values()))
            sec_map[name] = new
            renamed[name] = new

    return mat_map, sec_map, renamed


# ---------------------------------------------------------------------------
# Element mapping
# ---------------------------------------------------------------------------

def map_elements(base, other, node_map: dict, sec_map: dict) -> tuple:
    """Map ``other``'s element ids onto the unified id space.

    An imported element matches a base element when it joins the same welded node
    pair (unordered) with the same (reconciled) section; otherwise it is new.
    Returns ``(elem_map, new_elements)``.
    """
    elem_map: dict = {}
    new_elements: list = []

    base_by_conn = {}
    for e in base.bar_elements:
        base_by_conn[(frozenset((e.node_i, e.node_j)), e.section_name)] = e.id

    taken = set(base.bar_elements_by_id) | set(elem_map.values())
    for e in other.bar_elements:
        ni, nj = node_map.get(e.node_i, e.node_i), node_map.get(e.node_j, e.node_j)
        sec = sec_map.get(e.section_name, e.section_name)
        key = (frozenset((ni, nj)), sec)
        match = base_by_conn.get(key)
        if match is not None:
            elem_map[e.id] = match
        else:
            new_id = e.id
            if new_id in taken:
                new_id = _unique_id(e.id, taken)
            elem_map[e.id] = new_id
            taken.add(new_id)
            new_elements.append(new_id)

    return elem_map, new_elements


# ---------------------------------------------------------------------------
# Area (tri/quad) definitions, elements, and geometry objects
#
# The functions above (reconcile_definitions/map_elements) only ever covered
# bar materials/sections/elements -- a model built from tri/quad elements, or
# from geometry objects (GeoRectangle/GeoPolygon/GeoSegment/GeoArc/
# GeoMultisegment, which mesh into bars/tris/quads at solve time but are
# themselves defined only by node_ids, no coordinates of their own), silently
# lost everything but its bare node/support data on import_extend. These
# close that gap, following the exact same match-by-geometry-else-add pattern.
# ---------------------------------------------------------------------------

def _area_section_props(s) -> tuple:
    return (s.material_name, round(getattr(s, 'thickness', 0.0), 9),
            getattr(s, 'formulation', ''), bool(getattr(s, 'plane_strain', False)))


def reconcile_area_definitions(base, other) -> tuple:
    """tri_sections/quad_sections counterpart of reconcile_definitions().
    Returns ``(tri_sec_map, quad_sec_map, renamed)``."""
    tri_map: dict = {}
    quad_map: dict = {}
    renamed: dict = {}

    for name, s in other.tri_sections.items():
        bs = base.tri_sections.get(name)
        if bs is None:
            tri_map[name] = name
        elif _area_section_props(bs) == _area_section_props(s):
            tri_map[name] = name
        else:
            new = _unique_id(name, set(base.tri_sections) | set(tri_map.values()))
            tri_map[name] = new
            renamed[name] = new

    for name, s in other.quad_sections.items():
        bs = base.quad_sections.get(name)
        if bs is None:
            quad_map[name] = name
        elif _area_section_props(bs) == _area_section_props(s):
            quad_map[name] = name
        else:
            new = _unique_id(name, set(base.quad_sections) | set(quad_map.values()))
            quad_map[name] = new
            renamed[name] = new

    return tri_map, quad_map, renamed


def map_tri_elements(base, other, node_map: dict, tri_sec_map: dict) -> tuple:
    """The :func:`map_elements` counterpart for triangles: a triangle joining
    the same welded three nodes with the same (reconciled) section is the
    same triangle; otherwise it's new. Returns ``(elem_map, new_elements)``."""
    elem_map: dict = {}
    new_elements: list = []
    base_by_conn = {}
    for t in base.tri_elements:
        base_by_conn[(frozenset((t.node_i, t.node_j, t.node_k)),
                      t.section_name)] = t.id
    taken = set(base.tri_elements_by_id) | set(elem_map.values())
    for t in other.tri_elements:
        ni = node_map.get(t.node_i, t.node_i)
        nj = node_map.get(t.node_j, t.node_j)
        nk = node_map.get(t.node_k, t.node_k)
        sec = tri_sec_map.get(t.section_name, t.section_name)
        match = base_by_conn.get((frozenset((ni, nj, nk)), sec))
        if match is not None:
            elem_map[t.id] = match
        else:
            new_id = t.id
            if new_id in taken:
                new_id = _unique_id(t.id, taken)
            elem_map[t.id] = new_id
            taken.add(new_id)
            new_elements.append(new_id)
    return elem_map, new_elements


def map_quad_elements(base, other, node_map: dict, quad_sec_map: dict) -> tuple:
    """The four-node analogue of :func:`map_tri_elements`."""
    elem_map: dict = {}
    new_elements: list = []
    base_by_conn = {}
    for q in base.quad_elements:
        base_by_conn[(frozenset((q.node_i, q.node_j, q.node_k, q.node_l)),
                      q.section_name)] = q.id
    taken = set(base.quad_elements_by_id) | set(elem_map.values())
    for q in other.quad_elements:
        ni = node_map.get(q.node_i, q.node_i)
        nj = node_map.get(q.node_j, q.node_j)
        nk = node_map.get(q.node_k, q.node_k)
        nl = node_map.get(q.node_l, q.node_l)
        sec = quad_sec_map.get(q.section_name, q.section_name)
        match = base_by_conn.get((frozenset((ni, nj, nk, nl)), sec))
        if match is not None:
            elem_map[q.id] = match
        else:
            new_id = q.id
            if new_id in taken:
                new_id = _unique_id(q.id, taken)
            elem_map[q.id] = new_id
            taken.add(new_id)
            new_elements.append(new_id)
    return elem_map, new_elements


def map_geometry_objects(base, other, node_map: dict) -> dict:
    """Map ``other``'s geometry-object ids onto a unified id space. Unlike
    elements, objects are never matched against an existing one (there is no
    reliable "same object" test for a curve/surface definition) -- each is
    simply kept, renamed only on an id clash. Returns ``{other_id ->
    unified_id}``; the caller still has to deep-copy each object with its
    ``node_ids`` remapped through *node_map* (see import_extend)."""
    obj_map: dict = {}
    taken = set(base.geometry_objects)
    for oid in other.geometry_objects:
        new_id = oid
        if new_id in taken:
            new_id = _unique_id(oid, taken)
        obj_map[oid] = new_id
        taken.add(new_id)
    return obj_map


# ---------------------------------------------------------------------------
# Top-level import
# ---------------------------------------------------------------------------

def import_extend(base, other, weld: str = "coords", tol: float = 1e-6,
                  node_map: Optional[dict] = None):
    """Extend *base* with *other*: a new, larger Structure2D.

    Conceptually this is an **extension** of the base model, not a variant:
    coincident nodes are welded, and the imported nodes / bars / sections /
    materials / supports / loads are added to the base. Same-named load cases are
    shared (merged), so the imported dead load adds to the base's 'G', etc.

    Returns ``(unified, report)``. ``base`` and ``other`` are not mutated.
    (Variants are *sub-models* of the base and are defined in the workspace; an
    import that adds geometry is an extension and creates no variant.)
    """
    from .structure import Structure2D
    if isinstance(other, str):
        other = Structure2D.load(other)

    # A "plane" model's 3 DOFs/node (ux, uy, θz) and a "plate" model's
    # (w, θx, θy) are numerically the same shape but physically different
    # quantities -- welding one onto the other would silently mix them into
    # a meaningless combined model (a plane bar's axial force landing where
    # a plate bar's torsion is expected, etc.). Refuse instead, the same way
    # an incompatible weld= value is refused below.
    if getattr(other, "domain", "plane") != getattr(base, "domain", "plane"):
        raise ValueError(
            f"Cannot merge a '{other.domain}' model into a '{base.domain}' "
            "model — the two domains use the same DOF layout for physically "
            "different quantities.")

    unified = copy.deepcopy(base)
    report = WeldReport()

    # 1) Node welding.
    if weld == "coords":
        nmap, bits = weld_nodes(base, other, tol)
    elif weld == "name":
        nmap = {nid: nid for nid in other.nodes}
        bits = {'welded': [(n, n) for n in other.nodes if n in base.nodes],
                'added_nodes': [n for n in other.nodes if n not in base.nodes],
                'ambiguities': []}
    elif weld == "manual":
        if node_map is None:
            raise ValueError("weld='manual' requires node_map")
        nmap = dict(node_map)
        bits = {'welded': [(o, b) for o, b in nmap.items() if b in base.nodes],
                'added_nodes': [b for b in nmap.values() if b not in base.nodes],
                'ambiguities': []}
    else:
        raise ValueError("weld must be 'coords', 'name' or 'manual'")
    report.welded = bits['welded']
    report.added_nodes = bits['added_nodes']
    report.ambiguities = bits['ambiguities']

    # 2) Definitions and 3) elements (bars, then tri/quad).
    mat_map, sec_map, renamed = reconcile_definitions(base, other)
    tri_sec_map, quad_sec_map, area_renamed = reconcile_area_definitions(base, other)
    renamed.update(area_renamed)
    report.renamed_defs = renamed
    emap, new_elems = map_elements(base, other, nmap, sec_map)
    tri_emap, new_tris = map_tri_elements(base, other, nmap, tri_sec_map)
    quad_emap, new_quads = map_quad_elements(base, other, nmap, quad_sec_map)
    report.new_elements = new_elems + new_tris + new_quads
    obj_map = map_geometry_objects(base, other, nmap)

    # 4) New nodes.
    for oid, on in other.nodes.items():
        sid = nmap[oid]
        if sid not in unified.nodes:
            unified.add_node(sid, on.x, on.y)

    # 5) New materials / sections.
    for name, m in other.materials.items():
        nn = mat_map[name]
        if nn not in unified.materials:
            # _add_material_unchecked: *m* comes from another, already-open
            # model -- it was valid there (or, loaded permissively, never
            # checked at all), so merging it in is not the moment to start
            # enforcing add_material's class check (26/09/2026, same
            # reasoning as structure_io's plain loader).
            unified._add_material_unchecked(
                nn, m.elastic_modulus, m.unit_weight,
                alpha=getattr(m, 'alpha', 1e-5),
                unit_mass=getattr(m, 'unit_mass', None),
                material_type=getattr(m.material_type, 'value', None),
                design=dict(getattr(m, 'design', {}) or {}))
    for name, s in other.sections.items():
        nn = sec_map[name]
        if nn not in unified.sections:
            unified.add_section(nn, mat_map.get(s.material_name, s.material_name),
                                s.b, s.h,
                                area_override=s.area_override,
                                inertia_override=s.inertia_override,
                                profile_name=getattr(s, 'profile_name', None),
                                shape=getattr(s.shape, 'value', None),
                                tw=getattr(s, 'tw', 0.0), tf=getattr(s, 'tf', 0.0))
    for name, s in other.tri_sections.items():
        nn = tri_sec_map[name]
        if nn not in unified.tri_sections:
            unified.add_tri_section(
                nn, mat_map.get(s.material_name, s.material_name),
                thickness=s.thickness, plane_strain=s.plane_strain,
                formulation=s.formulation,
                rc_cover=s.rc_cover, rc_alpha_s=s.rc_alpha_s,
                rc_bar_phi=s.rc_bar_phi, rc_shear_min=s.rc_shear_min,
                rc_cover_top_x=s.rc_cover_top_x, rc_cover_top_y=s.rc_cover_top_y,
                rc_cover_bot_x=s.rc_cover_bot_x, rc_cover_bot_y=s.rc_cover_bot_y)
    for name, s in other.quad_sections.items():
        nn = quad_sec_map[name]
        if nn not in unified.quad_sections:
            unified.add_quad_section(
                nn, mat_map.get(s.material_name, s.material_name),
                thickness=s.thickness, plane_strain=s.plane_strain,
                formulation=s.formulation,
                rc_cover=s.rc_cover, rc_alpha_s=s.rc_alpha_s,
                rc_bar_phi=s.rc_bar_phi, rc_shear_min=s.rc_shear_min,
                rc_cover_top_x=s.rc_cover_top_x, rc_cover_top_y=s.rc_cover_top_y,
                rc_cover_bot_x=s.rc_cover_bot_x, rc_cover_bot_y=s.rc_cover_bot_y)

    # 6) New elements (bars, then tri/quad).
    for e in other.bar_elements:
        new_id = emap[e.id]
        if new_id not in unified.bar_elements_by_id:
            unified.add_bar_element(
                new_id, nmap.get(e.node_i, e.node_i), nmap.get(e.node_j, e.node_j),
                sec_map.get(e.section_name, e.section_name),
                hinge_i=e.hinge_i, hinge_j=e.hinge_j)
    for t in other.tri_elements:
        new_id = tri_emap[t.id]
        if new_id not in unified.tri_elements_by_id:
            unified.add_tri_element(
                new_id, nmap.get(t.node_i, t.node_i), nmap.get(t.node_j, t.node_j),
                nmap.get(t.node_k, t.node_k),
                tri_sec_map.get(t.section_name, t.section_name))
    for q in other.quad_elements:
        new_id = quad_emap[q.id]
        if new_id not in unified.quad_elements_by_id:
            unified.add_quad_element(
                new_id, nmap.get(q.node_i, q.node_i), nmap.get(q.node_j, q.node_j),
                nmap.get(q.node_k, q.node_k), nmap.get(q.node_l, q.node_l),
                quad_sec_map.get(q.section_name, q.section_name))

    # 6b) Geometry objects — no coordinates of their own (only node_ids, into
    # nodes already added above), so a deep copy with ids remapped through
    # nmap/obj_map is enough; each keeps its own materialize/mesh-at-solve
    # behaviour untouched.
    for oid, obj in other.geometry_objects.items():
        new_oid = obj_map[oid]
        if new_oid in unified.geometry_objects:
            continue
        new_obj = copy.deepcopy(obj)
        new_obj.id = new_oid
        new_obj.node_ids = [nmap.get(n, n) for n in getattr(obj, "node_ids", [])]
        unified.geometry_objects[new_oid] = new_obj

    # 7) Merge supports — add for nodes not already supported in the base.
    supported = {a.node_id for a in unified.support_assignments}
    for sname, sp in other.supports.items():
        if sname not in unified.supports:
            unified.add_support(sname, ux=sp.ux, uy=sp.uy, tz=sp.tz)
    for a in other.support_assignments:
        nid = nmap.get(a.node_id, a.node_id)
        if nid not in supported:
            unified.assign_support(nid, a.support_name)
            supported.add(nid)

    # 8) Merge loads — same-named load cases are shared.
    for lc in other.load_cases:
        if lc.id not in unified.load_cases_by_id:
            unified.add_load_case(lc.id, self_weight_factor=lc.self_weight_factor,
                                  action_type=getattr(lc.action_type, 'value', None))
    for pl in other.point_loads:
        unified.add_point_load(nmap.get(pl.node_id, pl.node_id), pl.load_case_id,
                               fx=pl.fx, fy=pl.fy, mz=pl.mz)
    for dl in other.distributed_loads:
        unified.add_distributed_load(emap.get(dl.element_id, dl.element_id),
                                     dl.load_case_id, fxe=dl.fxe, fxd=dl.fxd,
                                     fye=dl.fye, fyd=dl.fyd, coord_sys=dl.coord_sys)
    for epl in getattr(other, 'element_point_loads', []):
        unified.add_element_point_load(emap.get(epl.element_id, epl.element_id),
                                       epl.load_case_id, a=epl.a, fx=epl.fx,
                                       fy=epl.fy, mz=epl.mz, coord_sys=epl.coord_sys)
    return unified, report


def import_model(base, other, overlay_id: str,
                 mode: str = "variant",
                 weld: str = "coords", tol: float = 1e-6,
                 node_map: Optional[dict] = None):
    """Fold ``other`` into ``base``'s id space and return
    ``(unified, overlay, support_set, report)``.

    unified     : new Structure2D = base ∪ other (shared ids).
    overlay     : a Variant (mode='variant') describing the imported model
                  (active elements + ``support_set_id == overlay_id``).
    support_set : the SupportSet to pass to ``solve_variants`` for this overlay,
                  i.e. ``solve_variants(unified, [overlay], {overlay_id: support_set})``.
    report      : WeldReport.

    ``base`` and ``other`` are not mutated. ``mode='phase'`` is reserved for
    Phase 3 (construction phases) and raises NotImplementedError here.
    """
    from .structure import Structure2D
    from .models import Variant, SupportSet, SupportAssignment

    if mode not in ("variant", "phase"):
        raise ValueError("mode must be 'variant' or 'phase'")
    if mode == "phase":
        raise NotImplementedError(
            "mode='phase' (construction phases) lands in Phase 3.")

    if isinstance(other, str):
        other = Structure2D.load(other)

    # See the matching check in import_extend: a plane and a plate model use
    # the same 3-DOF/node layout for physically different quantities, so
    # welding one onto the other would silently produce a meaningless model.
    if getattr(other, "domain", "plane") != getattr(base, "domain", "plane"):
        raise ValueError(
            f"Cannot import a '{other.domain}' model into a '{base.domain}' "
            "model — the two domains use the same DOF layout for physically "
            "different quantities.")

    unified = copy.deepcopy(base)
    report = WeldReport()

    # 1) Node welding.
    if weld == "coords":
        nmap, bits = weld_nodes(base, other, tol)
    elif weld == "name":
        nmap = {nid: nid for nid in other.nodes}
        bits = {'welded': [(n, n) for n in other.nodes if n in base.nodes],
                'added_nodes': [n for n in other.nodes if n not in base.nodes],
                'ambiguities': []}
    elif weld == "manual":
        if node_map is None:
            raise ValueError("weld='manual' requires node_map")
        nmap = dict(node_map)
        bits = {'welded': [(o, b) for o, b in nmap.items() if b in base.nodes],
                'added_nodes': [b for b in nmap.values() if b not in base.nodes],
                'ambiguities': []}
    else:
        raise ValueError("weld must be 'coords', 'name' or 'manual'")
    report.welded = bits['welded']
    report.added_nodes = bits['added_nodes']
    report.ambiguities = bits['ambiguities']

    # 2) Definitions.
    mat_map, sec_map, renamed = reconcile_definitions(base, other)
    tri_sec_map, quad_sec_map, area_renamed = reconcile_area_definitions(base, other)
    renamed.update(area_renamed)
    report.renamed_defs = renamed

    # 3) Elements (bars, then tri/quad).
    emap, new_elems = map_elements(base, other, nmap, sec_map)
    tri_emap, new_tris = map_tri_elements(base, other, nmap, tri_sec_map)
    quad_emap, new_quads = map_quad_elements(base, other, nmap, quad_sec_map)
    report.new_elements = new_elems + new_tris + new_quads
    obj_map = map_geometry_objects(base, other, nmap)

    # 4) Insert new nodes.
    for oid, on in other.nodes.items():
        sid = nmap[oid]
        if sid not in unified.nodes:
            unified.add_node(sid, on.x, on.y)

    # 5) Insert new materials / sections (under reconciled names).
    for name, m in other.materials.items():
        nn = mat_map[name]
        if nn not in unified.materials:
            # _add_material_unchecked: *m* comes from another, already-open
            # model -- it was valid there (or, loaded permissively, never
            # checked at all), so merging it in is not the moment to start
            # enforcing add_material's class check (26/09/2026, same
            # reasoning as structure_io's plain loader).
            unified._add_material_unchecked(
                nn, m.elastic_modulus, m.unit_weight,
                alpha=getattr(m, 'alpha', 1e-5),
                unit_mass=getattr(m, 'unit_mass', None),
                material_type=getattr(m.material_type, 'value', None),
                design=dict(getattr(m, 'design', {}) or {}))
    for name, s in other.sections.items():
        nn = sec_map[name]
        if nn not in unified.sections:
            unified.add_section(nn, mat_map.get(s.material_name, s.material_name),
                                s.b, s.h,
                                area_override=s.area_override,
                                inertia_override=s.inertia_override,
                                profile_name=getattr(s, 'profile_name', None),
                                shape=getattr(s.shape, 'value', None),
                                tw=getattr(s, 'tw', 0.0), tf=getattr(s, 'tf', 0.0))
    for name, s in other.tri_sections.items():
        nn = tri_sec_map[name]
        if nn not in unified.tri_sections:
            unified.add_tri_section(
                nn, mat_map.get(s.material_name, s.material_name),
                thickness=s.thickness, plane_strain=s.plane_strain,
                formulation=s.formulation,
                rc_cover=s.rc_cover, rc_alpha_s=s.rc_alpha_s,
                rc_bar_phi=s.rc_bar_phi, rc_shear_min=s.rc_shear_min,
                rc_cover_top_x=s.rc_cover_top_x, rc_cover_top_y=s.rc_cover_top_y,
                rc_cover_bot_x=s.rc_cover_bot_x, rc_cover_bot_y=s.rc_cover_bot_y)
    for name, s in other.quad_sections.items():
        nn = quad_sec_map[name]
        if nn not in unified.quad_sections:
            unified.add_quad_section(
                nn, mat_map.get(s.material_name, s.material_name),
                thickness=s.thickness, plane_strain=s.plane_strain,
                formulation=s.formulation,
                rc_cover=s.rc_cover, rc_alpha_s=s.rc_alpha_s,
                rc_bar_phi=s.rc_bar_phi, rc_shear_min=s.rc_shear_min,
                rc_cover_top_x=s.rc_cover_top_x, rc_cover_top_y=s.rc_cover_top_y,
                rc_cover_bot_x=s.rc_cover_bot_x, rc_cover_bot_y=s.rc_cover_bot_y)

    # 6) Insert new elements (those that did not weld onto a base element).
    for e in other.bar_elements:
        new_id = emap[e.id]
        if new_id not in unified.bar_elements_by_id:
            unified.add_bar_element(
                new_id, nmap.get(e.node_i, e.node_i), nmap.get(e.node_j, e.node_j),
                sec_map.get(e.section_name, e.section_name),
                hinge_i=e.hinge_i, hinge_j=e.hinge_j)
    for t in other.tri_elements:
        new_id = tri_emap[t.id]
        if new_id not in unified.tri_elements_by_id:
            unified.add_tri_element(
                new_id, nmap.get(t.node_i, t.node_i), nmap.get(t.node_j, t.node_j),
                nmap.get(t.node_k, t.node_k),
                tri_sec_map.get(t.section_name, t.section_name))
    for q in other.quad_elements:
        new_id = quad_emap[q.id]
        if new_id not in unified.quad_elements_by_id:
            unified.add_quad_element(
                new_id, nmap.get(q.node_i, q.node_i), nmap.get(q.node_j, q.node_j),
                nmap.get(q.node_k, q.node_k), nmap.get(q.node_l, q.node_l),
                quad_sec_map.get(q.section_name, q.section_name))

    # 6b) Geometry objects — see the matching step in import_extend: no
    # coordinates of their own, just node_ids into nodes already added above.
    for oid, obj in other.geometry_objects.items():
        new_oid = obj_map[oid]
        if new_oid in unified.geometry_objects:
            continue
        new_obj = copy.deepcopy(obj)
        new_obj.id = new_oid
        new_obj.node_ids = [nmap.get(n, n) for n in getattr(obj, "node_ids", [])]
        unified.geometry_objects[new_oid] = new_obj

    # 7) Bring over support *definitions* used by other, then build a SupportSet.
    for sname, sp in other.supports.items():
        if sname not in unified.supports:
            unified.add_support(sname, ux=sp.ux, uy=sp.uy, tz=sp.tz)
    sset = SupportSet(
        id=overlay_id,
        assignments=[SupportAssignment(node_id=nmap.get(a.node_id, a.node_id),
                                       support_name=a.support_name)
                     for a in other.support_assignments],
        node_springs={nmap.get(k, k): copy.deepcopy(v)
                      for k, v in other.node_springs.items()},
        element_springs={emap.get(k, k): copy.deepcopy(v)
                         for k, v in other.element_springs.items()},
    )

    # 8) Import other's load cases + loads under a prefixed namespace.
    _import_loads(unified, other, overlay_id, nmap, emap)

    overlay = Variant(
        id=overlay_id,
        description=f"Imported model ({len(new_elems) + len(new_tris) + len(new_quads)} new elem)",
        active_elements=set(emap.values()) | set(tri_emap.values()) | set(quad_emap.values()),
        support_set_id=overlay_id)

    return unified, overlay, sset, report


def _import_loads(unified, other, prefix, nmap, emap):
    """Copy other's load cases and loads into unified under prefixed case ids."""
    for lc in other.load_cases:
        cid = f"{prefix}:{lc.id}"
        if cid in unified.load_cases_by_id:
            continue
        unified.add_load_case(cid, self_weight_factor=lc.self_weight_factor,
                              action_type=getattr(lc.action_type, 'value', None))
    for pl in other.point_loads:
        unified.add_point_load(nmap.get(pl.node_id, pl.node_id),
                               f"{prefix}:{pl.load_case_id}",
                               fx=pl.fx, fy=pl.fy, mz=pl.mz)
    for dl in other.distributed_loads:
        unified.add_distributed_load(emap.get(dl.element_id, dl.element_id),
                                     f"{prefix}:{dl.load_case_id}",
                                     fxe=dl.fxe, fxd=dl.fxd, fye=dl.fye, fyd=dl.fyd,
                                     coord_sys=dl.coord_sys)
    for epl in getattr(other, 'element_point_loads', []):
        unified.add_element_point_load(emap.get(epl.element_id, epl.element_id),
                                       f"{prefix}:{epl.load_case_id}",
                                       a=epl.a, fx=epl.fx, fy=epl.fy, mz=epl.mz,
                                       coord_sys=epl.coord_sys)


def import_as_variant(base, other, overlay_id: str, **kw):
    """Thin alias for ``import_model(..., mode='variant')``."""
    return import_model(base, other, overlay_id, mode="variant", **kw)
