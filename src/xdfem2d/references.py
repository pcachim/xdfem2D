"""Inventory of the references between the entities of a :class:`Structure2D`.

Every id or name a model entity stores about another one (a load's load case, an
element's nodes, a section's material, ...) is declared once here, as a *site*.
Renaming and removing an entity, and checking that nothing dangles, all walk the
same table, so a new kind of load or element is covered by adding one line
instead of by remembering to edit a dozen ``rename_*`` / ``remove_*`` methods.

Kinds of entity (``kind``): ``node``, ``bar``, ``tri``, ``quad``, ``load_case``,
``analysis_case``, ``combination``, ``material``, ``section`` (bar section),
``area_section`` (triangle/quad section, whose names are paired), ``support``
and ``object`` (geometry object).

A site says where the id is held and what happens to the holder when the entity
is removed (``on_remove``):

``drop``     the holder is removed (a load on a deleted load case);
``strip``    only the id is taken out of the holder's list / dict / set;
``clear``    the field is blanked (a case that named a deleted modal case);
``null``     the field is set to ``None`` (an optional reference);
``cascade``  the holder entity is removed with its own references (the bars of
             a deleted node);
``block``    the removal is refused while the holder exists (a material still
             used by a section).

``container`` may be a dotted path to reach nested holders
(``"construction_sequences.phases"`` is every phase of every sequence).
"""
from __future__ import annotations

from typing import NamedTuple


class Site(NamedTuple):
    kind: str            # kind of the referenced entity
    container: str       # Structure2D attribute holding the holders
    attr: str            # holder field that carries the id
    mode: str = "scalar"   # scalar | list | terms | keys | set | key
    on_remove: str = "drop"
    #: ``reference_problems`` already reports a dangling one of these.
    checked: bool = False


S = Site

#: Every place where a model entity names another one.
SITES: tuple[Site, ...] = (
    # ── nodes ──────────────────────────────────────────────────────────
    S("node", "bar_elements", "node_i", on_remove="cascade"),
    S("node", "bar_elements", "node_j", on_remove="cascade"),
    S("node", "tri_elements", "node_i", on_remove="cascade"),
    S("node", "tri_elements", "node_j", on_remove="cascade"),
    S("node", "tri_elements", "node_k", on_remove="cascade"),
    S("node", "quad_elements", "node_i", on_remove="cascade"),
    S("node", "quad_elements", "node_j", on_remove="cascade"),
    S("node", "quad_elements", "node_k", on_remove="cascade"),
    S("node", "quad_elements", "node_l", on_remove="cascade"),
    S("node", "tri_edge_loads", "node_a"),
    S("node", "tri_edge_loads", "node_b"),
    S("node", "quad_edge_loads", "node_a"),
    S("node", "quad_edge_loads", "node_b"),
    S("node", "surface_edge_loads", "node_a"),
    S("node", "surface_edge_loads", "node_b"),
    S("node", "support_assignments", "node_id", checked=True),
    S("node", "point_loads", "node_id", checked=True),
    S("node", "support_settlements", "node_id"),
    S("node", "nodal_masses", "node_id"),
    S("node", "punch_columns", "node_id"),
    S("node", "node_springs", "node_id", mode="key", checked=True),
    S("node", "constraints", "master"),
    S("node", "constraints", "terms", mode="terms"),
    S("node", "constraints", "slaves", mode="list", on_remove="strip"),
    S("node", "constraints", "nodes", mode="list", on_remove="strip"),
    S("node", "geometry_objects", "node_ids", mode="list", on_remove="strip"),
    S("node", "support_sets", "restraints", mode="keys", on_remove="strip"),
    S("node", "support_sets", "node_springs", mode="keys", on_remove="strip"),
    S("node", "support_sets.assignments", "node_id"),
    # ── bars ───────────────────────────────────────────────────────────
    S("bar", "distributed_loads", "element_id", checked=True),
    S("bar", "element_point_loads", "element_id", checked=True),
    S("bar", "temperature_loads", "element_id"),
    S("bar", "element_springs", "element_id", mode="key", checked=True),
    # a variant or a phase lists bars, triangles and objects in one set, so
    # these are carried and stripped but never reported as dangling
    S("bar", "variants", "active_elements", mode="set", on_remove="strip",
      checked=True),
    S("bar", "construction_sequences.phases", "active_elements", mode="set",
      on_remove="strip", checked=True),
    S("bar", "support_sets", "element_springs", mode="keys", on_remove="strip"),
    # ── triangles ──────────────────────────────────────────────────────
    S("tri", "tri_edge_loads", "tri_id"),
    S("tri", "tri_temperature_loads", "tri_id"),
    S("tri", "tri_area_loads", "tri_id"),
    S("tri", "tri_area_springs", "tri_id"),
    # ── quads ──────────────────────────────────────────────────────────
    S("quad", "quad_edge_loads", "quad_id"),
    S("quad", "quad_temperature_loads", "quad_id"),
    S("quad", "quad_area_loads", "quad_id"),
    S("quad", "quad_area_springs", "quad_id"),
    # ── load cases ─────────────────────────────────────────────────────
    S("load_case", "point_loads", "load_case_id", checked=True),
    S("load_case", "distributed_loads", "load_case_id", checked=True),
    S("load_case", "element_point_loads", "load_case_id", checked=True),
    S("load_case", "support_settlements", "load_case_id"),
    S("load_case", "temperature_loads", "load_case_id"),
    S("load_case", "tri_edge_loads", "load_case_id"),
    S("load_case", "quad_edge_loads", "load_case_id"),
    S("load_case", "surface_edge_loads", "load_case_id"),
    S("load_case", "tri_area_loads", "load_case_id"),
    S("load_case", "quad_area_loads", "load_case_id"),
    S("load_case", "surface_area_loads", "load_case_id"),
    S("load_case", "tri_temperature_loads", "load_case_id"),
    S("load_case", "quad_temperature_loads", "load_case_id"),
    S("load_case", "area_temperature_loads", "load_case_id"),
    S("load_case", "line_temperature_loads", "load_case_id"),
    S("load_case", "line_distributed_loads", "load_case_id"),
    S("load_case", "analysis_cases", "coefficients", mode="keys",
      on_remove="strip", checked=True),
    S("load_case", "construction_sequences.phases", "applied_cases",
      mode="list", on_remove="strip"),
    S("load_case", "construction_sequences.phases", "case_factors",
      mode="keys", on_remove="strip"),
    # ── analysis cases ─────────────────────────────────────────────────
    S("analysis_case", "load_combinations", "coefficients", mode="keys",
      on_remove="strip", checked=True),
    S("analysis_case", "analysis_cases", "coefficients", mode="keys",
      on_remove="strip", checked=True),
    S("analysis_case", "analysis_cases", "modal_case_id", on_remove="clear",
      checked=True),
    S("analysis_case", "analysis_cases", "stored_stiffness_id",
      on_remove="clear", checked=True),
    S("analysis_case", "nodal_masses", "mass_case_id", checked=True),
    # ── combinations (a combination may combine others) ────────────────
    S("combination", "load_combinations", "coefficients", mode="keys",
      on_remove="strip"),
    # ── materials and sections ─────────────────────────────────────────
    S("material", "sections", "material_name", on_remove="block", checked=True),
    S("material", "tri_sections", "material_name", on_remove="block",
      checked=True),
    S("material", "quad_sections", "material_name", on_remove="block",
      checked=True),
    S("material", "concrete_materials", "material_name", mode="key"),
    S("section", "bar_elements", "section_name", on_remove="block",
      checked=True),
    S("section", "geometry_objects", "section_name", on_remove="block"),
    S("area_section", "tri_elements", "section_name", on_remove="block",
      checked=True),
    S("area_section", "quad_elements", "section_name", on_remove="block",
      checked=True),
    S("area_section", "geometry_objects", "tri_section_name",
      on_remove="block"),
    # ── supports ───────────────────────────────────────────────────────
    S("support", "support_assignments", "support_name", checked=True),
    S("support", "support_sets.assignments", "support_name"),
    # ── support sets ───────────────────────────────────────────────────
    S("support_set", "variants", "support_set_id", on_remove="null"),
    S("support_set", "construction_sequences.phases", "support_set_id",
      on_remove="null"),
    # ── geometry objects ───────────────────────────────────────────────
    S("object", "surface_edge_loads", "object_id"),
    S("object", "surface_area_loads", "object_id"),
    S("object", "surface_area_springs", "object_id"),
    S("object", "area_temperature_loads", "object_id"),
    S("object", "line_temperature_loads", "object_id"),
    S("object", "line_distributed_loads", "object_id"),
    S("object", "line_element_springs", "object_id"),
)

_BY_KIND: dict[str, list[Site]] = {}
for _s in SITES:
    _BY_KIND.setdefault(_s.kind, []).append(_s)

#: Readable names of the containers, for messages.
_LABEL = {
    "bar_elements": "bar", "tri_elements": "triangle", "quad_elements": "quad",
    "tri_edge_loads": "triangle edge load", "quad_edge_loads": "quad edge load",
    "surface_edge_loads": "surface edge load",
    "support_assignments": "support", "point_loads": "point load",
    "support_settlements": "support settlement", "nodal_masses": "nodal mass",
    "punch_columns": "punching column", "node_springs": "node spring",
    "constraints": "constraint", "geometry_objects": "geometry object",
    "distributed_loads": "distributed load",
    "element_point_loads": "element point load",
    "temperature_loads": "temperature load", "element_springs": "element spring",
    "variants": "variant", "tri_temperature_loads": "triangle temperature load",
    "tri_area_loads": "triangle area load", "tri_area_springs": "triangle area spring",
    "quad_temperature_loads": "quad temperature load",
    "quad_area_loads": "quad area load", "quad_area_springs": "quad area spring",
    "surface_area_loads": "surface area load",
    "area_temperature_loads": "area temperature load",
    "line_temperature_loads": "line temperature load",
    "line_distributed_loads": "line distributed load",
    "analysis_cases": "analysis case", "load_combinations": "combination",
    "sections": "section", "tri_sections": "triangle section",
    "quad_sections": "quad section", "concrete_materials": "concrete material",
    "surface_area_springs": "surface area spring",
    "line_element_springs": "line spring",
    "support_sets": "support set", "support_sets.assignments": "support set support",
    "construction_sequences.phases": "construction phase",
}
_NAME_ATTRS = ("id", "name", "node_id", "element_id", "tri_id", "quad_id",
               "object_id", "material_name", "master")


class ReferenceInUse(ValueError):
    """Removing an entity that other entities still depend on."""


# ── helpers ────────────────────────────────────────────────────────────

def _items(coll) -> list:
    return list(coll.values()) if isinstance(coll, dict) else list(coll)


def _collections(struc, container: str) -> list:
    """The mutable list / dict objects that hold the holders of *container*."""
    parts = container.split(".")
    cur = [getattr(struc, parts[0], None)]
    for p in parts[1:]:
        nxt = []
        for coll in cur:
            if coll is None:
                continue
            for h in _items(coll):
                c = getattr(h, p, None)
                if c is not None:
                    nxt.append(c)
        cur = nxt
    return [c for c in cur if c is not None]


def _holders(struc, site: Site) -> list:
    items = [h for coll in _collections(struc, site.container)
             for h in _items(coll)]
    if site.container == "geometry_objects":
        items = [h for h in items if hasattr(h, site.attr)]
    return items


def _match(holder, site: Site, id_) -> bool:
    val = getattr(holder, site.attr, None)
    if site.mode in ("scalar", "key"):
        return val == id_
    if site.mode == "list":
        return bool(val) and id_ in val
    if site.mode == "terms":
        return any(t and t[0] == id_ for t in (val or ()))
    if site.mode in ("keys", "set"):
        return bool(val) and id_ in val
    return False


def _describe(holder, site: Site) -> str:
    who = next((str(getattr(holder, a)) for a in _NAME_ATTRS
                if getattr(holder, a, "")), "")
    label = _LABEL.get(site.container, site.container)
    return f"{label} '{who}'" if who else label


def _drop(struc, site: Site, holder) -> None:
    for cont in _collections(struc, site.container):
        if isinstance(cont, dict):
            for k in [k for k, v in cont.items() if v is holder]:
                cont.pop(k)
        else:
            cont[:] = [h for h in cont if h is not holder]


# ── queries ────────────────────────────────────────────────────────────

def users(struc, kind: str, id_: str) -> list[tuple[Site, object]]:
    """Every ``(site, holder)`` that refers to entity *id_* of *kind*."""
    return [(s, h) for s in _BY_KIND.get(kind, ())
            for h in _holders(struc, s) if _match(h, s, id_)]


def users_text(struc, kind: str, id_: str, only_blocking: bool = False,
               policies: tuple = ()) -> list[str]:
    """Readable names of the holders that refer to *id_* (sorted, unique).

    *only_blocking* keeps the holders that would refuse its removal; *policies*
    keeps those whose ``on_remove`` is one of the given ones (e.g.
    ``("cascade",)`` for the elements that stand on a node)."""
    keep = set(policies) | ({"block"} if only_blocking else set())
    out = {_describe(h, s) for s, h in users(struc, kind, id_)
           if not keep or s.on_remove in keep}
    return sorted(out)


# ── rename ─────────────────────────────────────────────────────────────

def rename(struc, kind: str, old: str, new: str) -> None:
    """Carry every reference to *old* over to *new*. Containers keyed by the id
    are re-keyed; the entity's own table is the caller's business."""
    if old == new:
        return
    for s in _BY_KIND.get(kind, ()):
        if s.mode == "key":
            for cont in _collections(struc, s.container):
                if isinstance(cont, dict) and old in cont:
                    h = cont.pop(old)
                    setattr(h, s.attr, new)
                    cont[new] = h
            continue
        for h in _holders(struc, s):
            if not _match(h, s, old):
                continue
            val = getattr(h, s.attr)
            if s.mode == "scalar":
                setattr(h, s.attr, new)
            elif s.mode == "list":
                setattr(h, s.attr, [new if x == old else x for x in val])
            elif s.mode == "terms":
                setattr(h, s.attr, [((new,) + tuple(t[1:])) if t and t[0] == old
                                    else t for t in val])
            elif s.mode == "keys":
                val[new] = val.pop(old)
            elif s.mode == "set":
                val.discard(old)
                val.add(new)


# ── redirect (merge) ───────────────────────────────────────────────────

def redirect(struc, kind: str, old: str, new: str) -> None:
    """Point every reference to *old* at the existing entity *new* (merging two
    nodes into one). Like :func:`rename`, but where *new* already has an entry
    in a keyed container its own entry is kept and *old*'s is dropped."""
    if old == new:
        return
    for s in _BY_KIND.get(kind, ()):
        if s.mode == "key":
            for cont in _collections(struc, s.container):
                if isinstance(cont, dict) and old in cont:
                    h = cont.pop(old)
                    if new not in cont:
                        setattr(h, s.attr, new)
                        cont[new] = h
            continue
        for h in _holders(struc, s):
            if not _match(h, s, old):
                continue
            val = getattr(h, s.attr)
            if s.mode == "scalar":
                setattr(h, s.attr, new)
            elif s.mode == "list":
                setattr(h, s.attr, [new if x == old else x for x in val])
            elif s.mode == "terms":
                setattr(h, s.attr, [((new,) + tuple(t[1:])) if t and t[0] == old
                                    else t for t in val])
            elif s.mode == "keys":
                v = val.pop(old)
                val.setdefault(new, v)
            elif s.mode == "set":
                val.discard(old)
                val.add(new)


# ── remove ─────────────────────────────────────────────────────────────

def remove(struc, kind: str, id_: str) -> None:
    """Apply each site's ``on_remove`` for entity *id_* that is going away.

    Raises :class:`ReferenceInUse`, before touching anything, when a ``block``
    site still refers to it."""
    blockers = users_text(struc, kind, id_, only_blocking=True)
    if blockers:
        raise ReferenceInUse(
            f"{kind.replace('_', ' ')} '{id_}' is used by: "
            + ", ".join(blockers) + ".")
    # Cascades first: removing a bar, triangle or quad takes its own
    # references along (through its own remove_* method).
    for s, h in users(struc, kind, id_):
        if s.on_remove != "cascade":
            continue
        eid = getattr(h, "id", None)
        if s.container == "bar_elements" and eid in struc.bar_elements_by_id:
            struc.remove_element(eid)
        elif s.container == "tri_elements" and eid in struc.tri_elements_by_id:
            struc.remove_tri_element(eid)
        elif s.container == "quad_elements" and eid in struc.quad_elements_by_id:
            struc.remove_quad_element(eid)
    for s in _BY_KIND.get(kind, ()):
        if s.on_remove in ("cascade", "block"):
            continue
        for h in _holders(struc, s):
            if not _match(h, s, id_):
                continue
            if s.on_remove == "drop":
                _drop(struc, s, h)
            elif s.on_remove == "clear":
                setattr(h, s.attr, "")
            elif s.on_remove == "null":
                setattr(h, s.attr, None)
            elif s.on_remove == "strip":
                val = getattr(h, s.attr)
                if s.mode == "list":
                    setattr(h, s.attr, [x for x in val if x != id_])
                elif s.mode == "keys":
                    val.pop(id_, None)
                elif s.mode == "set":
                    val.discard(id_)


# ── dangling references ────────────────────────────────────────────────

def _exists(struc, kind: str, id_: str) -> bool:
    tables = {
        "node": lambda: id_ in struc.nodes,
        "bar": lambda: id_ in struc.bar_elements_by_id,
        "tri": lambda: id_ in struc.tri_elements_by_id,
        "quad": lambda: id_ in struc.quad_elements_by_id,
        "load_case": lambda: id_ in struc.load_cases_by_id,
        "analysis_case": lambda: id_ in struc.analysis_cases_by_id,
        "combination": lambda: any(c.id == id_ for c in struc.load_combinations),
        "material": lambda: id_ in struc.materials,
        "section": lambda: id_ in struc.sections,
        "area_section": lambda: (id_ in struc.tri_sections
                                 or id_ in struc.quad_sections),
        "support": lambda: id_ in struc.supports,
        "support_set": lambda: id_ in struc.support_sets,
        "object": lambda: id_ in struc.geometry_objects,
    }
    return tables[kind]()


def _ids_in(holder, site: Site) -> list:
    val = getattr(holder, site.attr, None)
    if site.mode in ("scalar", "key"):
        return [val] if val else []
    if site.mode == "terms":
        return [t[0] for t in (val or ()) if t]
    if site.mode in ("list", "keys", "set"):
        return list(val or ())
    return []


def dangling(struc, include_checked: bool = False) -> list[str]:
    """Readable findings for every reference whose target does not exist.

    The sites ``reference_problems`` already reports are skipped unless
    *include_checked*. Node / bar / triangle / quad targets are only decidable
    on the editable mesh, so they are skipped when the model has geometry
    objects (whose mesh is generated at solve time). References to geometry
    objects are not checked at all: the compiled mesh the solver sees has
    already replaced the objects by elements."""
    editable = not getattr(struc, "geometry_objects", None)
    found: dict[tuple, list] = {}            # (container, kind) -> [(holder, ref)]
    for s in SITES:
        if s.checked and not include_checked:
            continue
        if s.kind in ("node", "bar", "tri", "quad") and not editable:
            continue
        if s.kind == "object":
            continue    # a compiled mesh no longer carries its source objects
        if s.mode == "keys" and s.kind in ("analysis_case", "combination"):
            continue                  # a key may be a load case, case or combo
        for h in _holders(struc, s):
            for ref in _ids_in(h, s):
                if ref and not _exists(struc, s.kind, ref):
                    found.setdefault((s.container, s.kind), []).append((h, s, ref))
    P: list[str] = []
    for (container, kind), items in found.items():
        label = _LABEL.get(container, container)
        what = kind.replace("_", " ")
        if len(items) <= 3:
            for h, s, ref in items:
                P.append(f"{_describe(h, s)} → {what} '{ref}'")
        else:
            refs = sorted({ref for _h, _s, ref in items})
            shown = ", ".join(f"'{r}'" for r in refs[:5])
            more = ", ..." if len(refs) > 5 else ""
            P.append(f"{len(items)} {label} references → {what} that does not "
                     f"exist ({shown}{more})")
    return P


def purge(struc) -> list[str]:
    """Remove every reference that points at something that does not exist,
    wherever the site allows it (``drop`` / ``strip`` / ``clear``): the load on
    a deleted load case, the id of a deleted node in a constraint, ... Returns
    one readable line per kind of thing removed. References held by an element
    to a missing node, section or material (``cascade`` / ``block``) are left
    for the user — they say something is wrong with the model itself."""
    editable = not getattr(struc, "geometry_objects", None)
    removed: dict[str, int] = {}
    for s in SITES:
        if s.on_remove in ("cascade", "block") or s.kind == "object":
            continue
        if s.kind in ("node", "bar", "tri", "quad") and not editable:
            continue
        if s.mode == "keys" and s.kind in ("analysis_case", "combination"):
            continue
        if s.container in ("variants", "construction_sequences.phases") \
                and s.mode == "set":
            continue
        for h in _holders(struc, s):
            bad = [r for r in _ids_in(h, s) if r and not _exists(struc, s.kind, r)]
            if not bad:
                continue
            key = f"{_LABEL.get(s.container, s.container)} → {s.kind.replace('_', ' ')}"
            if s.on_remove == "drop":
                _drop(struc, s, h)
                removed[key] = removed.get(key, 0) + 1
            else:
                for r in bad:
                    if s.mode == "list":
                        setattr(h, s.attr, [x for x in getattr(h, s.attr) if x != r])
                    elif s.mode == "keys":
                        getattr(h, s.attr).pop(r, None)
                    elif s.mode == "scalar":
                        setattr(h, s.attr, None if s.on_remove == "null" else "")
                    removed[key] = removed.get(key, 0) + 1
    return [f"{n} {k}" for k, n in removed.items()]
