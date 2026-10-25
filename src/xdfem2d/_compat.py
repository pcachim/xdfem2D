"""Compatibility / resolver helpers for the `create_*` convenience layer.

See dev/XDFEM2D_ENGINE.md for the design this supports. Every helper here is
pure and Qt-free; `create_*` methods on `Structure2D` call these to resolve
permissive arguments before delegating to an `add_*` method, so the JSON a
model writes never depends on which layer built it.
"""
from __future__ import annotations



def auto_name(prefix: str, collection) -> str:
    """Generate a unique ``"{prefix}{n}"`` name for an entity created
    without an explicit id, e.g. ``auto_name("B", self.bar_elements_by_id)``
    -> ``"B1"``, ``"B2"``, ... Numbering is scoped to ``collection`` only,
    so bar (``"B*"``) and triangle (``"T*"``) ids never collide with each
    other even though both start counting at 1.
    """
    n = 1
    while f"{prefix}{n}" in collection:
        n += 1
    return f"{prefix}{n}"


def as_node_id(node) -> str:
    """Resolve a ``Node`` object or a raw id to the id string."""
    return getattr(node, "id", node)


def as_case_coefficients(coefficients):
    """``{case: factor}`` with every key an id: a key may be the case object
    (``LoadCase``, ``AnalysisCase``, ``LoadCombination``) or its id, like every
    other argument of the ``create_*`` calls. ``None`` stays ``None``."""
    if coefficients is None:
        return None
    return {getattr(k, "id", k): v for k, v in dict(coefficients).items()}


def as_section_name(section) -> str:
    """Resolve a ``Section``/``TriSection`` object or a raw name to its name."""
    return getattr(section, "name", section)


def as_material_name(material) -> str:
    """Resolve a ``Material`` object or a raw name to its name."""
    return getattr(material, "name", material)


def as_object_id(obj) -> str:
    """Resolve a geometry object (GeoRectangle, GeoPolygon, GeoSegment, ...)
    or a raw id to its id string -- the object_id counterpart of
    as_node_id. create_polygon/add_geo_* return the object precisely so a
    caller can chain off it (create_polygon's own docstring says "keep the
    return", e.g. for .node_ids), so a later call reusing that same return
    value where an object_id is expected is not a typo, it is the natural
    next line -- and self.geometry_objects.get(obj) crashed with
    "unhashable type" instead of working, since these are plain (non-frozen)
    dataclasses (26/09/2026)."""
    return getattr(obj, "id", obj)


def as_object_id_list(targets) -> "tuple[list, bool]":
    """Same normalisation as as_node_id_list, named for callers whose
    targets are bars/triangles/quads/geometry objects rather than nodes --
    the underlying code only ever needs an ``.id``, so one implementation
    already covers both (27/09/2026, added for create_temperature: a list
    of bar ids crashed with "unhashable type: 'list'" the same way an
    un-resolved geometry object used to, see as_object_id's own docstring)."""
    return as_node_id_list(targets)


def as_points(outline) -> list[tuple[float, float]]:
    """Normalise a flexible point-list argument into ``[(x, y), ...]``.

    Accepted forms (mirrors the ``outline`` convention already used by
    ``create_polygon``):
      - ``[(x, y), (x, y), ...]``            — pairs
      - ``[x1, y1, x2, y2, ...]``             — flat list
      - ``[{"x": .., "y": ..}, ...]``         — dicts
      - ``[[x1, x2, ...], [y1, y2, ...]]``    — two columns (3+ points)
    """
    pts = list(outline)
    if not pts:
        return []
    first = pts[0]
    # Two columns: [[x...], [y...]]
    if (len(pts) == 2 and isinstance(first, (list, tuple))
            and len(pts[1]) == len(first) and len(first) > 2):
        xs, ys = pts
        return [(float(x), float(y)) for x, y in zip(xs, ys)]
    # Dicts: [{"x":.., "y":..}, ...]
    if isinstance(first, dict):
        return [(float(p["x"]), float(p["y"])) for p in pts]
    # Pairs: [(x, y), ...]
    if isinstance(first, (list, tuple)):
        return [(float(x), float(y)) for x, y in pts]
    # Flat: [x1, y1, x2, y2, ...]
    flat = [float(v) for v in pts]
    return list(zip(flat[0::2], flat[1::2]))


def as_node_id_list(nodes) -> tuple[list, bool]:
    """Normalise a support/list target into ``(ids, was_single)``.

    ``nodes`` may be a single raw id, a single ``Node`` object, or a
    list/tuple/set of either. Returns the resolved id list plus whether
    the input was a single ref (so the caller can return a scalar instead
    of a one-item list, keeping today's return shape for the common single-
    node call). A bare string is never iterated character-by-character —
    it is always a single id, not a list of one-letter ids.
    """
    single = isinstance(nodes, str) or hasattr(nodes, "id")
    if single:
        return [as_node_id(nodes)], True
    try:
        items = list(nodes)
    except TypeError:
        return [as_node_id(nodes)], True
    return [as_node_id(n) for n in items], False


def alias(target):
    """Class-body helper: ``old_name = alias(new_method)``, documenting the
    aliasing intent at the call site (functionally identical to a plain
    ``old_name = new_method`` assignment, the pattern already used for
    ``add_geo_line = add_geo_segment`` etc.)."""
    return target
