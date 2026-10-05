"""The simplified convenience API (facade) on Structure2D.

Thin, lossless wrappers over the generic methods: a magnitude + a named
direction instead of signed components and a reference frame, and a support
created-and-assigned in one call. They read ``self.domain`` and adapt (plane vs
plate). The load/support tests need no eurocodepy; the section shortcuts do
(they read the Eurocode grade databases).
"""
import context  # noqa: F401

import pytest

from xdfem2d import Structure2D


def _beam(domain="plane"):
    s = Structure2D(domain=domain)
    s.add_node("N1", 0.0, 0.0)
    s.add_node("N2", 4.0, 0.0)
    return s


# ── loads ────────────────────────────────────────────────────────────────────

def _last_dl(s):
    return s.distributed_loads[-1]


def test_bar_load_plane_directions():
    s = _beam()
    s.add_material("M", 30e6, 25.0, material_type="Concrete")
    s.add_section("S", "M", 0.3, 0.5)
    s.add_bar_element("E1", "N1", "N2", "S")
    s.add_load_case("LC1")
    s.create_uniform_load("E1", "LC1", 10, "down")
    assert _last_dl(s).fye == -10
    s.create_uniform_load("E1", "LC1", 10, "up")
    assert _last_dl(s).fye == 10
    s.create_uniform_load("E1", "LC1", 7, "perp")
    perp = _last_dl(s)
    assert perp.coord_sys == "local" and perp.fye == 7
    s.create_uniform_load("E1", "LC1", 3, "axial")
    axial = _last_dl(s)
    assert axial.coord_sys == "local" and axial.fxe == 3
    # magnitude is taken as positive: a negative q still loads downward
    s.create_uniform_load("E1", "LC1", -10, "down")
    assert _last_dl(s).fye == -10


def test_bar_point_load_fraction_and_node_load():
    s = _beam()
    s.add_material("M", 30e6, 25.0)
    s.add_section("S", "M", 0.3, 0.5)
    s.add_bar_element("E1", "N1", "N2", "S")
    s.add_load_case("LC1")
    pl = s.create_bar_point_load("E1", "LC1", 20, at=0.25, direction="down")
    assert pl.a == pytest.approx(1.0) and pl.fy == -20      # 0.25 · 4 m
    assert s.create_node_load("N2", "LC1", 5, "down").fy == -5
    with pytest.raises(ValueError):
        s.create_bar_point_load("E1", "LC1", 5, at=1.5)        # fraction out of range


def test_plate_loads_are_transverse():
    s = _beam("plate")
    s.add_material("M", 30e6, 25.0)
    s.add_section("S", "M", 0.3, 0.5)
    s.add_bar_element("E1", "N1", "N2", "S")
    s.add_load_case("LC1")
    # plate transverse loads: the distributed load stores fz in the fye slot,
    # the nodal load stores fz in the fx slot (per the generic aliases).
    s.create_uniform_load("E1", "LC1", 8, "down")
    assert _last_dl(s).fye == -8
    assert s.create_node_load("N2", "LC1", 6, "down").fx == -6
    for bad in ("perp", "axial"):
        with pytest.raises(ValueError):
            s.create_uniform_load("E1", "LC1", 5, bad)


def test_node_load_left_and_right():
    """A model asked for 'a horizontal load' had no direction that meant
    it — 'left'/'right' close that gap for the plane domain."""
    s = _beam()
    s.add_material("M", 30e6, 25.0)
    s.add_section("S", "M", 0.3, 0.5)
    s.add_bar_element("E1", "N1", "N2", "S")
    s.add_load_case("LC1")
    assert s.create_node_load("N2", "LC1", 10, "right").fx == 10
    assert s.create_node_load("N2", "LC1", 10, "left").fx == -10
    # magnitude is taken as positive: a negative p still points the same way
    assert s.create_node_load("N2", "LC1", -10, "right").fx == 10


def test_node_load_left_right_are_plane_only():
    s = _beam("plate")
    s.add_material("M", 30e6, 25.0)
    s.add_section("S", "M", 0.3, 0.5)
    s.add_bar_element("E1", "N1", "N2", "S")
    s.add_load_case("LC1")
    for bad in ("left", "right"):
        with pytest.raises(ValueError):
            s.create_node_load("N2", "LC1", 10, bad)


# ── supports ────────────────────────────────────────────────────────────────

def test_pin_fix_plane_create_and_assign():
    s = _beam()
    s.pin("N1")
    s.fix("N2")
    assert s.supports["PIN"].ux and s.supports["PIN"].uy and not s.supports["PIN"].tz
    assert s.supports["FIXED"].ux and s.supports["FIXED"].uy and s.supports["FIXED"].tz
    assert {a.node_id: a.support_name for a in s.support_assignments} == \
        {"N1": "PIN", "N2": "FIXED"}


def test_roller_plane_and_dedup():
    s = _beam()
    s.add_node("N3", 8.0, 0.0)
    s.roller("N2", free="x")
    s.roller("N3", free="x")                 # same definition → one support
    assert s.supports["ROLLER-X"].uy and not s.supports["ROLLER-X"].ux
    assert sum(1 for k in s.supports if k == "ROLLER-X") == 1
    assert len([a for a in s.support_assignments if a.support_name == "ROLLER-X"]) == 2


def test_plate_supports_and_symm():
    s = _beam("plate")
    s.fix("N1")
    s.symm("N2", axis="x")
    # plate w/tx/ty are stored in the ux/uy/tz slots
    assert s.supports["CLAMPED"].ux and s.supports["CLAMPED"].uy and s.supports["CLAMPED"].tz
    assert s.supports["SYM-X"].uy and not s.supports["SYM-X"].ux \
        and not s.supports["SYM-X"].tz


def test_support_domain_guards():
    plane, plate = _beam(), _beam("plate")
    with pytest.raises(ValueError):
        plane.symm("N1")                     # symm is plate-only
    with pytest.raises(ValueError):
        plate.roller("N1")                   # roller is plane-only
    plane.add_node("N3", 0.0, 4.0)
    plane.add_material("M", 30e6, 25.0)
    plane.add_tri_section("S", "M", thickness=0.2)
    plane.add_tri_element("T1", "N1", "N2", "N3", "S")
    plane.add_load_case("LC1")
    with pytest.raises(ValueError):
        plane.create_uniform_load("T1", "LC1", 5)    # pressure is plate-only


# ── material / section shortcuts (need the eurocodepy grade databases) ───────

def test_rc_section_shortcut():
    s = _beam()
    s.add_rc_section("B", 0.3, 0.5, concrete="C30/37", steel="B500B")
    sec = s.sections["B"]
    mat = s.materials[sec.material_name]
    assert mat.design["fck"] > 0 and mat.design["fyk"] > 0
    assert sec.b == 0.3 and sec.h == 0.5
    # a second RC section of the same grades reuses the material
    s.add_rc_section("B2", 0.25, 0.5, concrete="C30/37", steel="B500B")
    assert s.sections["B2"].material_name == sec.material_name


def test_steel_section_shortcut():
    s = _beam()
    s.add_steel_section("St", "IPE300", grade="S275")
    sec = s.sections["St"]
    assert sec.profile_name == "IPE300"
    assert sec.area_override and sec.area_override > 0
    assert s.materials[sec.material_name].design["fy"] > 0


# ── areas: flexible points, panel, edge support ─────────────────────────────

def test_coerce_points_forms():
    s = Structure2D()
    cp = s._coerce_points
    tri = [(0.0, 0.0), (4.0, 0.0), (4.0, 3.0)]
    assert cp([(0, 0), (4, 0), (4, 3)]) == tri                 # pairs (tuples)
    assert cp([[0, 0], [4, 0], [4, 3]]) == tri                 # pairs (lists)
    assert cp([0, 0, 4, 0, 4, 3]) == tri                       # flat
    assert cp([{"x": 0, "y": 0}, {"x": 4, "y": 0}, {"x": 4, "y": 3}]) == tri
    assert cp([[0, 4, 4], [0, 0, 3]]) == tri                   # two columns (≥3)
    # two points as list-of-lists stays pairs, not columns
    assert cp([[0, 0], [4, 3]]) == [(0.0, 0.0), (4.0, 3.0)]
    with pytest.raises(ValueError):
        cp([0, 0, 4])                                          # odd flat count


def test_coerce_points_accepts_node_ids_objects_and_mixed_lists():
    """An outline entry may be an existing node's id or Node object, mixed
    freely with coordinates -- except inside the flat [x1, y1, x2, y2, ...]
    form, which stays coordinates-only (there is no room in it for an id)."""
    s = Structure2D()
    n0 = s.create_node(0.0, 0.0, id="N0")
    n4 = s.create_node(4.0, 0.0, id="N4")
    cp = s._coerce_points
    assert cp(["N0", "N4", (4.0, 3.0)]) == [(0.0, 0.0), (4.0, 0.0), (4.0, 3.0)]
    assert cp([n0, n4, (4.0, 3.0)]) == [(0.0, 0.0), (4.0, 0.0), (4.0, 3.0)]
    assert cp(["N0", n4, {"x": 4.0, "y": 3.0}]) == [
        (0.0, 0.0), (4.0, 0.0), (4.0, 3.0)]
    with pytest.raises(ValueError):
        cp(["N0", "does-not-exist", (4.0, 3.0)])
    # a flat numeric list never treats a value as an id: [0, 0, 4] above is
    # already covered as an odd-count error; an id string cannot appear
    # inside that form at all since pts[0] being numeric routes to it.


def test_panel_rectangle_and_edge_support():
    s = Structure2D(domain="plate")
    obj = s.add_panel([0, 0, 5, 4], thickness=0.2, id="SL")       # 2-corner rectangle
    assert "SL_sec" in s.tri_sections and s.tri_sections["SL_sec"].thickness == 0.2
    assert len(obj.node_ids) == 4
    s.support_edge("SL", "all", kind="pin")
    # support_edge now writes the object's per-edge edge_supports list (the
    # general case of pin_edge), not corner-node assignments — the reliable
    # route the list was added to provide. Plate 'pin' -> SIMPLE.
    name = s.support_for(True, False, False)
    assert obj.edge_supports == [name] * 4
    assert not s.support_assignments


def test_panel_material_accepts_name_and_material_object():
    """create_polygon's material= used to only accept an unresolved
    Eurocode class string (e.g. "C30/37") and always ran it through
    _rc_material, raising ValueError for an already-existing material name
    or a Material object. It now resolves material the same permissive way
    as create_area_section/create_bar_section: an existing name or a
    Material object is used directly (no duplicate created); only an
    unrecognised string falls through to the Eurocode auto-create path."""
    from xdfem2d import Structure2D as _S

    # (a) regression: bare Eurocode class string still works
    s = _S(domain="plate")
    s.create_polygon([(0, 0), (6, 0), (6, 4), (0, 4)],
                      thickness=0.2, material="C30/37", id="Slab0")
    assert "Slab0_sec" in s.tri_sections

    # (b) an existing material *name* is reused, not treated as an unknown class
    s2 = _S(domain="plate")
    s2.create_rc_material(concrete="C30/37", steel="B500B", name="ConcreteMat")
    n_materials_before = len(s2.materials)
    s2.create_polygon([(0, 0), (6, 0), (6, 4), (0, 4)],
                       thickness=0.2, material="ConcreteMat", id="Slab1")
    assert s2.tri_sections["Slab1_sec"].material_name == "ConcreteMat"
    assert len(s2.materials) == n_materials_before   # no duplicate created

    # (c) an actual Material object is accepted too
    s3 = _S(domain="plate")
    mat = s3.create_rc_material(concrete="C30/37", steel="B500B", name="ConcreteMat3")
    s3.create_polygon([(0, 0), (6, 0), (6, 4), (0, 4)],
                       thickness=0.2, material=mat, id="Slab2")
    assert s3.tri_sections["Slab2_sec"].material_name == "ConcreteMat3"


def test_panel_named_edges_work_with_a_4corner_outline():
    """A script that spells a rectangle out as 4 corners (bottom-left,
    bottom-right, top-right, top-left — the natural order) used to build a
    GeoPolygon, not a GeoRectangle, and support_edge('bottom'/... ) raised
    ValueError at runtime with script_check reporting nothing wrong. Named
    edges now work off node count, not the class, as long as the 4 points
    are given in that order."""
    s = Structure2D(domain="plane")
    obj = s.create_polygon([(0, 0), (3, 0), (3, 15), (0, 15)],
                        thickness=0.22, id="W")
    assert len(obj.node_ids) == 4
    s.support_edge("W", "bottom", kind="fix")   # 'bottom' is edge index 0
    name = s.support_for(True, True, True)       # FIXED in plane
    assert obj.edge_supports[0] == name
    assert all(e == "free" for i, e in enumerate(obj.edge_supports) if i != 0)


def test_edge_load_tang_sign_follows_winding():
    """The concrete case from the confusion this documents: 'tang' on a
    rectangle's 'top' edge pushes left (-X), not right, because 'top' runs
    top-right -> top-left. create_node_load is the unambiguous alternative
    for a load with a real global direction."""
    from xdfem2d.loads import _edge_load_fxy
    s = Structure2D(domain="plane")
    obj = s.create_polygon([(0, 0), (3, 15)], thickness=0.22, id="W")
    el = s.create_edge_load("W", "top", p=10, direction="tang")
    el = el[0] if isinstance(el, list) else el
    na, nb = s.nodes[el.node_a], s.nodes[el.node_b]
    fx, fy = _edge_load_fxy(el, na, nb, thickness=1.0)
    assert fx < 0 and fy == pytest.approx(0.0)   # points left (-X), not right


def test_panel_polygon_edge_index_and_domain():
    w = Structure2D(domain="plane")
    ow = w.add_panel([[0, 4, 4, 0], [0, 0, 3, 3]], thickness=0.3, id="W")  # columns
    assert "W_sec" in w.tri_sections and len(ow.node_ids) == 4
    w.support_edge("W", 0, kind="pin")                         # edge v0→v1
    pin = w.support_for(True, True, False)                     # PIN in plane
    assert ow.edge_supports[0] == pin
    # roller is plane-only (ok here); symm would be plate-only
    w.support_edge("W", 2, kind="roller-x")
    assert ow.edge_supports[2] != "free"
    with pytest.raises(ValueError):
        w.support_edge("W", 1, kind="symm-x")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
