"""The reference inventory (xdfem2d.references): renaming and removing an entity
carries every reference to it, and a dangling one is reported."""
import pytest

import context  # noqa: F401
from xdfem2d import Structure2D, references
from xdfem2d.models import ElementPointLoad


def _model():
    s = Structure2D(domain="plane")
    s.add_material("M", elastic_modulus=3e7, unit_weight=0.0, poisson=0.2)
    s.add_section("S", "M", b=0.3, h=0.5)
    s.add_area_section("A", "M", thickness=0.2)
    for i, (x, y) in enumerate([(0, 0), (4, 0), (4, 4), (0, 4), (8, 0), (8, 4)]):
        s.add_node(f"N{i}", x, y)
    s.add_tri_element("T0", "N0", "N1", "N2", "A")
    s.add_quad_element("Q0", "N1", "N4", "N5", "N2", "A")
    s.add_bar_element("E0", "N0", "N3", "S")
    s.add_support("P", ux=True, uy=True, tz=True)
    s.assign_support("N0", "P")
    s.add_load_case("LC")
    s.add_point_load("N3", "LC", 0, -5, 0)
    s.add_distributed_load("E0", "LC", fye=-5, fyd=-5)
    s.element_point_loads.append(ElementPointLoad("E0", "LC", 2.0, 0.0, -3.0, 0.0))
    s.add_temperature_load("E0", "LC", 10.0, 0.0)
    s.add_tri_edge_load("TE", "T0", "N0", "N1", "LC", fx=1.0)
    s.add_quad_edge_load("QE", "Q0", "N1", "N4", "LC", fx=1.0)
    s.add_tri_temperature_load("T0", "LC", 10, 10, 10)
    s.add_quad_temperature_load("Q0", "LC", 10)
    s.add_support_settlement("N0", "LC", ux=0.001)
    s.add_rigid_link("N0", ["N3"])
    return s


def test_every_reference_in_a_sane_model_resolves():
    assert _model().reference_problems() == []


def test_rename_bar_carries_element_point_loads_and_springs():
    s = _model()
    s.add_element_spring("E0", 1.0, 1.0)
    s.rename_bar_element("E0", "EX")
    assert [e.element_id for e in s.element_point_loads] == ["EX"]
    assert [t.element_id for t in s.temperature_loads] == ["EX"]
    assert list(s.element_springs) == ["EX"] and s.element_springs["EX"].element_id == "EX"
    assert s.reference_problems() == []


def test_rename_node_follows_triangles_edge_loads_and_constraints():
    s = _model()
    s.rename_node("N0", "NX")
    assert s.tri_elements[0].node_i == "NX"
    assert (s.tri_edge_loads[0].node_a == "NX")
    assert s.constraints["RIGID1"].master == "NX"
    assert s.reference_problems() == []


def test_rename_area_elements_carry_edge_loads():
    s = _model()
    s.rename_tri_element("T0", "TX")
    s.rename_quad_element("Q0", "QX")
    assert s.tri_edge_loads[0].tri_id == "TX"
    assert s.quad_edge_loads[0].quad_id == "QX"
    assert s.quad_temperature_loads[0].quad_id == "QX"


def test_rename_load_case_reaches_every_kind_of_load():
    s = _model()
    s.rename_load_case("LC", "LX")
    for coll in (s.point_loads, s.distributed_loads, s.element_point_loads,
                 s.temperature_loads, s.tri_edge_loads, s.quad_edge_loads,
                 s.tri_temperature_loads, s.quad_temperature_loads,
                 s.support_settlements):
        assert coll and all(x.load_case_id == "LX" for x in coll)
    assert s.reference_problems() == []


def test_remove_load_case_drops_every_kind_of_load():
    s = _model()
    s.remove_load_case("LC")
    for coll in (s.point_loads, s.distributed_loads, s.element_point_loads,
                 s.temperature_loads, s.tri_edge_loads, s.quad_edge_loads,
                 s.tri_temperature_loads, s.quad_temperature_loads,
                 s.support_settlements):
        assert coll == []


def test_remove_bar_drops_its_loads():
    s = _model()
    s.remove_element("E0")
    assert s.distributed_loads == [] and s.element_point_loads == []
    assert s.temperature_loads == []


def test_remove_node_takes_its_elements_loads_and_constraints():
    s = _model()
    s.remove_node("N0")
    assert s.tri_elements == [] and s.bar_elements == []
    assert s.support_settlements == [] and s.support_assignments == []
    assert s.tri_edge_loads == [] and s.constraints == {}
    assert s.reference_problems() == []


def test_remove_tri_and_quad_take_their_loads():
    s = _model()
    s.remove_tri_element("T0"); s.remove_quad_element("Q0")
    assert (s.tri_edge_loads, s.tri_temperature_loads,
            s.quad_edge_loads, s.quad_temperature_loads) == ([], [], [], [])


def test_material_rename_follows_all_section_kinds_and_concrete_link():
    s = _model()
    s.add_concrete_material("M", "C30/37", "A500NR")
    s.rename_material("M", "X")
    assert s.sections["S"].material_name == "X"
    assert s.tri_sections["A"].material_name == "X"
    assert s.quad_sections["A"].material_name == "X"     # was left behind
    assert "X" in s.concrete_materials
    assert s.concrete_materials["X"].material_name == "X"
    assert s.reference_problems() == []


def test_material_in_use_cannot_be_removed():
    s = _model()
    with pytest.raises(references.ReferenceInUse):
        s.remove_material("M")
    assert "M" in s.materials
    assert references.users_text(s, "material", "M", only_blocking=True) == [
        "quad section 'A'", "section 'S'", "triangle section 'A'"]


def test_section_rename_and_remove_guard():
    s = _model()
    s.rename_section("S", "S2")
    assert s.bar_elements[0].section_name == "S2"
    with pytest.raises(references.ReferenceInUse):
        s.remove_section("S2")


def test_area_section_rename_and_remove_guard():
    s = _model()
    with pytest.raises(references.ReferenceInUse):
        s.remove_area_section("A")
    s.rename_area_section("A", "B")
    assert s.tri_elements[0].section_name == "B"
    assert s.quad_elements[0].section_name == "B"


def test_remove_analysis_case_clears_what_names_it():
    s = _model()
    s.add_load_case("L2")
    s.add_load_combination("C", {"LC": 1.0, "L2": 1.0})
    s.remove_analysis_case("L2")
    assert "L2" not in s.load_combinations[0].coefficients
    s.rename_analysis_case("LC", "LY")
    assert "LY" in s.load_combinations[0].coefficients


def test_dangling_references_are_reported_and_grouped():
    s = _model()
    s.point_loads[0].node_id = "NOPE"
    s.tri_temperature_loads[0].load_case_id = "GONE"
    s.support_settlements[0].load_case_id = "GONE"
    msgs = s.reference_problems()
    assert any("NOPE" in m for m in msgs)
    assert any("GONE" in m for m in msgs)
    # many of one kind collapse into one line
    for i in range(10):
        s.add_support_settlement("N0", f"G{i}", ux=0.0)
    grouped = [m for m in s.reference_problems() if "support settlement" in m]
    assert len(grouped) == 1 and "references" in grouped[0]


def test_new_load_without_a_case_refuses_to_run():
    s = _model()
    s.add_support("F", ux=True, uy=True, tz=True)
    s.tri_edge_loads[0].load_case_id = "GONE"
    with pytest.raises(ValueError, match="do not exist"):
        s.calculate()


# ── nested holders: support sets, variants, construction phases ─────────

def _with_phasing(s):
    from xdfem2d.models import (ConstructionPhase, ConstructionSequence,
                                SupportSet, Variant)
    s.add_support_set(SupportSet(id="SS", restraints={"N0": (True, True, True),
                                                      "N3": (True, False, False)},
                                 element_springs={}))
    s.add_variant(Variant(id="V1", support_set_id="SS", active_elements={"E0"}))
    ph = ConstructionPhase(id="P1", active_elements={"E0"},
                           applied_cases=["LC"], case_factors={"LC": 1.5},
                           support_set_id="SS")
    s.add_construction_sequence(ConstructionSequence(id="SEQ", phases=[ph]))
    return ph


def test_rename_node_and_bar_reach_support_sets_variants_and_phases():
    s = _model(); ph = _with_phasing(s)
    s.rename_node("N0", "NX")
    assert set(s.support_sets["SS"].restraints) == {"NX", "N3"}
    s.rename_bar_element("E0", "EY")
    assert s.variants["V1"].active_elements == {"EY"}
    assert ph.active_elements == {"EY"}


def test_rename_and_remove_load_case_reach_phases():
    s = _model(); ph = _with_phasing(s)
    s.rename_load_case("LC", "LZ")
    assert ph.applied_cases == ["LZ"] and "LZ" in ph.case_factors
    s.remove_load_case("LZ")
    assert ph.applied_cases == [] and ph.case_factors == {}


def test_remove_support_set_clears_variants_and_phases():
    s = _model(); ph = _with_phasing(s)
    s.remove_support_set("SS")
    assert s.variants["V1"].support_set_id is None and ph.support_set_id is None


def test_remove_node_strips_it_from_support_sets():
    s = _model(); _with_phasing(s)
    s.remove_node("N0")
    assert "N0" not in s.support_sets["SS"].restraints


# ── purge ───────────────────────────────────────────────────────────────

def test_purge_removes_orphans_and_reports_them():
    s = _model()
    s.tri_edge_loads[0].load_case_id = "GONE"
    s.support_settlements[0].load_case_id = "GONE"
    s.point_loads[0].node_id = "NOPE"
    assert s.reference_problems()
    summary = s.purge_dangling_references()
    assert summary and s.reference_problems() == []
    assert s.tri_edge_loads == [] and s.support_settlements == [] and s.point_loads == []
    assert len(s.element_point_loads) == 1          # the healthy ones stay
    assert s.purge_dangling_references() == []


def test_purge_leaves_what_is_wrong_with_the_model_itself():
    s = _model()
    s.bar_elements[0].section_name = "MISSING"
    assert s.purge_dangling_references() == []
    assert s.bar_elements[0].section_name == "MISSING"


def test_redirect_merges_a_node_into_another_everywhere():
    s = _model()
    s.add_node_spring("N3", 1.0, 1.0, 0.0)
    references.redirect(s, "node", "N3", "N2")
    assert s.bar_elements[0].node_j == "N2"
    assert s.point_loads[0].node_id == "N2"
    assert s.constraints["RIGID1"].slaves == ["N2"]
    assert "N2" in s.node_springs and "N3" not in s.node_springs
    # a quad corner and a quad edge load follow too (the old merge forgot quads)
    references.redirect(s, "node", "N4", "N1")
    assert s.quad_elements[0].node_j == "N1"
    assert s.quad_edge_loads[0].node_b == "N1"
