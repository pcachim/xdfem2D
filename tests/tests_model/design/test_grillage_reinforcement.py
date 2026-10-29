"""Phase 9C — EC2 reinforcement of grillage bars (bending + shear + torsion).

A grillage bar bends out of plane, carries shear, and — unlike a plane beam —
St-Venant torsion (GJ). Its end-force "N" slot is the torsion T. The design adds
EC2 §6.3 torsion (closed stirrups + longitudinal steel) to the bending/shear and
checks the shear-torsion interaction (Eq. 6.29).

Requires the project's eurocodepy (with ``calc_torsion`` and ``calc_asl_nm``);
skipped otherwise. The torsion formulae themselves are unit-tested inside
eurocodepy (tests/test_torsion.py).
"""
import pytest

import context  # noqa: F401

from eurocodepy.ec2 import uls
if not (hasattr(uls, "calc_torsion") and hasattr(uls, "calc_asl_nm")):
    pytest.skip("eurocodepy lacks calc_torsion / calc_asl_nm",
                allow_module_level=True)

from xdfem2d import Structure2D
from xdfem2d.rc_design import design_concrete_sections

FCK, FYK = 30.0, 500.0
B, H, L, NSEG = 0.30, 0.50, 4.0, 4


def _cantilever(load):
    s = Structure2D(domain="plate")
    s.add_material("C", elastic_modulus=30e6, unit_weight=0.0, poisson=0.2,
                   material_type="Concrete", design={"fck": FCK, "fyk": FYK})
    s.add_section("B", "C", b=B, h=H, rc_cover=0.05)
    for k in range(NSEG + 1):
        s.add_node(f"N{k}", L * k / NSEG, 0.0)
    for k in range(NSEG):
        s.add_bar_element(f"E{k}", f"N{k}", f"N{k + 1}", "B")
    s.add_support("FIX", w=True, tx=True, ty=True)
    s.assign_support("N0", "FIX")
    s.add_load_case("G")
    s.add_point_load(f"N{NSEG}", "G", **load)
    s.add_load_combination("ULS", {"G": 1.35})
    return s


def test_grillage_rows_carry_torsion():
    # Tip transverse load (bending + shear) plus a tip torque (torsion).
    s = _cantilever({"fz": -30.0, "mx": 40.0})
    rows = design_concrete_sections(s, s.calculate())
    assert rows and all(r.get("kind") == "grillage" for r in rows)
    gov = next(r for r in rows if r["governing"])
    # Torsion produces its own longitudinal steel and its demand is reported.
    assert gov["T_Ed"] != 0.0
    assert gov["Asl_tor"] > 0.0
    assert "interaction" in gov and gov["interaction"] > 0.0
    # The "axial" slot is not treated as an axial force in a grillage.
    assert gov["N_Ed"] == 0.0


def test_torsion_adds_to_the_shear_stirrups():
    """With torsion, the stirrups exceed the shear-only value (2·Asw,tor added)."""
    with_t = _cantilever({"fz": -30.0, "mx": 40.0})
    no_t = _cantilever({"fz": -30.0})
    r_t = design_concrete_sections(with_t, with_t.calculate())
    r_n = design_concrete_sections(no_t, no_t.calculate())
    # Compare the fixed-end (max shear) location.
    aw_t = max(r["Asw_s"] for r in r_t)
    aw_n = max(r["Asw_s"] for r in r_n)
    assert aw_t > aw_n
    # A pure bending/shear grillage has no torsion steel.
    assert all(r.get("Asl_tor", 0.0) == 0.0 for r in r_n)


def test_report_has_a_grillage_reinforcement_table():
    from xdfem2d.report_io import _results_tables
    s = _cantilever({"fz": -30.0, "mx": 40.0})
    res = s.calculate()
    from xdfem2d.rc_design import design_and_store
    design_and_store(s, res)
    titles = [t[0] for t in _results_tables(res, domain="plate")]
    assert "Grillage reinforcement (EC2 bending + shear + torsion)" in titles
    assert "Reinforcement (EC2 design)" not in titles
