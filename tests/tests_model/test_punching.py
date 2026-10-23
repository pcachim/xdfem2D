"""Phase 9D — EC2 §6.4 punching-shear verification at slab columns.

A column punches the slab locally; the check runs over the explicit PunchColumn
list. These build a flat slab on interior columns, verify the design stresses
and verdicts against the eurocodepy punch formulae, and check the reaction/force
source, the reinforcement-ratio input, the data round trip and the report.
"""
import context  # noqa: F401
import pytest

import eurocodepy.ec2.uls  # noqa: F401

from xdfem2d import Structure2D
from xdfem2d.punching import design_punching, design_punching_and_store
from xdfem2d.rc_design import design_tri_and_store

FCK, FYK = 30.0, 500.0


def _flat_slab(n=8, t=0.25, L=6.0, pz=-12.0, bx=0.40, by=0.40,
               position="center"):
    s = Structure2D(domain="plate")
    s.add_material("C", elastic_modulus=33e6, unit_weight=0.0, poisson=0.2,
                   material_type="Concrete", design={"fck": FCK, "fyk": FYK})
    s.add_plate_section("S", "C", thickness=t,
                        rc_cover_bot_x=0.035, rc_cover_bot_y=0.045,
                        rc_cover_top_x=0.035, rc_cover_top_y=0.045)
    ids = {}
    for j in range(n + 1):
        for i in range(n + 1):
            nid = f"N{i}_{j}"
            ids[(i, j)] = nid
            s.add_node(nid, L * i / n, L * j / n)
    e = 0
    for j in range(n):
        for i in range(n):
            a, b = ids[(i, j)], ids[(i + 1, j)]
            c, d = ids[(i + 1, j + 1)], ids[(i, j + 1)]
            s.add_tri_element(f"T{e}", a, b, c, "S"); e += 1
            s.add_tri_element(f"T{e}", a, c, d, "S"); e += 1
    cols = [(n // 4, n // 4), (3 * n // 4, n // 4),
            (n // 4, 3 * n // 4), (3 * n // 4, 3 * n // 4)]
    s.add_support("COL", w=True)
    for (i, j) in cols:
        s.assign_support(ids[(i, j)], "COL")
    s.add_load_case("G", self_weight_factor=0.0)
    for tid in list(s.tri_elements_by_id):
        s.add_area_load(tid, "G", pz=pz)
    s.add_load_combination("ULS", {"G": 1.35})
    for k, (i, j) in enumerate(cols):
        s.add_punch_column(f"C{k}", ids[(i, j)], shape="rectangular",
                           bx=bx, by=by, position=position)
    return s, ids


def test_punching_force_comes_from_the_reaction():
    s, _ = _flat_slab()
    res = s.calculate()
    rows = design_punching(s, res)
    assert len(rows) == 4
    # ΣN_Ed over the four columns balances the factored applied load.
    total = sum(r["N_Ed"] for r in rows)
    assert abs(total - 1.35 * 12.0 * 36.0) < 1.0
    # A light slab on generous columns is well within resistance.
    assert all(not r["needs_reinf"] and 0.0 < r["utilization"] < 1.0
               for r in rows)


def test_ved_matches_the_eurocodepy_formula():
    from eurocodepy.ec2.uls import calc_vedp
    s, _ = _flat_slab()
    res = s.calculate()
    row = design_punching(s, res)[0]
    dv = row["d"] * 1e3
    v = float(calc_vedp(row["N_Ed"], 0.0, 0.0, dv, 400.0, 400.0,
                        position="center"))
    assert abs(row["v_Ed"] - v) < 1e-6


def test_small_column_and_heavy_load_needs_reinforcement():
    # A small column under a heavy load punches: v_Ed exceeds v_Rd,c.
    s, _ = _flat_slab(pz=-60.0, bx=0.20, by=0.20)
    res = s.calculate()
    rows = design_punching(s, res)
    assert any(r["needs_reinf"] and r["utilization"] > 1.0 for r in rows)


def test_explicit_force_overrides_the_reaction():
    s, _ = _flat_slab()
    s.punch_columns[0].force = 1000.0            # impose a large design force
    res = s.calculate()
    row = next(r for r in design_punching(s, res) if r["column"] == "C0")
    assert abs(row["N_Ed"] - 1000.0) < 1e-9


def test_flexural_ratio_from_the_slab_design_raises_resistance():
    s, _ = _flat_slab(pz=-30.0)
    res = s.calculate()
    v_default = design_punching(s, res, edition="2023")[0]["v_Rdc"]
    # Design the slab flexure first: a realistic rho_l (> the 0.5% default here)
    # increases the punching resistance.
    design_tri_and_store(s, res)
    row = design_punching(s, res, edition="2023")[0]
    assert row["rho_l"] != 0.005                 # taken from the design now
    assert row["v_Rdc"] >= v_default


def test_beta_min_floors_the_eccentricity_factor():
    # Supports with no transferred moment give β at its ~1.05 floor; a user
    # beta_min (EC2 §6.4.3(6) simplified 1.15/1.4/1.5) raises it, and v_Ed with it.
    s, _ = _flat_slab()
    r0 = design_punching(s, s.calculate(), edition="2023")[0]
    assert abs(r0["beta"] - 1.05) < 0.01          # the computed floor
    for c in s.punch_columns:
        c.beta_min = 1.15
    r1 = design_punching(s, s.calculate(), edition="2023")[0]
    assert abs(r1["beta"] - 1.15) < 1e-9
    assert r1["v_Ed"] > r0["v_Ed"]
    assert abs(r1["v_Ed"] / r0["v_Ed"] - 1.15 / 1.05) < 1e-6


def test_max_resistance_and_outer_perimeter():
    # EN 1992-1-1:2023: τRd,max = η_sys·τRd,c (default η_sys = 1.5). A light
    # slab is below it (no crushing) and needs no reinforcement (u_out,ef None).
    s, _ = _flat_slab()
    r = design_punching(s, s.calculate(), edition="2023")[0]
    assert abs(r["v_Rd_max"] - 1.5 * r["v_Rdc"]) < 1e-9
    assert r["v_Ed"] < r["v_Rd_max"] and r["crushing"] is False
    assert r["u_out_eff"] is None
    # A heavy load on a small column needs reinforcement, and u_out,ef — the
    # perimeter beyond which it is no longer needed — is defined and past u1.
    s2, _ = _flat_slab(pz=-60.0, bx=0.20, by=0.20)
    bad = max(design_punching(s2, s2.calculate(), edition="2023"),
              key=lambda x: x["utilization"])
    assert bad["needs_reinf"] and bad["u_out_eff"] is not None
    assert bad["u_out_eff"] > bad["u1"]

def test_crushing_when_demand_exceeds_max_resistance():
    # η_sys = 1.0 lowers τRd,max to τRd,c; a section that needs reinforcement
    # then reads as crushing (demand above the capped resistance).
    s, _ = _flat_slab(pz=-60.0, bx=0.20, by=0.20)
    rows = design_punching(s, s.calculate(), eta_sys=1.0, edition="2023")
    assert any(r["crushing"] for r in rows)


def test_circular_column_is_supported():
    s, _ = _flat_slab()
    s.punch_columns[0].shape = "circular"
    s.punch_columns[0].bx = 0.45                 # diameter
    res = s.calculate()
    row = next(r for r in design_punching(s, res) if r["column"] == "C0")
    assert row["v_Ed"] > 0.0


def test_store_and_report():
    from xdfem2d.report_io import _results_tables
    s, _ = _flat_slab()
    res = s.calculate()
    payload = design_punching_and_store(s, res)
    assert payload is not None and res["punching"]["rows"]
    titles = [t[0] for t in _results_tables(res, domain="plate")]
    assert "Punching shear (EN 1992-1-1:2023)" in titles


def test_round_trip_of_the_columns():
    from xdfem2d.structure_io import _from_dict, _to_dict
    s, ids = _flat_slab()
    s.punch_columns[0].position = "corner"
    s.punch_columns[0].dx = 0.15
    s.punch_columns[0].beta_min = 1.5
    s2 = _from_dict(_to_dict(s))
    a, b = s.punch_columns[0], s2.punch_columns[0]
    assert (b.id, b.node_id, b.shape, b.bx, b.by, b.position, b.dx,
            b.beta_min) == \
           (a.id, a.node_id, a.shape, a.bx, a.by, a.position, a.dx, a.beta_min)
