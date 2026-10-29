"""Phase 9B — EC2 flexural reinforcement of slabs (Wood-Armer + calc_asl).

A plate (DKT/MITC3) triangle is reinforced in four orthogonal directions —
bottom/top × x/y — each sized from the Wood-Armer design moment already
recovered per element, with its own effective depth d = t − cover. These check
the design against hand calculation, the sagging/hogging split, the per-cover
effective depth, and the domain dispatch.

The EC2 formulae live in eurocodepy, a required dependency.
"""
import math

import context  # noqa: F401
import pytest

import eurocodepy.ec2.uls  # noqa: F401

from xdfem2d import Structure2D
from xdfem2d.rc_design import (design_concrete_slabs, design_tri_and_store,
                               _fctm)

FCK, FYK = 30.0, 500.0


def _slab(n=8, t=0.25, L=5.0, pz=-15.0, support="simple", covers=None):
    s = Structure2D(domain="plate")
    s.add_material("C", elastic_modulus=33e6, unit_weight=0.0, poisson=0.2,
                   material_type="Concrete", design={"fck": FCK, "fyk": FYK})
    ckw = covers or dict(rc_cover_top_x=0.03, rc_cover_top_y=0.04,
                         rc_cover_bot_x=0.03, rc_cover_bot_y=0.04)
    s.add_plate_section("S", "C", thickness=t, **ckw)
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
    if support == "clamped":
        s.add_support("E", w=True, tx=True, ty=True)
    else:
        s.add_support("E", w=True)
    for (i, j), nid in ids.items():
        if i in (0, n) or j in (0, n):
            s.assign_support(nid, "E")
    s.add_load_case("G", self_weight_factor=0.0)
    for tid in list(s.tri_elements_by_id):
        s.add_area_load(tid, "G", pz=pz)
    s.add_load_combination("ULS", {"G": 1.35})
    return s, ids


def test_bottom_steel_matches_hand_calc_where_it_governs():
    s, _ = _slab(pz=-40.0)                      # heavy load → required governs
    rows = design_concrete_slabs(s, s.calculate())
    row = max(rows, key=lambda r: r["Asx_bot"])
    # As ≈ M / (0.9 d fyd) — the classic lever-arm estimate; calc_asl is exact
    # so it sits a few percent below this. d = t − cover = 0.25 − 0.03.
    d, fyd = 0.22, FYK / 1.15 * 1e3            # [kN/m²]
    hand = row["mx_bot"] / (0.9 * d * fyd)     # [m²/m]
    assert 0.90 < row["Asx_bot"] / hand < 1.0


def test_minimum_reinforcement_floors_small_moments():
    s, _ = _slab(pz=-4.0)                       # light load → min governs
    rows = design_concrete_slabs(s, s.calculate())
    as_min = max(0.26 * _fctm(FCK) / FYK, 0.0013) * (0.25 - 0.03)
    # Every tensioned bottom-x reinforcement is at least the EC2 minimum.
    tensioned = [r for r in rows if r["mx_bot"] > 0.0]
    assert tensioned
    assert all(r["Asx_bot"] >= as_min - 1e-12 for r in tensioned)


def test_simply_supported_slab_has_real_sagging_bottom_steel():
    s, _ = _slab(support="simple", pz=-60.0)      # heavy → bending governs
    rows = design_concrete_slabs(s, s.calculate())
    gov = next(r for r in rows if r["governing"])
    assert gov["Asx_bot"] > 0.0 and gov["Asy_bot"] > 0.0
    # The largest bottom demand comes from a genuine sagging moment, well above
    # the minimum — i.e. bending, not just the minimum floor.
    as_min = max(0.26 * _fctm(FCK) / FYK, 0.0013) * (0.25 - 0.03)
    max_bot = max(r["Asx_bot"] for r in rows)
    assert max_bot > 1.5 * as_min
    # (Per-element plate moments carry a twisting component mxy that Wood-Armer
    # turns into some top steel too; that is expected with unsmoothed per-element
    # recovery — see the design docstring — and not asserted against here.)


def test_clamped_slab_needs_top_steel_at_the_edges():
    s, _ = _slab(support="clamped", pz=-20.0)
    rows = design_concrete_slabs(s, s.calculate())
    assert any(r["Asx_top"] > 0.0 or r["Asy_top"] > 0.0 for r in rows)


def test_smaller_cover_gives_less_steel_for_equal_moment():
    # bot_x has a smaller cover than bot_y → larger d → less steel, at the
    # triangle where mx_bot ≈ my_bot.
    s, _ = _slab(pz=-30.0,
                 covers=dict(rc_cover_bot_x=0.03, rc_cover_bot_y=0.05,
                             rc_cover_top_x=0.03, rc_cover_top_y=0.05))
    rows = design_concrete_slabs(s, s.calculate())
    # The centre triangles carry mx_bot ≈ my_bot (symmetry) and both large; pick
    # the row with the largest of the smaller moment.
    near = max(rows, key=lambda r: min(r["mx_bot"], r["my_bot"]))
    assert near["mx_bot"] > 0 and abs(near["mx_bot"] - near["my_bot"]) < 1e-3
    # Equal moment, but bot_x has the smaller cover → larger d → less steel.
    assert near["Asx_bot"] < near["Asy_bot"]


def test_design_tri_and_store_dispatches_to_slabs_for_plate():
    s, _ = _slab(pz=-20.0)
    res = s.calculate()
    payload = design_tri_and_store(s, res)
    assert payload is not None
    rows = res["tri_reinforcement"]["rows"]
    assert rows and all(r.get("kind") == "slab" for r in rows)
    # Membrane-only keys are absent; slab moment keys are present.
    r = rows[0]
    assert "mx_bot" in r and "n_xx" not in r


def test_report_has_a_slab_reinforcement_table():
    from xdfem2d.report_io import _results_tables
    s, _ = _slab(pz=-20.0)
    res = s.calculate()
    design_tri_and_store(s, res)
    titles = [t[0] for t in _results_tables(res, domain="plate")]
    assert "Slab reinforcement (EC2 flexure)" in titles
    assert "Shell reinforcement (EC2 design)" not in titles
