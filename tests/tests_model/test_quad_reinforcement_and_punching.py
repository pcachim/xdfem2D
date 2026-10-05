"""dev/IMPLEMENT_QUAD.md Phase 7: quad-aware design (rc_design.py) and
punching-shear checks (punching.py) end to end.

Uses ``eurocodepy`` (the EC2 formulae live there), a required dependency. The
pure-Python ``punching._slab_section_at``/``_rho_l_at`` unit tests live
separately in test_punching_quad_lookup.py.
"""
import context  # noqa: F401
import pytest

from xdfem2d import Structure2D

FCK, FYK = 30.0, 500.0

import eurocodepy.ec2.uls  # noqa: F401

from xdfem2d.rc_design import (design_concrete_planes, design_concrete_slabs,
                               design_tri_and_store)
from xdfem2d.punching import design_punching, design_punching_and_store


def _quad_slab(n=6, t=0.25, L=4.0, pz=-30.0):
    """A DKT4 quad-only slab, structured grid, simply supported at the
    perimeter — the quad analogue of test_slab_reinforcement.py's ``_slab``."""
    s = Structure2D(domain="plate")
    s.add_material("C", elastic_modulus=33e6, unit_weight=0.0, poisson=0.2,
                   material_type="Concrete", design={"fck": FCK, "fyk": FYK})
    s.add_quad_section("S", "C", thickness=t, formulation="DKT4",
                       rc_cover_top_x=0.03, rc_cover_top_y=0.04,
                       rc_cover_bot_x=0.03, rc_cover_bot_y=0.04)
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
            s.add_quad_element(f"Q{e}", a, b, c, d, "S"); e += 1
    s.add_support("E", w=True)
    for (i, j), nid in ids.items():
        if i in (0, n) or j in (0, n):
            s.assign_support(nid, "E")
    s.add_load_case("G", self_weight_factor=0.0)
    for qid in list(s.quad_elements_by_id):
        s.add_area_load(qid, "G", pz=pz)
    s.add_load_combination("ULS", {"G": 1.35})
    return s, ids


def _q4_wall(L=2.0, H=2.0, n=4, sx_load=200.0):
    """A Q4 membrane wall, uniaxial tension along the top edge, for the
    Q4/QM6 branch of design_concrete_planes."""
    s = Structure2D()
    s.add_material("C", elastic_modulus=33e6, unit_weight=0.0, poisson=0.2,
                   material_type="Concrete", design={"fck": FCK, "fyk": FYK})
    s.add_quad_section("S", "C", thickness=0.2, formulation="Q4")
    ids = {}
    for i in range(n + 1):
        for j in range(2):
            nid = f"N{i}_{j}"
            ids[(i, j)] = nid
            s.add_node(nid, L * i / n, H * j)
    for i in range(n):
        a, b = ids[(i, 0)], ids[(i + 1, 0)]
        c, d = ids[(i + 1, 1)], ids[(i, 1)]
        s.add_quad_element(f"Q{i}", a, b, c, d, "S")
    s.add_support("FIX", ux=True, uy=True)
    for j in range(2):
        s.assign_support(ids[(0, j)], "FIX")
    s.add_load_case("T")
    for j in range(2):
        s.add_point_load(ids[(n, j)], "T", fx=sx_load / 2.0)
    # design_concrete_planes/design_concrete_slabs only read
    # results['combinations'] (see their own source) — an analysis case
    # alone, with no combination, gives them nothing to iterate.
    s.add_load_combination("ULS", {"T": 1.0})
    return s, ids


class TestDesignConcreteSlabsOnQuads:

    def test_bottom_steel_is_produced_for_quad_rows(self):
        s, _ = _quad_slab(pz=-40.0)
        rows = design_concrete_slabs(s, s.calculate())
        assert rows
        assert all(r["kind"] == "slab" for r in rows)
        gov = next(r for r in rows if r["governing"])
        assert gov["Asx_bot"] > 0.0 and gov["Asy_bot"] > 0.0
        # The row's 'triangle' key holds the quad's own id.
        assert gov["triangle"] in s.quad_elements_by_id

    def test_q4_quads_are_excluded_from_the_slab_design(self):
        # A Q4 (membrane) quad section is invalid in a plate-domain model
        # (Structure2D.domain_problems() rejects it), so this cannot be
        # exercised through a real calculate() on one structure — it builds
        # both a DKT4 and a Q4 quad section directly and fabricates the
        # 'results' dict design_concrete_slabs reads, the same way the cuts
        # tests fabricate tri_stress records, to isolate the formulation
        # filter from domain validation.
        s = Structure2D(domain="plate")
        s.add_material("C", elastic_modulus=33e6, unit_weight=0.0, poisson=0.2,
                       material_type="Concrete", design={"fck": FCK, "fyk": FYK})
        s.add_quad_section("SLAB", "C", thickness=0.25, formulation="DKT4",
                           rc_cover_top_x=0.03, rc_cover_top_y=0.04,
                           rc_cover_bot_x=0.03, rc_cover_bot_y=0.04)
        s.add_quad_section("MEMB", "C", thickness=0.25, formulation="Q4")
        s.add_node("A", 0, 0); s.add_node("B", 1, 0)
        s.add_node("C", 1, 1); s.add_node("D", 0, 1)
        s.add_quad_element("Q1", "A", "B", "C", "D", "SLAB")
        s.add_node("E", 2, 0); s.add_node("F", 2, 1)
        s.add_quad_element("Q2", "B", "E", "F", "C", "MEMB")
        results = {"combinations": {"ULS": {"tri_stress": {
            "Q1": {"mx_bot": 5.0, "my_bot": 5.0, "mx_top": 0.0, "my_top": 0.0,
                   "mx": 5.0, "my": 5.0, "mxy": 0.0, "formulation": "DKT4"},
            "Q2": {"sx": 100.0, "sy": 0.0, "txy": 0.0, "formulation": "Q4"},
        }}}}
        rows = design_concrete_slabs(s, results)
        ids_in_rows = {r["triangle"] for r in rows}
        assert "Q1" in ids_in_rows
        assert "Q2" not in ids_in_rows


class TestDesignConcreteTrianglesOnQuads:

    def test_membrane_design_runs_for_q4_quads(self):
        s, _ = _q4_wall()
        rows = design_concrete_planes(s, s.calculate())
        assert rows
        assert all("n_xx" in r for r in rows)
        gov = next(r for r in rows if r["governing"])
        assert gov["triangle"] in s.quad_elements_by_id
        assert gov["Asx"] >= 0.0

    def test_dkt4_quads_are_excluded_from_the_membrane_design(self):
        # Same reasoning as test_q4_quads_are_excluded_from_the_slab_design:
        # a DKT4 quad section is invalid in a plane-domain model, so this
        # isolates the formulation filter with a fabricated 'results' dict
        # rather than a real calculate().
        s = Structure2D()
        s.add_material("C", elastic_modulus=33e6, unit_weight=0.0, poisson=0.2,
                       material_type="Concrete", design={"fck": FCK, "fyk": FYK})
        s.add_quad_section("MEMB", "C", thickness=0.2, formulation="Q4")
        s.add_quad_section("PLATE", "C", thickness=0.2, formulation="DKT4")
        s.add_node("A", 0, 0); s.add_node("B", 1, 0)
        s.add_node("C", 1, 1); s.add_node("D", 0, 1)
        s.add_quad_element("Q1", "A", "B", "C", "D", "MEMB")
        s.add_node("E", 2, 0); s.add_node("F", 2, 1)
        s.add_quad_element("Q2", "B", "E", "F", "C", "PLATE")
        results = {"combinations": {"ULS": {"tri_stress": {
            "Q1": {"sx": 100.0, "sy": 0.0, "txy": 0.0, "formulation": "Q4"},
            "Q2": {"mx_bot": 5.0, "my_bot": 5.0, "mx_top": 0.0, "my_top": 0.0,
                   "mx": 5.0, "my": 5.0, "mxy": 0.0, "formulation": "DKT4"},
        }}}}
        rows = design_concrete_planes(s, results)
        ids_in_rows = {r["triangle"] for r in rows}
        assert "Q1" in ids_in_rows
        assert "Q2" not in ids_in_rows


class TestDesignTriAndStoreDispatchesQuadsToo:

    def test_mixed_tri_and_quad_slab_gets_rows_for_both(self):
        s = Structure2D(domain="plate")
        s.add_material("C", elastic_modulus=33e6, unit_weight=0.0, poisson=0.2,
                       material_type="Concrete", design={"fck": FCK, "fyk": FYK})
        s.add_tri_section("TS", "C", thickness=0.25, formulation="DKT",
                          rc_cover_top_x=0.03, rc_cover_top_y=0.04,
                          rc_cover_bot_x=0.03, rc_cover_bot_y=0.04)
        s.add_quad_section("QS", "C", thickness=0.25, formulation="DKT4",
                           rc_cover_top_x=0.03, rc_cover_top_y=0.04,
                           rc_cover_bot_x=0.03, rc_cover_bot_y=0.04)
        # Left half: two quads; right half: their diagonal-split triangle
        # twins — a genuinely mixed slab.
        pts = {"A": (0, 0), "B": (2, 0), "C": (2, 2), "D": (0, 2),
               "E": (4, 0), "F": (4, 2)}
        for nid, (x, y) in pts.items():
            s.add_node(nid, x, y)
        s.add_quad_element("Q1", "A", "B", "C", "D", "QS")
        s.add_tri_element("T1", "B", "E", "F", "TS")
        s.add_tri_element("T2", "B", "F", "C", "TS")
        s.add_support("E", w=True)
        for nid in ("A", "D", "E", "F"):
            s.assign_support(nid, "E")
        s.add_load_case("G", self_weight_factor=0.0)
        s.add_area_load("Q1", "G", pz=-30.0)
        s.add_area_load("T1", "G", pz=-30.0)
        s.add_area_load("T2", "G", pz=-30.0)
        s.add_load_combination("ULS", {"G": 1.35})

        res = s.calculate()
        payload = design_tri_and_store(s, res)
        assert payload is not None
        rows = res["tri_reinforcement"]["rows"]
        ids_in_rows = {r["triangle"] for r in rows}
        assert "Q1" in ids_in_rows
        assert {"T1", "T2"} & ids_in_rows


class TestPunchingOnQuadOnlySlab:

    def _flat_quad_slab(self, n=8, t=0.25, L=6.0, pz=-12.0, bx=0.40, by=0.40):
        s = Structure2D(domain="plate")
        s.add_material("C", elastic_modulus=33e6, unit_weight=0.0, poisson=0.2,
                       material_type="Concrete", design={"fck": FCK, "fyk": FYK})
        s.add_quad_section("S", "C", thickness=t, formulation="DKT4",
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
                s.add_quad_element(f"Q{e}", a, b, c, d, "S"); e += 1
        cols = [(n // 4, n // 4), (3 * n // 4, n // 4),
                (n // 4, 3 * n // 4), (3 * n // 4, 3 * n // 4)]
        s.add_support("COL", w=True)
        for (i, j) in cols:
            s.assign_support(ids[(i, j)], "COL")
        s.add_load_case("G", self_weight_factor=0.0)
        for qid in list(s.quad_elements_by_id):
            s.add_area_load(qid, "G", pz=pz)
        s.add_load_combination("ULS", {"G": 1.35})
        for k, (i, j) in enumerate(cols):
            s.add_punch_column(f"C{k}", ids[(i, j)], shape="rectangular",
                               bx=bx, by=by, position="center")
        return s, ids

    def test_punching_check_runs_on_a_quad_only_slab(self):
        s, _ = self._flat_quad_slab()
        res = s.calculate()
        rows = design_punching(s, res)
        assert len(rows) == 4
        assert all(r["N_Ed"] > 0.0 for r in rows)

    def test_rho_l_uses_the_quad_slab_design(self):
        s, _ = self._flat_quad_slab(pz=-30.0)
        res = s.calculate()
        v_default = design_punching(s, res, edition="2023")[0]["v_Rdc"]
        design_tri_and_store(s, res)
        row = design_punching(s, res, edition="2023")[0]
        assert row["rho_l"] != 0.005
        assert row["v_Rdc"] >= v_default

    def test_store_and_report(self):
        from xdfem2d.report_io import _results_tables
        s, _ = self._flat_quad_slab()
        res = s.calculate()
        payload = design_punching_and_store(s, res)
        assert payload is not None and res["punching"]["rows"]
        titles = [t[0] for t in _results_tables(res, domain="plate")]
        assert "Punching shear (EN 1992-1-1:2023)" in titles
