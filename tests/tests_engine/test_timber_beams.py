"""Timber beams (dev/TIMBER_BEAMS.md, phase 0): the bars sharing a ``beam`` tag
are one beam -- path, spans, persistence and the tag-driven EC5 design."""
import os
import tempfile
import unittest

from context import Structure2D  # noqa: F401  (adds src/ to sys.path)

from xdfem2d import timber_design as T
from xdfem2d.structure_io import load_structure_json, save_structure_json
from xdfem2d.structure_io_checked import load_structure_json_checked
from xdfem2d.timber_beams import resolve_timber_beams


def _is_timber(s):
    return lambda b: T._is_timber(s, s.sections.get(b.section_name))


def _model(points, supports=()):
    """Chain of bars E1.. through *points* [(x, y)...], nodes n0.., timber."""
    s = Structure2D()
    for i, (x, y) in enumerate(points):
        s.add_node(f"n{i}", x, y)
    s.add_material("T", 11e6, 4.2, material_type="Timber",
                   design={"class": "GL24h"})
    s.add_section("GL", "T", b=0.14, h=0.6)
    for i in range(len(points) - 1):
        s.add_bar_element(f"E{i + 1}", f"n{i}", f"n{i + 1}", "GL")
    s.add_support("PIN", ux=True, uy=True)
    for n in supports:
        s.assign_support(n, "PIN")
    return s


def _results(ids, m_end=100.0):
    return {"element_forces": {}, "combinations": {"ULS1": {"element_forces": {
        i: {"i": [0, 40, -50], "j": [0, -40, m_end]} for i in ids}}}}


class TestResolve(unittest.TestCase):
    def test_pitched_chain_is_one_beam_ordered_left_to_right(self):
        # apex at n2: two bars rising, two falling -- not collinear
        s = _model([(0, 0), (3, 1), (6, 2), (9, 1), (12, 0)])
        s.assign_beam(["E4", "E3", "E2", "E1"])        # any order
        beams, warns = resolve_timber_beams(s, _is_timber(s))
        self.assertEqual(warns, [])
        self.assertEqual(len(beams), 1)
        b = beams[0]
        self.assertEqual(b.id, "V1")
        self.assertEqual(b.bar_ids, ["E1", "E2", "E3", "E4"])
        self.assertEqual(b.member.node_ids, ["n0", "n1", "n2", "n3", "n4"])
        self.assertAlmostEqual(b.member.length, 2 * 2 * (3 ** 2 + 1) ** 0.5)
        self.assertEqual(len(b.spans), 1)              # no support inside

    def test_right_to_left_input_is_reoriented(self):
        s = _model([(12, 0), (6, 2), (0, 0)])          # drawn right to left
        s.assign_beam(["E1", "E2"])
        b = resolve_timber_beams(s, _is_timber(s))[0][0]
        self.assertEqual(b.member.node_ids, ["n2", "n1", "n0"])

    def test_spans_split_at_supports_not_at_section_change(self):
        s = _model([(0, 0), (4, 0), (8, 0), (12, 0)], supports=("n0", "n2"))
        s.add_section("GL2", "T", b=0.14, h=0.5)
        s.bar_elements_by_id["E3"].section_name = "GL2"
        s.assign_beam(["E1", "E2", "E3"])
        b = resolve_timber_beams(s, _is_timber(s))[0][0]
        self.assertEqual([sp.bar_ids for sp in b.spans], [["E1", "E2"], ["E3"]])
        self.assertEqual(b.bar_ids, ["E1", "E2", "E3"])    # section change: one beam

    def test_branch_is_refused_with_warning(self):
        s = _model([(0, 0), (4, 0), (8, 0)])
        s.add_node("t", 4, 3)
        s.add_bar_element("E9", "n1", "t", "GL")
        s.assign_beam(["E1", "E2", "E9"])
        beams, warns = resolve_timber_beams(s, _is_timber(s))
        self.assertEqual(beams, [])
        self.assertEqual(len(warns), 1)

    def test_disconnected_pieces_become_separate_beams(self):
        s = _model([(0, 0), (4, 0), (8, 0), (12, 0)])
        s.assign_beam(["E1", "E3"])
        beams, _ = resolve_timber_beams(s, _is_timber(s))
        self.assertEqual([b.id for b in beams], ["V1.1", "V1.2"])

    def test_non_timber_bars_are_ignored_with_warning(self):
        s = _model([(0, 0), (4, 0), (8, 0)])
        s.add_material("C", 33e6, 25.0, material_type="Concrete",
                       design={"fck": 30, "fyk": 500})
        s.add_section("CONC", "C", b=0.3, h=0.5)
        s.bar_elements_by_id["E2"].section_name = "CONC"
        s.assign_beam(["E1", "E2"])
        beams, warns = resolve_timber_beams(s, _is_timber(s))
        self.assertEqual(beams[0].bar_ids, ["E1"])
        self.assertEqual(len(warns), 1)


class TestPersistence(unittest.TestCase):
    def test_timber_params_roundtrip_and_survive_prune(self):
        s = _model([(0, 0), (4, 0)])
        tag = s.assign_beam(["E1"])
        s.beams[tag]["timber"] = {"lateral_spacing": 1.5}
        s.clear_beam(["E1"])
        self.assertIn(tag, s.beams)                    # params keep the entry
        s.assign_beam(["E1"], tag=tag)
        with tempfile.TemporaryDirectory() as t:
            p = os.path.join(t, "m.json")
            save_structure_json(s, p)
            for loader in (load_structure_json, load_structure_json_checked):
                r = loader(p)
                r = r[0] if isinstance(r, tuple) else r
                self.assertEqual(r.beams[tag]["timber"],
                                 {"lateral_spacing": 1.5})

    def test_no_timber_key_when_unused(self):
        s = _model([(0, 0), (4, 0)])
        s.assign_beam(["E1"])
        from xdfem2d.structure_io import _to_dict
        self.assertNotIn("timber", _to_dict(s)["beams"]["V1"])


class TestDesign(unittest.TestCase):
    def test_tagged_pitched_beam_is_one_member_row(self):
        s = _model([(0, 0), (3, 1), (6, 2), (9, 1), (12, 0)],
                   supports=("n0", "n4"))
        ids = ["E1", "E2", "E3", "E4"]
        s.assign_beam(ids, name="Main")
        out = T.design_timber_members(s, _results(ids))
        self.assertEqual([r["member"] for r in out["members"]], ["V1"])
        self.assertEqual(out["members"][0]["elements"], "E1, E2, E3, E4")
        self.assertAlmostEqual(out["members"][0]["length"],
                               2 * 2 * (3 ** 2 + 1) ** 0.5)
        self.assertNotIn("warnings", out)

    def test_untagged_keeps_the_per_member_grouping(self):
        s = _model([(0, 0), (3, 1), (6, 2), (9, 1), (12, 0)],
                   supports=("n0", "n4"))
        ids = ["E1", "E2", "E3", "E4"]
        out = T.design_timber_members(s, _results(ids))
        self.assertEqual(sorted(r["member"] for r in out["members"]),
                         ["M1", "M2"])               # kink at the apex

    def test_lateral_spacing_sets_the_ltb_length(self):
        def util(spacing):
            s = _model([(0, 0), (6, 0), (12, 0)], supports=("n0", "n2"))
            tag = s.assign_beam(["E1", "E2"])
            if spacing:
                s.beams[tag]["timber"] = {"lateral_spacing": spacing}
            r = T.design_timber_members(s, _results(["E1", "E2"], 400.0))
            return r["members"][0]["util_bending"]
        # a shorter unrestrained length can only help (k_crit grows)
        self.assertLessEqual(util(1.0), util(None))
        self.assertLess(util(1.0), util(12.0))

    def test_branch_warning_reaches_the_result(self):
        s = _model([(0, 0), (4, 0), (8, 0)], supports=("n0", "n2"))
        s.add_node("t", 4, 3)
        s.add_bar_element("E9", "n1", "t", "GL")
        s.assign_beam(["E1", "E2", "E9"])
        out = T.design_timber_members(s, _results(["E1", "E2", "E9"]))
        self.assertEqual(len(out["warnings"]), 1)


if __name__ == "__main__":
    unittest.main()


# ── phases 1-2: geometry read from the model, EC5 §6.4 ───────────────────────

import math  # noqa: E402

from xdfem2d.timber_beams import beam_geometry  # noqa: E402


def _tapered(pitch_deg=5.0, h0=0.5, n=6, span=12.0):
    """Double tapered beam: bottom horizontal at y=0, top rising to the middle.
    The bars follow the centroid; each has the depth of its mid-point."""
    s = Structure2D()
    s.add_material("T", 11e6, 4.2, material_type="Timber",
                   design={"class": "GL24h"})
    tn = math.tan(math.radians(pitch_deg))

    def depth(x):
        return h0 + tn * min(x, span - x)
    xs = [span * k / n for k in range(n + 1)]
    for i, x in enumerate(xs):
        s.add_node(f"n{i}", x, depth(x) / 2)
    for i in range(n):
        xm = 0.5 * (xs[i] + xs[i + 1])
        s.add_section(f"S{i}", "T", b=0.14, h=round(depth(xm), 6))
        s.add_bar_element(f"E{i + 1}", f"n{i}", f"n{i + 1}", f"S{i}")
    s.add_support("PIN", ux=True, uy=True)
    s.assign_support("n0", "PIN")
    s.assign_support(f"n{n}", "PIN")
    return s, [f"E{i + 1}" for i in range(n)]


def _geom(s, ids, **params):
    tag = s.assign_beam(ids)
    if params:
        s.beams[tag]["timber"] = params
    beams, _ = resolve_timber_beams(s, _is_timber(s))
    return beams[0], beam_geometry(s, beams[0])


class TestGeometry(unittest.TestCase):
    def test_double_tapered_is_read_from_the_bars(self):
        s, ids = _tapered()
        _, g = _geom(s, ids)
        self.assertEqual(g.depth_kind, "tapered")
        self.assertEqual(g.taper_edge, "top")          # bottom is the flat one
        self.assertEqual(g.kind, "double_tapered")
        self.assertEqual(list(g.apex.nodes), ["n3"])
        an = g.apex.nodes["n3"]
        self.assertAlmostEqual(an.h, 0.5 + 6 * math.tan(math.radians(5)), 4)
        self.assertAlmostEqual(math.degrees(an.alpha_ap), 5.0, 1)
        self.assertIsNone(an.r_in)
        self.assertEqual(g.apex.inner, "bottom")        # ridge: centre below
        for bg in g.bars.values():                      # angle between edges
            self.assertAlmostEqual(math.degrees(bg.alpha), 5.0, 1)

    def test_depth_at_a_station_is_interpolated_not_the_bar_value(self):
        s, ids = _tapered()
        _, g = _geom(s, ids)
        bg = g.bars["E1"]                                # x = 0 … 2
        self.assertAlmostEqual(bg.h_at(0.0), 0.5, 4)
        self.assertAlmostEqual(bg.h_at(bg.length), 0.5 + 2 * math.tan(
            math.radians(5)), 3)

    def test_constant_depth_and_single_step(self):
        s = _model([(0, 0), (4, 0), (8, 0)])
        s.assign_beam(["E1", "E2"])
        b, _ = resolve_timber_beams(s, _is_timber(s))
        self.assertEqual(beam_geometry(s, b[0]).depth_kind, "constant")
        s.add_section("GL2", "T", b=0.14, h=0.5)
        s.bar_elements_by_id["E2"].section_name = "GL2"
        self.assertEqual(beam_geometry(s, b[0]).depth_kind, "step")
        self.assertIsNone(beam_geometry(s, b[0]).taper_edge)

    def test_override_taper_edge(self):
        s, ids = _tapered()
        _, g = _geom(s, ids, taper_edge="bottom")
        self.assertEqual(g.taper_edge, "bottom")

    def test_constant_depth_pitched_ridge(self):
        s = _model([(0, 0), (3, 1), (6, 2), (9, 1), (12, 0)])
        _, g = _geom(s, ["E1", "E2", "E3", "E4"])
        self.assertEqual(g.kind, "double_tapered")
        self.assertAlmostEqual(math.degrees(g.apex.nodes["n2"].alpha_ap),
                               math.degrees(math.atan(1 / 3)), 1)

    def test_arc_gives_a_curved_beam_with_its_radius(self):
        R, step = 20.0, math.radians(5)
        pts = [(R * math.sin(k * step), R * math.cos(k * step) - R)
               for k in range(-6, 7)]
        s = _model(pts)
        _, g = _geom(s, [f"E{i + 1}" for i in range(12)])
        self.assertEqual(g.kind, "curved")
        r = [a.r_in for a in g.apex.nodes.values()]
        self.assertTrue(all(abs(x - (R - 0.3)) < 0.3 for x in r), r)
        self.assertEqual(len(g.apex.curved_bars), 12)

    def test_radius_override(self):
        R, step = 20.0, math.radians(5)
        s = _model([(R * math.sin(k * step), R * math.cos(k * step) - R)
                    for k in range(-6, 7)])
        _, g = _geom(s, [f"E{i + 1}" for i in range(12)], r_in=15.0)
        self.assertTrue(all(a.r_in == 15.0 for a in g.apex.nodes.values()))

    def test_coarse_taper_is_warned(self):
        s, ids = _tapered(pitch_deg=20.0, n=4)
        tb, _ = _geom(s, ids)
        g = beam_geometry(s, tb, step_warn=0.05)
        self.assertTrue(any("too coarsely" in w for w in g.warnings))

    def test_circular_section_has_no_geometry(self):
        s = _model([(0, 0), (4, 0)])
        s.add_section("CIR", "T", b=0.3, h=0.3,
                      shape=__import__("xdfem2d").models.SectionShape.CIRCULAR)
        s.bar_elements_by_id["E1"].section_name = "CIR"
        s.assign_beam(["E1"])
        b, _ = resolve_timber_beams(s, _is_timber(s))
        self.assertIsNone(beam_geometry(s, b[0]))


class TestSpecialDesign(unittest.TestCase):
    def _design(self, s, ids, m=300.0, tag_params=None):
        tag = s.assign_beam(ids)
        if tag_params:
            s.beams[tag]["timber"] = tag_params
        res = {"element_forces": {}, "combinations": {"ULS1": {"element_forces": {
            i: {"i": [0, 40, m], "j": [0, -40, m]} for i in ids}}}}
        return T.design_timber_members(s, res)

    def test_double_tapered_reports_taper_and_apex_checks(self):
        s, ids = _tapered()
        out = self._design(s, ids)
        row = out["members"][0]
        self.assertEqual(row["beam_kind"], "double_tapered")
        for k in ("taper", "apex_bending", "apex_t90", "apex_inter"):
            self.assertGreater(row["special"][k]["util"], 0.0, k)
        # the apex node is reported with where it governs
        self.assertGreaterEqual(row["util_bending"], row["special"]["taper"]["util"])
        self.assertGreaterEqual(row["util_shear"], row["special"]["apex_t90"]["util"])

    def test_tension_edge_governs_over_compression_edge(self):
        s, ids = _tapered()
        sag = self._design(s, ids, m=300.0)["members"][0]["special"]["taper"]["util"]
        s2, ids2 = _tapered()
        hog = self._design(s2, ids2, m=-300.0)["members"][0]["special"]["taper"]["util"]
        # sagging puts tension on the bottom (flat) edge, hogging on the cut one
        self.assertGreater(hog, sag)

    def test_no_tension_perpendicular_when_hogging_at_a_ridge(self):
        s, ids = _tapered()
        sag = self._design(s, ids, m=300.0)["members"][0]["special"]
        s2, ids2 = _tapered()
        hog = self._design(s2, ids2, m=-300.0)["members"][0]["special"]
        self.assertIn("apex_t90", sag)
        self.assertNotIn("apex_t90", hog)       # curvature increases: no t90

    def test_untagged_or_straight_beams_have_no_special_block(self):
        s = _model([(0, 0), (6, 0), (12, 0)], supports=("n0", "n2"))
        tag = s.assign_beam(["E1", "E2"])
        res = _results(["E1", "E2"])
        row = T.design_timber_members(s, res)["members"][0]
        self.assertNotIn("special", row)
        self.assertNotIn("beam_kind", row)


class TestAgainstDirectFormulas(unittest.TestCase):
    def test_engine_equals_the_formulas_at_known_points(self):
        from eurocodepy.ec5 import uls
        from eurocodepy.ec5.materials import (LoadDuration, ServiceClass,
                                              TimberClass)
        s, ids = _tapered()
        s.assign_beam(ids)
        res = {"element_forces": {}, "combinations": {"ULS1": {
            "element_forces": {i: {"i": [0, 40, 300.0], "j": [0, -40, 300.0]}
                               for i in ids}}}}
        sp = T.design_timber_members(s, res)["members"][0]["special"]
        t = TimberClass("GL24h")
        t.safety = 1.25                      # Code-preferences default, glulam
        a, svc, ld = (math.radians(5), ServiceClass.SC1,
                      LoadDuration.MediumDuration)
        # sagging: the flat (bottom) edge is in tension, the cut edge in
        # compression; the thinnest section (h = 0.5 m, bar E1 end i) governs
        r = uls.check_tapered_edge(300.0, 0.14, 0.5, a, False, t, svc, ld)
        self.assertAlmostEqual(sp["taper"]["util"], r.util_cut, delta=0.01)
        h_ap = 0.5 + 6 * math.tan(a)
        ap = uls.check_apex_zone(uls.ApexInput(
            300, 40, 0.14, h_ap, a, None, 33, "double_tapered", True,
            beam_volume=sum(0.14 * 2 * (0.5 + 2 * math.tan(a) * (k + 0.5)
                                        if k < 3 else
                                        0.5 + 2 * math.tan(a) * (5 - k + 0.5))
                            for k in range(6))), t, svc, ld)
        self.assertAlmostEqual(sp["apex_bending"]["util"], ap.util_bending,
                               delta=0.01)
        self.assertAlmostEqual(sp["apex_t90"]["util"], ap.util_tension90,
                               delta=0.01)
        self.assertAlmostEqual(sp["apex_inter"]["util"], ap.util_interaction,
                               delta=0.01)


class TestReport(unittest.TestCase):
    def _out(self, m=300.0):
        s, ids = _tapered()
        s.assign_beam(ids)
        res = {"element_forces": {}, "combinations": {"ULS1": {
            "element_forces": {i: {"i": [0, 40, m], "j": [0, -40, m]}
                               for i in ids}}}}
        return T.design_timber_members(s, res, with_reports=True)

    def test_report_has_geometry_and_the_6_4_sections_at_their_points(self):
        out = self._out()
        rep = out["reports"]["V1"]
        titles = [s["title"] for s in rep["sections"]]
        self.assertIn("Beam geometry (read from the model)", titles)
        taper = [t for t in titles if t.startswith("Tapered edge (§6.4.2)")]
        apex = [t for t in titles if t.startswith("Apex zone (§6.4.3)")]
        self.assertEqual((len(taper), len(apex)), (1, 1))   # apex traced once
        row = out["members"][0]["special"]
        self.assertTrue(taper[0].endswith(row["taper"]["where"]))
        self.assertTrue(apex[0].endswith(row["apex_t90"]["where"]))
        # the §6.4 blocks come before the combined utilisation
        self.assertLess(max(titles.index(t) for t in taper + apex),
                        min((i for i, t in enumerate(titles)
                             if t.startswith("Combined")), default=len(titles)))

    def test_traced_numbers_are_the_reported_utilisations(self):
        out = self._out()
        rep = out["reports"]["V1"]
        sp = out["members"][0]["special"]
        steps = {st["symbol"]: st["value"] for s in rep["sections"]
                 for st in s["steps"]}
        self.assertAlmostEqual(
            steps["σ_m,α,d / (k_m,α·f_m,d)"], sp["taper"]["util"], places=6)
        self.assertAlmostEqual(
            steps["τ_d/f_v,d + σ_t,90,d/(k_dis·k_vol·f_t,90,d)"],
            sp["apex_inter"]["util"], places=6)

    def test_no_special_sections_for_a_straight_beam(self):
        s = _model([(0, 0), (6, 0), (12, 0)], supports=("n0", "n2"))
        s.assign_beam(["E1", "E2"])
        out = T.design_timber_members(s, _results(["E1", "E2"]),
                                      with_reports=True)
        titles = [x["title"] for x in out["reports"]["V1"]["sections"]]
        self.assertFalse([t for t in titles if "§6.4" in t or "geometry" in t])


class TestScopeOfTheClauses(unittest.TestCase):
    """EN 1995-1-1 §6.4.3(1)-(4): apex clause for glulam/LVL only; the
    single-taper check outside the apex zone; M_ap and α_ap of the middle."""

    def _geom_and_bars(self):
        from xdfem2d.timber_beams import special_point_checks
        from eurocodepy.ec5 import uls
        from eurocodepy.ec5.materials import (LoadDuration, ServiceClass,
                                              TimberClass)
        s, ids = _tapered()
        tb, g = _geom(s, ids)
        t = TimberClass("GL24h")

        def at(bar, label):
            return special_point_checks(uls, g, s.bar_elements_by_id[bar],
                                        label, 10.0, 100.0, t,
                                        ServiceClass.SC1,
                                        LoadDuration.MediumDuration)
        return g, at

    def test_taper_check_is_skipped_inside_the_apex_zone(self):
        g, at = self._geom_and_bars()
        h_ap = g.apex.nodes["n3"].h
        self.assertNotIn("taper", at("E3", "j"))       # at the apex node
        self.assertNotIn("taper", at("E4", "i"))
        self.assertIn("taper", at("E3", "i"))          # 2 m away > h_ap/2
        self.assertGreater(2.0, h_ap / 2)
        self.assertIn("apex_bending", at("E3", "j"))   # the apex check is there
        self.assertNotIn("apex_bending", at("E3", "i"))

    def test_arc_checks_only_the_middle_of_the_zone(self):
        R, step = 20.0, math.radians(5)
        s = _model([(R * math.sin(k * step), R * math.cos(k * step) - R)
                    for k in range(-6, 7)])
        _, g = _geom(s, [f"E{i + 1}" for i in range(12)])
        self.assertEqual(list(g.apex.nodes), ["n6"])
        arc = 11 * 2 * R * math.sin(step / 2)           # 11 chords of the arc
        self.assertAlmostEqual(g.apex.zone_volume, 0.14 * 0.6 * arc, delta=0.01)
        self.assertEqual(len(g.apex.curved_bars), 12)

    def test_even_arc_checks_the_two_central_nodes(self):
        R, step = 20.0, math.radians(5)
        s = _model([(R * math.sin(k * step), R * math.cos(k * step) - R)
                    for k in range(-5, 6)])          # 11 nodes, 9 turning
        _, g = _geom(s, [f"E{i + 1}" for i in range(10)])
        self.assertEqual(len(g.apex.nodes), 1)
        s = _model([(R * math.sin(k * step), R * math.cos(k * step) - R)
                    for k in range(-5, 5)])          # 10 nodes, 8 turning
        _, g = _geom(s, [f"E{i + 1}" for i in range(9)])
        self.assertEqual(len(g.apex.nodes), 2)

    def test_warning_for_apex_clause_on_solid_timber(self):
        s, ids = _tapered()
        s.materials["T"].design = {"class": "C24"}
        s.assign_beam(ids)
        res = {"element_forces": {}, "combinations": {"ULS1": {
            "element_forces": {i: {"i": [0, 4, 30.0], "j": [0, -4, 30.0]}
                               for i in ids}}}}
        out = T.design_timber_members(s, res)
        self.assertTrue(any("glued laminated timber and LVL" in w
                            for w in out.get("warnings", [])))


class TestProfile(unittest.TestCase):
    def _run(self, with_profile=True):
        s, ids = _tapered()
        s.assign_beam(ids)
        q, L, n = 20.0, 12.0, 6
        elems, dist = {}, {}
        for k, i in enumerate(ids):
            x0 = L * k / n
            xs_ = [x0 + 2.0 * j / 4 for j in range(5)]          # 5 stations / bar
            m = [q * x * (L - x) / 2 for x in xs_]
            v = [q * (L / 2 - x) for x in xs_]
            dist[i] = {"x": [x - x0 for x in xs_], "N": [0.0] * 5,
                       "V": v, "M": m}
            elems[i] = {"i": [0, v[0], m[0]], "j": [0, -v[-1], m[-1]]}
        res = {"element_forces": {}, "combinations": {"ULS1": {
            "element_forces": elems}}, "combo_distribution": {"ULS1": dist}}
        return T.design_timber_members(s, res, with_profile=with_profile)["members"][0]

    def test_profile_and_outline_only_when_asked(self):
        self.assertNotIn("profile", self._run(False))
        self.assertIn("profile", self._run())

    def test_profile_runs_along_the_whole_beam(self):
        row = self._run()
        ss = [p["s"] for p in row["profile"]]
        self.assertEqual(ss, sorted(ss))
        self.assertAlmostEqual(ss[0], 0.0, 3)
        self.assertAlmostEqual(ss[-1], row["outline"]["length"], 2)
        for k in ("bending", "shear", "taper", "apex"):
            self.assertTrue(all(k in p for p in row["profile"]))

    def test_profile_peak_is_the_member_utilisation(self):
        row = self._run()
        self.assertAlmostEqual(max(p["bending"] for p in row["profile"]),
                               row["util_bending"], places=6)
        self.assertAlmostEqual(max(p["shear"] for p in row["profile"]),
                               row["util_shear"], places=6)

    def test_taper_is_absent_in_the_apex_zone_and_apex_only_there(self):
        row = self._run()
        apex_s = row["outline"]["apex"][0]
        mid = min(row["profile"], key=lambda p: abs(p["s"] - apex_s))
        self.assertEqual(mid["taper"], 0.0)
        self.assertGreater(mid["apex"], 0.0)
        far = row["profile"][0]
        self.assertEqual(far["apex"], 0.0)

    def test_outline_is_the_beam_as_drawn(self):
        o = self._run()["outline"]
        self.assertEqual(len(o["top"]), len(o["bottom"]), 7)
        for t_, b_ in zip(o["top"], o["bottom"]):
            self.assertGreater(t_[1], b_[1])
        self.assertAlmostEqual(o["bottom"][0][1], 0.0, 3)          # flat soffit
        self.assertAlmostEqual(o["bottom"][3][1], 0.0, 3)
        self.assertAlmostEqual(o["supports"][0], 0.0)
        self.assertAlmostEqual(o["supports"][-1], o["length"], 3)
        self.assertEqual(o["kind"], "double_tapered")


class TestProfileDoesNotChangeTheResult(unittest.TestCase):
    def test_results_do_not_depend_on_with_profile(self):
        a = TestProfile()._run(with_profile=False)
        b = TestProfile()._run(with_profile=True)
        for k in ("util_bending", "util_shear", "utilization", "passed"):
            self.assertEqual(a[k], b[k], k)
        self.assertEqual(a["special"], b["special"])

    def test_standard_check_uses_the_local_depth_of_a_tapered_beam(self):
        # constant moment: a thin end must be more stressed than a thick one,
        # inside the same bar (the bar's own depth is the mid-bar value)
        s, ids = _tapered()
        s.assign_beam(ids)
        m = 100.0
        res = {"element_forces": {}, "combinations": {"ULS1": {"element_forces": {
            i: {"i": [0, 0, m], "j": [0, 0, m]} for i in ids}}}}
        out = T.design_timber_members(s, res, with_profile=True)["members"][0]
        first = [p for p in out["profile"] if p["s"] < 2.1]
        self.assertGreater(first[0]["bending"], first[-1]["bending"])


# ── automatic detection and ordering ─────────────────────────────────────────

from xdfem2d.timber_beams import order_bars_along, suggest_timber_beams  # noqa: E402


def _suggest(s):
    return suggest_timber_beams(s, lambda b: T._is_timber(
        s, s.sections.get(b.section_name)))


class TestSuggest(unittest.TestCase):
    def test_pitched_chain_is_one_beam_and_posts_are_left_out(self):
        s = _model([(0, 3), (3, 4), (6, 5), (9, 4), (12, 3)])
        for i, x in enumerate((0, 12)):
            s.add_node(f"p{i}", x, 0)
            s.add_bar_element(f"P{i}", f"p{i}", f"n{4 * i}", "GL")
        self.assertEqual(_suggest(s), [["E1", "E2", "E3", "E4"]])

    def test_through_beam_keeps_going_and_the_branch_starts_its_own(self):
        s = _model([(0, 0), (4, 0), (8, 0)])
        s.add_node("u", 8, 0.5)           # shallow secondary beam off n1
        s.add_bar_element("E9", "n1", "u", "GL")
        got = _suggest(s)
        self.assertIn(["E1", "E2"], got)
        self.assertIn(["E9"], got)
        self.assertEqual(len(got), 2)

    def test_a_sharp_turn_does_not_join(self):
        s = _model([(0, 0), (4, 0)])
        s.add_node("v", 4, 4)             # 90° turn, but also a 90° slope
        s.add_bar_element("E2", "n1", "v", "GL")
        self.assertEqual(_suggest(s), [["E1"]])        # E2 is steep: left out
        s2 = _model([(0, 0), (4, 0), (4.0 + 3.0, 3.0)])  # 45° up: a 45° turn
        self.assertEqual(_suggest(s2), [["E1", "E2"]])
        s3 = _model([(0, 0), (4, 0), (4.0 - 3.0, 3.0)])  # 135° turn: refused
        self.assertEqual(sorted(_suggest(s3)), [["E1"], ["E2"]])

    def test_beams_are_ordered_bottom_to_top_then_left_to_right(self):
        s = _model([(0, 0), (4, 0)])
        for name, pts in (("a", [(10, 0), (14, 0)]), ("b", [(0, 3), (4, 3)])):
            for k, (x, y) in enumerate(pts):
                s.add_node(f"{name}{k}", x, y)
            s.add_bar_element(f"F{name}", f"{name}0", f"{name}1", "GL")
        self.assertEqual(_suggest(s), [["E1"], ["Fa"], ["Fb"]])

    def test_each_beam_runs_left_to_right_whatever_the_drawing_direction(self):
        s = _model([(12, 0), (6, 1), (0, 0)])         # drawn right to left
        self.assertEqual(_suggest(s), [["E2", "E1"]])

    def test_other_materials_and_sharp_corners_are_not_joined(self):
        s = _model([(0, 0), (4, 1), (8, 0), (4, -1)])      # a shallow diamond
        s.add_bar_element("E4", "n3", "n0", "GL")
        s.add_material("C", 33e6, 25.0, material_type="Concrete",
                       design={"fck": 30, "fyk": 500})
        s.add_section("CONC", "C", b=0.3, h=0.5)
        s.add_node("c0", 20, 0)
        s.add_node("c1", 24, 0)
        s.add_bar_element("C1", "c0", "c1", "CONC")
        flat = [b for chain in _suggest(s) for b in chain]
        self.assertNotIn("C1", flat)                 # concrete: not timber
        # the corners at both ends turn 152°: the diamond never closes
        self.assertEqual(sorted(map(sorted, _suggest(s))),
                         [["E1", "E2"], ["E3", "E4"]])

    def test_columns_flagged_are_left_out(self):
        s = _model([(0, 0), (4, 0)])
        s.bar_elements_by_id["E1"].is_column = True
        self.assertEqual(_suggest(s), [])


class TestOrderAlong(unittest.TestCase):
    def test_chain_is_ordered_from_the_left_end(self):
        s = _model([(12, 0), (6, 1), (0, 0)])
        self.assertEqual(order_bars_along(s, ["E1", "E2"]), ["E2", "E1"])

    def test_pitched_chain_in_any_input_order(self):
        s = _model([(0, 0), (3, 1), (6, 2), (9, 1), (12, 0)])
        self.assertEqual(order_bars_along(s, ["E3", "E1", "E4", "E2"]),
                         ["E1", "E2", "E3", "E4"])

    def test_branch_falls_back_to_midpoints(self):
        s = _model([(0, 0), (4, 0), (8, 0)])
        s.add_node("t", 4, 3)
        s.add_bar_element("E9", "n1", "t", "GL")
        got = order_bars_along(s, ["E9", "E2", "E1"])
        self.assertEqual(sorted(got), ["E1", "E2", "E9"])
        self.assertEqual(got, order_bars_along(s, ["E1", "E9", "E2"]))   # stable

    def test_unknown_ids_are_ignored(self):
        s = _model([(0, 0), (4, 0)])
        self.assertEqual(order_bars_along(s, ["E1", "NOPE"]), ["E1"])


class TestEdition2025(unittest.TestCase):
    """EN 1995-1-1:2025 §8.3.3 / §8.3.4 through the design (ec5_edition=2025)."""

    P25 = {"ec5_edition": "2025"}

    def _design(self, m=300.0, prefs=None, pitch=5.0, with_reports=False):
        s, ids = _tapered(pitch_deg=pitch)
        s.assign_beam(ids)
        res = {"element_forces": {}, "combinations": {"ULS1": {
            "element_forces": {i: {"i": [0, 40, m], "j": [0, -40, m]}
                               for i in ids}}}}
        return T.design_timber_members(s, res, prefs=prefs,
                                       with_reports=with_reports)

    def test_the_apex_uses_the_2025_factors(self):
        a04 = self._design()["members"][0]["special"]
        a25 = self._design(prefs=self.P25)["members"][0]["special"]
        self.assertNotAlmostEqual(a04["apex_t90"]["util"],
                                  a25["apex_t90"]["util"], places=3)
        # the taper check is the same for glulam (k_τ 0.75 / 1.5 as in 2004)
        self.assertAlmostEqual(a04["taper"]["util"], a25["taper"]["util"],
                               delta=0.01 * a04["taper"]["util"])

    def test_engine_equals_the_2025_formulas(self):
        from eurocodepy.ec5 import uls2025 as u
        from eurocodepy.ec5.materials import (LoadDuration, ServiceClass,
                                              TimberClass)
        sp = self._design(prefs=self.P25)["members"][0]["special"]
        t = TimberClass("GL24h")
        t.safety = 1.25
        a = math.radians(5)
        h_ap = 0.5 + 6 * math.tan(a)
        ap = u.check_apex_zone(u.ApexInput(300, 40, 0.14, h_ap, a, None, 33,
                                           "double_tapered", True), t,
                               ServiceClass.SC1, LoadDuration.MediumDuration)
        self.assertAlmostEqual(sp["apex_t90"]["util"], ap.util_tension90,
                               delta=0.01)
        self.assertAlmostEqual(sp["apex_inter"]["util"], ap.util_interaction,
                               delta=0.01)

    def test_report_cites_the_2025_clauses(self):
        rep = self._design(prefs=self.P25, with_reports=True)["reports"]["V1"]
        titles = [s["title"] for s in rep["sections"]]
        self.assertTrue(any(t.startswith("Apex zone (§8.3.4)") for t in titles))
        self.assertTrue(any(t.startswith("Tapered edge (§8.3.3)") for t in titles))
        self.assertFalse(any("§6.4" in t for t in titles))
        clauses = {st["clause"] for s_ in rep["sections"] for st in s_["steps"]}
        self.assertIn("EN 1995-1-1:2025 Eq. 8.68", clauses)

    def test_reinforcement_is_advised_not_required_above_60_percent(self):
        out = self._design(m=2500.0, prefs=self.P25)
        row = out["members"][0]
        self.assertIn("8.3.1(4)", row["reinforcement_advised"])
        self.assertIn("recommended", row["reinforcement_advised"])
        warn = [w for w in out["warnings"] if "reinforcement" in w][0]
        self.assertIn("optional but advised", warn)
        self.assertNotIn("required", warn)
        self.assertGreater(row["special"]["apex_t90"]["util"], 0.0)  # still checked
        low = self._design(m=5.0, prefs=self.P25)
        self.assertNotIn("reinforcement_advised", low["members"][0])

    def test_2004_does_not_report_reinforcement(self):
        out = self._design(m=2500.0)
        self.assertNotIn("reinforcement_advised", out["members"][0])

    def test_taper_over_15_degrees_is_warned_only_in_2025(self):
        w25 = self._design(pitch=20.0, prefs=self.P25).get("warnings", [])
        w04 = self._design(pitch=20.0).get("warnings", [])
        self.assertTrue(any("15°" in w for w in w25))
        self.assertFalse(any("15°" in w for w in w04))

    def test_scope_warning_names_the_2025_clause(self):
        s, ids = _tapered()
        s.materials["T"].design = {"class": "C24"}
        s.assign_beam(ids)
        res = {"element_forces": {}, "combinations": {"ULS1": {
            "element_forces": {i: {"i": [0, 4, 30.0], "j": [0, -4, 30.0]}
                               for i in ids}}}}
        out = T.design_timber_members(s, res, prefs=self.P25)
        self.assertTrue(any("§8.3.4" in w for w in out["warnings"]))


class TestReinforcedApex(unittest.TestCase):
    def _design(self, params, prefs, m=2500.0):
        s, ids = _tapered()
        tag = s.assign_beam(ids)
        s.beams[tag]["timber"] = params
        res = {"element_forces": {}, "combinations": {"ULS1": {
            "element_forces": {i: {"i": [0, 40, m], "j": [0, -40, m]}
                               for i in ids}}}}
        return T.design_timber_members(s, res, prefs=prefs, with_reports=True)

    P25 = {"ec5_edition": "2025"}

    def test_2025_reinforced_does_not_apply_eqs_8_67_and_8_70(self):
        plain = self._design({}, self.P25)["members"][0]
        rf = self._design({"reinforced": True}, self.P25)["members"][0]
        self.assertIn("apex_t90", plain["special"])
        self.assertNotIn("apex_t90", rf["special"])
        self.assertNotIn("apex_inter", rf["special"])
        self.assertIn("apex_bending", rf["special"])          # 8.58 stays
        self.assertLess(rf["util_shear"], plain["util_shear"])
        self.assertNotIn("reinforcement_advised", rf)         # it is reinforced

    def test_force_is_reported_per_metre_and_per_element(self):
        out = self._design({"reinforced": True, "rf_spacing": 0.4}, self.P25)
        rf = out["members"][0]["reinforced"]
        self.assertGreater(rf["F_t90_rf_per_m"], 0.0)
        self.assertAlmostEqual(rf["F_t90_rf_Ed"], rf["F_t90_rf_per_m"] * 0.4)
        w = [x for x in out["warnings"] if "reinforced" in x][0]
        self.assertIn("8.3.4.2(1)", w)
        self.assertIn("not verified", w)

    def test_report_has_the_force_not_the_unreinforced_checks(self):
        rep = self._design({"reinforced": True, "rf_spacing": 0.4},
                           self.P25)["reports"]["V1"]
        symbols = {st["symbol"] for s_ in rep["sections"] for st in s_["steps"]}
        self.assertIn("F_t,90,rf,Ed", symbols)
        self.assertNotIn("σ_t,90,d / (k_dis·k_vol·f_t,90,d)", symbols)

    def test_2004_ignores_it_with_a_warning(self):
        out = self._design({"reinforced": True}, None)
        row = out["members"][0]
        self.assertIn("apex_t90", row["special"])             # still checked
        self.assertNotIn("reinforced", row)
        self.assertTrue(any("ignored" in w for w in out["warnings"]))

    def test_readout_mentions_it(self):
        from xdfem2d.timber_beams import describe_beam
        s, ids = _tapered()
        tag = s.assign_beam(ids)
        s.beams[tag]["timber"] = {"reinforced": True, "rf_spacing": 0.4}
        txt = "\n".join(describe_beam(s, tag))
        self.assertIn("reinforced", txt)
        self.assertIn("0.400", txt)


class TestPlaneReinforcement(unittest.TestCase):
    P25 = {"ec5_edition": "2025"}
    PL = {"b_r": 0.06, "t_r": 0.012, "f_td": 200.0, "f_b2d": 0.4}

    def _design(self, m=2500.0, **extra):
        s, ids = _tapered()
        tag = s.assign_beam(ids)
        s.beams[tag]["timber"] = dict(
            reinforced=True, rf_spacing=0.3, rf_plane=dict(self.PL), **extra)
        res = {"element_forces": {}, "combinations": {"ULS1": {
            "element_forces": {i: {"i": [0, 40, m], "j": [0, -40, m]}
                               for i in ids}}}}
        return T.design_timber_members(s, res, prefs=self.P25,
                                       with_reports=True, with_profile=True)

    def test_the_reinforcement_is_verified_and_counts_in_the_utilisation(self):
        out = self._design()
        row = out["members"][0]
        rf = row["reinforced"]
        self.assertIn("util", rf)
        self.assertEqual(rf["governing"], "bond")
        h_ap = 0.5 + 6 * math.tan(math.radians(5))
        f_rd = 0.4e3 * (h_ap / 2) * 0.06                  # l_r = h_ap/2
        self.assertAlmostEqual(rf["util"], rf["F_t90_rf_Ed"] / f_rd, delta=0.02)
        self.assertAlmostEqual(row["special"]["apex_rf"]["util"], rf["util"],
                               places=6)
        self.assertGreaterEqual(row["util_shear"], rf["util"])
        w = [x for x in out["warnings"] if "reinforced" in x][0]
        self.assertIn("is verified", w)

    def test_a_weak_reinforcement_fails_the_member(self):
        s, ids = _tapered()
        tag = s.assign_beam(ids)
        s.beams[tag]["timber"] = {"reinforced": True, "rf_spacing": 0.3,
                                  "rf_plane": dict(self.PL, f_b2d=0.01)}
        res = {"element_forces": {}, "combinations": {"ULS1": {
            "element_forces": {i: {"i": [0, 40, 300.0], "j": [0, -40, 300.0]}
                               for i in ids}}}}
        row = T.design_timber_members(s, res, prefs=self.P25)["members"][0]
        self.assertGreater(row["reinforced"]["util"], 1.0)
        self.assertFalse(row["passed"])

    def test_without_the_data_only_the_force_is_reported(self):
        s, ids = _tapered()
        tag = s.assign_beam(ids)
        s.beams[tag]["timber"] = {"reinforced": True, "rf_spacing": 0.3}
        res = {"element_forces": {}, "combinations": {"ULS1": {
            "element_forces": {i: {"i": [0, 40, 300.0], "j": [0, -40, 300.0]}
                               for i in ids}}}}
        out = T.design_timber_members(s, res, prefs=self.P25)
        self.assertNotIn("util", out["members"][0]["reinforced"])
        self.assertTrue(any("not verified" in w for w in out["warnings"]))

    def test_report_and_profile_carry_the_check(self):
        out = self._design()
        rep = out["reports"]["V1"]
        syms = {st["symbol"] for s_ in rep["sections"] for st in s_["steps"]}
        self.assertIn("F_t,90,rf,Ed / F_t,90,rf,Rd", syms)
        prof = out["members"][0]["profile"]
        self.assertGreater(max(p["apex"] for p in prof), 0.0)

    def test_l_r_override_and_curved_zone_length(self):
        a = self._design(m=300.0)["members"][0]["reinforced"]["util"]
        s, ids = _tapered()
        tag = s.assign_beam(ids)
        s.beams[tag]["timber"] = {"reinforced": True, "rf_spacing": 0.3,
                                  "rf_plane": dict(self.PL, l_r_l=0.1)}
        res = {"element_forces": {}, "combinations": {"ULS1": {
            "element_forces": {i: {"i": [0, 40, 300.0], "j": [0, -40, 300.0]}
                               for i in ids}}}}
        b = T.design_timber_members(s, res, prefs=self.P25)["members"][0]
        self.assertGreater(b["reinforced"]["util"], a)       # shorter l_r
        R, step = 20.0, math.radians(5)
        sc = _model([(R * math.sin(k * step), R * math.cos(k * step) - R)
                     for k in range(-6, 7)])
        _, g = _geom(sc, [f"E{i + 1}" for i in range(12)])
        self.assertAlmostEqual(g.apex.zone_length, 11 * 2 * R * math.sin(step / 2),
                               delta=0.01)


class TestDowelReinforcement(unittest.TestCase):
    P25 = {"ec5_edition": "2025"}
    DW = {"d": 8.0, "d1": 5.0, "f_uk": 920.0, "gamma_r": 1.25, "gamma_m2": 1.25}

    def _design(self, m=300.0, a1=0.3, dw=None, **extra):
        s, ids = _tapered()
        tag = s.assign_beam(ids)
        s.beams[tag]["timber"] = dict(
            reinforced=True, rf_spacing=a1, rf_type="dowel",
            rf_dowel=dict(self.DW if dw is None else dw), **extra)
        res = {"element_forces": {}, "combinations": {"ULS1": {
            "element_forces": {i: {"i": [0, 40, m], "j": [0, -40, m]}
                               for i in ids}}}}
        return T.design_timber_members(s, res, prefs=self.P25, with_reports=True)

    def test_screws_are_verified_with_section_11_2_2(self):
        out = self._design()
        rf = out["members"][0]["reinforced"]
        self.assertIn("util", rf)
        self.assertIn(rf["governing"], ("steel", "withdrawal"))
        self.assertAlmostEqual(
            out["members"][0]["special"]["apex_rf"]["util"], rf["util"], places=6)
        syms = {st["symbol"] for s_ in out["reports"]["V1"]["sections"]
                for st in s_["steps"]}
        self.assertIn("F_ax,t,d", syms)
        self.assertIn("f_w,k", syms)

    def test_a_weak_screw_fails_the_member(self):
        out = self._design(m=2500.0, dw=dict(self.DW, f_uk=100.0))
        self.assertGreater(out["members"][0]["reinforced"]["util"], 1.0)
        self.assertFalse(out["members"][0]["passed"])

    def test_spacing_limits_and_table_limits_are_warned(self):
        w = self._design(a1=0.2)["warnings"]
        self.assertTrue(any("250 mm" in x and "8.3.4.2(4)" in x for x in w))
        w = self._design(dw=dict(self.DW, d=30.0))["warnings"]
        self.assertTrue(any("outside 3.5-22 mm" in x for x in w))
        self.assertFalse(any("250 mm" in x for x in self._design(a1=0.3)["warnings"]))

    def test_partial_factors_default_to_table_4_4(self):
        dw = {k: v for k, v in self.DW.items() if k not in ("gamma_r", "gamma_m2")}
        a = self._design(dw=dw)["members"][0]["reinforced"]["util"]
        b = self._design(dw=dict(dw, gamma_r=1.30, gamma_m2=1.25))["members"][0][
            "reinforced"]["util"]
        c = self._design(dw=dict(dw, gamma_r=1.6, gamma_m2=1.6))["members"][0][
            "reinforced"]["util"]
        self.assertEqual(a, b)
        self.assertGreater(c, a)

    def test_without_the_steel_strength_only_the_force_is_reported(self):
        dw = {k: v for k, v in self.DW.items() if k != "f_uk"}
        out = self._design(dw=dw)
        self.assertNotIn("util", out["members"][0]["reinforced"])
        self.assertTrue(any("not verified" in w for w in out["warnings"]))

    def test_the_type_selects_the_check(self):
        s, ids = _tapered()
        tag = s.assign_beam(ids)
        s.beams[tag]["timber"] = {
            "reinforced": True, "rf_spacing": 0.3, "rf_type": "plane",
            "rf_plane": {"b_r": 0.06, "t_r": 0.012, "f_td": 200.0, "f_b2d": 0.4},
            "rf_dowel": dict(self.DW)}
        res = {"element_forces": {}, "combinations": {"ULS1": {
            "element_forces": {i: {"i": [0, 40, 300.0], "j": [0, -40, 300.0]}
                               for i in ids}}}}
        row = T.design_timber_members(s, res, prefs=self.P25)["members"][0]
        self.assertEqual(row["reinforced"]["governing"], "bond")    # plane

    def test_readout_describes_the_dowel_reinforcement(self):
        from xdfem2d.timber_beams import describe_beam
        s, ids = _tapered()
        tag = s.assign_beam(ids)
        s.beams[tag]["timber"] = {"reinforced": True, "rf_type": "dowel",
                                  "rf_dowel": dict(self.DW)}
        txt = "\n".join(describe_beam(s, tag))
        self.assertIn("dowel-type reinforcement", txt)
        self.assertIn("f_uk = 920", txt)


class TestPlaneDefaultsAndEditionParameters(unittest.TestCase):
    P25 = {"ec5_edition": "2025"}

    def test_plane_bond_strength_and_length_default_from_the_standard(self):
        def util(pl):
            s, ids = _tapered()
            tag = s.assign_beam(ids)
            s.beams[tag]["timber"] = {"reinforced": True, "rf_spacing": 0.3,
                                      "rf_plane": pl}
            res = {"element_forces": {}, "combinations": {"ULS1": {
                "element_forces": {i: {"i": [0, 40, 300.0], "j": [0, -40, 300.0]}
                                   for i in ids}}}}
            return T.design_timber_members(s, res, prefs=self.P25)["members"][0][
                "reinforced"]["util"]
        base = {"b_r": 0.06, "t_r": 0.012, "f_td": 200.0}
        k_mod = 0.8                                        # GL, SC1, medium-term
        explicit = util(dict(base, f_b2d=k_mod * 0.75 / 1.30))
        self.assertAlmostEqual(util(base), explicit, places=9)
        self.assertGreater(util(dict(base, f_b2d=0.2)), explicit)

    def _glulam(self, service_class, prefs, grade="GL24h"):
        s, ids = _model([(0, 0), (6, 0), (12, 0)], supports=("n0", "n2")), ["E1", "E2"]
        s.materials["T"].design = {"class": grade}
        s.sections["GL"].timber_service_class = service_class
        tag = s.assign_beam(ids)
        s.beams[tag]["timber"] = {"lateral_spacing": 0.5}    # k_crit = 1
        return T.design_timber_members(s, _results(ids, 400.0), prefs=prefs)

    def test_the_2025_k_mod_of_sc3_is_used(self):
        u04 = self._glulam("SC3", None)["members"][0]["util_bending"]
        u25 = self._glulam("SC3", self.P25)["members"][0]["util_bending"]
        # medium-term SC3: 0.65 (2004) -> 0.70 (2025)
        self.assertAlmostEqual(u25 / u04, 0.65 / 0.70, delta=0.01)
        # SC1 is the same in both editions
        self.assertAlmostEqual(
            self._glulam("SC1", None)["members"][0]["util_bending"],
            self._glulam("SC1", self.P25)["members"][0]["util_bending"],
            delta=1e-6)

    def test_sc4_exists_only_in_2025_and_not_for_glulam(self):
        ok = self._glulam("SC4", self.P25, grade="C24")
        self.assertEqual(len(ok["members"]), 1)               # solid timber, SC4
        no04 = self._glulam("SC4", None, grade="C24")
        self.assertEqual(no04["members"], [])
        self.assertTrue(any("SC4" in w and "not designed" in w for w in no04["warnings"]))
        gl = self._glulam("SC4", self.P25)                     # GL: no SC4 k_mod
        self.assertEqual(gl["members"], [])
        self.assertTrue(any("SC4" in w for w in gl["warnings"]))
