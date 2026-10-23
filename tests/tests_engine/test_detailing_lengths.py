"""Anchorage / lap geometry of a beam detail; the formulas are eurocodepy's
(tests/test_ec2_anchorage.py there), here we check the geometry fed to them."""
import copy
import math
import unittest

import numpy as np
from eurocodepy.ec2.uls import (
    beam_cover_distance, bond_conditions_beam, design_anchorage_length,
    lap_length,
)

from context import Structure2D  # noqa: F401  (adds src/ to sys.path)

from beam_params_util import loose
from xdfem2d import design_beam_bars, design_concrete_sections
from xdfem2d import detailing as D
from xdfem2d import detailing_lengths as L


def _design(**kw):
    s = Structure2D()
    for i, x in enumerate((0.0, 5.0, 10.0), 1):
        s.add_node(f"n{i}", x, 0.0)
    s.add_material("C", elastic_modulus=33e6, unit_weight=25.0,
                   material_type="Concrete", design={"fck": 30, "fyk": 500})
    s.add_section("CONC", "C", b=0.3, h=0.5)
    s.add_bar_element("E1", "n1", "n2", "CONC")
    s.add_bar_element("E2", "n2", "n3", "CONC")
    s.add_support("PIN", ux=True, uy=True)
    for n in ("n1", "n2", "n3"):
        s.assign_support(n, "PIN")
    x = np.linspace(0.0, 5.0, 11)
    z = np.zeros_like(x)
    m1 = 150.0 * (1 - ((x - 2.5) / 2.5) ** 2)
    m2 = -90.0 * (1 - x / 5.0) ** 2
    res = {"element_forces": {}, "combinations": {"ULS1": {"element_forces": {
        "E1": {"i": [0, 60, 0], "j": [0, -60, 0]},
        "E2": {"i": [0, 30, -90], "j": [0, -30, 0]}}}},
        "combo_distribution": {"ULS1": {
            "E1": {"x": x, "N": z, "V": 2.0 * np.gradient(m1, x), "M": m1},
            "E2": {"x": x, "N": z, "V": 2.0 * np.gradient(m2, x), "M": m2}}}}
    inputs = []
    rows = design_beam_bars(s, design_concrete_sections(s, res), res,
                            inputs_out=inputs, **loose(kw))
    (key, detail), = D.details_from_rows(rows, inputs).items()
    return detail, inputs[0]


def _flat(detail):
    return [z for sp in detail.segments[0].spans for z in sp.zones]


class TestBarLengths(unittest.TestCase):
    def setUp(self):
        self.detail, self.inp = _design(zones=(0.25, 0.5, 0.25))
        self.seg = self.detail.segments[0]

    def test_it_asks_eurocodepy_with_the_beam_geometry(self):
        sec = self.inp.section
        r = L.bar_lengths(self.inp, "bottom", 16.0, 4)
        cd = beam_cover_distance(sec["b"], sec["cover"] + sec["stirrup"], 4, 16.0)
        ref = design_anchorage_length(16.0, sec["fck"], sec["fyk"], cd,
                                      good_bond=True, shape="straight")
        lap = lap_length(16.0, sec["fck"], sec["fyk"], cd, rho1=50.0,
                         good_bond=True)
        self.assertAlmostEqual(r.cd, cd)
        self.assertEqual((r.bond, r.shape), ("good", "straight"))
        self.assertAlmostEqual(r.lbd, ref["lbd"])
        self.assertAlmostEqual(r.lb_rqd, ref["lb_rqd"])
        self.assertAlmostEqual(r.l0, lap["l0"])
        self.assertAlmostEqual(r.fbd, ref["fbd"])

    def test_top_bars_of_a_deep_beam_have_poor_bond(self):
        top = L.bar_lengths(self.inp, "top", 16.0, 4)
        bot = L.bar_lengths(self.inp, "bottom", 16.0, 4)
        self.assertEqual(top.bond, bond_conditions_beam(500.0, "top"))
        self.assertEqual((top.bond, bot.bond), ("poor", "good"))
        self.assertGreater(top.lbd, bot.lbd)
        self.assertGreater(top.l0, bot.l0)
        shallow = copy.deepcopy(self.inp)
        shallow.section["h"] = 250.0
        self.assertEqual(L.bar_lengths(shallow, "top", 16.0, 4).bond, "good")

    def test_more_bars_in_a_layer_make_the_bars_longer(self):
        few = L.bar_lengths(self.inp, "bottom", 16.0, 2)
        many = L.bar_lengths(self.inp, "bottom", 16.0, 6)
        self.assertLess(many.cd, few.cd)
        self.assertGreaterEqual(many.lbd, few.lbd)

    def test_bent_ends_and_the_partial_factors_follow_the_inputs(self):
        bent = copy.deepcopy(self.inp)
        bent.params["anchorage_shape"] = "bent"
        a = L.bar_lengths(self.inp, "bottom", 16.0, 4)
        b = L.bar_lengths(bent, "bottom", 16.0, 4)
        self.assertEqual(b.shape, "bent")
        sec = bent.section
        ref = design_anchorage_length(
            16.0, sec["fck"], sec["fyk"], b.cd, good_bond=True, shape="bent")
        self.assertAlmostEqual(b.lbd, ref["lbd"])            # eurocodepy's number
        # Table 8.2: a hook only helps when cd > 3φ; with the bars this close
        # (cd = 26.7 < 48) the bent end is NOT shorter than the straight one
        self.assertGreaterEqual(b.lbd, a.lbd)
        # with a wide cover (cd > 3φ) it is
        wide_b = copy.deepcopy(bent)
        wide_s = copy.deepcopy(self.inp)
        for inp in (wide_b, wide_s):
            inp.section["cover"] = 60.0
        self.assertLess(L.bar_lengths(wide_b, "bottom", 12.0, 1).lbd,
                        L.bar_lengths(wide_s, "bottom", 12.0, 1).lbd)
        loose = copy.deepcopy(self.inp)
        loose.section["gamma_s"] = 1.0                   # higher f_yd -> longer
        self.assertGreater(L.bar_lengths(loose, "bottom", 16.0, 4).lb_rqd,
                           a.lb_rqd)
        lap25 = copy.deepcopy(self.inp)
        lap25.params["lap_percentage"] = 25.0            # alpha6 = 1 -> shorter lap
        self.assertLess(L.bar_lengths(lap25, "bottom", 16.0, 4).l0, a.l0)

    def test_the_inputs_carry_the_partial_factors(self):
        self.assertEqual(self.inp.section["gamma_c"], 1.5)
        self.assertEqual(self.inp.section["gamma_s"], 1.15)

    def test_bad_face(self):
        with self.assertRaises(ValueError):
            L.bar_lengths(self.inp, "side", 16.0)

    def test_table_for_every_diameter_used(self):
        rows = L.segment_lengths(self.seg, self.inp)
        used = {(f, float(d)) for z in _flat(self.detail)
                for f in D.FACES for _n, d in z.layers(f)}
        self.assertEqual({(r.face, r.diameter) for r in rows}, used)
        self.assertEqual([r.face for r in rows],
                         sorted((r.face for r in rows),
                                key=lambda f: {"top": 0, "bottom": 1}[f]))
        for r in rows:
            n = max(n for z in _flat(self.detail)
                    for n, d in z.layers(r.face) if float(d) == r.diameter)
            self.assertEqual(r.n_bars, n)
            self.assertGreaterEqual(r.lbd, r.lb_min)
            self.assertGreaterEqual(r.l0, r.l0_min)


class TestCurtailments(unittest.TestCase):
    def setUp(self):
        self.detail, self.inp = _design(zones=(0.25, 0.5, 0.25))
        self.seg = copy.deepcopy(self.detail.segments[0])
        for z in _flat_zones(self.seg):                      # a clean slate
            z.bottom = [(4, 16.0)]
            z.top = [(2, 10.0)]

    def lbd(self, face, d, n):
        return L.bar_lengths(self.inp, face, d, n).lbd

    def test_equal_zones_have_no_curtailment(self):
        self.assertEqual(L.curtailments(self.seg, self.inp), [])

    def test_dropped_bars_extend_into_the_next_zone(self):
        z = _flat_zones(self.seg)
        z[1].bottom = [(6, 16.0)]                           # zone 2 has 2 more
        cs = [c for c in L.curtailments(self.seg, self.inp)
              if c.face == "bottom"]
        # 4 -> 6 : two bars START before the cut; 6 -> 4: two bars STOP after
        self.assertEqual({(c.direction, c.n_bars) for c in cs},
                         {(-1, 2), (+1, 2)})
        stop = next(c for c in cs if c.direction == +1)
        start = next(c for c in cs if c.direction == -1)
        self.assertAlmostEqual(stop.x_cut, z_x1(self.seg, 1))
        self.assertAlmostEqual(stop.lbd, self.lbd("bottom", 16.0, 6))
        self.assertAlmostEqual(stop.x_end, stop.x_cut + stop.lbd / 1000.0)
        self.assertAlmostEqual(start.x_end, start.x_cut - start.lbd / 1000.0)
        self.assertEqual(stop.layer, 0)
        self.assertFalse(stop.outside or start.outside)

    def test_layers_and_diameter_changes(self):
        z = _flat_zones(self.seg)
        z[1].bottom = [(4, 16.0), (2, 16.0)]                # a 2nd layer appears
        cs = [c for c in L.curtailments(self.seg, self.inp)
              if c.face == "bottom" and c.layer == 1]
        self.assertEqual({(c.direction, c.n_bars) for c in cs},
                         {(-1, 2), (+1, 2)})
        z[1].bottom = [(4, 20.0)]                           # same count, new Ø
        cs = L.curtailments(self.seg, self.inp)
        ds = {(c.direction, c.diameter) for c in cs
              if c.face == "bottom" and c.layer == 0}
        self.assertEqual(ds, {(+1, 16.0), (-1, 20.0), (-1, 16.0), (+1, 20.0)})

    def test_an_extension_that_leaves_the_beam_is_flagged(self):
        ed = D.DetailEditor(self.seg, self.inp)
        ed.set_layers("T1", 0, "bottom", [(2, 16.0)])
        ed.set_layers("T1", 1, "bottom", [(6, 16.0)])
        ed.move_cut("T1", 0, 0.2)                           # cut 20 cm from the end
        cs = [c for c in L.curtailments(ed.segment, self.inp)
              if c.face == "bottom" and c.direction == -1 and c.x_cut < 0.3]
        self.assertTrue(cs)
        self.assertTrue(all(c.outside and c.x_end < 0 for c in cs))
        issues = L.length_issues(ed.segment, self.inp)
        self.assertTrue(any(i.code == "CURTAIL_BEYOND"
                            and i.severity == "warning" for i in issues))
        # they reach the window's list as warnings, never errors
        rep = ed.checks()
        self.assertIn("CURTAIL_BEYOND", rep.codes())
        self.assertNotIn("CURTAIL_BEYOND", {i.code for i in rep.errors})

    def test_the_proposal_has_curtailments_and_stays_valid(self):
        detail, inp = _design(zones=(0.25, 0.5, 0.25))
        cs = L.curtailments(detail.segments[0], inp)
        self.assertTrue(cs)
        for c in cs:
            self.assertGreater(c.lbd, 0.0)
            self.assertIn(c.direction, (-1, 1))
        self.assertTrue(D.DetailEditor(detail.segments[0], inp).checks().ok)


def _flat_zones(seg):
    return [z for sp in seg.spans for z in sp.zones]


def z_x1(seg, idx):
    """Absolute x of the end of the idx-th zone (flat order)."""
    off = 0.0
    n = 0
    for sp in seg.spans:
        for z in sp.zones:
            if n == idx:
                return off + z.x1
            n += 1
        off += sp.length
    raise IndexError(idx)


class TestEndsAndLaps(unittest.TestCase):
    def setUp(self):
        self.detail, self.inp = _design(zones=(0.25, 0.5, 0.25))
        self.seg = self.detail.segments[0]

    def test_end_anchorages(self):
        rows = L.end_anchorages(self.seg, self.inp)
        self.assertEqual({r["end"] for r in rows}, {"start", "end"})
        self.assertEqual({r["face"] for r in rows}, {"top", "bottom"})
        first = _flat(self.detail)[0]
        for r in rows:
            if r["end"] == "start":
                lay = first.layers(r["face"])[r["layer"]]
                self.assertEqual((r["n_bars"], r["diameter"]),
                                 (lay[0], float(lay[1])))
            self.assertAlmostEqual(
                r["lbd"], L.bar_lengths(self.inp, r["face"], r["diameter"],
                                        r["n_bars"]).lbd)

    def test_lap_count(self):
        self.assertEqual(L.lap_count(10.0, 12.0, 800.0), 0)       # one bar suffices
        self.assertEqual(L.lap_count(12.0, 12.0, 800.0), 0)
        self.assertEqual(L.lap_count(30.0, 12.0, 800.0), 2)       # 12 + 11.2 + 11.2
        self.assertEqual(L.lap_count(23.2, 12.0, 800.0), 1)       # 12 + 11.2 exactly
        self.assertEqual(L.lap_count(23.3, 12.0, 800.0), 2)       # a third bar
        with self.assertRaises(ValueError):
            L.lap_count(30.0, 0.0, 800.0)
        with self.assertRaises(ValueError):
            L.lap_count(30.0, 0.5, 800.0)                         # lap > bar

    def test_through_laps(self):
        out = L.through_laps(self.seg, self.inp)
        self.assertTrue(out)
        for r in out:
            n_max = max(n for z in _flat(self.detail)
                        for n, d in z.layers(r["face"]) if float(d) == r["diameter"])
            bl = L.bar_lengths(self.inp, r["face"], r["diameter"], n_max)
            self.assertAlmostEqual(r["l0"], bl.l0)
            self.assertAlmostEqual(r["length"], 10.0)
            self.assertEqual(r["n_laps"], L.lap_count(10.0, 12.0, r["l0"]))
        short = copy.deepcopy(self.inp)
        short.params["bar_length"] = 4.0                          # more laps
        more = L.through_laps(self.seg, short)
        self.assertTrue(all(m["n_laps"] > o["n_laps"]
                            for m, o in zip(more, out)))

    def test_no_through_laps_when_the_outer_diameter_changes(self):
        seg = copy.deepcopy(self.seg)
        zs = _flat_zones(seg)
        zs[0].bottom = [(4, 16.0)]
        zs[1].bottom = [(4, 20.0)]
        faces = {r["face"] for r in L.through_laps(seg, self.inp)}
        self.assertNotIn("bottom", faces)


if __name__ == "__main__":
    unittest.main()
