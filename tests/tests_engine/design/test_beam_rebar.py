"""Tests for beam_rebar: areas -> bars, As,min and diameter uniformisation."""
import unittest

from xdfem2d.beam_rebar import (
    Beam, Span, bars_for_area, beam_reinforcement, fctm, max_bars_per_layer,
)


def _beam(spans):
    return Beam(width=300, height=600, cover=30, stirrup_diameter=8,
                fck=30, fyk=500, spans=spans)


class BeamRebarTests(unittest.TestCase):
    def test_as_min_ec2(self):
        b = _beam([])
        d = 600 - 30 - 8 - 8
        expected = max(0.26 * fctm(30) / 500 * 300 * d, 0.0013 * 300 * d)
        self.assertAlmostEqual(b.as_min, expected)
        self.assertAlmostEqual(b.as_min, 0.26 * fctm(30) / 500 * 300 * d)

    def test_area_is_covered_and_fits(self):
        b = _beam([])
        lay = bars_for_area(1450.0, b)
        self.assertGreaterEqual(lay.area, 1450.0)
        for g in lay.layers:
            self.assertLessEqual(g.n, max_bars_per_layer(b, g.diameter))

    def test_infeasible_raises(self):
        with self.assertRaises(ValueError):
            bars_for_area(1e6, _beam([]))

    def test_as_min_applied_to_bottom(self):
        b = _beam([Span(5.0, 100.0, 0.0, "T1")])
        res = beam_reinforcement(b)
        self.assertGreaterEqual(res.spans[0].bottom.area, b.as_min)
        self.assertTrue(res.warnings)

    def test_top_hangers_when_no_requirement(self):
        res = beam_reinforcement(_beam([Span(5.0, 900.0, 0.0)]))
        self.assertGreaterEqual(res.spans[0].top.n_bars, 2)

    def test_uniform_single_diameter(self):
        spans = [Span(5.0, 820.0, 310.0, "T1"), Span(4.0, 1450.0, 900.0, "T2"),
                 Span(5.0, 700.0, 250.0, "T3")]
        res = beam_reinforcement(_beam(spans), max_diameters_beam=1)
        self.assertEqual(len(res.diameters("bottom")), 1)
        self.assertEqual(len(res.diameters("top")), 1)
        for s in res.spans:
            self.assertGreaterEqual(s.bottom.area, s.as_bottom_design)
            self.assertGreaterEqual(s.top.area, s.as_top_design)

    def test_at_most_k_diameters(self):
        spans = [Span(5.0, 500.0, 200.0), Span(4.0, 2600.0, 1500.0)]
        res = beam_reinforcement(_beam(spans), max_diameters_beam=2)
        self.assertLessEqual(len(res.diameters("bottom")), 2)

    def test_through_bars_in_every_span(self):
        spans = [Span(5.0, 820.0, 310.0), Span(4.0, 2600.0, 1500.0),
                 Span(5.0, 700.0, 250.0)]
        res = beam_reinforcement(_beam(spans), n_through=2)
        for face in ("bottom", "top"):
            d = res.through[face]
            self.assertIsNotNone(d)
            for s_ in res.spans:
                outer = getattr(s_, face).layers[0]
                self.assertEqual(outer.diameter, d)
                self.assertGreaterEqual(outer.n, 2)

    def test_through_disabled(self):
        res = beam_reinforcement(_beam([Span(5.0, 900.0, 0.0)]), n_through=0)
        self.assertIsNone(res.through["bottom"])

    def test_rule_allows_odd_but_never_one_bar(self):
        b = _beam([])
        lay = bars_for_area(550.0, b, diameters=(16,))
        self.assertEqual(lay.n_bars, 3)              # 3Ø16 = 603 (odd >= 3)
        for g in bars_for_area(1.0, b, diameters=(16,), min_bars=1).layers:
            self.assertGreaterEqual(g.n, 2)          # never a single bar
        even = bars_for_area(550.0, b, diameters=(16,), symmetry="even")
        self.assertEqual(even.n_bars, 4)

    def test_symmetric_pairs(self):
        from xdfem2d.beam_rebar import _pair_symmetric
        for a, b in ((4, 2), (5, 3), (5, 2), (5, 4), (3, 2), (6, 4), (3, 0)):
            self.assertTrue(_pair_symmetric(a, b), (a, b))
        for a, b in ((4, 3), (6, 3), (4, 1), (2, 1)):
            self.assertFalse(_pair_symmetric(a, b), (a, b))

    def test_adjacent_zones_are_symmetric(self):
        from xdfem2d.beam_rebar import layouts_compatible
        spans = [Span(1.0, 1500.0, 900.0), Span(3.0, 400.0, 200.0),
                 Span(1.0, 1100.0, 600.0), Span(3.0, 300.0, 100.0)]
        res = beam_reinforcement(_beam(spans))
        for face in ("bottom", "top"):
            lays = [getattr(s_, face) for s_ in res.spans]
            for x, y in zip(lays, lays[1:]):
                self.assertTrue(layouts_compatible(x, y), (str(x), str(y)))
            for lay in lays:
                for g in lay.layers:
                    self.assertTrue(g.n == 2 or g.n >= 3)

    def test_symmetry_none_allows_single_bars(self):
        lay = bars_for_area(150.0, _beam([]), diameters=(16,), min_bars=1,
                            symmetry="none")
        self.assertEqual(lay.n_bars, 1)

    def test_as_max_warning(self):
        b = _beam([Span(5.0, 7000.0, 0.0)])
        res = beam_reinforcement(b, max_layers=3)
        self.assertTrue(any("As,max" in w for w in res.warnings)
                        or res.spans[0].bottom.area <= b.as_max)


if __name__ == "__main__":
    unittest.main()
