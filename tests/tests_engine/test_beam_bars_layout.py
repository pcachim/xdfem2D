"""layout_beams: rows of design_beam_bars -> drawable groups on one x axis."""
import unittest

from context import Structure2D  # noqa: F401  (adds src/ to sys.path)

from xdfem2d import design_beam_bars, design_concrete_sections
from beam_params_util import loose
from xdfem2d.beam_bars_layout import layout_beams


def model(section_change=False):
    s = Structure2D()
    for nid, x in (("a", 0), ("b", 5), ("c", 10)):
        s.add_node(nid, x, 3)
    s.add_material("C", elastic_modulus=33e6, unit_weight=25.0,
                   material_type="Concrete", design={"fck": 30, "fyk": 500})
    s.add_section("CONC", "C", b=0.3, h=0.5)
    s.add_section("CONC2", "C", b=0.3, h=0.6)
    s.add_bar_element("E1", "a", "b", "CONC")
    s.add_bar_element("E2", "b", "c", "CONC2" if section_change else "CONC")
    s.add_support("P", ux=True, uy=True)
    for n in "abc":
        s.assign_support(n, "P")
    return s


def rows_of(s, **kw):
    res = {"element_forces": {}, "combinations": {"ULS1": {"element_forces": {
        "E1": {"i": [0, 40, -50], "j": [0, -40, 100]},
        "E2": {"i": [0, 40, -50], "j": [0, -40, 100]}}}}}
    return design_beam_bars(s, design_concrete_sections(s, res), res,
                            **loose(kw))


class TestLayout(unittest.TestCase):
    def test_continuous_beam_is_one_group_one_segment(self):
        rows = rows_of(model())
        (g,) = layout_beams(rows)
        self.assertEqual(len(g["segments"]), 1)
        self.assertEqual(g["breaks"], [])
        self.assertAlmostEqual(g["length"], 10.0)
        spans = g["segments"][0]["spans"]
        self.assertEqual([round(sp["x0"], 6) for sp in spans], [0.0, 5.0])
        self.assertAlmostEqual(spans[-1]["x1"], 10.0)

    def test_zones_tile_the_group_without_gaps(self):
        (g,) = layout_beams(rows_of(model(), zones=(0.25, 0.5, 0.25)))
        zones = [z for s in g["segments"] for sp in s["spans"]
                 for z in sp["zones"]]
        self.assertEqual(len(zones), 6)
        self.assertAlmostEqual(zones[0]["x0"], 0.0)
        for a, b in zip(zones, zones[1:]):
            self.assertAlmostEqual(a["x1"], b["x0"])
        self.assertAlmostEqual(zones[-1]["x1"], 10.0)
        self.assertTrue(all(z["bottom_layers"] and z["top_layers"] for z in zones))

    def test_tagged_beam_with_section_change_shows_a_break(self):
        s = model(section_change=True)
        s.assign_beam(["E1", "E2"], tag="V1", name="Floor")
        (g,) = layout_beams(rows_of(s))
        self.assertEqual([x["name"] for x in g["segments"]],
                         ["Floor.1", "Floor.2"])
        (b,) = g["breaks"]
        self.assertAlmostEqual(b["x"], 5.0)
        self.assertEqual((b["before"], b["after"]), ("Floor.1", "Floor.2"))
        self.assertAlmostEqual(g["length"], 10.0)

    def test_untagged_split_beams_are_separate_groups(self):
        groups = layout_beams(rows_of(model(section_change=True)))
        self.assertEqual(len(groups), 2)
        self.assertTrue(all(g["breaks"] == [] for g in groups))

    def test_segments_sorted_naturally(self):
        rows = [{"beam": f"V1.{n}", "beam_tag": "V1", "span": f"M{n}",
                 "x0": 0.0, "x1": 2.0, "bottom": "", "top": ""}
                for n in (10, 2, 1)]
        (g,) = layout_beams(rows)
        self.assertEqual([s["name"] for s in g["segments"]],
                         ["V1.1", "V1.2", "V1.10"])

    def test_empty(self):
        self.assertEqual(layout_beams([]), [])


if __name__ == "__main__":
    unittest.main()
