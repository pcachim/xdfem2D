"""element_info: property / result / design rows for one bar, area or object."""
import unittest

from context import Structure2D  # noqa: F401  (adds src/ to sys.path)

from xdfem2d import element_info as EI


def _model():
    s = Structure2D()
    s.add_node("a", 0, 0); s.add_node("b", 4, 0)
    s.add_material("S", elastic_modulus=210e6, unit_weight=78.5,
                   material_type="Steel", design={"fy": 275})
    s.add_material("C", elastic_modulus=33e6, unit_weight=25.0,
                   material_type="Concrete", design={"fck": 30, "fyk": 500})
    s.add_section("SEC", "S", b=0.2, h=0.4)
    s.add_bar_element("E1", "a", "b", "SEC")
    s.add_geo_segment("L1", 0, 3, 6, 3, section_name="SEC", divisions=3)
    s.add_tri_section("T", "C", thickness=0.2)
    s.add_node("c", 0, 1)
    s.add_tri_element("TR1", "a", "b", "c", "T")
    s.add_geo_rectangle("R1", (0, 5), (4, 7), section_name="T", target_size=1.0,
                        prefer_quad=False)
    return s


def _labels(rows):
    return {(g, l): t for g, l, t in rows}


class TestProperties(unittest.TestCase):
    def test_bar(self):
        r = _labels(EI.property_rows(_model(), "bar", "E1"))
        self.assertEqual(r[("Element", "Length")], "4 m")
        self.assertIn(("Section", "A"), r)
        self.assertIn(("Material", "E"), r)

    def test_area(self):
        r = _labels(EI.property_rows(_model(), "area", "TR1"))
        self.assertEqual(r[("Element", "Area")], "2 m²")
        self.assertEqual(r[("Section", "Thickness t")], "200 mm")
        g = "Section (per metre width)"
        self.assertEqual(r[(g, "I = t³/12")], "0.0006667 m⁴/m")
        self.assertEqual(r[(g, "Wel = t²/6")], "0.006667 m³/m")
        self.assertEqual(r[(g, "A = t")], "0.2 m²/m")

    def test_line_object(self):
        r = _labels(EI.property_rows(_model(), "object", "L1"))
        self.assertEqual(r[("Object", "Length")], "6 m")
        self.assertEqual(r[("Mesh", "Generated bars")], "3")

    def test_surface_object(self):
        r = _labels(EI.property_rows(_model(), "object", "R1"))
        self.assertEqual(r[("Object", "Area")], "8 m²")
        self.assertIn(("Mesh", "Triangles / quads"), r)

    def test_missing(self):
        self.assertEqual(EI.property_rows(_model(), "bar", "nope"), [])
        self.assertEqual(EI.property_rows(_model(), "object", "nope"), [])


class TestResultsDesign(unittest.TestCase):
    def _res(self):
        ef = {"E1": {"i": [1, 2, 3], "j": [4, 5, 6]},
              "E2": {"i": [9, 9, 9], "j": [9, 9, 9]}}
        return {"combinations": {"ULS1": {"element_forces": ef,
                                          "displacements": {}, "reactions": {}}},
                "steel_design": {"elements": {
                    "E1": {"bending": .5, "shear": .1, "torsion": 0,
                           "buckling": .7, "combined": .7}}}}

    def test_filtered_to_target(self):
        t = dict((x[0], x) for x in
                 EI.result_tables(_model(), self._res(), "bar", "E1"))
        rows = t["Element forces"][2]
        self.assertEqual({r[2] for r in rows}, {"E1"})

    def test_no_results(self):
        self.assertEqual(EI.result_tables(_model(), None, "bar", "E1"), [])
        self.assertEqual(EI.design_tables(_model(), {}, "bar", "E1"), [])

    def test_steel_ratio(self):
        t = EI.design_tables(_model(), self._res(), "bar", "E1")
        self.assertEqual(t[0][0], "Steel utilisation (EC3)")
        self.assertEqual(t[0][2][0][0], "E1")
        self.assertEqual(EI.design_tables(_model(), self._res(), "bar", "E2"), [])


def _many_model():
    s = _model()
    s.add_node("c2", 8, 0)
    s.add_section("SEC2", "S", b=0.3, h=0.5)
    s.add_bar_element("E2", "b", "c2", "SEC2")
    s.add_bar_element("E3", "a", "c2", "SEC")
    return s


def _many_results():
    def ef(v):
        return {"i": [v, v, v], "j": [-v, -v, -v]}
    return {"combinations": {
        "ULS1": {"element_forces": {"E1": ef(1), "E2": ef(5), "E3": ef(3)},
                 "displacements": {}, "reactions": {}}},
        "steel_design": {"elements": {
            "E1": {"combined": 0.5}, "E2": {"combined": 1.2},
            "E3": {"combined": 0.9}}}}


class TestMany(unittest.TestCase):
    def test_bar_properties_totals_and_groups(self):
        r = _labels(EI.property_rows_many(_many_model(), "bar", ["E1", "E2", "E3"]))
        self.assertEqual(r[("Selection", "Bars")], "3")
        self.assertEqual(r[("Selection", "Total length")], "16 m")
        self.assertIn(("By section", "SEC"), r)
        self.assertIn(("By section", "SEC2"), r)
        self.assertIn(("Section", "Properties"), r)      # mixed → no inertias
        self.assertNotIn(("Section", "A"), r)

    def test_single_section_shows_section_properties(self):
        r = _labels(EI.property_rows_many(_many_model(), "bar", ["E1", "E3"]))
        self.assertIn(("Section", "A"), r)

    def test_area_properties(self):
        s = _model()
        r = _labels(EI.property_rows_many(s, "area", ["TR1"]))
        self.assertEqual(r[("Selection", "Total area")], "2 m²")

    def test_envelope(self):
        t = EI.result_summary(_many_model(), _many_results(), "bar",
                              ["E1", "E2"])
        title, hdr, rows = t[0]
        self.assertIn("envelope", title)
        self.assertEqual(len(rows), 1)
        row = dict(zip(hdr, rows[0]))
        self.assertEqual(row["N [kN] min"], -5)
        self.assertEqual(row["N [kN] max"], 5)
        self.assertEqual(row["Peak |M [kNm]| at"], "E2")

    def test_design_summary_ranked_with_failures(self):
        t = EI.design_summary(_many_model(), _many_results(), "bar",
                              ["E1", "E2", "E3"])
        title, hdr, rows = t[0]
        self.assertIn("1 of 3 above 1", title)
        self.assertEqual([r[0] for r in rows], ["E2", "E3", "E1"])

    def test_scope_excludes_unselected(self):
        t = EI.result_tables_many(_many_model(), _many_results(), "bar", ["E1"])
        ids = {r[2] for _n, _h, rows in t if _n == "Element forces" for r in rows}
        self.assertEqual(ids, {"E1"})

    def test_no_results(self):
        self.assertEqual(EI.result_summary(_many_model(), None, "bar", ["E1"]), [])
        self.assertEqual(EI.design_summary(_many_model(), {}, "bar", ["E1"]), [])


class TestNodes(unittest.TestCase):
    def _m(self):
        s = _many_model()
        s.add_support("PIN", ux=True, uy=True, tz=False)
        s.assign_support("a", "PIN")
        s.add_node_spring("b", 100.0, 200.0, 0.0)
        return s

    def _res(self):
        return {"combinations": {"ULS1": {
            "displacements": {"a": [0, 0, 0.01], "b": [0.001, -0.02, 0.0]},
            "reactions": {"a": [5, 6, 0]},
            "spring_forces": {"b": [0.1, -4.0, 0]},
            "element_forces": {}}}}

    def test_properties(self):
        r = _labels(EI.property_rows(self._m(), "node", "a"))
        self.assertIn("restrains ux, uy", r[("Support", "Restraint")])
        self.assertEqual(r[("Connectivity", "Bars")], "E1, E3")
        r = _labels(EI.property_rows(self._m(), "node", "b"))
        self.assertEqual(r[("Spring", "ky")], "200 kN/m")

    def test_results_single(self):
        t = dict((x[0], x) for x in
                 EI.result_tables(self._m(), self._res(), "node", "b"))
        self.assertEqual({r[2] for r in t["Displacements"][2]}, {"b"})
        self.assertEqual(t["Spring forces"][2][0][4], -4.0)
        self.assertNotIn("Reactions", t)

    def test_many_summary_and_properties(self):
        s = self._m()
        r = _labels(EI.property_rows_many(s, "node", ["a", "b"]))
        self.assertEqual(r[("Selection", "With node spring")], "1")
        env = dict((x[0].split(" —")[0], x) for x in
                   EI.result_summary(s, self._res(), "node", ["a", "b"]))
        title, hdr, rows = env["Displacements"]
        row = dict(zip(hdr, rows[0]))
        self.assertEqual(row["uy [m] min"], -0.02)
        self.assertEqual(row["Peak |uy [m]| at"], "b")


if __name__ == "__main__":
    unittest.main()
