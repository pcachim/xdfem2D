"""Model statistics — the quantity take-off (xdfem2d.model_stats)."""
from __future__ import annotations

import unittest

import context  # noqa: F401
from xdfem2d import Structure2D
from xdfem2d.model_stats import compute_model_stats, model_stats_rows


def _beam_and_plate():
    """Two bars (0.3×0.5, L=6) + two CST triangles (t=0.1, area 2), one
    concrete material γ=25, ρ=2.5."""
    s = Structure2D()
    s.add_material("C", 30e6, 25.0, unit_mass=2.5)
    s.add_section("S", "C", b=0.3, h=0.5)                 # A = 0.15
    s.add_tri_section("T", "C", thickness=0.1)
    s.add_node("A", 0, 0); s.add_node("M", 3, 0); s.add_node("B", 6, 0)
    s.add_bar_element("E1", "A", "M", "S")
    s.add_bar_element("E2", "M", "B", "S")
    s.add_node("q0", 10, 0); s.add_node("q1", 12, 0)
    s.add_node("q2", 12, 1); s.add_node("q3", 10, 1)
    s.add_tri_element("t1", "q0", "q1", "q2", "T")
    s.add_tri_element("t2", "q0", "q2", "q3", "T")
    return s


class TestCompute(unittest.TestCase):

    def setUp(self):
        self.st = compute_model_stats(_beam_and_plate())
        self.t = self.st["total"]

    def test_counts(self):
        self.assertEqual(self.t["elements"], 4)
        self.assertEqual(self.t["bars"], 2)
        self.assertEqual(self.t["triangles"], 2)
        self.assertEqual(self.t["cst"], 2)
        self.assertEqual(self.t["allman"], 0)

    def test_length_and_area(self):
        self.assertAlmostEqual(self.t["length"], 6.0)
        self.assertAlmostEqual(self.t["area"], 2.0)
        self.assertAlmostEqual(self.t["area_cst"], 2.0)
        self.assertAlmostEqual(self.t["area_allman"], 0.0)

    def test_volume_weight_mass(self):
        # bars 0.15·6 = 0.9 m³, triangles 2·0.1 = 0.2 m³
        self.assertAlmostEqual(self.t["volume_bars"], 0.9)
        self.assertAlmostEqual(self.t["volume_triangles"], 0.2)
        self.assertAlmostEqual(self.t["volume"], 1.1)
        self.assertAlmostEqual(self.t["weight"], 1.1 * 25.0)     # 27.5 kN
        self.assertAlmostEqual(self.t["mass"], 1.1 * 2.5)        # 2.75 t

    def test_breakdowns(self):
        mat = self.st["by_material"]["C"]
        self.assertEqual(mat["count"], 4)
        self.assertAlmostEqual(mat["length"], 6.0)
        self.assertAlmostEqual(mat["area"], 2.0)
        self.assertAlmostEqual(mat["weight"], 27.5)
        self.assertAlmostEqual(self.st["by_bar_section"]["S"]["length"], 6.0)
        self.assertAlmostEqual(self.st["by_tri_section"]["T"]["area"], 2.0)


class TestMassFallbackAndAllman(unittest.TestCase):

    def test_mass_derived_from_weight_when_unit_mass_unset(self):
        s = Structure2D()
        s.add_material("C", 30e6, 25.0)                   # unit_mass None → γ/g
        s.add_section("S", "C", b=0.3, h=0.5)
        s.add_node("A", 0, 0); s.add_node("B", 4, 0)
        s.add_bar_element("E", "A", "B", "S")
        t = compute_model_stats(s)["total"]
        vol = 0.15 * 4
        self.assertAlmostEqual(t["mass"], vol * 25.0 / 9.81)

    def test_allman_area_is_split_out(self):
        s = Structure2D()
        s.add_material("C", 30e6, 0.0, poisson=0.2)
        s.add_tri_section("T", "C", thickness=0.1, formulation="Allman")
        s.add_node("1", 0, 0); s.add_node("2", 2, 0); s.add_node("3", 0, 1)
        s.add_tri_element("t", "1", "2", "3", "T")
        t = compute_model_stats(s)["total"]
        self.assertEqual(t["allman"], 1)
        self.assertAlmostEqual(t["area_allman"], 1.0)
        self.assertAlmostEqual(t["area_cst"], 0.0)


class TestObjectsUseCompiledMesh(unittest.TestCase):

    def test_polyline_object_counts_generated_bars(self):
        s = Structure2D()
        s.add_material("C", 30e6, 25.0)
        s.add_section("S", "C", b=0.3, h=0.5)
        s.add_geo_polyline("L", [(0, 0), (3, 0), (6, 0)], section_name="S",
                           divisions=2)
        self.assertEqual(len(s.bar_elements), 0)         # authored: none yet
        t = compute_model_stats(s)["total"]
        self.assertEqual(t["bars"], 4)                   # compiled: 4
        self.assertAlmostEqual(t["length"], 6.0)


class TestRows(unittest.TestCase):

    def test_rows_have_three_fields_and_headers(self):
        rows = model_stats_rows(_beam_and_plate())
        self.assertTrue(all(len(r) == 3 for r in rows))
        headers = [r[0] for r in rows if r[1] is None and r[2] is None]
        for h in ("Counts", "Length", "Area",
                  "Volume", "Weight", "Mass"):
            self.assertIn(h, headers)

    def test_a_known_row_value(self):
        rows = model_stats_rows(_beam_and_plate())
        d = {r[0]: (r[1], r[2]) for r in rows}
        self.assertEqual(d["Total elements"], (4, "—"))
        val, unit = d["Total weight"]
        self.assertAlmostEqual(val, 27.5)
        self.assertEqual(unit, "kN")


class TestPlateDomainLabels(unittest.TestCase):
    """A plate model's take-off names its parts as a plate's: line elements are
    Grid members, area elements are DKT Slabs — not Beams, Panels or CST."""

    def _plate(self):
        s = Structure2D(domain="plate")
        s.add_material("C", 30e6, 25.0, unit_mass=2.5)
        s.add_plate_section("Sl", "C", thickness=0.2)
        s.add_section("Rib", "C", b=0.3, h=0.5)
        for i, (x, y) in enumerate([(0, 0), (4, 0), (4, 3), (0, 3)]):
            s.add_node(f"N{i}", float(x), float(y))
        s.add_tri_element("T1", "N0", "N1", "N2", "Sl")
        s.add_tri_element("T2", "N0", "N2", "N3", "Sl")
        s.add_bar_element("B1", "N0", "N1", "Rib")
        return s

    def test_rows_use_the_plate_vocabulary(self):
        labels = [r[0] for r in model_stats_rows(self._plate())]
        joined = "\n".join(labels)
        self.assertIn("Grid elements", joined)
        self.assertIn("Slab elements", joined)
        self.assertIn("plate-bending slabs", joined)
        # None of the plane-only wording leaks in.
        for plane_only in ("Beam elements", "Panel elements",
                           "CST triangles", "Allman triangles"):
            self.assertNotIn(plane_only, joined)

    def test_counts_and_amounts_are_still_computed(self):
        st = compute_model_stats(self._plate())
        self.assertEqual(st["total"]["triangles"], 2)
        self.assertEqual(st["total"]["bars"], 1)
        # Slab area 12 m² (two triangles of a 4×3 panel), volume 12·0.2.
        self.assertAlmostEqual(st["total"]["area"], 12.0)
        self.assertAlmostEqual(st["total"]["volume_triangles"], 2.4)


class TestPlaceholderExclusion(unittest.TestCase):
    """Rigid/Dummy count and add length, and now also weight/mass — their
    reserved material is pinned to unit_weight=unit_mass=0, so that is always
    genuinely zero, not a number worth hiding. Area/volume alone still leave
    them out: Rigid's section area is a fictitious override that would
    pollute those two with a number that means nothing physically."""

    def _model(self):
        s = Structure2D()
        s.add_material("C", 30e6, 25.0)
        s.add_section("S", "C", b=0.3, h=0.5)
        # A rigid placeholder section (huge area) so its volume would dominate.
        s.add_material("Rigid", 1e10, 0.0, material_type="Other")
        s.add_section("Rigid", "Rigid", b=0.0, h=0.0,
                      area_override=100.0, inertia_override=1000.0)
        s.add_node("A", 0, 0); s.add_node("B", 6, 0); s.add_node("C", 9, 0)
        s.add_bar_element("E1", "A", "B", "S")       # real, L=6
        s.add_bar_element("R1", "B", "C", "Rigid")   # placeholder, L=3
        return s

    def test_counts_and_length_include_placeholder(self):
        t = compute_model_stats(self._model())["total"]
        self.assertEqual(t["bars"], 2)
        self.assertAlmostEqual(t["length"], 9.0)      # 6 + 3

    def test_volume_weight_mass_exclude_placeholder(self):
        t = compute_model_stats(self._model())["total"]
        self.assertAlmostEqual(t["volume"], 0.15 * 6)  # only the real bar
        self.assertAlmostEqual(t["weight"], 0.9 * 25.0)

    def test_rows_keep_placeholder_only_out_of_area_and_volume(self):
        rows = model_stats_rows(self._model())
        by_label_unit = {(r[0], r[2]): r[1] for r in rows}
        labels = [r[0] for r in rows]
        self.assertIn("Elements · beam section Rigid", labels)
        self.assertIn("  beam section Rigid",
                      [r[0] for r in rows if r[2] == "m"])
        # Weight/mass rows for Rigid are shown now (real material, always 0),
        # unlike volume, which still hides it (fictitious section area).
        self.assertEqual(by_label_unit.get(("  beam section Rigid", "kN")), 0.0)
        self.assertNotIn(("  beam section Rigid", "m³"), by_label_unit)
        # The exclusion is stated, not silent — one row says how many and why.
        self.assertIn(
            ("Rigid/Dummy elements (excluded from area/volume)", "—"),
            [(l, u) for l, v, u in rows])

    def test_exclude_can_be_overridden(self):
        # Passing an empty set brings the placeholder back into the amounts.
        t = compute_model_stats(self._model(), exclude_amounts=frozenset())["total"]
        self.assertGreater(t["volume"], 100.0)         # 100 m² × 3 m dominates


class TestModelInfoInReportTables(unittest.TestCase):
    """The take-off joins the model tables, so it lists in the export dialog and
    is written out with the rest."""

    def test_model_tables_include_model_info(self):
        from xdfem2d.report_io import _model_tables
        tables = _model_tables(_beam_and_plate())
        by_title = {t[0]: t for t in tables}
        self.assertIn("Model info", by_title)
        _title, header, rows = by_title["Model info"]
        self.assertEqual(header, ["Label", "Value", "Units"])
        self.assertTrue(any(r[0] == "Total elements" for r in rows))


if __name__ == "__main__":
    unittest.main()
