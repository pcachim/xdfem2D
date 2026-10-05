"""Tests for the EC2 column design driver (``design_concrete_columns``).

Requires eurocodepy to be importable (xdfem2D delegates the column checks
to ``eurocodepy.ec2.uls.column``). Run with: python -m unittest discover -s tests
"""
import unittest

from context import Structure2D  # noqa: F401  (adds src/ to sys.path)

from xdfem2d.rc_design import design_concrete_columns, suggest_column_reinforcement

FCK, FYK = 30.0, 500.0


class TestDesignConcreteColumns(unittest.TestCase):
    def _struc(self, length=3.0, is_column=True):
        s = Structure2D()
        s.add_node("n1", 0.0, 0.0)
        s.add_node("n2", 0.0, length)
        s.add_material("C", elastic_modulus=33e6, unit_weight=25.0,
                       material_type="Concrete",
                       design={"fck": FCK, "fyk": FYK})
        s.add_section("COL", "C", b=0.30, h=0.30)
        s.add_bar_element("E1", "n1", "n2", "COL", is_column=is_column)
        return s

    def _results(self, n_axial=-500.0, m_end=40.0):
        # xdfem2D internal convention: tension-positive axial. A compressive
        # column load is therefore negative here.
        return {
            "element_forces": {},
            "combinations": {
                "ULS1": {"element_forces": {
                    "E1": {"i": [n_axial, 0, m_end], "j": [-n_axial, 0, m_end]},
                }},
            },
        }

    def test_only_flagged_columns_are_designed(self):
        struc = self._struc(is_column=False)
        out = design_concrete_columns(struc, self._results())
        self.assertEqual(out["members"], [])
        self.assertEqual(out["elements"], {})

    def test_column_bars_excluded_from_ordinary_section_design(self):
        """The GUI merged 'Concrete members (ULS)' / 'Concrete columns (ULS)'
        into one action: a bar with effective is_column=True must be
        designed by design_concrete_columns only, never also picked up by
        design_concrete_sections's ordinary flexure/shear design — see
        dev/RC_COLUMN_DESIGN.md."""
        from xdfem2d.rc_design import design_concrete_sections
        struc = self._struc(is_column=True)
        col_out = design_concrete_columns(struc, self._results())
        self.assertEqual(len(col_out["members"]), 1)
        self.assertIn("E1", col_out["elements"])
        sec_rows = design_concrete_sections(struc, self._results())
        self.assertEqual(sec_rows, [])

    def test_non_column_bars_still_get_ordinary_section_design(self):
        """The flip side: a bar NOT flagged is_column is untouched by the
        exclusion and still goes through design_concrete_sections as
        before."""
        from xdfem2d.rc_design import design_concrete_sections
        struc = self._struc(is_column=False)
        sec_rows = design_concrete_sections(struc, self._results())
        self.assertTrue(sec_rows)

    def test_section_is_column_default_is_used_when_bar_does_not_override(self):
        """dev/BUCKLING_COLUMN_PERSISTENCE.md: is_column now defaults on the
        Section (bars using a "column section" are columns unless a bar
        explicitly overrides it). A bar built without ``is_column=`` leaves
        the new Optional[bool] override at None, which must fall back to
        whatever the section says -- not silently to False."""
        s = Structure2D()
        s.add_node("n1", 0.0, 0.0)
        s.add_node("n2", 0.0, 3.0)
        s.add_material("C", elastic_modulus=33e6, unit_weight=25.0,
                       material_type="Concrete",
                       design={"fck": FCK, "fyk": FYK})
        s.add_section("COL", "C", b=0.30, h=0.30, is_column=True)
        s.add_bar_element("E1", "n1", "n2", "COL")   # no is_column override
        out = design_concrete_columns(s, self._results())
        self.assertEqual(len(out["members"]), 1)

    def test_bar_override_wins_over_section_default(self):
        """An explicit is_column=False on the bar must suppress the column
        check even when the section itself defaults to is_column=True."""
        s = Structure2D()
        s.add_node("n1", 0.0, 0.0)
        s.add_node("n2", 0.0, 3.0)
        s.add_material("C", elastic_modulus=33e6, unit_weight=25.0,
                       material_type="Concrete",
                       design={"fck": FCK, "fyk": FYK})
        s.add_section("COL", "C", b=0.30, h=0.30, is_column=True)
        s.add_bar_element("E1", "n1", "n2", "COL", is_column=False)
        out = design_concrete_columns(s, self._results())
        self.assertEqual(out["members"], [])

    def test_section_rc_phi_ef_and_n_bars_feed_the_column_check(self):
        """rc_phi_ef/rc_n_bars/rc_second_order_method now live on the
        Section, not the bar -- confirm they are actually read from there
        (a section with a much higher phi_ef should raise 2nd-order
        moments and thus the reported utilization)."""
        def _struc_with_phi(phi_ef):
            s = Structure2D()
            s.add_node("n1", 0.0, 0.0)
            s.add_node("n2", 0.0, 6.0)
            s.add_material("C", elastic_modulus=33e6, unit_weight=25.0,
                           material_type="Concrete",
                           design={"fck": FCK, "fyk": FYK})
            s.add_section("COL", "C", b=0.30, h=0.30, is_column=True,
                          rc_phi_ef=phi_ef)
            s.add_bar_element("E1", "n1", "n2", "COL")
            return s

        results = self._results(n_axial=-900.0, m_end=40.0)
        out_low = design_concrete_columns(_struc_with_phi(0.0), results)
        out_high = design_concrete_columns(_struc_with_phi(4.0), results)
        self.assertGreater(out_high["members"][0]["utilization"],
                           out_low["members"][0]["utilization"])

    def test_short_column_produces_a_result(self):
        struc = self._struc(length=2.0)
        out = design_concrete_columns(struc, self._results())
        self.assertEqual(len(out["members"]), 1)
        row = out["members"][0]
        self.assertIn("E1", row["elements"])
        self.assertGreater(row["utilization"], 0.0)
        self.assertIn("E1", out["elements"])
        self.assertAlmostEqual(out["elements"]["E1"], row["utilization"])

    def test_slender_column_reports_larger_utilization_than_short(self):
        out_short = design_concrete_columns(self._struc(length=2.0), self._results())
        out_slender = design_concrete_columns(self._struc(length=6.0), self._results())
        u_short = out_short["members"][0]["utilization"]
        u_slender = out_slender["members"][0]["utilization"]
        self.assertGreater(u_slender, u_short)

    def test_heavily_loaded_slender_column_fails(self):
        struc = self._struc(length=7.0)
        out = design_concrete_columns(struc, self._results(n_axial=-1400.0, m_end=60.0))
        row = out["members"][0]
        self.assertFalse(row["passed"])
        self.assertGreater(row["utilization"], 1.0)

    def test_with_reports_attaches_a_trace(self):
        struc = self._struc(length=3.0)
        out = design_concrete_columns(struc, self._results(), with_reports=True)
        self.assertIn("reports", out)
        member_id = out["members"][0]["member"]
        self.assertIn(member_id, out["reports"])

    def test_combination_filter(self):
        struc = self._struc(length=3.0)
        results = self._results()
        results["combinations"]["ULS2"] = results["combinations"]["ULS1"]
        out = design_concrete_columns(struc, results, combinations=["ULS2"])
        self.assertEqual(out["members"][0]["combination"], "ULS2")


class TestSuggestColumnReinforcement(unittest.TestCase):
    def _struc(self, length=3.0):
        s = Structure2D()
        s.add_node("n1", 0.0, 0.0)
        s.add_node("n2", 0.0, length)
        s.add_material("C", elastic_modulus=33e6, unit_weight=25.0,
                       material_type="Concrete",
                       design={"fck": FCK, "fyk": FYK})
        s.add_section("COL", "C", b=0.30, h=0.30)
        s.add_bar_element("E1", "n1", "n2", "COL", is_column=True)
        return s

    def _results(self, n_axial=-300.0, m_end=20.0):
        return {
            "element_forces": {},
            "combinations": {
                "ULS1": {"element_forces": {
                    "E1": {"i": [n_axial, 0, m_end], "j": [-n_axial, 0, m_end]},
                }},
            },
        }

    def test_suggests_a_passing_layout(self):
        struc = self._struc()
        out = suggest_column_reinforcement(struc, self._results())
        self.assertEqual(len(out["members"]), 1)
        row = out["members"][0]
        self.assertTrue(row["passed"])
        self.assertIsNotNone(row["n_bars_suggested"])
        self.assertGreaterEqual(row["n_bars_suggested"], 4)
        self.assertIn("E1", out["elements"])

    def test_impossible_column_reports_failure_not_a_crash(self):
        struc = self._struc(length=9.0)
        out = suggest_column_reinforcement(
            struc, self._results(n_axial=-2500.0, m_end=200.0))
        row = out["members"][0]
        self.assertFalse(row["passed"])
        self.assertIn("error", row)


class TestCircularColumn(unittest.TestCase):
    """Circular concrete columns (dev/RC_COLUMN_DESIGN.md Fase 5 item 6) —
    SectionShape.CIRCULAR (diameter in ``b``, ``h`` ignored, same convention
    as the elastic properties in models.section_area_inertia)."""

    def _struc(self, length=3.0, diameter=0.40):
        from xdfem2d.models import SectionShape
        s = Structure2D()
        s.add_node("n1", 0.0, 0.0)
        s.add_node("n2", 0.0, length)
        s.add_material("C", elastic_modulus=33e6, unit_weight=25.0,
                       material_type="Concrete",
                       design={"fck": FCK, "fyk": FYK})
        s.add_section("COL", "C", b=diameter, h=diameter,
                      shape=SectionShape.CIRCULAR)
        s.add_bar_element("E1", "n1", "n2", "COL", is_column=True)
        return s

    def _results(self, n_axial=-400.0, m_end=30.0):
        return {
            "element_forces": {},
            "combinations": {
                "ULS1": {"element_forces": {
                    "E1": {"i": [n_axial, 0, m_end], "j": [-n_axial, 0, m_end]},
                }},
            },
        }

    def test_circular_column_is_designed(self):
        struc = self._struc()
        out = design_concrete_columns(struc, self._results())
        self.assertEqual(len(out["members"]), 1)
        row = out["members"][0]
        self.assertIn("E1", out["elements"])
        self.assertGreater(row["utilization"], 0.0)
        # A circular column bends the same about y and z by symmetry.
        self.assertAlmostEqual(row["MRd_y"], row["MRd_z"], places=6)

    def test_circular_column_suggestion_uses_symmetric_circular_layout(self):
        struc = self._struc()
        out = suggest_column_reinforcement(struc, self._results())
        row = out["members"][0]
        self.assertTrue(row["passed"])
        self.assertGreaterEqual(row["n_bars_suggested"], 3)

    def test_circular_column_with_reports_attaches_a_trace(self):
        struc = self._struc()
        out = design_concrete_columns(struc, self._results(), with_reports=True)
        member_id = out["members"][0]["member"]
        rep = out["reports"][member_id]
        titles = [sec["title"] for sec in rep["sections"]]
        self.assertTrue(any("M-N interaction" in t for t in titles))


class TestGenericSectionColumn(unittest.TestCase):
    """A column section defined by area_override/inertia_override (not real
    b×h/diameter geometry) has nothing for the fibre integrator to check
    against, so it is skipped rather than treated as a b×h rectangle it may
    not be. dev/RC_COLUMN_DESIGN.md."""

    def _struc(self, length=3.0):
        s = Structure2D()
        s.add_node("n1", 0.0, 0.0)
        s.add_node("n2", 0.0, length)
        s.add_material("C", elastic_modulus=33e6, unit_weight=25.0,
                       material_type="Concrete",
                       design={"fck": FCK, "fyk": FYK})
        s.add_section("COL", "C", b=0.30, h=0.30,
                      area_override=0.09, inertia_override=0.000675)
        s.add_bar_element("E1", "n1", "n2", "COL", is_column=True)
        return s

    def _results(self, n_axial=-400.0, m_end=30.0):
        return {
            "element_forces": {},
            "combinations": {
                "ULS1": {"element_forces": {
                    "E1": {"i": [n_axial, 0, m_end], "j": [-n_axial, 0, m_end]},
                }},
            },
        }

    def test_design_concrete_columns_skips_it(self):
        struc = self._struc()
        out = design_concrete_columns(struc, self._results())
        self.assertEqual(out["members"], [])
        self.assertEqual(out["elements"], {})

    def test_suggest_column_reinforcement_reports_it_visibly(self):
        struc = self._struc()
        out = suggest_column_reinforcement(struc, self._results())
        self.assertEqual(len(out["members"]), 1)
        row = out["members"][0]
        self.assertFalse(row["passed"])
        self.assertIn("error", row)
        self.assertNotIn("E1", out["elements"])


if __name__ == "__main__":
    unittest.main()
