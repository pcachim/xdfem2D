"""Tests for the GUI-free report/export builders (xdfem2d.report_io).

These were moved out of the GUI so they can be tested without PySide6. The
builders turn a solved results dict (and the model) into tables, plain-text
report lines, and Excel/Word/PDF documents.
"""
import os
import tempfile
import unittest

from context import Structure2D  # noqa: F401  (adds src/ to sys.path)
from xdfem2d import report_io


def _solved():
    s = Structure2D()
    s.add_material("M", elastic_modulus=30e6, unit_weight=25.0)
    s.add_section("S", "M", b=0.3, h=0.6)
    s.add_node("N1", 0.0, 0.0)
    s.add_node("N2", 5.0, 0.0)
    s.add_bar_element("E1", "N1", "N2", "S")
    s.add_support("PIN", ux=True, uy=True)
    s.add_support("ROLLER", ux=False, uy=True)
    s.assign_support("N1", "PIN")
    s.assign_support("N2", "ROLLER")
    s.add_load_case("LC")
    s.add_distributed_load("E1", "LC", fye=-10.0, fyd=-10.0)
    s.add_analysis_case("AC", "Linear", {"LC": 1.0})
    return s, s.calculate()


class TestTables(unittest.TestCase):
    def setUp(self):
        self.s, self.r = _solved()

    def test_results_tables_shape(self):
        tables = report_io._results_tables(self.r)
        titles = [t[0] for t in tables]
        self.assertIn("Displacements", titles)
        self.assertIn("Reactions", titles)
        for _title, header, rows in tables:
            for row in rows:
                self.assertEqual(len(row), len(header))  # rectangular

    def test_model_tables_have_nodes_and_elements(self):
        titles = [t[0] for t in report_io._model_tables(self.s)]
        self.assertIn("Nodes", titles)
        # A plane model's line elements are Beams (plate: Grid) — topology names.
        self.assertIn("Beams", titles)

    def test_model_table_titles_follow_the_domain(self):
        # Plane model: Beams / Panels / Panel sections.
        plane = [t[0] for t in report_io._model_tables(self.s)]
        assert "Beams" in plane
        # Plate model: Grid / Slabs / Slab sections.
        p = Structure2D(domain='plate')
        p.add_material("M", elastic_modulus=30e6, unit_weight=25.0, poisson=0.2)
        p.add_plate_section("SL", "M", thickness=0.2)
        p.add_node("N1", 0.0, 0.0); p.add_node("N2", 1.0, 0.0)
        p.add_node("N3", 0.0, 1.0)
        p.add_tri_element("T1", "N1", "N2", "N3", "SL")
        titles = [t[0] for t in report_io._model_tables(p)]
        assert "Slabs" in titles and "Slab sections" in titles
        assert "Beams" not in titles and "Panels" not in titles

    def test_results_tables_are_domain_aware(self):
        from xdfem2d.templates import _tpl_slab
        s = _tpl_slab('simply', Lx=4.0, Ly=4.0, nx=4, ny=4)
        s.add_analysis_case('AC', 'Linear', {'SW': 1.0})
        r = s.calculate()
        plate = report_io._results_tables(r, domain='plate')
        titles = [t[0] for t in plate]
        # Area forces replaces Panel stresses (Slab forces/Triangle stresses,
        # pre-"Panel"/"Area" rename) for a plate; displacement header reads w/θ.
        assert "Area forces" in titles and "Panel stresses" not in titles
        disp = next(t for t in plate if t[0] == "Displacements")
        assert "w [m]" in disp[1] and "θx [rad]" in disp[1]
        sf = next(t for t in plate if t[0] == "Area forces")
        assert "mx [kNm/m]" in sf[1] and "m*x,bot" in sf[1]
        # Plane is unchanged (default) — it reports "Panel stresses" instead.
        plane = [t[0] for t in report_io._results_tables(self.r)]
        assert "Panel stresses" in plane or "Displacements" in plane

    def test_edge_loads_appear_in_the_model_tables(self):
        s = Structure2D()
        s.add_material("M", elastic_modulus=30e6, unit_weight=25.0)
        s.add_tri_section("W", "M", thickness=0.2)
        s.add_node("N1", 0.0, 0.0); s.add_node("N2", 1.0, 0.0)
        s.add_node("N3", 0.0, 1.0)
        s.add_tri_element("T1", "N1", "N2", "N3", "W")
        s.add_load_case("LC")
        s.add_tri_edge_load("EL1", "T1", "N1", "N2", "LC", fx=0.0, fy=-5.0)
        tbl = {t[0]: t for t in report_io._model_tables(s)}
        assert "Edge loads" in tbl
        assert tbl["Edge loads"][2]          # has at least one row

    def test_tables_to_lines_fixed_decimals_and_alignment(self):
        hdr = ["Node", "ux [m]", "rz [rad]"]
        rows = [["N1", 0.123456, -0.5], ["N10", 12.5, 0.001]]
        lines = report_io._tables_to_lines([("Displacements", hdr, rows)])
        body = [l for l in lines if l.strip().startswith(("N1", "N10"))]
        # [m] → 3 decimals, [rad] → 5 decimals, numeric columns right-aligned.
        assert "0.123" in body[0] and "-0.50000" in body[0]
        assert "12.500" in body[1] and "0.00100" in body[1]
        # Right-alignment: the two value columns line up on the decimal point.
        assert body[0].index(".") == body[1].index(".")

    def test_tables_to_lines_splits_wide_tables_with_cont(self):
        hdr = ["ID"] + [f"c{i} [kN]" for i in range(12)]
        rows = [["E1"] + [float(i) for i in range(12)]]
        lines = report_io._tables_to_lines([("Element forces", hdr, rows)],
                                           max_width=60)
        assert any("(cont.)" in l for l in lines)
        # The key column (ID) repeats in each group — E1 appears more than once.
        assert sum(l.count("E1") for l in lines) >= 2

    def test_split_report_tables_main_vs_annex(self):
        from xdfem2d.templates import _tpl_slab
        s = _tpl_slab('simply', Lx=3.0, Ly=3.0, nx=3, ny=3)
        s.add_analysis_case('AC', 'Linear', {'SW': 1.0})
        r = s.calculate()
        mt = report_io._model_tables(s, include_project=True, include_model=True)
        rt = report_io._results_tables(r, domain='plate')
        main, annex = report_io.split_report_tables(mt, rt)
        mtitles = [t[0] for t in main]
        atitles = [t[0] for t in annex]
        assert "Materials" in mtitles and "Model info" in mtitles
        assert "Nodes" in atitles and "Area forces" in atitles
        assert "Nodes" not in mtitles and "Materials" not in atitles
        # Materials is curated (fewer columns than the raw model table).
        raw = next(t for t in mt if t[0] == "Materials")
        cur = next(t for t in main if t[0] == "Materials")
        assert len(cur[1]) < len(raw[1])

    def test_reaction_sum_is_total_load(self):
        # Sum of vertical reactions over the AC analysis case = applied load 50 kN.
        rows = report_io._results_reaction_sum_rows(self.r)
        ac_rows = [row for row in rows if "AC" in str(row)]
        self.assertTrue(ac_rows)
        ry = [r for r in ac_rows[0] if isinstance(r, (int, float))]
        self.assertTrue(any(abs(v - 50.0) < 1e-6 for v in ry))


class TestResultsLines(unittest.TestCase):
    """The plain-text results report shared by the GUI panel and exports."""

    def setUp(self):
        self.s, self.r = _solved()

    def test_returns_nonempty_lines(self):
        lines = report_io._build_results_lines(self.r)
        self.assertIsInstance(lines, list)
        self.assertTrue(any(line.strip() for line in lines))

    def test_reports_analysis_case(self):
        text = "\n".join(report_io._build_results_lines(self.r))
        self.assertIn("AC", text)            # the analysis case id appears
        self.assertRegex(text.upper(), r"REACTION|REAC")

    def test_reaction_value_present(self):
        text = "\n".join(report_io._build_results_lines(self.r))
        # Vertical reaction at each support is 25 kN under the 10 kN/m UDL.
        self.assertIn("25.0", text)


class TestFmtHelpers(unittest.TestCase):
    def test_fmt_disp_appends_rows(self):
        lines = []
        report_io._fmt_disp({"N1": [0.0, -0.001, 0.0002]}, lines)
        self.assertTrue(any("N1" in ln for ln in lines))

    def test_fmt_reac_with_sum(self):
        lines = []
        report_io._fmt_reac({"N1": [0.0, 25.0, 0.0], "N2": [0.0, 25.0, 0.0]},
                            lines, show_sum=True)
        self.assertTrue(any("N1" in ln for ln in lines))


class TestDocxExport(unittest.TestCase):
    def setUp(self):
        try:
            import docx  # noqa: F401
        except ImportError:
            self.skipTest("python-docx not installed")

    def test_docx_written(self):
        _s, r = _solved()
        tables = report_io._results_tables(r)
        blocks = [("table", t[0], t[1], t[2]) for t in tables]
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "report.docx")
            report_io._docx_from(path, "xdfem2D — Results", blocks)
            self.assertGreater(os.path.getsize(path), 0)


class TestPdfExport(unittest.TestCase):
    def setUp(self):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib not installed")

    def test_pdf_written(self):
        _s, r = _solved()
        lines = report_io._tables_to_lines(report_io._results_tables(r))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "report.pdf")
            report_io._export_results_to_pdf(lines, path)
            self.assertGreater(os.path.getsize(path), 0)
            with open(path, "rb") as f:
                self.assertEqual(f.read(5), b"%PDF-")


class TestReinforcementTables(unittest.TestCase):
    """dev/GRILLAGE_DESIGN.md §5.3/Phase 3: the grillage reinforcement table
    grows torsion columns, and an extra left/right column only when the
    section opted into perimeter-distributed Asl,tor (some row's
    Asl_tor_by_face has a non-zero side_left)."""

    def _bar_row(self):
        return {
            'element': 'E1', 'combination': 'ULS', 'location': 'i',
            'M_Ed': 50.0, 'N_Ed': 0.0, 'V_Ed': 80.0,
            'As_bot': 2e-4, 'As_top': 0.0, 'Asw_s': 3e-4,
            'cot': 2.5, 'crushing': False, 'governing': True,
        }

    def _grillage_row(self, side_left=0.0, side_right=0.0):
        return {
            'element': 'E1', 'combination': 'ULS', 'location': 'i',
            'M_Ed': 50.0, 'N_Ed': 0.0, 'V_Ed': 80.0, 'T_Ed': 20.0,
            'As_bot': 2e-4, 'As_top': 2e-4, 'Asw_s': 5e-4,
            'Asl_tor': 4e-4,
            'Asl_tor_by_face': {'top': 2e-4, 'bottom': 2e-4,
                                'side_left': side_left, 'side_right': side_right},
            'cot': 2.5, 'interaction': 0.6, 'crushing': False, 'governing': True,
            'kind': 'grillage',
        }

    def _table(self, rows):
        tables = report_io._results_tables({'reinforcement': {'rows': rows}})
        matches = [t for t in tables if t[0].startswith(
            ("Reinforcement", "Grillage reinforcement"))]
        self.assertEqual(len(matches), 1)
        return matches[0]

    def test_plane_bar_table_unaffected(self):
        title, headers, rows = self._table([self._bar_row()])
        self.assertNotIn("T_Ed [kNm]", headers)
        self.assertNotIn("Asl,tor [cm²]", headers)
        self.assertEqual(len(rows[0]), len(headers))

    def test_grillage_table_has_torsion_columns_but_no_side_column_by_default(self):
        title, headers, rows = self._table([self._grillage_row()])
        self.assertIn("Grillage", title)
        self.assertIn("T_Ed [kNm]", headers)
        self.assertIn("Asl,tor [cm²]", headers)
        self.assertNotIn("Asl,tor sides L/R [cm²]", headers)
        self.assertEqual(len(rows[0]), len(headers))

    def test_grillage_table_adds_side_column_when_perimeter_mode_is_active(self):
        title, headers, rows = self._table(
            [self._grillage_row(side_left=1e-4, side_right=1e-4)])
        self.assertIn("Asl,tor sides L/R [cm²]", headers)
        side_col = headers.index("Asl,tor sides L/R [cm²]")
        self.assertEqual(rows[0][side_col], "1.00 / 1.00")
        self.assertEqual(len(rows[0]), len(headers))


if __name__ == "__main__":
    unittest.main(verbosity=2)
