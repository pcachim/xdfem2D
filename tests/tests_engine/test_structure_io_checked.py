"""Validated JSON loader: reads to the end, collects every problem, and builds
what is valid instead of raising on the first bad entry."""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from xdfem2d import (Structure2D, save_structure_json,
                     load_structure_json_checked, format_report)


def _sample() -> Structure2D:
    s = Structure2D()
    s.add_node('N1', 0.0, 0.0); s.add_node('N2', 5.0, 0.0)
    s.add_material('C30', 30e6, 25.0)
    s.add_section('S1', 'C30', 0.3, 0.6)
    s.add_bar_element('E1', 'N1', 'N2', 'S1')
    s.add_support('PIN', ux=True, uy=True)
    s.assign_support('N1', 'PIN')
    s.add_load_case('LC1')
    # A complete model carries an analysis case; without one the resilient
    # loader now (correctly) folds a completeness warning into the report, so a
    # fixture meant to be "valid" must have it.
    s.add_analysis_case('AC1', 'Linear', coefficients={'LC1': 1.0})
    return s


class TestCheckedLoader(unittest.TestCase):

    def _write(self, tmp, data) -> str:
        p = Path(tmp) / "m.json"
        p.write_text(json.dumps(data), encoding='utf-8')
        return str(p)

    def _write_text(self, tmp, text) -> str:
        p = Path(tmp) / "m.json"
        p.write_text(text, encoding='utf-8')
        return str(p)

    def test_valid_file_reports_no_problems(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "m.json"
            save_structure_json(_sample(), p)
            struc, rep = load_structure_json_checked(p)
        self.assertTrue(rep.ok)
        self.assertEqual(rep.issues, [])
        self.assertEqual(rep.counts.get('nodes'), 2)
        self.assertEqual(len(struc.nodes), 2)

    def test_missing_key_is_reported_and_skipped(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "m.json"
            save_structure_json(_sample(), p)
            data = json.loads(Path(p).read_text())
            data['nodes'].append({'id': 'BAD'})          # no x, y
            p2 = self._write(tmp, data)
            struc, rep = load_structure_json_checked(p2)
        self.assertFalse(rep.ok)
        self.assertEqual(len(rep.errors), 1)
        self.assertEqual(rep.errors[0].section, 'nodes')
        self.assertIn('x', rep.errors[0].message)
        self.assertEqual(len(struc.nodes), 2)            # only good ones

    def test_bad_reference_is_reported_and_skipped(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "m.json"
            save_structure_json(_sample(), p)
            data = json.loads(Path(p).read_text())
            data['sections'].append(
                {'name': 'S9', 'material_name': 'NOPE', 'b': 0.3, 'h': 0.6})
            p2 = self._write(tmp, data)
            struc, rep = load_structure_json_checked(p2)
        self.assertEqual(len(rep.errors), 1)
        self.assertEqual(rep.errors[0].section, 'sections')
        self.assertNotIn('S9', struc.sections)

    def test_all_errors_collected_not_just_first(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "m.json"
            save_structure_json(_sample(), p)
            data = json.loads(Path(p).read_text())
            data['nodes'].append({'id': 'BAD'})
            data['sections'].append(
                {'name': 'S9', 'material_name': 'NOPE', 'b': 0.3, 'h': 0.6})
            p2 = self._write(tmp, data)
            _, rep = load_structure_json_checked(p2)
        self.assertEqual(len(rep.errors), 2)             # read to the end

    def test_syntax_error_recovered_and_reported(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "m.json"
            save_structure_json(_sample(), p)
            text = Path(p).read_text().replace('"nodes": [', '"nodes": [ ,', 1)
            p2 = self._write_text(tmp, text)
            struc, rep = load_structure_json_checked(p2)
        self.assertTrue(rep.parsed)                      # recovered past it
        self.assertEqual(len(rep.syntax_errors), 1)
        self.assertEqual(len(struc.nodes), 2)

    def test_unparseable_returns_empty_structure(self):
        with TemporaryDirectory() as tmp:
            p2 = self._write_text(tmp, "this is not json at all {{{")
            struc, rep = load_structure_json_checked(p2)
        self.assertFalse(rep.parsed)
        self.assertTrue(rep.syntax_errors)
        self.assertEqual(len(struc.nodes), 0)

    def test_format_report_is_plain_text(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "m.json"
            save_structure_json(_sample(), p)
            _, rep = load_structure_json_checked(p)
        txt = format_report(rep)
        self.assertIn("Nodes: 2", txt)
        self.assertIn("valid", txt.lower())


class TestBucklingColumnPersistence(unittest.TestCase):
    """Regression for dev/BUCKLING_COLUMN_PERSISTENCE.md fase 1: the K
    overrides, is_column (section default + per-bar override), and the
    column-check parameters must all survive a real Save -> Open round trip
    through the loader the GUI's File > Open actually uses
    (load_structure_json_checked), not just the plain load_structure_json.
    """

    def _struc(self):
        s = Structure2D()
        s.add_node('N1', 0.0, 0.0); s.add_node('N2', 0.0, 3.0)
        s.add_node('N3', 5.0, 0.0)
        s.add_material('C30', 30e6, 25.0, material_type='Concrete',
                       design={'fck': 30.0, 'fyk': 500.0})
        s.add_section('COL', 'C30', 0.3, 0.3, is_column=True, rc_phi_ef=1.8,
                      rc_n_bars=8, rc_second_order_method='nominal_stiffness',
                      rc_torsion_distribution='perimeter')
        s.add_section('BEAM', 'C30', 0.3, 0.5)
        e1 = s.add_bar_element('E1', 'N1', 'N2', 'COL')
        e1.sd_ky = 2.0; e1.sd_kz = 1.5; e1.sd_klt = 1.2; e1.sd_ltb = False
        # E2 explicitly overrides the section's is_column=True to False.
        s.add_bar_element('E2', 'N1', 'N3', 'COL', is_column=False)
        return s

    def _round_trip(self, loader):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "m.json"
            save_structure_json(self._struc(), p)
            result = loader(p)
        return result[0] if isinstance(result, tuple) else result

    def test_section_column_defaults_survive_checked_open(self):
        struc = self._round_trip(load_structure_json_checked)
        col = struc.sections['COL']
        self.assertTrue(col.is_column)
        self.assertAlmostEqual(col.rc_phi_ef, 1.8)
        self.assertEqual(col.rc_n_bars, 8)
        self.assertEqual(col.rc_second_order_method, 'nominal_stiffness')
        self.assertEqual(col.rc_torsion_distribution, 'perimeter')
        # A section without an explicit is_column keeps the (unrelated)
        # default False -- confirms absent-key compatibility, not just that
        # the flag round-trips when set.
        self.assertFalse(struc.sections['BEAM'].is_column)

    def test_bar_k_overrides_survive_checked_open(self):
        struc = self._round_trip(load_structure_json_checked)
        e1 = struc.bar_elements_by_id['E1']
        self.assertEqual(e1.sd_ky, 2.0)
        self.assertEqual(e1.sd_kz, 1.5)
        self.assertEqual(e1.sd_klt, 1.2)
        self.assertFalse(e1.sd_ltb)

    def test_bar_is_column_override_survives_checked_open(self):
        struc = self._round_trip(load_structure_json_checked)
        # E1 has no per-bar override -> None, follows the section (True).
        self.assertIsNone(struc.bar_elements_by_id['E1'].is_column)
        # E2 explicitly overrides to False.
        self.assertEqual(struc.bar_elements_by_id['E2'].is_column, False)

    def test_same_fields_survive_the_plain_loader_too(self):
        from xdfem2d import load_structure_json
        struc = self._round_trip(load_structure_json)
        self.assertTrue(struc.sections['COL'].is_column)
        self.assertEqual(struc.bar_elements_by_id['E1'].sd_ky, 2.0)
        self.assertEqual(struc.bar_elements_by_id['E2'].is_column, False)

    def test_object_level_design_fields_survive_checked_open(self):
        """dev/BUCKLING_COLUMN_PERSISTENCE.md fase 3 (estrutural): is_column/
        sd_ky/sd_kz/sd_klt/sd_ltb set directly on a GeoSegment object must
        survive a Save -> Open round trip through both loaders."""
        s = self._struc()
        col = s.add_geo_segment('COLOBJ', 5.0, 0.0, 5.0, 3.0, section_name='COL',
                                divisions=2)
        col.is_column = True
        col.sd_ky = 0.7; col.sd_kz = 0.85; col.sd_klt = 1.0; col.sd_ltb = False
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "m.json"
            save_structure_json(s, p)
            for loader in (load_structure_json_checked,
                          __import__('xdfem2d', fromlist=['load_structure_json']).load_structure_json):
                result = loader(p)
                struc = result[0] if isinstance(result, tuple) else result
                obj = struc.geometry_objects['COLOBJ']
                self.assertTrue(obj.is_column)
                self.assertEqual(obj.sd_ky, 0.7)
                self.assertEqual(obj.sd_kz, 0.85)
                self.assertEqual(obj.sd_klt, 1.0)
                self.assertFalse(obj.sd_ltb)


if __name__ == '__main__':
    unittest.main()
