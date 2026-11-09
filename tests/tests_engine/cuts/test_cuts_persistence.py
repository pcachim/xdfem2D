"""Phase-1 Cut tests: entity CRUD and JSON persistence round-trip.
No Qt is involved. See dev/CUT_PLAN.md."""
import os
import tempfile
import unittest

from context import Structure2D


def _beam():
    s = Structure2D()
    s.add_material("M", elastic_modulus=30e6, unit_weight=0.0)
    s.add_section("S", "M", b=0.3, h=0.6)
    s.add_node("N1", 0.0, 0.0)
    s.add_node("N2", 5.0, 0.0)
    s.add_bar_element("E1", "N1", "N2", "S")
    s.add_support("PIN", ux=True, uy=True)
    s.assign_support("N1", "PIN")
    s.add_load_case("LC")
    s.add_distributed_load("E1", "LC", fye=-10.0, fyd=-10.0)
    return s


class TestCutEntity(unittest.TestCase):
    def test_add_cut(self):
        s = _beam()
        cut = s.add_cut("C1", 2.5, -1.0, 2.5, 1.0, name="Mid span")
        self.assertEqual(len(s.cuts), 1)
        self.assertIs(s.cuts[0], cut)
        self.assertEqual(cut.id, "C1")
        self.assertEqual(cut.name, "Mid span")
        self.assertEqual((cut.x1, cut.y1, cut.x2, cut.y2), (2.5, -1.0, 2.5, 1.0))

    def test_add_cut_default_name(self):
        s = _beam()
        cut = s.add_cut("C1", 0.0, 0.0, 1.0, 0.0)
        self.assertEqual(cut.name, "")

    def test_add_duplicate_id_raises(self):
        s = _beam()
        s.add_cut("C1", 0.0, 0.0, 1.0, 0.0)
        with self.assertRaises(ValueError):
            s.add_cut("C1", 0.0, 0.0, 1.0, 1.0)

    def test_remove_cut(self):
        s = _beam()
        s.add_cut("C1", 0.0, 0.0, 1.0, 0.0)
        s.add_cut("C2", 0.0, 1.0, 1.0, 1.0)
        s.remove_cut("C1")
        self.assertEqual([c.id for c in s.cuts], ["C2"])

    def test_remove_missing_cut_is_noop(self):
        s = _beam()
        s.add_cut("C1", 0.0, 0.0, 1.0, 0.0)
        s.remove_cut("NOPE")
        self.assertEqual([c.id for c in s.cuts], ["C1"])

    def test_rename_cut(self):
        s = _beam()
        s.add_cut("C1", 0.0, 0.0, 1.0, 0.0)
        s.rename_cut("C1", "C2")
        self.assertEqual([c.id for c in s.cuts], ["C2"])

    def test_rename_cut_same_id_noop(self):
        s = _beam()
        s.add_cut("C1", 0.0, 0.0, 1.0, 0.0)
        s.rename_cut("C1", "C1")
        self.assertEqual([c.id for c in s.cuts], ["C1"])

    def test_rename_cut_missing_raises(self):
        s = _beam()
        with self.assertRaises(ValueError):
            s.rename_cut("NOPE", "C2")

    def test_rename_cut_duplicate_raises(self):
        s = _beam()
        s.add_cut("C1", 0.0, 0.0, 1.0, 0.0)
        s.add_cut("C2", 0.0, 1.0, 1.0, 1.0)
        with self.assertRaises(ValueError):
            s.rename_cut("C1", "C2")

    def test_check_references_ignores_cuts(self):
        s = _beam()
        s.add_cut("C1", 0.0, 0.0, 1.0, 0.0)
        # Should not raise: cuts carry no cross-references.
        s.check_references()


class TestCutRoundTrip(unittest.TestCase):
    def _save_load(self, s):
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            s.save(path)
            return Structure2D.load(path)
        finally:
            os.remove(path)

    def test_legacy_model_unaffected(self):
        s = _beam()
        r = self._save_load(s)
        self.assertEqual(r.cuts, [])

    def test_cut_round_trip(self):
        s = _beam()
        s.add_cut("C1", 2.5, -1.0, 2.5, 1.0, name="Mid span")
        r = self._save_load(s)
        self.assertEqual(len(r.cuts), 1)
        c = r.cuts[0]
        self.assertEqual(c.id, "C1")
        self.assertEqual(c.name, "Mid span")
        self.assertEqual((c.x1, c.y1, c.x2, c.y2), (2.5, -1.0, 2.5, 1.0))

    def test_multiple_cuts_round_trip(self):
        s = _beam()
        s.add_cut("C1", 0.0, 0.0, 1.0, 0.0)
        s.add_cut("C2", 0.0, 1.0, 1.0, 1.0, name="Second")
        r = self._save_load(s)
        self.assertEqual([c.id for c in r.cuts], ["C1", "C2"])

    def test_old_json_without_cuts_key_loads(self):
        # Simulate a file saved before the 'cuts' key existed.
        s = _beam()
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            s.save(path)
            import json
            with open(path) as f:
                data = json.load(f)
            self.assertIn('cuts', data)
            del data['cuts']
            with open(path, 'w') as f:
                json.dump(data, f)
            r = Structure2D.load(path)
            self.assertEqual(r.cuts, [])
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()
