"""Timber load duration: from each load case (explicit, else its action type)."""
import json
import os
import tempfile
import unittest

from xdfem2d import Structure2D
from xdfem2d import timber_design as T
from xdfem2d.models import ActionType, coerce_load_duration
from xdfem2d.structure_io import _to_dict, load_structure_json, save_structure_json


def _model(*cases):
    s = Structure2D()
    ids = []
    for cid, at, ld in cases:
        s.add_load_case(cid, 0.0, action_type=at, load_duration=ld)
        ids.append(cid)
    s.add_load_combination("C", coefficients={i: 1.0 for i in ids})
    return s


class TestLoadDuration(unittest.TestCase):
    def _gov(self, *cases):
        return T._combo_load_duration(_model(*cases), "C", None).name

    def test_every_action_type_has_a_duration(self):
        for a in ActionType:
            self.assertIn(a.value, T._ACTION_LOAD_DURATION)

    def test_automatic_from_action_type(self):
        self.assertEqual(self._gov(("A", "G", None), ("B", "W", None)),
                         "ShortDuration")
        self.assertEqual(self._gov(("A", "Q", None)), "MediumDuration")

    def test_explicit_overrides_action_type(self):
        self.assertEqual(self._gov(("A", "Q", "long")), "LongDuration")
        self.assertEqual(self._gov(("A", "G", None), ("B", "S", "medium")),
                         "MediumDuration")

    def test_coerce(self):
        self.assertEqual(coerce_load_duration("Short-term"), "ShortDuration")
        self.assertEqual(coerce_load_duration("perm"), "Permanent")
        self.assertEqual(coerce_load_duration("auto"), "")
        with self.assertRaises(ValueError):
            coerce_load_duration("forever")

    def test_roundtrip_and_default_not_written(self):
        s = _model(("A", "Q", "long"), ("B", "G", None))
        d = _to_dict(s)
        self.assertEqual(d["load_cases"][0]["load_duration"], "LongDuration")
        self.assertNotIn("load_duration", d["load_cases"][1])
        with tempfile.TemporaryDirectory() as t:
            p = os.path.join(t, "m.json")
            save_structure_json(s, p)
            r = load_structure_json(p)
        self.assertEqual(r.load_cases_by_id["A"].load_duration, "LongDuration")
        self.assertEqual(r.load_cases_by_id["B"].load_duration, "")


if __name__ == "__main__":
    unittest.main()
