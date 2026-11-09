"""Beam tags (``BarElement.beam``) and the ``Structure2D.beams`` registry:
API, persistence (all loaders + script export), splits, and the tag-driven
beam-bars design."""
import os
import tempfile
import unittest

from context import Structure2D  # noqa: F401  (adds src/ to sys.path)

from xdfem2d import design_beam_bars, design_concrete_sections
from xdfem2d import subdivide as S
from xdfem2d.rc_design import suggest_beams
from xdfem2d.structure_io import load_structure_json, save_structure_json
from xdfem2d.structure_io_checked import load_structure_json_checked
from xdfem2d.script_export import to_python


def _frame(columns=True):
    """Two-span beam (E1, E2) on three columns, plus a free spare bar E3."""
    s = Structure2D()
    for nid, x, y in (("a", 0, 3), ("b", 5, 3), ("c", 10, 3),
                      ("g1", 0, 0), ("g2", 5, 0), ("g3", 10, 0)):
        s.add_node(nid, x, y)
    s.add_material("C", elastic_modulus=33e6, unit_weight=25.0,
                   material_type="Concrete", design={"fck": 30, "fyk": 500})
    s.add_section("CONC", "C", b=0.3, h=0.5)
    s.add_section("CONC2", "C", b=0.3, h=0.6)
    s.add_section("COL", "C", b=0.3, h=0.3, is_column=True)
    s.add_bar_element("E1", "a", "b", "CONC")
    s.add_bar_element("E2", "b", "c", "CONC")
    if columns:
        for i, (g, n) in enumerate((("g1", "a"), ("g2", "b"), ("g3", "c")), 1):
            s.add_bar_element(f"C{i}", g, n, "COL")
    s.add_support("FIX", ux=True, uy=True, tz=True)
    for g in ("g1", "g2", "g3"):
        s.assign_support(g, "FIX")
    return s


def _rows(s, ids):
    res = {"element_forces": {}, "combinations": {"ULS1": {"element_forces": {
        i: {"i": [0, 40, -50], "j": [0, -40, 100]} for i in ids}}}}
    return design_concrete_sections(s, res), res


class TestBeamApi(unittest.TestCase):
    def test_assign_and_clear(self):
        s = _frame()
        tag = s.assign_beam(["E1", "E2"], name="Main")
        self.assertEqual(tag, "V1")
        self.assertEqual({e.beam for e in s.bar_elements if e.beam}, {"V1"})
        self.assertEqual(s.beams["V1"]["name"], "Main")
        self.assertEqual(s.next_beam_tag(), "V2")
        s.clear_beam(["E1", "E2"])
        self.assertTrue(all(e.beam is None for e in s.bar_elements))
        self.assertIn("V1", s.beams)           # named -> survives empty
        s.beams["V1"]["name"] = ""
        s.prune_beams()
        self.assertNotIn("V1", s.beams)

    def test_unknown_bar_rejected_and_model_untouched(self):
        s = _frame()
        with self.assertRaises(KeyError):
            s.assign_beam(["E1", "NOPE"])
        self.assertIsNone(s.bar_elements_by_id["E1"].beam)
        with self.assertRaises(ValueError):
            s.assign_beam([])


class TestBeamPersistence(unittest.TestCase):
    def _model(self):
        s = _frame()
        s.assign_beam(["E1", "E2"], tag="V7", name="Level 1")
        s.beams["V7"]["overrides"] = {"max_layers": 1, "n_through": 0}
        return s

    def _check(self, s2):
        self.assertEqual(s2.bar_elements_by_id["E1"].beam, "V7")
        self.assertEqual(s2.bar_elements_by_id["E2"].beam, "V7")
        self.assertIsNone(s2.bar_elements_by_id["C1"].beam)
        self.assertEqual(s2.beams, {"V7": {
            "name": "Level 1",
            "overrides": {"max_layers": 1, "n_through": 0}}})

    def test_json_roundtrip_both_loaders(self):
        s = self._model()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.json")
            save_structure_json(s, p)
            self._check(load_structure_json(p))
            s2, report = load_structure_json_checked(p)
            self._check(s2)

    def test_old_file_without_beams_loads(self):
        import json
        s = _frame()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.json")
            save_structure_json(s, p)
            data = json.load(open(p))
            data.pop("beams")
            for e in data["bar_elements"]:
                e.pop("beam")
            json.dump(data, open(p, "w"))
            for s2 in (load_structure_json(p), load_structure_json_checked(p)[0]):
                self.assertEqual(s2.beams, {})
                self.assertTrue(all(e.beam is None for e in s2.bar_elements))

    def test_script_export_round_trips(self):
        s = self._model()
        ns: dict = {}
        exec(compile(to_python(s), "<export>", "exec"), ns)
        rebuild = next(v for k, v in ns.items()
                       if callable(v) and k.startswith("build"))
        self._check(rebuild())

    def test_divide_bars_copies_the_tag(self):
        s = self._model()
        S.divide_bars(s, ["E1"], count=3)
        subs = [e for e in s.bar_elements if e.id.startswith("E1/")]
        self.assertEqual(len(subs), 3)
        self.assertTrue(all(e.beam == "V7" for e in subs))

    def test_tee_split_copies_the_tag(self):
        from xdfem2d.geo_expand import expand_geometry
        s = Structure2D()
        s.add_node("a", 0, 0)
        s.add_node("b", 10, 0)
        s.add_material("C", elastic_modulus=33e6, unit_weight=25.0,
                       material_type="Concrete")
        s.add_section("CONC", "C", b=0.3, h=0.5)
        s.add_bar_element("E1", "a", "b", "CONC", beam="V1")
        s.add_geo_polyline("D", [[5, 0], [5, 4]], section_name="CONC")
        compiled, _ = expand_geometry(s)
        pieces = [e for e in compiled.bar_elements if e.id.startswith("E1")]
        self.assertGreaterEqual(len(pieces), 2)          # E1 split at x = 5
        self.assertTrue(all(e.beam == "V1" for e in pieces))


class TestBeamObjects(unittest.TestCase):
    """Line objects carry the tag; the bars they generate inherit it."""

    def _model(self):
        s = _frame(columns=True)
        s.remove_element("E1")
        s.remove_element("E2")
        s.add_geo_multisegment("OBJ", [[0, 3], [5, 3], [10, 3]],
                               section_name="CONC", divisions=2)
        return s

    def test_assign_to_objects_and_registry(self):
        s = self._model()
        tag = s.assign_beam(object_ids=["OBJ"], name="Obj beam")
        self.assertEqual(tag, "V1")
        self.assertEqual(s.geometry_objects["OBJ"].beam, "V1")
        self.assertEqual(s.next_beam_tag(), "V2")     # object tags count
        s.clear_beam(object_ids=["OBJ"])
        self.assertIsNone(s.geometry_objects["OBJ"].beam)
        with self.assertRaises(KeyError):
            s.assign_beam(object_ids=["NOPE"])
        with self.assertRaises(ValueError):
            s.assign_beam()

    def test_generated_bars_inherit_the_tag_on_every_regeneration(self):
        from xdfem2d.geo_expand import expand_geometry
        s = self._model()
        s.assign_beam(object_ids=["OBJ"], tag="V1")
        for _ in range(2):                              # regenerate twice
            mesh, trace = expand_geometry(s)
            ids = trace["OBJ"]["elems"]
            self.assertGreaterEqual(len(ids), 4)
            self.assertTrue(all(mesh.bar_elements_by_id[b].beam == "V1"
                                for b in ids))
        self.assertEqual(set(mesh.beams), {"V1"})

    def test_persistence_and_export(self):
        s = self._model()
        s.assign_beam(object_ids=["OBJ"], tag="V3", name="N")
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.json")
            save_structure_json(s, p)
            for s2 in (load_structure_json(p),
                       load_structure_json_checked(p)[0]):
                self.assertEqual(s2.geometry_objects["OBJ"].beam, "V3")
        ns: dict = {}
        exec(compile(to_python(s), "<export>", "exec"), ns)
        rebuild = next(v for k, v in ns.items()
                       if callable(v) and k.startswith("build"))
        self.assertEqual(rebuild().geometry_objects["OBJ"].beam, "V3")

    def test_design_uses_the_object_tag(self):
        from xdfem2d.geo_expand import expand_geometry
        s = self._model()
        s.assign_beam(object_ids=["OBJ"], tag="V1", name="Floor")
        mesh, trace = expand_geometry(s)
        ids = trace["OBJ"]["elems"]
        rows, res = _rows(mesh, ids)
        out = design_beam_bars(mesh, rows, res)
        self.assertEqual({r["beam"] for r in out}, {"Floor"})
        self.assertEqual({r["beam_tag"] for r in out}, {"V1"})

    def test_a_tagged_object_can_join_a_real_bar_in_one_beam(self):
        from xdfem2d.geo_expand import expand_geometry
        s = _frame(columns=False)                      # real E1 (a-b), E2 (b-c)
        s.remove_element("E2")
        s.add_geo_multisegment("OBJ", [[5, 3], [10, 3]],
                               section_name="CONC", divisions=1)
        s.assign_beam(["E1"], tag="V1", object_ids=["OBJ"])
        mesh, trace = expand_geometry(s)
        ids = ["E1"] + trace["OBJ"]["elems"]
        rows, res = _rows(mesh, ids)
        out = design_beam_bars(mesh, rows, res)
        self.assertEqual({r["beam"] for r in out}, {"V1"})


class TestDesignByTags(unittest.TestCase):
    def test_suggest_beams_ignores_columns_and_orders_left_to_right(self):
        s = _frame()
        self.assertEqual(suggest_beams(s), [["E1", "E2"]])

    def test_untagged_beams_are_auto_named(self):
        s = _frame()
        rows, res = _rows(s, ("E1", "E2"))
        out = design_beam_bars(s, rows, res)
        self.assertEqual({r["beam"] for r in out}, {"B1"})
        self.assertEqual({r["beam_tag"] for r in out}, {None})

    def test_tag_groups_and_names_the_beam(self):
        s = _frame()
        s.assign_beam(["E1", "E2"], tag="V1", name="Floor 1")
        rows, res = _rows(s, ("E1", "E2"))
        out = design_beam_bars(s, rows, res)
        self.assertEqual({r["beam"] for r in out}, {"Floor 1"})
        self.assertEqual({r["beam_tag"] for r in out}, {"V1"})

    def test_a_tag_can_group_what_auto_detection_would_split(self):
        s = _frame(columns=False)
        s.bar_elements_by_id["E1"].hinge_j = True        # auto: two beams
        rows, res = _rows(s, ("E1", "E2"))
        self.assertEqual({r["beam"] for r in design_beam_bars(s, rows, res)},
                         {"B1", "B2"})
        s.assign_beam(["E1", "E2"], tag="V1")
        out = design_beam_bars(s, rows, res)
        # a hinge is still a break of continuity: tagged beam -> 2 segments
        self.assertEqual({r["beam"] for r in out}, {"V1.1", "V1.2"})
        self.assertTrue(all(any("segments" in n for n in r["notes"])
                            for r in out))

    def test_section_change_inside_a_tag_makes_segments(self):
        s = _frame()
        s.bar_elements_by_id["E2"].section_name = "CONC2"
        s.assign_beam(["E1", "E2"], tag="V1")
        rows, res = _rows(s, ("E1", "E2"))
        out = design_beam_bars(s, rows, res)
        self.assertEqual({r["beam"] for r in out}, {"V1.1", "V1.2"})

    def test_registry_overrides_reach_the_design(self):
        s = _frame()
        s.assign_beam(["E1", "E2"], tag="V1")
        s.beams["V1"]["overrides"] = {"diameters": [16], "max_diameters_beam": 1}
        rows, res = _rows(s, ("E1", "E2"))
        out = design_beam_bars(s, rows, res)
        self.assertTrue(all(g[1] == 16 for r in out
                            for g in r["bottom_layers"] + r["top_layers"]))

    def test_bad_override_key_is_reported(self):
        s = _frame()
        s.assign_beam(["E1", "E2"], tag="V1")
        s.beams["V1"]["overrides"] = {"max_layer": 1}
        rows, res = _rows(s, ("E1", "E2"))
        with self.assertRaises(ValueError):
            design_beam_bars(s, rows, res)

    def test_auto_names_skip_tag_names(self):
        s = _frame()
        s.add_node("d", 10, 8)
        s.add_node("e", 15, 8)
        s.add_bar_element("E9", "d", "e", "CONC")
        s.assign_beam(["E1", "E2"], tag="B1")             # user tag = auto name
        rows, res = _rows(s, ("E1", "E2", "E9"))
        names = {r["beam"] for r in design_beam_bars(s, rows, res)}
        self.assertEqual(names, {"B1", "B2"})


if __name__ == "__main__":
    unittest.main()


class TestSectionOverrides(unittest.TestCase):
    def test_persist_and_reach_the_design(self):
        s = _frame()
        s.sections["CONC"].beam_overrides = {"diameters": [16],
                                              "stirrup_diameter_mm": 10}
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.json")
            save_structure_json(s, p)
            for s2 in (load_structure_json(p),
                       load_structure_json_checked(p)[0]):
                self.assertEqual(s2.sections["CONC"].beam_overrides,
                                 {"diameters": [16], "stirrup_diameter_mm": 10})
                self.assertIsNone(s2.sections["CONC2"].beam_overrides)
        ns: dict = {}
        exec(compile(to_python(s), "<export>", "exec"), ns)
        rebuild = next(v for k, v in ns.items()
                       if callable(v) and k.startswith("build"))
        self.assertEqual(rebuild().sections["CONC"].beam_overrides,
                         {"diameters": [16], "stirrup_diameter_mm": 10})
        rows, res = _rows(s, ("E1", "E2"))
        out = design_beam_bars(s, rows, res)
        self.assertTrue(all(g[1] == 16 for r in out
                            for g in r["bottom_layers"] + r["top_layers"]))

    def test_beam_override_beats_section_override(self):
        s = _frame()
        s.sections["CONC"].beam_overrides = {"diameters": [16]}
        s.assign_beam(["E1", "E2"], tag="V1")
        s.beams["V1"]["overrides"] = {"diameters": [20, 25]}
        rows, res = _rows(s, ("E1", "E2"))
        out = design_beam_bars(s, rows, res)
        self.assertTrue(all(g[1] in (20, 25) for r in out
                            for g in r["bottom_layers"] + r["top_layers"]))


class TestAutoDetectHorizontalOnly(unittest.TestCase):
    """Automatic detection ignores steep members (unflagged columns, walls)."""

    def _model(self, top_y=3.0, dx=0.0):
        s = Structure2D()
        s.add_node("g", 0, 0)
        s.add_node("a", dx, top_y)        # steep bar g-a
        s.add_node("b", dx + 5, top_y)    # horizontal bar a-b
        s.add_material("C", elastic_modulus=33e6, unit_weight=25.0,
                       material_type="Concrete",
                       design={"fck": 30, "fyk": 500})
        s.add_section("CONC", "C", b=0.3, h=0.5)   # NOT flagged as column
        s.add_bar_element("STEEP", "g", "a", "CONC")
        s.add_bar_element("HORIZ", "a", "b", "CONC")
        s.add_support("FIX", ux=True, uy=True, tz=True)
        s.assign_support("g", "FIX")
        return s

    def test_vertical_bar_is_not_a_beam(self):
        s = self._model()
        self.assertEqual(suggest_beams(s), [["HORIZ"]])
        rows, res = _rows(s, ("STEEP", "HORIZ"))
        out = design_beam_bars(s, rows, res)
        self.assertEqual({e for r in out for e in r["elements"]}, {"HORIZ"})

    def test_limit_is_45_degrees_inclusive(self):
        at_45 = self._model(top_y=3.0, dx=3.0)       # 45°  -> still a beam
        self.assertIn(["STEEP"], suggest_beams(at_45))
        steep = self._model(top_y=3.0, dx=2.5)       # ~50°  -> excluded
        self.assertNotIn(["STEEP"], suggest_beams(steep))

    def test_an_explicit_tag_overrides_the_slope_filter(self):
        s = self._model()
        s.assign_beam(["STEEP"], tag="V1")           # user decides: it is a beam
        rows, res = _rows(s, ("STEEP", "HORIZ"))
        out = design_beam_bars(s, rows, res)
        self.assertEqual({r["beam"] for r in out if "STEEP" in r["elements"]},
                         {"V1"})
