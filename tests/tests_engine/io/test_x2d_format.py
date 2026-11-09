"""Versions of the .x2d container (xdfem2d.x2d_format): the manifest, files from
before versions, files from a newer version, and the chain of conversions."""
import json
import os
import tempfile
import unittest
import zipfile
import zlib
from pathlib import Path

from context import Structure2D  # noqa: F401  (adds src/ to sys.path)
from xdfem2d import save_x2d, load_x2d, x2d_format as fmt

FROZEN = Path(__file__).resolve().parents[2] / "data" / "formats"


def _model():
    s = Structure2D()
    s.add_material("M", elastic_modulus=30e6, unit_weight=25.0)
    s.add_section("S", "M", b=0.3, h=0.6)
    s.add_node("N1", 0.0, 0.0); s.add_node("N2", 5.0, 0.0)
    s.add_bar_element("E1", "N1", "N2", "S")
    s.add_support("PIN", ux=True, uy=True)
    s.assign_support("N1", "PIN")
    s.add_load_case("LC")
    s.add_distributed_load("E1", "LC", fye=-10.0, fyd=-10.0)
    s.add_analysis_case("AC", "Linear", {"LC": 1.0})
    return s


def _rewrite(src, dst, edit):
    """Copy a .x2d to *dst*, letting ``edit(name, bytes) -> bytes | None`` change
    or drop each entry."""
    with zipfile.ZipFile(src) as zi, zipfile.ZipFile(dst, "w") as zo:
        for n in zi.namelist():
            data = edit(n, zi.read(n))
            if data is not None:
                zo.writestr(n, data)


class _Tmp(unittest.TestCase):
    def setUp(self):
        self._d = tempfile.TemporaryDirectory()
        self.dir = Path(self._d.name)
        self.addCleanup(self._d.cleanup)

    def path(self, name):
        return self.dir / name


class TestManifest(_Tmp):
    def test_a_saved_file_carries_the_version_first(self):
        p = self.path("m.x2d")
        save_x2d(_model(), None, p)
        with zipfile.ZipFile(p) as zf:
            self.assertEqual(zf.namelist()[0], "manifest.json")
            m = json.loads(zf.read("manifest.json"))
        self.assertEqual(m["format_version"], fmt.FORMAT_VERSION)
        self.assertEqual(m["format"], "x2d")
        self.assertIn("structure.json", m["entries"])
        self.assertTrue(m["app_version"])

    def test_round_trip_is_not_reported_as_converted(self):
        p = self.path("m.x2d")
        save_x2d(_model(), None, p)
        s, _, _ = load_x2d(p)
        self.assertIsNone(s._format_converted_from)
        self.assertEqual(sorted(s.nodes), ["N1", "N2"])

    def test_same_model_gives_same_bytes(self):
        a, b = self.path("a.x2d"), self.path("b.x2d")
        save_x2d(_model(), None, a); save_x2d(_model(), None, b)
        self.assertEqual(a.read_bytes(), b.read_bytes())

    def test_a_file_without_manifest_is_version_1_and_current(self):
        p, q = self.path("m.x2d"), self.path("old.x2d")
        save_x2d(_model(), None, p)
        _rewrite(p, q, lambda n, d: None if n == "manifest.json" else d)
        s, _, _ = load_x2d(q)
        self.assertIsNone(s._format_converted_from)
        self.assertEqual(fmt.UNVERSIONED, 1)


class TestRefusals(_Tmp):
    def _with_manifest(self, manifest):
        p, q = self.path("m.x2d"), self.path("x.x2d")
        save_x2d(_model(), None, p)
        _rewrite(p, q, lambda n, d: manifest if n == "manifest.json" else d)
        return q

    def test_a_newer_version_is_refused_with_a_message(self):
        q = self._with_manifest(json.dumps(
            {"format_version": fmt.FORMAT_VERSION + 1,
             "app_version": "2099.1"}).encode())
        with self.assertRaises(fmt.X2DFormatError) as cm:
            load_x2d(q)
        msg = str(cm.exception)
        self.assertIn(str(fmt.FORMAT_VERSION + 1), msg)
        self.assertIn("2099.1", msg)
        self.assertIn("Update", msg)

    def test_an_unreadable_manifest_is_refused(self):
        with self.assertRaises(fmt.X2DFormatError):
            load_x2d(self._with_manifest(b"not json"))

    def test_a_manifest_without_a_whole_version_is_refused(self):
        for bad in (b'{"format_version": "2"}', b'{"format_version": 0}',
                    b'{"format_version": true}', b'{"x": 1}'):
            with self.assertRaises(fmt.X2DFormatError, msg=bad):
                load_x2d(self._with_manifest(bad))

    def test_it_is_a_value_error(self):
        self.assertTrue(issubclass(fmt.X2DFormatError, ValueError))


class TestConversionChain(_Tmp):
    """With FORMAT_VERSION raised to 3 and two conversions registered."""

    def setUp(self):
        super().setUp()
        self._saved = (fmt.FORMAT_VERSION, dict(fmt._MIGRATIONS))
        self.addCleanup(self._restore)
        fmt.FORMAT_VERSION = 3
        fmt._MIGRATIONS.clear()

    def _restore(self):
        fmt.FORMAT_VERSION = self._saved[0]
        fmt._MIGRATIONS.clear(); fmt._MIGRATIONS.update(self._saved[1])

    def _file_of_version(self, version, with_variant=True, results=None):
        from xdfem2d.workflows import new_variant
        from xdfem2d.structure_io import _to_dict
        s = _model()
        if with_variant:
            s.add_variant(new_variant(s, "V1"))
        snap = [{"name": "s", "time": 1.0,
                 "data": zlib.compress(json.dumps(_to_dict(s)).encode())}]
        p, q = self.path("cur.x2d"), self.path(f"v{version}.x2d")
        save_x2d(s, results, p, snapshots=snap)
        man = json.dumps({"format_version": version}).encode()
        _rewrite(p, q, lambda n, d: man if n == "manifest.json" else d)
        return q

    def test_conversions_run_in_order_on_every_model(self):
        seen = []

        @fmt.migration(1)
        def one(docs):
            seen.append(1)
            for m in docs.models():
                m["touched_by"] = m.get("touched_by", []) + [1]

        @fmt.migration(2)
        def two(docs):
            seen.append(2)
            for m in docs.models():
                m["touched_by"] = m.get("touched_by", []) + [2]

        models = []
        orig = fmt.Documents.models

        def spy(self):
            for m in orig(self):
                models.append(m); yield m
        fmt.Documents.models = spy
        self.addCleanup(setattr, fmt.Documents, "models", orig)

        s, _, _ = load_x2d(self._file_of_version(1))
        self.assertEqual(seen, [1, 2])
        self.assertEqual(s._format_converted_from, 1)
        # main model + the variant's own + the snapshot's, each converted twice
        self.assertEqual(len(models) // 2, 3)
        self.assertTrue(all(m["touched_by"] == [1, 2] for m in models[-3:]))

    def test_a_file_in_the_middle_runs_only_the_rest(self):
        seen = []
        fmt.migration(1)(lambda d: seen.append(1))
        fmt.migration(2)(lambda d: seen.append(2))
        s, _, _ = load_x2d(self._file_of_version(2))
        self.assertEqual(seen, [2])
        self.assertEqual(s._format_converted_from, 2)

    def test_a_missing_conversion_is_an_error_not_a_guess(self):
        fmt.migration(2)(lambda d: None)
        with self.assertRaises(fmt.X2DFormatError):
            load_x2d(self._file_of_version(1))

    def test_a_conversion_can_drop_the_stored_results(self):
        fmt.migration(1)(lambda d: setattr(d, "drop_results", True))
        fmt.migration(2)(lambda d: None)
        s = _model()
        res = s.calculate()
        _, results, _ = load_x2d(self._file_of_version(1, results=res))
        self.assertIsNone(results)
        _, kept, _ = load_x2d(self._file_of_version(3, results=res))
        self.assertIsNotNone(kept)

    def test_notes_say_what_was_done(self):
        fmt.migration(1)(lambda d: None)
        fmt.migration(2)(lambda d: None)
        docs = fmt.Documents({"structure.json": {}})
        fmt.migrate(docs, 1)
        self.assertEqual(docs.notes, ["converted from format version 1 to 2",
                                      "converted from format version 2 to 3"])

    def test_two_conversions_from_one_version_are_refused(self):
        fmt.migration(1)(lambda d: None)
        with self.assertRaises(ValueError):
            fmt.migration(1)(lambda d: None)


class TestFrozenVersion1(unittest.TestCase):
    """A file of version 1, as every file was before versions existed (no
    manifest), is never edited: it must open as it does now."""

    PATH = FROZEN / "v1" / "variants_areas_snapshot.x2d"

    def test_it_has_no_manifest(self):
        with zipfile.ZipFile(self.PATH) as zf:
            self.assertNotIn("manifest.json", zf.namelist())

    def test_it_opens_with_everything_it_holds(self):
        s, results, view = load_x2d(self.PATH)
        self.assertIsNone(results)
        self.assertEqual(view, {"zoom": 1.0})
        self.assertEqual(sorted(s.nodes), ["N1", "N2", "N3", "N4", "N5", "N6"])
        self.assertEqual(sorted(s.tri_elements_by_id), ["T1", "T2", "T3"])
        self.assertEqual([e.id for e in s.bar_elements], ["E1"])
        self.assertIn("AC", [a.id for a in s.analysis_cases])
        v = s.variants["V1"]
        self.assertEqual(v.active_elements, {"E1", "T1", "T2"})
        self.assertEqual(sorted(v.model.tri_elements_by_id), ["T1", "T2"])
        self.assertEqual([sn["name"] for sn in s._loaded_snapshots], ["before"])

    def test_it_saves_in_the_current_version_and_opens_again(self):
        s, _, _ = load_x2d(self.PATH)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "again.x2d"
            save_x2d(s, None, p)
            with zipfile.ZipFile(p) as zf:
                self.assertEqual(json.loads(zf.read("manifest.json"))
                                 ["format_version"], fmt.FORMAT_VERSION)
            s2, _, _ = load_x2d(p)
        self.assertEqual(s2.variants["V1"].active_elements, {"E1", "T1", "T2"})


class TestVariantAreaChoiceSurvivesSaving(_Tmp):
    """The load-time repair of a variant's active set must not drop the areas
    the variant chose."""

    def test_round_trip_keeps_areas(self):
        from xdfem2d.workflows import new_variant
        s = _model()
        s.add_tri_section("TS", "M", thickness=0.2)
        s.add_node("N3", 5.0, 3.0)
        s.add_tri_element("T1", "N1", "N2", "N3", "TS")
        s.add_variant(new_variant(s, "V1", active_elements={"E1", "T1"}))
        p = self.path("v.x2d")
        save_x2d(s, None, p)
        s2, _, _ = load_x2d(p)
        self.assertEqual(s2.variants["V1"].active_elements, {"E1", "T1"})
        self.assertIn("T1", s2.variants["V1"].model.tri_elements_by_id)


if __name__ == "__main__":
    unittest.main()
