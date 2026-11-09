"""Detailing core (no Qt): data contract, required steel, checks, editing with
undo/redo, reconciliation and the self-contained package."""
import copy
import json
import unittest

import numpy as np

from context import Structure2D  # noqa: F401  (adds src/ to sys.path)

from beam_params_util import loose
from xdfem2d import design_beam_bars, design_concrete_sections
from xdfem2d import detailing as D


def _design(**kw):
    """Two-span continuous beam on three supports (stations from diagrams)."""
    s = Structure2D()
    for i, x in enumerate((0.0, 5.0, 10.0), 1):
        s.add_node(f"n{i}", x, 0.0)
    s.add_material("C", elastic_modulus=33e6, unit_weight=25.0,
                   material_type="Concrete", design={"fck": 30, "fyk": 500})
    s.add_section("CONC", "C", b=0.3, h=0.5)
    s.add_bar_element("E1", "n1", "n2", "CONC")
    s.add_bar_element("E2", "n2", "n3", "CONC")
    s.add_support("PIN", ux=True, uy=True)
    for n in ("n1", "n2", "n3"):
        s.assign_support(n, "PIN")
    x = np.linspace(0.0, 5.0, 11)
    z = np.zeros_like(x)
    res = {"element_forces": {}, "combinations": {"ULS1": {"element_forces": {
        "E1": {"i": [0, 60, 0], "j": [0, -60, 0]},
        "E2": {"i": [0, 30, -90], "j": [0, -30, 0]}}}},
        "combo_distribution": {"ULS1": {
            # M parabolas and the shear they imply (V = dM/dx, x2 so the
            # demand varies along the beam and above the minimum stirrups)
            "E1": {"x": x, "N": z,
                   "V": 2.0 * np.gradient(
                       150.0 * (1 - ((x - 2.5) / 2.5) ** 2), x),
                   "M": 150.0 * (1 - ((x - 2.5) / 2.5) ** 2)},
            "E2": {"x": x, "N": z,
                   "V": 2.0 * np.gradient(-90.0 * (1 - x / 5.0) ** 2, x),
                   "M": -90.0 * (1 - x / 5.0) ** 2}}}}
    rows_in = design_concrete_sections(s, res)
    inputs = []
    rows = design_beam_bars(s, rows_in, res, inputs_out=inputs, **loose(kw))
    return s, rows, inputs


def _proposal(**kw):
    s, rows, inputs = _design(**kw)
    (key, detail), = D.details_from_rows(rows).items()
    D.stamp(detail, inputs)
    return detail, inputs[0]


def _zone(x0, x1, bottom, top, origin="auto"):
    return D.ZoneDetail(x0, x1, origin, list(bottom), list(top))


class TestInputsAndProposal(unittest.TestCase):
    def test_design_hands_over_the_inputs(self):
        _, rows, inputs = _design()
        self.assertEqual(len(inputs), 1)
        inp = inputs[0]
        self.assertEqual([s.id for s in inp.spans], ["T1", "T2"])
        self.assertAlmostEqual(inp.length(), 10.0)
        xs = [t[0] for t in inp.stations]
        self.assertEqual(xs, sorted(xs))
        self.assertAlmostEqual(inp.section["b"], 300.0)
        self.assertEqual(inp.params_obj.symmetry, "rule")
        self.assertEqual(inp.name, rows[0]["beam"])

    def test_inputs_round_trip_and_signature(self):
        inp = _design()[2][0]
        back = D.SegmentInputs.from_dict(json.loads(json.dumps(inp.to_dict())))
        self.assertEqual(back.signature(), inp.signature())
        self.assertEqual(back.params_obj, inp.params_obj)
        other = copy.deepcopy(inp)
        other.stations[3] = (other.stations[3][0], other.stations[3][1] + 50,
                             other.stations[3][2])
        self.assertNotEqual(other.signature(), inp.signature())

    def test_bad_inputs_are_rejected(self):
        d = _design()[2][0].to_dict()
        d["section"].pop("fck")
        with self.assertRaises(D.DetailError):
            D.SegmentInputs.from_dict(d)

    def test_required_steel_matches_the_design(self):
        _, rows, inputs = _design()
        inp = inputs[0]
        (detail,) = D.details_from_rows(rows).values()
        am = inp.as_min
        for sp in detail.segments[0].spans:
            reqs = D.required_areas(inp, sp.id, sp.zones)
            raw = [r for r in rows if r["span"] == sp.id]
            raw.sort(key=lambda r: r["x0"])
            for (rb, rt), r in zip(reqs, raw):
                self.assertAlmostEqual(rb, max(r["As_bot_req"] * 1e6, am),
                                       places=4)
                self.assertGreaterEqual(rt, r["As_top_req"] * 1e6 - 1e-6)

    def test_proposal_structure_and_areas(self):
        _, rows, _ = _design()
        (detail,) = D.details_from_rows(rows).values()
        (seg,) = detail.segments
        self.assertEqual([s.id for s in seg.spans], ["T1", "T2"])
        self.assertEqual(D.validate_structure(seg), [])
        for sp in seg.spans:
            for z, r in zip(sp.zones, sorted(
                    (r for r in rows if r["span"] == sp.id),
                    key=lambda r: r["x0"])):
                self.assertEqual(z.origin, "auto")
                self.assertAlmostEqual(z.area("bottom") * 1e-6,
                                       r["As_bot_prov"], places=9)
                self.assertAlmostEqual(z.area("top") * 1e-6,
                                       r["As_top_prov"], places=9)

    def test_the_proposal_passes_its_own_checks(self):
        for kw in ({}, {"zones": (0.25, 0.5, 0.25)},
                   {"symmetry": "even"}, {"n_through": 3}):
            detail, inp = _proposal(**kw)
            rep = D.check_segment(detail.segments[0], inp)
            self.assertTrue(rep.ok, (kw, [i.message for i in rep.errors]))


class TestChecks(unittest.TestCase):
    def setUp(self):
        self.detail, self.inp = _proposal(zones=(0.25, 0.5, 0.25))
        self.seg = self.detail.segments[0]

    def check(self, mutate):
        seg = copy.deepcopy(self.seg)
        mutate(seg)
        return D.check_segment(seg, self.inp)

    def test_too_little_steel(self):
        rep = self.check(lambda s: setattr(s.spans[0].zones[1], "bottom",
                                           [(2, 10.0)]))
        self.assertIn("AS_LOW", rep.codes())
        self.assertFalse(rep.ok)

    def test_layer_rules(self):
        def one_bar(s):  s.spans[0].zones[1].bottom = [(1, 25.0)]
        self.assertIn("LAYER_COUNT", self.check(one_bar).codes())

        def too_many(s): s.spans[0].zones[1].bottom = [(20, 12.0)]
        self.assertIn("SPACING", self.check(too_many).codes())

        def order(s):    s.spans[0].zones[1].bottom = [(4, 16.0), (6, 16.0)]
        self.assertIn("LAYER_ORDER", self.check(order).codes())

        def layers(s):
            s.spans[0].zones[1].bottom = [(4, 16.0)] * 3
        self.assertIn("MAX_LAYERS", self.check(layers).codes())

        def empty(s):    s.spans[0].zones[1].top = []
        self.assertIn("NO_BARS", self.check(empty).codes())

    def test_unlisted_diameter_is_only_a_warning(self):
        def odd(s):
            s.spans[0].zones[1].bottom = [(6, 14.0)]       # 14 not listed
        rep = self.check(odd)
        self.assertIn("DIAMETER_UNLISTED", rep.codes())
        self.assertNotIn("DIAMETER_UNLISTED", {i.code for i in rep.errors})

    def test_as_max(self):
        def huge(s): s.spans[0].zones[1].bottom = [(8, 32.0), (8, 32.0)]
        self.assertIn("AS_MAX", self.check(huge).codes())

    def test_asymmetric_curtailment(self):
        def odd_drop(s):                    # 4 -> 3 is not symmetric
            s.spans[0].zones[0].bottom = [(4, 25.0)]
            s.spans[0].zones[1].bottom = [(3, 25.0)]
        rep = self.check(odd_drop)
        self.assertIn("CURTAILMENT", rep.codes())

    def test_through_bars(self):
        def other_d(s):
            s.spans[0].zones[1].bottom = [(6, 32.0)]       # outer layer changes Ø
        self.assertIn("THROUGH", self.check(other_d).codes())

    def test_topology_and_zones(self):
        def shorter(s): s.spans[0].length = 4.0
        self.assertIn("TOPOLOGY", self.check(shorter).codes())

        def gap(s):    s.spans[0].zones[1].x0 += 0.3
        self.assertIn("ZONES", self.check(gap).codes())


class TestEditor(unittest.TestCase):
    def setUp(self):
        self.detail, self.inp = _proposal(zones=(0.25, 0.5, 0.25))
        self.seg = self.detail.segments[0]
        self.ed = D.DetailEditor(self.seg, self.inp, self.seg)

    def test_set_zone_changes_both_faces_in_one_undo_step(self):
        self.ed.set_zone("T1", 1, bottom=[(6, 20.0)], top=[(4, 16.0)])
        z = self.ed.segment.span("T1").zones[1]
        self.assertEqual((z.bottom, z.top, z.origin),
                         ([(6, 20.0)], [(4, 16.0)], "manual"))
        self.assertTrue(self.ed.undo())
        self.assertFalse(self.ed.can_undo)                 # ONE step
        with self.assertRaises(D.DetailError):
            self.ed.set_zone("T1", 1)                      # nothing to set

    def test_set_layers_marks_manual_and_undoes(self):
        before = self.ed.segment.to_dict()
        self.ed.set_layers("T1", 1, "bottom", [(6, 20.0)])
        z = self.ed.segment.span("T1").zones[1]
        self.assertEqual((z.bottom, z.origin), ([(6, 20.0)], "manual"))
        self.assertTrue(self.ed.can_undo and not self.ed.can_redo)
        self.assertTrue(self.ed.undo())
        self.assertEqual(self.ed.segment.to_dict(), before)
        self.assertTrue(self.ed.redo())
        self.assertEqual(self.ed.segment.span("T1").zones[1].bottom, [(6, 20.0)])

    def test_the_original_is_not_touched(self):
        before = copy.deepcopy(self.seg.to_dict())
        self.ed.set_layers("T1", 0, "top", [(2, 10.0)])
        self.assertEqual(self.seg.to_dict(), before)

    def test_a_new_edit_clears_redo(self):
        self.ed.set_layers("T1", 0, "top", [(2, 10.0)])
        self.ed.undo()
        self.assertTrue(self.ed.can_redo)
        self.ed.set_layers("T1", 0, "top", [(4, 10.0)])
        self.assertFalse(self.ed.can_redo)

    def test_invalid_edits_raise_and_leave_everything_as_it_was(self):
        before = self.ed.segment.to_dict()
        for bad in (lambda: self.ed.set_layers("T1", 9, "top", [(2, 10.0)]),
                    lambda: self.ed.set_layers("T9", 0, "top", [(2, 10.0)]),
                    lambda: self.ed.set_layers("T1", 0, "side", [(2, 10.0)]),
                    lambda: self.ed.set_layers("T1", 0, "top", [(0, 10.0)]),
                    lambda: self.ed.split_zone("T1", 0, 99.0),
                    lambda: self.ed.merge_zones("T1", 2),
                    lambda: self.ed.move_cut("T1", 2, 1.0),
                    lambda: self.ed.move_cut("T1", 0, 4.9)):
            with self.assertRaises(D.DetailError):
                bad()
        self.assertEqual(self.ed.segment.to_dict(), before)
        self.assertFalse(self.ed.can_undo)

    def test_split_merge_and_move(self):
        n0 = len(self.ed.segment.span("T1").zones)
        self.ed.split_zone("T1", 1, 2.5)
        zs = self.ed.segment.span("T1").zones
        self.assertEqual(len(zs), n0 + 1)
        self.assertEqual((zs[1].x1, zs[2].x0), (2.5, 2.5))
        self.assertTrue(zs[1].origin == zs[2].origin == "manual")
        self.assertEqual(D.validate_structure(self.ed.segment), [])
        self.ed.move_cut("T1", 1, 2.0)
        zs = self.ed.segment.span("T1").zones
        self.assertEqual((zs[1].x1, zs[2].x0), (2.0, 2.0))
        self.ed.merge_zones("T1", 1)
        self.assertEqual(len(self.ed.segment.span("T1").zones), n0)
        self.assertEqual(D.validate_structure(self.ed.segment), [])

    def test_merge_keeps_the_stronger_layers(self):
        self.ed.set_layers("T1", 0, "bottom", [(2, 10.0)])
        self.ed.set_layers("T1", 1, "bottom", [(6, 25.0)])
        self.ed.merge_zones("T1", 0)
        self.assertEqual(self.ed.segment.span("T1").zones[0].bottom, [(6, 25.0)])

    def test_reset_zone_and_all(self):
        orig = copy.deepcopy(self.seg.to_dict())
        self.ed.set_layers("T1", 1, "bottom", [(6, 25.0)])
        self.ed.reset_zone("T1", 1)
        z = self.ed.segment.span("T1").zones[1]
        self.assertEqual(z.origin, "auto")                 # same limits
        self.assertEqual(self.ed.segment.to_dict(), orig)
        self.ed.split_zone("T1", 1, 2.5)
        self.ed.set_layers("T1", 1, "top", [(8, 25.0)])
        self.ed.reset_all()
        self.assertEqual(self.ed.segment.to_dict(), orig)
        self.assertTrue(self.ed.undo())                    # reset_all undoes

    def test_reset_needs_a_proposal(self):
        ed = D.DetailEditor(self.seg)
        with self.assertRaises(D.DetailError):
            ed.reset_all()

    def test_checks_follow_the_edits(self):
        self.assertTrue(self.ed.checks().ok)
        self.ed.set_layers("T1", 1, "bottom", [(2, 10.0)])
        self.assertIn("AS_LOW", self.ed.checks().codes())
        self.ed.undo()
        self.assertTrue(self.ed.checks().ok)
        self.assertEqual(D.DetailEditor(self.seg).checks().issues, [])  # no inputs


class TestSerialisation(unittest.TestCase):
    def test_round_trip(self):
        detail, _ = _proposal()
        doc = D.DetailDocument({detail.tag: detail})
        back = D.DetailDocument.loads(doc.dumps())
        self.assertEqual(back.to_dict(), doc.to_dict())
        self.assertIsInstance(back.beams[detail.tag].segments[0].spans[0]
                              .zones[0].bottom[0], tuple)

    def test_absent_document_is_empty(self):
        self.assertEqual(D.DetailDocument.from_dict(None).beams, {})
        self.assertEqual(D.DetailDocument.from_dict({}).beams, {})

    def test_newer_version_and_garbage_are_rejected(self):
        for bad in ({"version": 99, "beams": {}}, {"version": "x"}, [1],
                    {"beams": {"V1": {"segments": [{"name": "V1",
                     "spans": [{"id": "T1", "length": 5.0, "zones": [
                         {"x0": 0, "x1": 5, "origin": "weird"}]}]}]}}},
                    {"beams": {"V1": {"segments": [{"name": "V1",
                     "spans": [{"id": "T1", "length": 5.0, "zones": [
                         {"x0": 0, "x1": 4}]}]}]}}},           # does not tile
                    {"beams": {"V1": {"segments": [{"name": "V1",
                     "spans": [{"id": "T1", "length": 5.0, "zones": [
                         {"x0": 0, "x1": 5, "bottom": [[0, 12]]}]}]}]}}}):
            with self.assertRaises(D.DetailError, msg=bad):
                D.DetailDocument.from_dict(bad)
        with self.assertRaises(D.DetailError):
            D.DetailDocument.loads("{nope")

    def test_package_round_trip_and_version(self):
        detail, inp = _proposal()
        pk = D.BeamPackage([inp], D.DetailDocument({detail.tag: detail}),
                           D.DetailDocument({detail.tag: detail}))
        back = D.BeamPackage.loads(pk.dumps())
        self.assertEqual(back.to_dict(), pk.to_dict())
        # the loaded package is enough to check a detail, no model needed
        self.assertTrue(D.check_segment(back.detail.beams[detail.tag]
                                        .segments[0], back.inputs[0]).ok)
        self.assertIn(detail.tag, back.proposal.beams)
        with self.assertRaises(D.DetailError):
            D.BeamPackage.from_dict({"version": 99})
        # a package written before proposals existed still loads
        old = pk.to_dict()
        old.pop("proposal")
        self.assertEqual(D.BeamPackage.from_dict(old).proposal.beams, {})

    def test_rekey(self):
        detail, _ = _proposal()
        doc = D.DetailDocument({detail.tag: detail})
        D.rekey(doc, detail.tag, "V7")
        self.assertEqual(list(doc.beams), ["V7"])
        self.assertEqual(doc.beams["V7"].tag, "V7")
        doc.beams["V8"] = D.BeamDetail("V8")
        with self.assertRaises(D.DetailError):
            D.rekey(doc, "V7", "V8")
        with self.assertRaises(D.DetailError):
            D.rekey(doc, "nope", "V9")


class TestReconcile(unittest.TestCase):
    def setUp(self):
        self.detail, self.inp = _proposal(zones=(0.25, 0.5, 0.25))
        self.seg = self.detail.segments[0]

    def test_no_existing_detail_gives_the_proposal(self):
        out, rep = D.reconcile_segment(None, self.seg, self.inp)
        self.assertEqual(rep.status, "new")
        self.assertEqual(out.to_dict(), self.seg.to_dict())
        self.assertTrue(all(i.severity != "error" for i in rep.issues))

    def test_same_proposal_is_unchanged(self):
        out, rep = D.reconcile_segment(copy.deepcopy(self.seg), self.seg,
                                       self.inp)
        self.assertEqual(rep.status, "unchanged")
        self.assertEqual(rep.kept_manual, 0)

    def test_manual_zones_are_kept_and_auto_zones_follow(self):
        existing = copy.deepcopy(self.seg)
        ed = D.DetailEditor(existing, self.inp, self.seg)
        ed.set_layers("T1", 1, "bottom", [(8, 25.0)])           # manual
        new_prop = copy.deepcopy(self.seg)
        new_prop.spans[0].zones[1].bottom = [(2, 10.0)]          # ignored: manual
        new_prop.spans[0].zones[0].top = [(6, 20.0)]             # auto: follows
        out, rep = D.reconcile_segment(ed.segment, new_prop, self.inp)
        self.assertEqual(rep.status, "merged")
        zs = out.span("T1").zones
        self.assertEqual(zs[1].bottom, [(8, 25.0)])
        self.assertEqual(zs[1].origin, "manual")
        self.assertEqual(zs[0].top, [(6, 20.0)])
        self.assertEqual(rep.kept_manual, 1)
        self.assertEqual(D.validate_structure(out), [])

    def test_a_manual_zone_that_no_longer_complies_is_flagged_not_erased(self):
        existing = copy.deepcopy(self.seg)
        ed = D.DetailEditor(existing, self.inp, self.seg)
        ed.set_layers("T1", 1, "bottom", [(4, 16.0)])
        # the calculation now asks for much more steel everywhere
        hungry = copy.deepcopy(self.inp)
        hungry.stations = [(x, b * 4, t) for x, b, t in hungry.stations]
        out, rep = D.reconcile_segment(ed.segment, self.seg, hungry)
        self.assertEqual(out.span("T1").zones[1].bottom, [(4, 16.0)])
        flagged = [i for i in rep.issues
                   if i.code == "AS_LOW" and (i.span, i.zone) == ("T1", 1)]
        self.assertTrue(flagged)

    def test_changed_spans_make_the_detail_stale(self):
        existing = copy.deepcopy(self.seg)
        longer = copy.deepcopy(self.seg)
        longer.spans[0].length += 1.0
        longer.spans[0].zones[-1].x1 += 1.0
        out, rep = D.reconcile_segment(existing, longer, None)
        self.assertEqual(rep.status, "stale")
        self.assertEqual(out.to_dict(), existing.to_dict())     # untouched

    def test_document_level(self):
        detail, inp = self.detail, self.inp
        old = D.DetailDocument({"V9": D.BeamDetail("V9", [self.seg])})
        doc, reports = D.reconcile_document(old, {detail.tag: detail}, [inp])
        self.assertEqual(reports[detail.tag][self.seg.name].status, "new")
        self.assertEqual(reports["V9"][self.seg.name].status, "orphan")
        self.assertIn("V9", doc.beams)                         # never dropped
        self.assertIn(detail.tag, doc.beams)
        self.assertTrue(doc.beams[detail.tag].based_on)

    def test_outdated_tracks_the_inputs(self):
        detail, inp = self.detail, self.inp
        D.stamp(detail, [inp])
        self.assertFalse(D.is_outdated(detail, [inp]))
        other = copy.deepcopy(inp)
        other.stations[0] = (other.stations[0][0], 9999.0, 0.0)
        self.assertTrue(D.is_outdated(detail, [other]))
        self.assertFalse(D.is_outdated(D.BeamDetail("X"), [inp]))  # never stamped


# ── persistence in the model and in the .x2d ─────────────────────────────

import io
import os
import tempfile
import zipfile

from xdfem2d.file_io import load_x2d, save_x2d
from xdfem2d.structure_io import (_from_dict, _to_dict, load_structure_json,
                                  save_structure_json)
from xdfem2d.structure_io_checked import load_structure_json_checked
from xdfem2d.script_export import to_python


def _manual_model():
    """The design model with one manual edit stored on the tagged beam."""
    s, rows, inputs = _design()
    s.assign_beam(["E1", "E2"], tag="V1", name="Floor")
    s, rows, inputs = _redo(s)
    detail = D.details_from_rows(rows)["V1"]
    ed = D.DetailEditor(detail.segments[0], inputs[0], detail.segments[0])
    ed.set_layers("T1", 0, "bottom", [(6, 25.0)])
    detail.segments[0] = ed.segment
    D.stamp(detail, inputs)
    s.beam_detail = D.DetailDocument({"V1": detail})
    return s, rows, inputs


def _redo(s):
    """Re-run the design on an already built model *s* (same as _design)."""
    x = np.linspace(0.0, 5.0, 11)
    z = np.zeros_like(x)
    res = {"element_forces": {}, "combinations": {"ULS1": {"element_forces": {
        "E1": {"i": [0, 60, 0], "j": [0, -60, 0]},
        "E2": {"i": [0, 30, -90], "j": [0, -30, 0]}}}},
        "combo_distribution": {"ULS1": {
            "E1": {"x": x, "N": z, "V": z,
                   "M": 150.0 * (1 - ((x - 2.5) / 2.5) ** 2)},
            "E2": {"x": x, "N": z, "V": z,
                   "M": -90.0 * (1 - x / 5.0) ** 2}}}}
    inputs = []
    rows = design_beam_bars(s, design_concrete_sections(s, res), res,
                            inputs_out=inputs)
    return s, rows, inputs


class TestModelPersistence(unittest.TestCase):
    def test_empty_detail_leaves_the_model_dict_unchanged(self):
        s, _, _ = _design()
        self.assertNotIn("beam_detail", _to_dict(s))

    def test_survives_undo_snapshots(self):
        s, _, _ = _manual_model()
        back = _from_dict(_to_dict(s))                 # what undo/redo does
        self.assertEqual(back.beam_detail.to_dict(), s.beam_detail.to_dict())
        self.assertTrue(back.beam_detail.has_manual("V1"))

    def test_plain_json_and_checked_loader(self):
        s, _, _ = _manual_model()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.json")
            save_structure_json(s, p)
            for back in (load_structure_json(p),
                         load_structure_json_checked(p)[0]):
                self.assertEqual(back.beam_detail.to_dict(),
                                 s.beam_detail.to_dict())

    def test_x2d_has_its_own_entry(self):
        s, _, _ = _manual_model()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.x2d")
            save_x2d(s, None, p)
            with zipfile.ZipFile(p) as zf:
                self.assertIn("beam_detail.json", zf.namelist())
                self.assertNotIn("beam_detail",
                                 json.loads(zf.read("structure.json")))
            back, res, _view = load_x2d(p)
            self.assertEqual(back.beam_detail.to_dict(),
                             s.beam_detail.to_dict())
            self.assertEqual(back.beam_detail_load_error, "")
            self.assertIsNone(res)                    # no results needed

    def test_x2d_without_a_detail_has_no_entry_and_old_files_load(self):
        s, _, _ = _design()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.x2d")
            save_x2d(s, None, p)
            with zipfile.ZipFile(p) as zf:
                self.assertNotIn("beam_detail.json", zf.namelist())
            back, _r, _v = load_x2d(p)
            self.assertEqual(back.beam_detail.beams, {})

    def test_a_corrupt_detail_never_blocks_opening(self):
        s, _, _ = _manual_model()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.x2d")
            save_x2d(s, None, p)
            bad = os.path.join(d, "bad.x2d")
            with zipfile.ZipFile(p) as zin, zipfile.ZipFile(bad, "w") as zout:
                for n in zin.namelist():
                    data = zin.read(n)
                    if n == "beam_detail.json":
                        data = json.dumps({"version": 99}).encode()
                    zout.writestr(n, data)
            back, _r, _v = load_x2d(bad)
            self.assertEqual(back.beam_detail.beams, {})
            self.assertIn("newer", back.beam_detail_load_error)
            self.assertIn("V1", back.beams)                   # model intact

            garbage = os.path.join(d, "garbage.x2d")
            with zipfile.ZipFile(p) as zin, zipfile.ZipFile(garbage, "w") as zout:
                for n in zin.namelist():
                    zout.writestr(n, b"{nope" if n == "beam_detail.json"
                                  else zin.read(n))
            back, _r, _v = load_x2d(garbage)
            self.assertTrue(back.beam_detail_load_error)

    def test_checked_loader_reports_a_bad_detail(self):
        s, _, _ = _manual_model()
        data = _to_dict(s)
        data["beam_detail"] = {"version": 99}
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.json")
            json.dump(data, open(p, "w"))
            back, report = load_structure_json_checked(p)
            self.assertEqual(back.beam_detail.beams, {})
            bad = [i for i in report.issues if i.section == "beam_detail"]
            self.assertEqual(len(bad), 1)
            self.assertEqual(bad[0].severity, "error")
            self.assertIn("newer", bad[0].message)

    def test_script_export_round_trips_it(self):
        s, _, _ = _manual_model()
        ns: dict = {}
        exec(compile(to_python(s), "<export>", "exec"), ns)
        rebuild = next(v for k, v in ns.items()
                       if callable(v) and k.startswith("build"))
        self.assertEqual(rebuild().beam_detail.to_dict(),
                         s.beam_detail.to_dict())


class TestRefreshAfterDesign(unittest.TestCase):
    def test_nothing_stored_nothing_changes(self):
        s, rows, inputs = _design()
        summary, changed = D.refresh_model_detail(s, rows, inputs)
        self.assertFalse(changed)
        self.assertEqual(summary["beams"], 0)
        self.assertEqual(D.summary_text(summary), "")
        self.assertEqual(s.beam_detail.beams, {})     # no auto copies stored

    def test_manual_decision_is_kept_across_a_new_design(self):
        s, rows, inputs = _manual_model()
        before = s.beam_detail.beams["V1"].segments[0].span("T1").zones[0].bottom
        s2, rows2, inputs2 = _redo(s)                  # same model, new run
        summary, changed = D.refresh_model_detail(s2, rows2, inputs2)
        zone = s2.beam_detail.beams["V1"].segments[0].span("T1").zones[0]
        self.assertEqual((zone.bottom, zone.origin), (before, "manual"))
        self.assertEqual(summary["beams"], 1)
        self.assertEqual(summary["manual_zones"], 1)
        self.assertIn("manual detail kept", D.summary_text(summary))

    def test_a_manual_zone_that_stops_complying_is_reported(self):
        s, rows, inputs = _manual_model()
        zone = s.beam_detail.beams["V1"].segments[0].span("T1").zones[0]
        zone.bottom = [(2, 10.0)]                      # far too little steel
        summary, _ = D.refresh_model_detail(s, rows, inputs)
        self.assertGreaterEqual(summary["errors"], 1)
        self.assertIn("check error", D.summary_text(summary))
        self.assertEqual(zone.bottom, [(2, 10.0)])     # never erased

    def test_prune_auto(self):
        s, rows, inputs = _design()
        det = D.details_from_rows(rows)
        doc = D.DetailDocument(dict(det))
        self.assertEqual(doc.prune_auto(), list(det))
        self.assertEqual(doc.beams, {})
        m, _, _ = _manual_model()
        self.assertEqual(m.beam_detail.prune_auto(), [])


class TestPromotion(unittest.TestCase):
    def test_automatic_beam_gets_a_tag(self):
        from xdfem2d.rc_design import promote_auto_beam
        s, rows, _ = _design()
        name = rows[0]["beam"]                          # "B1": no tag yet
        self.assertIsNone(rows[0]["beam_tag"])
        tag = promote_auto_beam(s, rows, name)
        self.assertEqual(tag, "V1")
        self.assertEqual({e.beam for e in s.bar_elements}, {"V1"})
        # promoted: the next design groups by that tag and the key is stable
        _s, rows2, _ = _redo(s)
        self.assertEqual({r["beam_tag"] for r in rows2}, {"V1"})
        # an already tagged beam returns its own tag
        self.assertEqual(promote_auto_beam(s, rows2, rows2[0]["beam"]), "V1")
        with self.assertRaises(KeyError):
            promote_auto_beam(s, rows2, "nope")

    def test_object_spanning_two_beams_cannot_be_promoted(self):
        from xdfem2d.rc_design import assign_detected_beam, compiled_beam_index
        s = Structure2D()
        s.add_material("C", elastic_modulus=33e6, unit_weight=25.0,
                       material_type="Concrete", design={"fck": 30, "fyk": 500})
        s.add_section("CONC", "C", b=0.3, h=0.5)
        s.add_geo_multisegment("OBJ", [[0, 3], [5, 3], [10, 6]],
                               section_name="CONC", divisions=1)
        index = compiled_beam_index(s)
        mesh = index[0]
        ids = [b.id for b in mesh.bar_elements]
        # a single bar of a two-bar object: the object would be half-tagged
        self.assertIsNone(assign_detected_beam(s, [ids[0]], index))
        self.assertIsNotNone(assign_detected_beam(s, ids, index))
        self.assertEqual(s.geometry_objects["OBJ"].beam, "V1")


if __name__ == "__main__":   # keep the original entry point last
    unittest.main()


class TestReconcileByPosition(unittest.TestCase):
    def test_a_renamed_beam_keeps_its_detail(self):
        detail, inp = _proposal(zones=(0.25, 0.5, 0.25))
        ed = D.DetailEditor(detail.segments[0], inp, detail.segments[0])
        ed.set_layers("T1", 1, "bottom", [(8, 25.0)])
        stored = D.DetailDocument({detail.tag: D.BeamDetail(
            detail.tag, [ed.segment])})
        # the beam is renamed / promoted: the proposal's segment name changes
        renamed = copy.deepcopy(detail)
        renamed.segments[0].name = "Floor 1"
        inp2 = copy.deepcopy(inp)
        inp2.name = "Floor 1"
        doc, reports = D.reconcile_document(stored, {detail.tag: renamed}, [inp2])
        seg = doc.beams[detail.tag].segments[0]
        self.assertEqual(seg.name, "Floor 1")
        self.assertEqual(seg.span("T1").zones[1].bottom, [(8, 25.0)])
        self.assertEqual(reports[detail.tag]["Floor 1"].kept_manual, 1)

    def test_a_vanished_segment_is_kept_as_orphan(self):
        detail, inp = _proposal()
        two = D.BeamDetail(detail.tag, [copy.deepcopy(detail.segments[0]),
                                        copy.deepcopy(detail.segments[0])])
        two.segments[1].name = "extra"
        stored = D.DetailDocument({detail.tag: two})
        doc, reports = D.reconcile_document(stored, {detail.tag: detail}, [inp])
        self.assertEqual(len(doc.beams[detail.tag].segments), 2)
        self.assertEqual(reports[detail.tag]["extra"].status, "orphan")


class TestChartData(unittest.TestCase):
    def setUp(self):
        self.detail, self.inp = _proposal(zones=(0.25, 0.5, 0.25))
        self.seg = self.detail.segments[0]

    def test_envelope_is_the_shifted_station_maximum(self):
        for face, col in (("bottom", 1), ("top", 2)):
            env = D.envelope_profile(self.inp, face)
            self.assertEqual([x for x, _ in env],
                             sorted(t[0] for t in self.inp.stations))
            for x, v in env:                    # never below the raw station
                raw = max(t[col] for t in self.inp.stations
                          if abs(t[0] - x) < 1e-9)
                self.assertGreaterEqual(v + 1e-9, raw)
        shifted = copy.deepcopy(self.inp)
        shifted.a_l = 1.0
        wide = D.envelope_profile(shifted, "top")
        none = D.envelope_profile(self.inp, "top")
        self.assertGreaterEqual(sum(v for _, v in wide),
                                sum(v for _, v in none))
        with self.assertRaises(D.DetailError):
            D.envelope_profile(self.inp, "side")

    def test_zone_profile_matches_the_checks(self):
        prof = D.zone_profile(self.seg, self.inp)
        self.assertEqual(len(prof), sum(len(s.zones) for s in self.seg.spans))
        off = self.inp.offsets()
        for p in prof:
            sp = self.seg.span(p["span"])
            z = sp.zones[p["k"]]
            rb, rt = D.required_areas(self.inp, p["span"], sp.zones)[p["k"]]
            self.assertEqual((p["req_bottom"], p["req_top"]), (rb, rt))
            self.assertEqual(p["prov_bottom"], z.area("bottom"))
            self.assertAlmostEqual(p["x0"], off[p["span"]] + z.x0)
            self.assertGreaterEqual(p["prov_bottom"] + 1e-6, p["req_bottom"])
        for a, b in zip(prof, prof[1:]):           # tiles the segment
            self.assertAlmostEqual(a["x1"], b["x0"])

    def test_unknown_spans_are_skipped(self):
        other = copy.deepcopy(self.seg)
        other.spans[0].id = "ZZ"
        prof = D.zone_profile(other, self.inp)
        self.assertTrue(all(p["span"] != "ZZ" for p in prof))


# ── stirrups ─────────────────────────────────────────────────────────────

def _proposal_st(**kw):
    s, rows, inputs = _design(**kw)
    dets = D.details_from_rows(rows, inputs)
    (key, detail), = dets.items()
    D.stamp(detail, inputs)
    return detail, inputs[0]


class TestStirrups(unittest.TestCase):
    def setUp(self):
        self.detail, self.inp = _proposal_st(zones=(0.25, 0.5, 0.25))
        self.seg = self.detail.segments[0]

    # -- inputs ---------------------------------------------------------
    def test_inputs_carry_the_shear_stations(self):
        self.assertTrue(self.inp.shear)
        xs = [x for x, _ in self.inp.shear]
        self.assertEqual(xs, sorted(xs))
        self.assertGreater(max(v for _, v in self.inp.shear), 0.0)
        self.assertEqual(self.inp.section["alpha_s"], 90.0)
        self.assertTrue(self.inp.section["shear_min"])
        back = D.SegmentInputs.from_dict(json.loads(json.dumps(self.inp.to_dict())))
        self.assertEqual(back.shear, self.inp.shear)
        self.assertEqual(back.signature(), self.inp.signature())

    def test_older_inputs_without_shear_still_load_and_give_no_stirrups(self):
        d = self.inp.to_dict()
        d.pop("shear")
        old = D.SegmentInputs.from_dict(d)
        self.assertEqual(old.shear, [])
        self.assertEqual(D.propose_stirrups(old), {})
        self.assertEqual(D.check_segment(
            D.BeamDetail("x", [copy.deepcopy(self.seg)]).segments[0], old
        ).errors == [], True)

    def test_signature_follows_the_shear_demand(self):
        other = copy.deepcopy(self.inp)
        other.shear[0] = (other.shear[0][0], other.shear[0][1] + 500.0)
        self.assertNotEqual(other.signature(), self.inp.signature())

    # -- EC2 numbers ----------------------------------------------------
    def test_minimum_and_maximum_spacing(self):
        sec = self.inp.section
        want = 0.08 * (sec["fck"] ** 0.5) / sec["fyk"] * sec["b"] * 1000.0
        self.assertAlmostEqual(D.min_shear_floor(self.inp), want, places=6)
        self.assertAlmostEqual(D.stirrup_s_max(self.inp),
                               0.75 * self.inp.beam().d_eff, places=6)
        off = copy.deepcopy(self.inp)
        off.section["shear_min"] = False
        self.assertEqual(D.min_shear_floor(off), 0.0)
        inclined = copy.deepcopy(self.inp)
        inclined.section["alpha_s"] = 45.0                 # cot = 1 -> 2 * 0.75 d
        self.assertAlmostEqual(D.stirrup_s_max(inclined),
                               1.5 * self.inp.beam().d_eff, places=6)

    # -- the proposal ---------------------------------------------------
    def test_proposal_covers_every_span_without_gaps(self):
        for sp in self.seg.spans:
            self.assertTrue(sp.stirrups)
            self.assertAlmostEqual(sp.stirrups[0].x0, 0.0)
            self.assertAlmostEqual(sp.stirrups[-1].x1, sp.length)
            for a, b in zip(sp.stirrups, sp.stirrups[1:]):
                self.assertAlmostEqual(a.x1, b.x0)
            self.assertTrue(all(z.origin == "auto" for z in sp.stirrups))

    def test_proposal_respects_requirement_spacing_and_one_diameter(self):
        p = self.inp.params_obj
        smax = D.stirrup_s_max(self.inp)
        diams = set()
        for sp in self.seg.spans:
            reqs = D.stirrup_requirements(self.inp, sp.id, sp.stirrups)
            for z, req in zip(sp.stirrups, reqs):
                self.assertGreaterEqual(z.asw_s() + 1e-6, req)
                self.assertLessEqual(z.spacing, smax + 1e-6)
                self.assertIn(z.spacing, p.stirrup_spacings)
                self.assertEqual(z.legs, p.stirrup_legs)
                diams.add(z.diameter)
        self.assertEqual(len(diams), 1)                    # one Ø per segment
        self.assertLessEqual(diams.pop(), self.inp.section["stirrup"])

    def test_the_largest_spacing_that_works_is_used(self):
        for sp in self.seg.spans:
            reqs = D.stirrup_requirements(self.inp, sp.id, sp.stirrups)
            for z, req in zip(sp.stirrups, reqs):
                bigger = [s for s in self.inp.params_obj.stirrup_spacings
                          if s > z.spacing
                          and s <= D.stirrup_s_max(self.inp) + 1e-9]
                for s in bigger:                           # none of them covers it
                    a = z.legs * D.bar_area(z.diameter) / (s / 1000.0)
                    self.assertLess(a, req)

    def test_a_heavier_demand_needs_closer_or_bigger_stirrups(self):
        hungry = copy.deepcopy(self.inp)
        hungry.shear = [(x, v * 4.0 + 200.0) for x, v in hungry.shear]
        light = D.propose_stirrups(self.inp)
        heavy = D.propose_stirrups(hungry)
        area = lambda d: sum((z.x1 - z.x0) * z.asw_s()
                             for zs in d.values() for z in zs)
        self.assertGreater(area(heavy), area(light))

    def test_impossible_demand_gives_the_closest_effort_and_is_flagged(self):
        absurd = copy.deepcopy(self.inp)
        absurd.shear = [(x, 1e6) for x, _ in absurd.shear]
        prop = D.propose_stirrups(absurd)
        seg = copy.deepcopy(self.seg)
        for sp in seg.spans:
            sp.stirrups = prop[sp.id]
        self.assertIn("ASW_LOW", D.check_segment(seg, absurd).codes())

    def test_proposal_passes_its_own_checks(self):
        for kw in ({}, {"zones": (0.25, 0.5, 0.25)}, {"symmetry": "even"}):
            detail, inp = _proposal_st(**kw)
            rep = D.check_segment(detail.segments[0], inp)
            self.assertTrue(rep.ok, (kw, [i.message for i in rep.errors]))
            self.assertEqual([i for i in rep.issues
                              if i.layer == "stirrups"], [])

    def test_proposals_without_inputs_have_no_stirrups(self):
        _, rows, _ = _design()
        (d,) = D.details_from_rows(rows).values()
        self.assertTrue(all(not sp.stirrups for sp in d.segments[0].spans))

    # -- checks ---------------------------------------------------------
    def _check(self, mutate):
        seg = copy.deepcopy(self.seg)
        mutate(seg)
        return D.check_segment(seg, self.inp)

    def test_check_codes(self):
        # 2 legs Ø6 every 300 mm = 188 mm²/m, below even the minimum (263)
        def very_weak(s):
            z = s.spans[0].stirrups[0]
            z.diameter, z.spacing = 6.0, 300.0
        rep = self._check(very_weak)
        self.assertGreater(D.min_shear_floor(self.inp), 188.0)
        self.assertTrue(any(i.code == "ASW_LOW" and i.layer == "stirrups"
                            for i in rep.issues))

        def too_open(s): s.spans[0].stirrups[0].spacing = 600.0
        self.assertIn("S_MAX", self._check(too_open).codes())

        def one_leg(s):  s.spans[0].stirrups[0].legs = 1
        self.assertIn("STIRRUP_LEGS", self._check(one_leg).codes())

        def odd(s):
            z = s.spans[0].stirrups[0]
            z.diameter, z.spacing = 7.0, 133.0
        codes = self._check(odd).codes()
        self.assertIn("STIRRUP_DIAMETER", codes)
        self.assertIn("STIRRUP_SPACING", codes)

        def big(s):      s.spans[0].stirrups[0].diameter = 12.0
        rep = self._check(big)
        self.assertIn("STIRRUP_COVER", rep.codes())
        self.assertNotIn("STIRRUP_COVER", {i.code for i in rep.errors})

        def gap(s):      s.spans[0].stirrups[0].x1 -= 0.2
        zi = [i for i in self._check(gap).issues
              if i.code == "ZONES" and i.layer == "stirrups"]
        self.assertTrue(zi)

    def test_issues_say_which_layer_they_belong_to(self):
        def weak(s):
            s.spans[0].zones[1].bottom = [(2, 10.0)]
            z = s.spans[0].stirrups[0]
            z.legs, z.diameter, z.spacing = 2, 6.0, 300.0
        rep = self._check(weak)
        self.assertTrue(rep.for_zone("T1", 1, "bars"))
        self.assertTrue(all(i.layer == "bars" for i in rep.for_zone("T1", 1)))

    # -- editing --------------------------------------------------------
    def setup_editor(self):
        return D.DetailEditor(self.seg, self.inp, self.seg)

    def test_set_stirrups_and_undo(self):
        ed = self.setup_editor()
        before = ed.segment.to_dict()
        ed.set_stirrups("T1", 0, diameter=10, spacing=100)
        z = ed.segment.span("T1").stirrups[0]
        self.assertEqual((z.diameter, z.spacing, z.origin),
                         (10.0, 100.0, "manual"))
        self.assertEqual(z.legs, 2)                        # untouched
        ed.undo()
        self.assertEqual(ed.segment.to_dict(), before)
        for bad in (dict(), dict(legs=0), dict(diameter=-1), dict(spacing=0)):
            with self.assertRaises(D.DetailError):
                ed.set_stirrups("T1", 0, **bad)
        with self.assertRaises(D.DetailError):
            ed.set_stirrups("T1", 99, spacing=100)
        self.assertFalse(ed.can_undo)

    def test_split_merge_move_reset_stirrup_zones(self):
        ed = self.setup_editor()
        n0 = len(ed.segment.span("T1").stirrups)
        z0 = ed.segment.span("T1").stirrups[0]
        mid = 0.5 * (z0.x0 + z0.x1)
        ed.split_stirrup_zone("T1", 0, mid)
        zs = ed.segment.span("T1").stirrups
        self.assertEqual(len(zs), n0 + 1)
        self.assertTrue(zs[0].origin == zs[1].origin == "manual")
        self.assertEqual(D.validate_structure(ed.segment), [])
        ed.move_stirrup_cut("T1", 0, mid - 0.1)
        zs = ed.segment.span("T1").stirrups
        self.assertAlmostEqual(zs[0].x1, mid - 0.1)
        self.assertAlmostEqual(zs[0].x1, zs[1].x0)
        ed.set_stirrups("T1", 1, spacing=75)               # the heavier one
        ed.merge_stirrup_zones("T1", 0)
        merged = ed.segment.span("T1").stirrups[0]
        self.assertEqual(merged.spacing, 75.0)             # heavier kept
        self.assertEqual(len(ed.segment.span("T1").stirrups), n0)
        ed.reset_stirrup_zone("T1", 0)
        self.assertEqual(ed.segment.span("T1").stirrups[0].spacing,
                         self.seg.span("T1").stirrups[0].spacing)
        for bad in (lambda: ed.split_stirrup_zone("T1", 0, 99.0),
                    lambda: ed.merge_stirrup_zones("T1", 99),
                    lambda: ed.move_stirrup_cut("T1", 99, 1.0)):
            with self.assertRaises(D.DetailError):
                bad()

    def test_reset_all_restores_the_stirrups_too(self):
        ed = self.setup_editor()
        orig = copy.deepcopy(self.seg.to_dict())
        ed.set_stirrups("T1", 0, spacing=75)
        ed.set_layers("T1", 0, "top", [(2, 10.0)])
        ed.reset_all()
        self.assertEqual(ed.segment.to_dict(), orig)

    def test_stirrup_edits_count_as_manual_decisions(self):
        ed = self.setup_editor()
        ed.set_stirrups("T1", 0, spacing=75)
        doc = D.DetailDocument({"B1": D.BeamDetail("B1", [ed.segment])})
        self.assertTrue(doc.has_manual("B1"))
        self.assertEqual(doc.prune_auto(), [])             # kept: only stirrups edited
        auto = D.DetailDocument({"B1": D.BeamDetail("B1", [self.seg])})
        self.assertEqual(auto.prune_auto(), ["B1"])

    # -- persistence ----------------------------------------------------
    def test_round_trip_and_old_documents(self):
        doc = D.DetailDocument({"B1": D.BeamDetail("B1", [self.seg])})
        back = D.DetailDocument.loads(doc.dumps())
        self.assertEqual(back.to_dict(), doc.to_dict())
        self.assertIsInstance(back.beams["B1"].segments[0].spans[0]
                              .stirrups[0], D.StirrupZone)
        # a document written before stirrups existed: no key, no stirrups
        d = doc.to_dict()
        for sp in d["beams"]["B1"]["segments"][0]["spans"]:
            sp.pop("stirrups")
        old = D.DetailDocument.from_dict(d)
        self.assertTrue(all(not sp.stirrups
                            for sp in old.beams["B1"].segments[0].spans))
        # ... and spans without stirrups do not write the key
        self.assertNotIn("stirrups", old.to_dict()["beams"]["B1"]
                         ["segments"][0]["spans"][0])

    def test_bad_stirrups_are_rejected(self):
        d = D.DetailDocument({"B1": D.BeamDetail("B1", [self.seg])}).to_dict()
        sp = d["beams"]["B1"]["segments"][0]["spans"][0]
        for bad in ({"legs": 0}, {"diameter": 0}, {"spacing": -5},
                    {"origin": "x"}, {"x1": 0.0}):
            broken = copy.deepcopy(d)
            broken["beams"]["B1"]["segments"][0]["spans"][0]["stirrups"][0] \
                .update(bad)
            with self.assertRaises(D.DetailError, msg=bad):
                D.DetailDocument.from_dict(broken)
        broken = copy.deepcopy(d)
        broken["beams"]["B1"]["segments"][0]["spans"][0]["stirrups"][0]["x1"] -= 0.3
        with self.assertRaises(D.DetailError):             # does not tile
            D.DetailDocument.from_dict(broken)

    # -- reconciliation -------------------------------------------------
    def test_manual_stirrups_are_kept_and_auto_ones_follow(self):
        ed = self.setup_editor()
        ed.set_stirrups("T1", 0, diameter=10, spacing=100)        # manual
        new_prop = copy.deepcopy(self.seg)
        new_prop.spans[0].stirrups[0].spacing = 75.0              # ignored
        n = len(new_prop.spans[1].stirrups)
        new_prop.spans[1].stirrups[0].spacing = 75.0              # auto: follows
        out, rep = D.reconcile_segment(ed.segment, new_prop, self.inp)
        self.assertEqual(rep.status, "merged")
        self.assertEqual(out.span("T1").stirrups[0].spacing, 100.0)
        self.assertEqual(out.span("T2").stirrups[0].spacing, 75.0)
        self.assertEqual(len(out.span("T2").stirrups), n)
        self.assertEqual(D.validate_structure(out), [])

    def test_unchanged_when_nothing_changed(self):
        out, rep = D.reconcile_segment(copy.deepcopy(self.seg), self.seg,
                                       self.inp)
        self.assertEqual(rep.status, "unchanged")

    def test_a_manual_stirrup_that_no_longer_complies_is_flagged(self):
        ed = self.setup_editor()
        ed.set_stirrups("T1", 0, diameter=6, spacing=200)
        hungry = copy.deepcopy(self.inp)
        hungry.shear = [(x, v * 6.0 + 300.0) for x, v in hungry.shear]
        out, rep = D.reconcile_segment(ed.segment, self.seg, hungry)
        self.assertEqual(out.span("T1").stirrups[0].spacing, 200.0)   # kept
        self.assertTrue([i for i in rep.issues
                         if i.code == "ASW_LOW" and i.layer == "stirrups"])

    def test_a_proposal_that_gains_stirrups_fills_a_detail_without_them(self):
        bare = copy.deepcopy(self.seg)
        for sp in bare.spans:
            sp.stirrups = []
        out, _ = D.reconcile_segment(bare, self.seg, self.inp)
        self.assertTrue(all(sp.stirrups for sp in out.spans))
        # and the other way round: stored stirrups survive a proposal without
        stored = copy.deepcopy(self.seg)
        no_prop = copy.deepcopy(self.seg)
        for sp in no_prop.spans:
            sp.stirrups = []
        out, _ = D.reconcile_segment(stored, no_prop, None)
        self.assertEqual(out.to_dict(), stored.to_dict())

    def test_refresh_document_uses_the_stirrup_proposal(self):
        s, rows, inputs = _manual_model()
        key = "V1"
        detail = s.beam_detail.beams[key]
        for sp in detail.segments[0].spans:
            sp.stirrups = []                                  # an older detail
        doc, reports = D.refresh_document(s.beam_detail, rows, inputs)
        self.assertTrue(all(sp.stirrups for sp in
                            doc.beams[key].segments[0].spans))

    # -- the chart data -------------------------------------------------
    def test_stirrup_profile_matches_the_checks(self):
        prof = D.stirrup_profile(self.seg, self.inp)
        self.assertEqual(len(prof), sum(len(s.stirrups)
                                        for s in self.seg.spans))
        off = self.inp.offsets()
        for p in prof:
            sp = self.seg.span(p["span"])
            z = sp.stirrups[p["k"]]
            req = D.stirrup_requirements(self.inp, p["span"], sp.stirrups)[p["k"]]
            self.assertEqual((p["required"], p["provided"]), (req, z.asw_s()))
            self.assertAlmostEqual(p["x0"], off[p["span"]] + z.x0)
            self.assertAlmostEqual(p["s_max"], D.stirrup_s_max(self.inp))
        for a, b in zip(prof, prof[1:]):
            self.assertAlmostEqual(a["x1"], b["x0"])
        bare = copy.deepcopy(self.seg)
        for sp in bare.spans:
            sp.stirrups = []
        self.assertEqual(D.stirrup_profile(bare, self.inp), [])


# ── the proposal as a function of the inputs; parameters ─────────────────

class TestProposalFromInputs(unittest.TestCase):
    def setUp(self):
        s, self.rows, self.inputs = _design(zones=(0.25, 0.5, 0.25))
        self.inp = self.inputs[0]

    def test_it_is_what_the_design_reports(self):
        from_rows = D.details_from_rows(self.rows, self.inputs)
        from_inputs = D.proposals_from_inputs(self.inputs)
        self.assertEqual({k: v.to_dict() for k, v in from_rows.items()},
                         {k: v.to_dict() for k, v in from_inputs.items()})

    def test_the_serialised_inputs_give_the_same_proposal(self):
        back = D.SegmentInputs.from_dict(json.loads(json.dumps(self.inp.to_dict())))
        a = D.propose_segment(self.inp)
        b = D.propose_segment(back)
        self.assertEqual(a.segment.to_dict(), b.segment.to_dict())
        self.assertEqual(a.through, b.through)

    def test_proposal_parts(self):
        prop = D.propose_segment(self.inp)
        self.assertIsNone(prop.error)
        flat = [z for sp in prop.segment.spans for z in sp.zones]
        self.assertEqual(len(prop.zones), len(flat))
        for zp, z in zip(prop.zones, flat):
            self.assertEqual((zp.x0, zp.x1), (z.x0, z.x1))
            self.assertTrue(z.bottom and z.top and z.origin == "auto")
        self.assertTrue(all(sp.stirrups for sp in prop.segment.spans))
        self.assertTrue(D.check_segment(prop.segment, self.inp).ok)
        self.assertIn(prop.through["bottom"], {d for z in flat
                                               for _n, d in z.bottom})

    def test_no_layout_fits_gives_an_error_and_empty_zones(self):
        absurd = copy.deepcopy(self.inp)
        absurd.stations = [(x, 1e7, 1e7) for x, _b, _t in absurd.stations]
        prop = D.propose_segment(absurd)
        self.assertIsNotNone(prop.error)
        self.assertTrue(prop.notes)
        self.assertTrue(all(not z.bottom and not z.top
                            for sp in prop.segment.spans for z in sp.zones))

    def test_changing_a_parameter_in_the_inputs_equals_designing_with_it(self):
        """The window recalculates a proposal after a parameter change; it must
        be the proposal the design itself would give with that parameter."""
        cases = [dict(diameters=(16, 20, 25)), dict(n_through=3),
                 dict(shift_d=0.5), dict(stirrup_diameter_mm=10),
                 dict(symmetry="even"), dict(cutoff_levels=4),
                 dict(zones=(0.5, 0.5)), dict(top_min_ratio=1.0),
                 dict(max_layers=1, max_diameters_beam=1)]
        base = _design()[2][0]                       # default parameters
        for kw in cases:
            _s, _rows, ins = _design(**kw)
            want = D.propose_segment(ins[0]).segment.to_dict()
            params = copy.deepcopy(base.params_obj)
            from dataclasses import replace
            fields = {k: v for k, v in kw.items() if k != "zones"}
            if "zones" in kw:
                fields.update(zone_mode="fixed", zones=tuple(kw["zones"]))
            refit = D.refit_inputs(base, replace(params, **fields))
            got = D.propose_segment(refit).segment.to_dict()
            self.assertEqual(got, want, kw)


class TestRefit(unittest.TestCase):
    def setUp(self):
        self.inp = _design()[2][0]

    def test_idempotent(self):
        again = D.refit_inputs(self.inp, self.inp.params_obj)
        self.assertEqual(again.to_dict(), self.inp.to_dict())
        self.assertEqual(again.signature(), self.inp.signature())

    def test_stirrup_assumption_and_shift_recompute_cover_and_a_l(self):
        from dataclasses import replace
        sec = self.inp.section
        self.assertIn("rc_cover", sec)
        new = D.refit_inputs(self.inp, replace(
            self.inp.params_obj, stirrup_diameter_mm=10.0, shift_d=0.5))
        self.assertAlmostEqual(new.section["stirrup"], 10.0)
        self.assertAlmostEqual(new.section["rc_cover"], sec["rc_cover"])
        # the bar's centre stays where the mechanical cover puts it
        self.assertAlmostEqual(
            new.section["cover"] + 10.0 + new.params["d_bar_est"] / 2.0,
            sec["rc_cover"])
        self.assertAlmostEqual(new.a_l, 0.5 * (sec["h"] - sec["rc_cover"]) / 1000.0)
        self.assertEqual(new.params["shift_d"], 0.5)
        # what is not a parameter being changed is kept
        self.assertEqual(new.params["d_bar_est"], self.inp.params["d_bar_est"])
        self.assertEqual(new.params["dg"], self.inp.params["dg"])
        self.assertEqual(new.stations, self.inp.stations)

    def test_old_inputs_without_rc_cover_are_handled(self):
        old = copy.deepcopy(self.inp)
        old.section.pop("rc_cover")
        a = D.refit_inputs(old, old.params_obj)
        self.assertAlmostEqual(a.section["rc_cover"],
                               self.inp.section["rc_cover"])


class TestParameterLayers(unittest.TestCase):
    def test_layers_resolve_like_resolve_beam_params(self):
        from xdfem2d.beam_bars_params import BeamBarParams, resolve_beam_params
        lay = D.ParameterLayers(
            defaults={"beam_n_through": 3, "beam_stirrup_legs": 4},
            sections={"CONC": {"max_layers": 3}},
            beams={"V1": {"max_layers": 1, "stirrup_legs": 2}})
        p = lay.effective("V1", "CONC")
        self.assertEqual((p.n_through, p.max_layers, p.stirrup_legs),
                         (3, 1, 2))                      # beam > section > defaults
        q = lay.effective("V9", "CONC")
        self.assertEqual((q.n_through, q.max_layers, q.stirrup_legs), (3, 3, 4))
        self.assertEqual(lay.effective("V9", "OTHER").max_layers, 2)
        # inherited: the level above
        self.assertEqual(lay.inherited("beam", "V1", "CONC").max_layers, 3)
        self.assertEqual(lay.inherited("section", section_name="CONC").n_through, 3)
        self.assertEqual(lay.inherited("defaults"), BeamBarParams())
        with self.assertRaises(D.DetailError):
            lay.inherited("galaxy")

    def test_base_keeps_the_assumed_bar_and_aggregate(self):
        from xdfem2d.beam_bars_params import BeamBarParams
        from dataclasses import replace
        base = replace(BeamBarParams(), d_bar_est=20.0, dg=25.0)
        p = D.ParameterLayers().effective("V1", "CONC", base)
        self.assertEqual((p.d_bar_est, p.dg), (20.0, 25.0))

    def test_round_trip_and_validation(self):
        lay = D.ParameterLayers({"beam_diameters": [16, 20]},
                                {"CONC": {"diameters": [12]}},
                                {"V1": {"zones": (0.5, 0.5)}})
        back = D.ParameterLayers.from_dict(json.loads(json.dumps(lay.to_dict())))
        self.assertEqual(back.to_dict(), lay.to_dict())
        self.assertEqual(D.ParameterLayers.from_dict(None).defaults, {})
        for bad in ({"defaults": {"nope": 1}},
                    {"sections": {"S": {"n_through": 2}}},      # not section-level
                    {"beams": {"V": {"dg": 5}}},
                    {"beams": {"V": {"max_layers": 0}}},        # fails validation
                    [1]):
            with self.assertRaises(D.DetailError, msg=bad):
                D.ParameterLayers.from_dict(bad)

    def test_refit_for_layers(self):
        s, rows, inputs = _design()
        tag = inputs[0].tag
        lay = D.ParameterLayers({"beam_stirrup_mm": 10},
                                beams={tag: {"n_through": 4}})
        out = D.refit_for_layers(inputs, lay)
        self.assertEqual(out[0].params["n_through"], 4)
        self.assertEqual(out[0].section["stirrup"], 10)
        self.assertEqual(inputs[0].params["n_through"], 2)       # input untouched

    def test_package_carries_the_layers_and_old_packages_load(self):
        detail, inp = _proposal()
        lay = D.ParameterLayers({"beam_n_through": 3}, beams={"B1": {"shift_d": 0.4}})
        pk = D.BeamPackage([inp], D.DetailDocument({detail.tag: detail}),
                           D.DetailDocument({detail.tag: detail}), lay)
        back = D.BeamPackage.loads(pk.dumps())
        self.assertEqual(back.parameters.to_dict(), lay.to_dict())
        old = pk.to_dict()
        old.pop("parameters")
        self.assertEqual(D.BeamPackage.from_dict(old).parameters.to_dict(),
                         D.ParameterLayers().to_dict())
        self.assertEqual(inp.section_name, "CONC")               # the name travels
        self.assertEqual(back.inputs[0].section_name, "CONC")
