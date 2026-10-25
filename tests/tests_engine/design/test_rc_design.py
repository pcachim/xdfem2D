"""Tests for the concrete (EC2) reinforcement design.

Covers the ``RCSection`` single-section design (which delegates to eurocodepy)
and the structure-level ``design_concrete_sections`` driver: section-type
selection, per-combination rows, governing flag, envelope handling and
combination filtering.

Requires eurocodepy to be importable (xdfem2D delegates the EC2 formulae to it).
Run with:  python -m unittest discover -s tests
"""
import unittest

from context import Structure2D  # noqa: F401  (adds src/ to sys.path)

from beam_params_util import bbp
from xdfem2d.rc_design import RCSection, design_concrete_sections
from xdfem2d.models import SectionType

FCK, FYK = 30.0, 500.0


def _ef(forces):
    """Build an element_forces map {elem: {'i':[N,V,M], 'j':[N,V,M]}}."""
    return forces


class TestRCSection(unittest.TestCase):
    def setUp(self):
        self.sec = RCSection(0.30, 0.50, 0.05, FCK, FYK, 1.5, 1.15, 1.0)

    def test_pure_bending_positive_steel(self):
        r = self.sec.flexural_reinforcement(150.0, 0.0)
        self.assertGreater(r["As_bot"], 0.0)
        self.assertEqual(r["As_top"], 0.0)
        # As in m²; a C30 0.3x0.5 beam at 150 kNm needs ~8 cm².
        self.assertAlmostEqual(r["As_bot"] * 1e4, 8.21, delta=0.2)

    def test_compression_reduces_steel(self):
        bending = self.sec.flexural_reinforcement(150.0, 0.0)
        # Ned positive = tension in xdfem2D; pass negative for compression.
        comp = self.sec.flexural_reinforcement(150.0, -200.0)
        self.assertLess(comp["As_bot"], bending["As_bot"])

    def test_tension_increases_steel(self):
        bending = self.sec.flexural_reinforcement(150.0, 0.0)
        tens = self.sec.flexural_reinforcement(150.0, 200.0)
        self.assertGreater(tens["As_bot"], bending["As_bot"])

    def test_sagging_steel_on_bottom(self):
        r = self.sec.flexural_reinforcement(150.0, 0.0)
        self.assertGreater(r["As_bot"], 0.0)
        self.assertEqual(r["As_top"], 0.0)

    def test_hogging_steel_on_top(self):
        # Negative (hogging) moment must place the tension steel on the TOP face.
        r = self.sec.flexural_reinforcement(-150.0, 0.0)
        self.assertGreater(r["As_top"], 0.0)
        self.assertEqual(r["As_bot"], 0.0)

    def test_sagging_hogging_symmetry(self):
        # Symmetric section: ±M (with the same axial) mirror bottom/top.
        rs = self.sec.flexural_reinforcement(150.0, -200.0)
        rh = self.sec.flexural_reinforcement(-150.0, -200.0)
        self.assertAlmostEqual(rs["As_bot"], rh["As_top"], places=6)
        self.assertAlmostEqual(rs["As_top"], rh["As_bot"], places=6)

    def test_compression_reduces_both_faces(self):
        # Axial compression reduces the tension steel for sagging AND hogging.
        bend = self.sec.flexural_reinforcement(150.0, 0.0)["As_bot"]
        sag = self.sec.flexural_reinforcement(150.0, -200.0)["As_bot"]
        hog = self.sec.flexural_reinforcement(-150.0, -200.0)["As_top"]
        self.assertLess(sag, bend)
        self.assertLess(hog, bend)

    def test_ordinary_case_has_no_note(self):
        self.assertEqual(self.sec.flexural_reinforcement(150.0, 0.0)["note"], "")

    def test_compression_controlled_uses_strain_compatibility(self):
        # N = 1800 kN compression (xdfem2D: tension positive) on a 30x50 beam
        # is far outside the simplified method: it used to return As_min on the
        # tension face and a heavily over-designed compression face.
        r = self.sec.flexural_reinforcement(300.0, -1800.0)
        self.assertEqual(r["note"], "")          # exact result, nothing to warn
        from eurocodepy.ec2.uls import calc_asl_nm
        old = calc_asl_nm(0.30, 0.50, 0.05, 0.05, 300.0, 1800.0, FCK, FYK,
                          1.5, 1.15)
        self.assertLess(r["As_bot"] + r["As_top"],
                        0.85 * (old["As1"] + old["As2"]) * 1e-4)

    def test_strain_fallback_puts_tension_face_by_moment_sign(self):
        sag = self.sec.flexural_reinforcement(300.0, -1800.0)
        hog = self.sec.flexural_reinforcement(-300.0, -1800.0)
        self.assertAlmostEqual(sag["As_bot"], hog["As_top"], places=9)
        self.assertAlmostEqual(sag["As_top"], hog["As_bot"], places=9)

    def test_oversized_axial_load_is_flagged(self):
        r = self.sec.flexural_reinforcement(300.0, -9000.0)
        self.assertTrue(r["note"])
        self.assertNotIn("compression-controlled)", r["note"])   # a warning

    def test_low_shear_is_minimum_stirrups(self):
        s = self.sec.shear_reinforcement(40.0, As_long=8e-4)
        self.assertEqual(s["mode"], "no_shear_reinf")
        self.assertGreater(s["Asw_s"], 0.0)

    def test_high_shear_needs_stirrups(self):
        s = self.sec.shear_reinforcement(300.0, cotg_theta=2.5, As_long=8e-4)
        self.assertEqual(s["mode"], "stirrups")
        self.assertGreater(s["Asw_s"], 0.0)
        self.assertIsNotNone(s["VRd_max"])


class TestDesignConcreteSections(unittest.TestCase):
    def _struc(self):
        s = Structure2D()
        s.add_node("n1", 0.0, 0.0)
        s.add_node("n2", 5.0, 0.0)
        s.add_node("n3", 10.0, 0.0)
        s.add_material("C", elastic_modulus=33e6, unit_weight=25.0,
                       material_type="Concrete",
                       design={"fck": FCK, "fyk": FYK})
        # The section's design type now derives from its material: a steel
        # section is one whose material is steel.
        s.add_material("S", elastic_modulus=210e6, unit_weight=78.5,
                       material_type="Steel")
        s.add_section("CONC", "C", b=0.3, h=0.5)
        s.add_section("STL", "S", b=0.2, h=0.3)
        s.add_bar_element("E1", "n1", "n2", "CONC")
        s.add_bar_element("E2", "n2", "n3", "STL")
        return s

    def _results(self):
        return {
            "element_forces": {},
            "combinations": {
                "ULS1": {"element_forces": _ef({
                    "E1": {"i": [0, 40, 0], "j": [0, -40, 150]},
                    "E2": {"i": [0, 10, 0], "j": [0, -10, 30]},
                })},
                "ULS2": {"element_forces": _ef({
                    "E1": {"i": [0, 30, 0], "j": [0, -30, 90]},
                })},
            },
        }

    def test_only_concrete_sections_designed(self):
        rows = design_concrete_sections(self._struc(), self._results())
        elems = {r["element"] for r in rows}
        self.assertIn("E1", elems)
        self.assertNotIn("E2", elems)   # steel section excluded

    def test_rows_carry_flexure_note(self):
        rows = design_concrete_sections(self._struc(), self._results())
        self.assertTrue(all("note" in r for r in rows))
        self.assertTrue(all(r["note"] == "" for r in rows))    # plain bending

    def test_oversized_axial_bar_row_reports_note(self):
        res = {"element_forces": {}, "combinations": {"ULS1": {
            "element_forces": _ef({"E1": {"i": [-9000, 0, 300],
                                          "j": [9000, 0, 300]}})}}}   # j: sign flips
        rows = design_concrete_sections(self._struc(), res)
        self.assertTrue(rows)
        self.assertTrue(all(r["note"] for r in rows))    # section far too small

    def test_low_axial_force_keeps_simplified_result(self):
        # nu <= 0.05 -> the hand-checkable simplified numbers, unchanged.
        from eurocodepy.ec2.uls import calc_asl_nm
        rcs = RCSection(0.30, 0.50, 0.05, FCK, FYK, 1.5, 1.15, 1.0)
        r = rcs.flexural_reinforcement(150.0, -100.0)     # 100 kN compression
        ref = calc_asl_nm(0.30, 0.50, 0.05, 0.05, 150.0, 100.0, FCK, FYK, 1.5, 1.15)
        self.assertAlmostEqual(r["As_bot"], ref["As1"] * 1e-4, places=12)

    def test_rows_for_every_combination(self):
        rows = design_concrete_sections(self._struc(), self._results())
        combos = {r["combination"] for r in rows}
        self.assertEqual(combos, {"ULS1", "ULS2"})

    def test_one_governing_row_per_element(self):
        rows = design_concrete_sections(self._struc(), self._results())
        gov = [r for r in rows if r["governing"]]
        self.assertEqual(len(gov), 1)
        self.assertEqual(gov[0]["combination"], "ULS1")  # 150 kNm governs

    def test_combination_filter(self):
        rows = design_concrete_sections(self._struc(), self._results(), combinations=["ULS2"])
        combos = {r["combination"] for r in rows}
        self.assertEqual(combos, {"ULS2"})

    def test_envelope_combination_expands_to_max_min(self):
        struc = self._struc()
        results = {
            "element_forces": {},
            "combinations": {
                "ENV": {"element_forces": {
                    "max": {"E1": {"i": [0, 50, 0], "j": [0, -50, 150]}},
                    "min": {"E1": {"i": [0, -50, 0], "j": [0, 50, -30]}},
                }},
            },
        }
        rows = design_concrete_sections(struc, results)
        combos = {r["combination"] for r in rows}
        self.assertEqual(combos, {"ENV (max)", "ENV (min)"})

    def test_generic_section_shape_is_still_designed(self):
        """SectionShape.GENERIC (the default for any section that never sets
        `shape` explicitly, including this test's own plain b×h sections) is
        NOT a signal to skip -- only a section defined purely by A/I
        overrides is (see test_area_inertia_override_section_is_skipped)."""
        struc = self._struc()
        for sec in struc.sections.values():
            from xdfem2d.models import SectionShape
            self.assertEqual(sec.shape, SectionShape.GENERIC)
        rows = design_concrete_sections(struc, self._results())
        self.assertIn("E1", {r["element"] for r in rows})

    def test_area_inertia_override_section_is_skipped(self):
        """A concrete section defined by area_override/inertia_override (no
        real b×h geometry -- e.g. an imported/idealised member) has nothing
        for the flexure/shear design to integrate against, so it is skipped
        rather than designed against a meaningless b×h."""
        struc = self._struc()
        struc.sections["CONC"].area_override = 0.15
        struc.sections["CONC"].inertia_override = 0.003
        rows = design_concrete_sections(struc, self._results())
        self.assertNotIn("E1", {r["element"] for r in rows})


class TestDesignConcreteSectionsGrillage(unittest.TestCase):
    """Regression baseline for the grillage (domain='plate') shear+torsion
    path, captured BEFORE the migration proposed in
    dev/GRILLAGE_DESIGN.md phase 2 (moving the combination logic into
    eurocodepy.ec2.uls.shear_torsion). These tests pin today's numbers and
    known quirks so the migration can be checked against them; they do not
    assert that the current behaviour is normatively ideal (see the
    docstrings below for the two points flagged in the design doc: the
    top/bottom 50/50 torsion-steel split in §2.4, and the cot-theta
    divergence between shear and torsion in §2.3/§7).
    """

    def _grillage_struc(self, cotg_theta=None):
        s = Structure2D(domain="plate")
        s.add_node("n1", 0.0, 0.0)
        s.add_node("n2", 5.0, 0.0)
        s.add_material("C", elastic_modulus=33e6, unit_weight=25.0,
                       material_type="Concrete",
                       design={"fck": FCK, "fyk": FYK})
        s.add_section("CONC", "C", b=0.3, h=0.5)
        s.add_bar_element("E1", "n1", "n2", "CONC")
        if cotg_theta is not None:
            for elem in s.bar_elements:
                elem.rc_cotg_theta = cotg_theta
        return s

    def _plane_struc(self):
        s = Structure2D()
        s.add_node("n1", 0.0, 0.0)
        s.add_node("n2", 5.0, 0.0)
        s.add_material("C", elastic_modulus=33e6, unit_weight=25.0,
                       material_type="Concrete",
                       design={"fck": FCK, "fyk": FYK})
        s.add_section("CONC", "C", b=0.3, h=0.5)
        s.add_bar_element("E1", "n1", "n2", "CONC")
        return s

    @staticmethod
    def _results(t, v, m):
        # i-end as given; j-end negated (nodal convention, see _critical_points).
        return {
            "element_forces": {},
            "combinations": {"ULS": {"element_forces": {
                "E1": {"i": [t, v, m], "j": [-t, -v, -m]},
            }}},
        }

    def _governing(self, rows):
        gov = [r for r in rows if r["governing"]]
        self.assertEqual(len(gov), 1)
        return gov[0]

    def test_grillage_rows_carry_torsion_specific_fields(self):
        rows = design_concrete_sections(
            self._grillage_struc(), self._results(t=20.0, v=80.0, m=100.0))
        self.assertTrue(rows)
        for r in rows:
            self.assertEqual(r["kind"], "grillage")
            for key in ("T_Ed", "Asw_tor_s", "Asl_tor", "TRd_max", "interaction"):
                self.assertIn(key, r)

    def test_plane_domain_rows_have_no_torsion_fields(self):
        # Same section/forces, but domain='plane' (default): T stays 0 and none
        # of the grillage-only keys are added (dev/GRILLAGE_DESIGN.md §4 —
        # torsion only exists in the plate domain today, Option A).
        rows = design_concrete_sections(
            self._plane_struc(), self._results(t=20.0, v=80.0, m=100.0))
        self.assertTrue(rows)
        for r in rows:
            self.assertNotIn("kind", r)
            self.assertNotIn("T_Ed", r)
            self.assertEqual(r["N_Ed"], 20.0)  # the "T" slot is read as axial N here

    def test_pure_torsion_splits_asl_tor_50_50_onto_flexural_bot_and_top(self):
        """Captures the §2.4 simplification as today's actual behaviour: even
        with no bending moment, Asl_tor is added 50/50 onto whatever
        As_bot/As_top the flexural design alone already produced -- it is
        NOT distributed around the section perimeter. This is the baseline
        the §2.4/Phase-3 perimeter-distribution fix must be compared
        against; it is not being asserted as normatively complete.

        A separate quirk falls out of this: ``flexural_reinforcement(0, 0)``
        itself returns a non-zero ``As_bot`` (apparently a minimum-
        reinforcement default that always lands on the bottom face), so
        As_bot and As_top end up genuinely different even under pure
        torsion -- the two faces are not symmetric just because M=0.
        """
        struc = self._grillage_struc()
        rows = design_concrete_sections(struc, self._results(t=15.0, v=0.0, m=0.0))
        row = rows[0]
        self.assertGreater(row["Asl_tor"], 0.0)

        sec = struc.sections["CONC"]
        rcs = RCSection(sec.b, sec.h, sec.rc_cover, FCK, FYK, 1.5, 1.15, 1.0)
        flex_zero = rcs.flexural_reinforcement(0.0, 0.0)

        self.assertAlmostEqual(row["As_bot"],
                               flex_zero["As_bot"] + row["Asl_tor"] / 2.0, places=9)
        self.assertAlmostEqual(row["As_top"],
                               flex_zero["As_top"] + row["Asl_tor"] / 2.0, places=9)
        # Documented quirk: not symmetric, because flex_zero itself isn't.
        self.assertNotAlmostEqual(row["As_bot"], row["As_top"], places=6)

    def test_low_interaction_does_not_trip_crushing_from_torsion(self):
        rows = design_concrete_sections(
            self._grillage_struc(), self._results(t=5.0, v=20.0, m=50.0))
        gov = self._governing(rows)
        self.assertLess(gov["interaction"], 1.0)
        self.assertFalse(gov["crushing"])

    def test_high_shear_and_torsion_trip_crushing_via_interaction(self):
        rows = design_concrete_sections(
            self._grillage_struc(), self._results(t=200.0, v=250.0, m=50.0))
        gov = self._governing(rows)
        self.assertGreater(gov["interaction"], 1.0)
        self.assertTrue(gov["crushing"])

    def test_interaction_equals_torsion_util_plus_shear_ratio(self):
        # Confirms dev/GRILLAGE_DESIGN.md §2.3: the interaction is exactly
        # the linear EC2 Eq. 6.29 sum (T_Ed/T_Rd,max + V_Ed/V_Rd,max) -- not
        # an elliptical or otherwise adjusted combination.
        rows = design_concrete_sections(
            self._grillage_struc(), self._results(t=40.0, v=120.0, m=60.0))
        gov = self._governing(rows)
        expected = (gov["T_Ed"] / gov["TRd_max"]
                    + abs(gov["V_Ed"]) / gov["VRd_max"])
        self.assertAlmostEqual(gov["interaction"], expected, places=9)

    def test_migration_fixes_the_cot_theta_divergence(self):
        """Historical note: before the dev/GRILLAGE_DESIGN.md §5.2 migration,
        this test documented a real bug -- eurocodepy's ``ShearInput`` has no
        cot-theta field at all (``eurocode2_shear_check`` always sweeps its
        own flattest strut internally), so ``RCSection.torsion_reinforcement``
        computing T_Rd,max from the user's ``elem.rc_cotg_theta`` directly
        could diverge from whatever angle shear had actually used --
        violating EC2 §6.3.2(3). See the Phase-1 commit for the version of
        this test that demonstrated the divergence against the pre-migration
        code path.

        Now that ``design_concrete_sections`` calls
        ``RCSection.shear_torsion_reinforcement`` (which forces both
        verifications through the SAME cot θ by construction -- see
        eurocodepy's ``ec2.uls.shear_torsion``), ``elem.rc_cotg_theta`` is no
        longer read for grillage members at all: this test now confirms
        that changing it has NO effect on the grillage design, which is the
        correct fixed behaviour, not a leftover no-op to be cleaned up.
        """
        rows_a = design_concrete_sections(
            self._grillage_struc(cotg_theta=1.0),
            self._results(t=50.0, v=150.0, m=50.0))
        rows_b = design_concrete_sections(
            self._grillage_struc(cotg_theta=2.5),
            self._results(t=50.0, v=150.0, m=50.0))
        gov_a, gov_b = self._governing(rows_a), self._governing(rows_b)

        self.assertEqual(gov_a["cot"], gov_b["cot"])
        self.assertEqual(gov_a["TRd_max"], gov_b["TRd_max"])
        self.assertEqual(gov_a["interaction"], gov_b["interaction"])

    def test_rows_carry_the_new_asl_tor_by_face_field(self):
        rows = design_concrete_sections(
            self._grillage_struc(), self._results(t=15.0, v=0.0, m=0.0))
        row = rows[0]
        faces = row["Asl_tor_by_face"]
        for key in ("top", "bottom", "side_left", "side_right"):
            self.assertIn(key, faces)
        # default distribution_mode is "top_bottom" -- unchanged results.
        self.assertEqual(faces["side_left"], 0.0)
        self.assertEqual(faces["side_right"], 0.0)
        self.assertAlmostEqual(faces["top"] + faces["bottom"],
                               row["Asl_tor"], places=9)

    def test_rcsection_shear_torsion_reinforcement_delegates_to_eurocodepy(self):
        # Direct unit test of the new method (not just through the
        # design_concrete_sections orchestrator): it must return the same
        # combined fields eurocode2_shear_torsion_check computes.
        from eurocodepy.ec2.uls import ShearTorsionInput, eurocode2_shear_torsion_check

        sec = RCSection(0.30, 0.50, 0.05, FCK, FYK, 1.5, 1.15, 1.0)
        result = sec.shear_torsion_reinforcement(120.0, 30.0, As_long=8e-4)

        inp = ShearTorsionInput(b=0.30, h=0.50, cover=0.05, fck=FCK, fyk=FYK,
                                gamma_c=1.5, gamma_s=1.15, alpha_cc=1.0,
                                as_long=8e-4)
        expected = eurocode2_shear_torsion_check(inp, 120.0, 30.0)

        self.assertEqual(result["Asw_s"], expected.asw_total_s)
        self.assertEqual(result["Asl_tor"], expected.asl_tor_total)
        self.assertEqual(result["interaction"], expected.interaction)
        self.assertEqual(result["crushing"], expected.crushing)

    def test_perimeter_distribution_populates_side_faces_end_to_end(self):
        """Phase 3 (dev/GRILLAGE_DESIGN.md sec 2.4/6): setting
        Section.rc_torsion_distribution = "perimeter" on the grillage
        section must flow all the way through design_concrete_sections()
        into non-zero side_left/side_right entries in Asl_tor_by_face --
        while the default ("top_bottom") stays exactly as before (see
        test_rows_carry_the_new_asl_tor_by_face_field above).
        """
        struc = self._grillage_struc()
        struc.sections["CONC"].rc_torsion_distribution = "perimeter"
        rows = design_concrete_sections(
            struc, self._results(t=15.0, v=0.0, m=0.0))
        row = rows[0]
        faces = row["Asl_tor_by_face"]
        self.assertGreater(faces["side_left"], 0.0)
        self.assertGreater(faces["side_right"], 0.0)
        self.assertAlmostEqual(
            faces["top"] + faces["bottom"] + faces["side_left"] + faces["side_right"],
            row["Asl_tor"], places=9)


if __name__ == "__main__":
    unittest.main()


class TestDesignBeamBars(unittest.TestCase):
    """design_beam_bars: rows (areas) -> bars, spans chained over supports."""

    def _struc(self, sec2="CONC"):
        s = Structure2D()
        for i, x in enumerate((0.0, 5.0, 10.0), 1):
            s.add_node(f"n{i}", x, 0.0)
        s.add_material("C", elastic_modulus=33e6, unit_weight=25.0,
                       material_type="Concrete",
                       design={"fck": FCK, "fyk": FYK})
        s.add_section("CONC", "C", b=0.3, h=0.5)
        s.add_section("CONC2", "C", b=0.3, h=0.6)
        s.add_bar_element("E1", "n1", "n2", "CONC")
        s.add_bar_element("E2", "n2", "n3", sec2)
        s.add_support("PIN", ux=True, uy=True)
        s.add_support("ROLLER", ux=False, uy=True)
        s.assign_support("n1", "PIN")
        s.assign_support("n2", "ROLLER")
        s.assign_support("n3", "ROLLER")
        return s

    def _rows(self, s):
        res = {"element_forces": {}, "combinations": {"ULS1": {
            "element_forces": {
                "E1": {"i": [0, 40, 0], "j": [0, -40, 150]},
                "E2": {"i": [0, 30, -90], "j": [0, -30, 20]}}}}}
        return design_concrete_sections(s, res)

    def test_two_spans_form_one_beam(self):
        from xdfem2d.rc_design import design_beam_bars
        s = self._struc()
        out = design_beam_bars(s, self._rows(s), params=bbp(zones=(1.0,)))
        self.assertEqual(len(out), 2)
        self.assertEqual({r["beam"] for r in out}, {"B1"})
        for r in out:
            self.assertGreaterEqual(r["As_bot_prov"], r["As_bot_req"])
            self.assertGreaterEqual(r["As_top_prov"], r["As_top_req"])
            self.assertTrue(r["bottom"])
            self.assertIsNotNone(r["through_bot"])

    def test_different_sections_are_separate_beams(self):
        from xdfem2d.rc_design import design_beam_bars
        s = self._struc(sec2="CONC2")
        out = design_beam_bars(s, self._rows(s), params=bbp(zones=(1.0,)))
        self.assertEqual({r["beam"] for r in out}, {"B1", "B2"})

    def test_zones_split_top_and_bottom(self):
        from xdfem2d.rc_design import design_beam_bars
        s = self._struc()
        out = design_beam_bars(s, self._rows(s),
                              params=bbp(zones=(0.25, 0.5, 0.25)))
        self.assertEqual(len(out), 6)
        self.assertEqual([r["zone"] for r in out], [1, 2, 3, 1, 2, 3])
        e1, e2 = out[:3], out[3:]
        # E1 sags (M=150 at j is the max-M station at x=5 m -> right zone);
        # E2 hogs (-90 at its start -> left zone, over the support n2).
        self.assertGreater(e2[0]["As_top_req"], e2[1]["As_top_req"])
        self.assertGreater(e1[2]["As_bot_req"], e1[1]["As_bot_req"] - 1e-12)
        for r in out:
            self.assertGreaterEqual(r["As_bot_prov"], r["As_bot_req"])
            self.assertGreaterEqual(r["As_top_prov"], r["As_top_req"])
        self.assertAlmostEqual(sum(r["length"] for r in e1), 5.0)
        self.assertAlmostEqual(e1[1]["x0"], 1.25)

    def _results_dist(self):
        import numpy as np
        x = np.linspace(0.0, 5.0, 11)
        # E1 sagging parabola (peak 150 at x=2.5), E2 hogging at its start.
        m1 = 150.0 * (1 - ((x - 2.5) / 2.5) ** 2)
        m2 = -90.0 * (1 - x / 5.0) ** 2
        z = np.zeros_like(x)
        return {"element_forces": {}, "combinations": {
            "ULS1": {"element_forces": {
                "E1": {"i": [0, 60, 0], "j": [0, -60, 0]},
                "E2": {"i": [0, 30, -90], "j": [0, -30, 0]}}}},
            "combo_distribution": {"ULS1": {
                "E1": {"x": x, "N": z, "V": z, "M": m1},
                "E2": {"x": x, "N": z, "V": z, "M": m2}}}}

    def test_stations_drive_the_zones(self):
        from xdfem2d.rc_design import design_beam_bars
        s = self._struc()
        res = self._results_dist()
        rows = design_concrete_sections(s, res)
        Z = (0.25, 0.5, 0.25)
        out = design_beam_bars(s, rows, res, params=bbp(zones=Z))
        e1, e2 = out[:3], out[3:]
        # E1: peak sagging in the middle zone (support zones: 25 % rule/min)
        self.assertGreater(e1[1]["As_bot_req"], e1[0]["As_bot_req"])
        self.assertGreater(e1[1]["As_bot_req"], e1[2]["As_bot_req"])
        # E2: hogging concentrated in the left (support) zone only
        self.assertGreater(e2[0]["As_top_req"], e2[1]["As_top_req"])
        self.assertGreater(e2[0]["As_top_req"], e2[2]["As_top_req"])
        # shift rule widens the hogging envelope into the next zone
        out_s = design_beam_bars(s, rows, res,
                                 params=bbp(zones=Z, shift_d=1.0))
        self.assertGreaterEqual(out_s[4]["As_top_req"], out[4]["As_top_req"])

    def test_cutoff_zones_follow_the_envelope(self):
        from xdfem2d.rc_design import design_beam_bars
        s = self._struc()
        res = self._results_dist()
        rows = design_concrete_sections(s, res)
        out = design_beam_bars(s, rows, res, params=bbp(
            cutoff_levels=3, min_zone_length=0.4))
        by = {}
        for r in out:
            by.setdefault(r["span"], []).append(r)
        for zs in by.values():
            # zones tile the span without gaps and cover every requirement
            self.assertAlmostEqual(zs[0]["x0"], 0.0)
            self.assertAlmostEqual(zs[-1]["x1"], 5.0)
            for a, b in zip(zs, zs[1:]):
                self.assertAlmostEqual(a["x1"], b["x0"])
            for r in zs:
                self.assertGreaterEqual(r["length"], 0.4 - 1e-9)
                self.assertGreaterEqual(r["As_bot_prov"], r["As_bot_req"])
                self.assertGreaterEqual(r["As_top_prov"], r["As_top_req"])
        # a parabola quantised in steps needs more than one zone per span
        self.assertTrue(all(len(zs) > 1 for zs in by.values()))
        # hogging over the support (start of E2) is stepped down along E2
        top = [r["As_top_req"] for r in sorted(
            (r for r in out if r["span"] == out[-1]["span"]),
            key=lambda r: r["x0"])]
        self.assertEqual(top, sorted(top, reverse=True))
        self.assertGreater(top[0], top[-1])

    def test_cutoff_single_zone_when_envelope_flat(self):
        from xdfem2d.rc_design import design_beam_bars
        s = self._struc()
        out = design_beam_bars(s, self._rows(s),
                              params=bbp(cutoff_levels=1))
        self.assertEqual(len(out), 2)         # rows only + 1 level -> 1 zone

    def test_through_bars_in_all_zones(self):
        from xdfem2d.rc_design import design_beam_bars
        s = self._struc()
        out = design_beam_bars(s, self._rows(s),
                              params=bbp(zones=(0.25, 0.5, 0.25)))
        for face, key in (("bottom_layers", "through_bot"),
                          ("top_layers", "through_top")):
            for r in out:
                n, d = r[face][0]
                self.assertEqual(d, r[key])
                self.assertGreaterEqual(n, 2)


class TestBeamChaining(unittest.TestCase):
    """design_beam_bars: which spans form a beam, orientation and numbering."""

    def _model(self, nodes, bars, supports=(), columns=()):
        """nodes {id: (x, y)}, bars [(id, ni, nj, section, kw)]."""
        s = Structure2D()
        for nid, (x, y) in nodes.items():
            s.add_node(nid, x, y)
        s.add_material("C", elastic_modulus=33e6, unit_weight=25.0,
                       material_type="Concrete",
                       design={"fck": FCK, "fyk": FYK})
        s.add_section("CONC", "C", b=0.3, h=0.5)
        s.add_section("CONC2", "C", b=0.3, h=0.6)
        s.add_section("COL", "C", b=0.3, h=0.3, is_column=True)
        for bid, ni, nj, sec, kw in bars:
            s.add_bar_element(bid, ni, nj, sec, **kw)
        s.add_support("FIX", ux=True, uy=True, tz=True)
        for n in supports:
            s.assign_support(n, "FIX")
        return s

    def _beams(self, s, beam_ids):
        from xdfem2d.rc_design import design_beam_bars
        res = {"element_forces": {}, "combinations": {"ULS1": {
            "element_forces": {b: {"i": [0, 40, -50], "j": [0, -40, 100]}
                               for b in beam_ids}}}}
        rows = design_concrete_sections(s, res)
        out = design_beam_bars(s, rows, res)
        by = {}
        for r in out:
            by.setdefault(r["beam"], []).append(r)
        return {k: sorted({e for r in v for e in r["elements"]})
                for k, v in by.items()}, out

    def test_beam_continues_over_columns(self):
        nodes = {"a": (0, 3), "b": (5, 3), "c": (10, 3),
                 "g1": (0, 0), "g2": (5, 0), "g3": (10, 0)}
        bars = [("E1", "a", "b", "CONC", {}), ("E2", "b", "c", "CONC", {}),
                ("C1", "g1", "a", "COL", {}), ("C2", "g2", "b", "COL", {}),
                ("C3", "g3", "c", "COL", {})]
        s = self._model(nodes, bars, supports=("g1", "g2", "g3"))
        beams, _ = self._beams(s, ("E1", "E2"))
        self.assertEqual(beams, {"B1": ["E1", "E2"]})   # columns ignored

    def test_t_joins_only_the_collinear_pair(self):
        # third member leaves b at ~27° (auto-detection is horizontal-only)
        nodes = {"a": (0, 3), "b": (5, 3), "c": (10, 3), "d": (8, 4.5)}
        bars = [("E1", "a", "b", "CONC", {}), ("E2", "b", "c", "CONC", {}),
                ("E3", "b", "d", "CONC", {})]
        s = self._model(nodes, bars, supports=("a", "c"))
        beams, _ = self._beams(s, ("E1", "E2", "E3"))
        self.assertEqual(beams, {"B1": ["E1", "E2"], "B2": ["E3"]})

    def test_hinge_breaks_continuity(self):
        nodes = {"a": (0, 3), "b": (5, 3), "c": (10, 3)}
        bars = [("E1", "a", "b", "CONC", {"hinge_j": True}),
                ("E2", "b", "c", "CONC", {})]
        s = self._model(nodes, bars, supports=("a", "b", "c"))
        beams, _ = self._beams(s, ("E1", "E2"))
        self.assertEqual(beams, {"B1": ["E1"], "B2": ["E2"]})

    def test_section_change_breaks_continuity(self):
        nodes = {"a": (0, 3), "b": (5, 3), "c": (10, 3)}
        bars = [("E1", "a", "b", "CONC", {}), ("E2", "b", "c", "CONC2", {})]
        s = self._model(nodes, bars, supports=("a", "b", "c"))
        beams, _ = self._beams(s, ("E1", "E2"))
        self.assertEqual(beams, {"B1": ["E1"], "B2": ["E2"]})

    def test_orientation_left_to_right_whatever_the_bar_direction(self):
        nodes = {"a": (0, 3), "b": (5, 3), "c": (10, 3)}
        # both bars drawn right -> left
        bars = [("E1", "b", "a", "CONC", {}), ("E2", "c", "b", "CONC", {})]
        s = self._model(nodes, bars, supports=("a", "b", "c"))
        _, out = self._beams(s, ("E1", "E2"))
        self.assertEqual(out[0]["elements"], ["E1"])       # leftmost span first
        self.assertEqual(out[-1]["elements"], ["E2"])
        self.assertAlmostEqual(out[0]["x0"], 0.0)
        xs = [(r["x0"], r["x1"]) for r in out if r["span"] == out[0]["span"]]
        self.assertEqual(xs, sorted(xs))

    def test_numbering_bottom_to_top_then_left_to_right(self):
        nodes = {"a": (0, 6), "b": (5, 6), "c": (0, 3), "d": (5, 3),
                 "e": (8, 3), "f": (13, 3)}
        bars = [("TOP", "a", "b", "CONC", {}),        # defined first, upper
                ("R", "e", "f", "CONC", {}),           # lower level, right
                ("L", "c", "d", "CONC", {})]           # lower level, left
        s = self._model(nodes, bars, supports=tuple(nodes))
        beams, _ = self._beams(s, ("TOP", "R", "L"))
        self.assertEqual(beams, {"B1": ["L"], "B2": ["R"], "B3": ["TOP"]})

    def test_same_level_within_a_millimetre(self):
        nodes = {"a": (0, 3.0), "b": (5, 3.0), "c": (8, 3.0004),
                 "d": (13, 3.0004)}
        bars = [("R", "c", "d", "CONC", {}), ("L", "a", "b", "CONC", {})]
        s = self._model(nodes, bars, supports=tuple(nodes))
        beams, _ = self._beams(s, ("R", "L"))
        self.assertEqual(beams, {"B1": ["L"], "B2": ["R"]})   # by x, not noise
