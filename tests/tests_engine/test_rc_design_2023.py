"""EN 1992-1-1:2023 concrete design in xdfem2D: members (flexure, shear, torsion),
membranes, slabs and punching, selected with the code edition ('2004' default).

The 2023 formulae live in eurocodepy (ec2.uls2023); these tests check that the
app selects them, feeds them the right data and keeps the 2004 path unchanged.
"""
import math
import unittest
import warnings

from context import Structure2D  # noqa: F401  (adds src/ to sys.path)

import eurocodepy as ec
from eurocodepy.ec2 import uls2023
from xdfem2d.punching import design_punching
from xdfem2d.rc_design import (RCSection, design_concrete_planes,
                               design_concrete_sections, design_concrete_slabs,
                               design_tri_and_store, edition_kwargs,
                               reinforcement_envelope)

FCK, FYK = 30.0, 500.0


def _material(s, fck=FCK):
    s.add_material("C", elastic_modulus=33e6, unit_weight=25.0, poisson=0.2,
                   material_type="Concrete", design={"fck": fck, "fyk": FYK})


class TestEditionKwargs(unittest.TestCase):
    def test_defaults_and_prefs(self):
        self.assertEqual(edition_kwargs(None),
                         {"edition": "2004", "annex": None, "k_tc": 1.0})
        self.assertEqual(
            edition_kwargs({"ec2_edition": "2023", "ec2_annex": "PT", "ec2_k_tc": 0.85}),
            {"edition": "2023", "annex": "PT", "k_tc": 0.85})

    def test_unknown_edition_is_refused(self):
        with self.assertRaisesRegex(ValueError, "edition"):
            RCSection(0.3, 0.5, 0.05, FCK, FYK, edition="2024")


class TestRCSection2023(unittest.TestCase):
    def setUp(self):
        self.s04 = RCSection(0.30, 0.50, 0.05, FCK, FYK, 1.5, 1.15, 1.0)
        self.s23 = RCSection(0.30, 0.50, 0.05, FCK, FYK, 1.5, 1.15, 1.0, edition="2023")

    def test_fcd_is_eta_cc_k_tc_fck_over_gamma_c(self):
        self.assertEqual(self.s04.alpha_cc, 1.0)
        self.assertAlmostEqual(self.s23.fcd, 1.0 * 30.0e3 / 1.5)        # η_cc = 1 up to C40
        hi = RCSection(0.3, 0.5, 0.05, 60.0, FYK, edition="2023")
        self.assertAlmostEqual(hi.alpha_cc, (40.0 / 60.0) ** (1 / 3))
        other = RCSection(0.3, 0.5, 0.05, FCK, FYK, edition="2023", k_tc=0.85)
        self.assertAlmostEqual(other.alpha_cc, 0.85)
        # the 2004 coefficient is ignored in 2023
        ignored = RCSection(0.3, 0.5, 0.05, FCK, FYK, 1.5, 1.15, 0.85, edition="2023")
        self.assertEqual(ignored.alpha_cc, 1.0)
        self.assertEqual(self.s23.gamma_v, 1.4)

    def test_flexure_uses_the_2023_fcd(self):
        hi04 = RCSection(0.3, 0.5, 0.05, 45.0, FYK, edition="2004")
        hi23 = RCSection(0.3, 0.5, 0.05, 45.0, FYK, edition="2023", k_tc=0.85)
        self.assertGreater(hi23.flexural_reinforcement(300.0)["As_bot"],
                           hi04.flexural_reinforcement(300.0)["As_bot"])
        # same stress block up to C50/60 and fcd: identical steel
        same = RCSection(0.3, 0.5, 0.05, FCK, FYK, edition="2023")
        self.assertAlmostEqual(same.flexural_reinforcement(150.0)["As_bot"],
                               self.s04.flexural_reinforcement(150.0)["As_bot"])

    def test_shear_is_the_2023_check(self):
        r = self.s23.shear_reinforcement(150.0, 0.0, As_long=0.0015)
        ref = uls2023.eurocode2_shear_check(uls2023.ShearInput(
            b=0.3, d=0.45, fck=FCK, fyk=FYK, as_long=0.0015), 150.0)
        self.assertAlmostEqual(r["Asw_s"], ref.asw_s)
        self.assertAlmostEqual(r["VRd_max"], ref.vrd_max)
        self.assertEqual(r["mode"], "stirrups")
        self.assertEqual(r["cot"], 2.5)
        old = self.s04.shear_reinforcement(150.0, 0.0, As_long=0.0015)
        # the same stirrups at cot θ = 2.5, but another strut resistance (ν 0.5 vs 0.6(1−fck/250))
        self.assertNotAlmostEqual(r["VRd_max"], old["VRd_max"], places=3)
        # near the strut limit the angle and the stirrups differ
        r2 = self.s23.shear_reinforcement(450.0, 0.0, As_long=0.0015)
        o2 = self.s04.shear_reinforcement(450.0, 0.0, As_long=0.0015)
        self.assertLess(r2["cot"], 2.5)
        self.assertNotAlmostEqual(r2["Asw_s"], o2["Asw_s"], places=7)

    def test_low_shear_keeps_the_contract_of_the_no_reinforcement_branch(self):
        r = self.s23.shear_reinforcement(20.0, 0.0, As_long=0.0015)
        self.assertEqual(r["mode"], "no_shear_reinf")
        self.assertEqual(r["mode_2023"], "no_verification")
        self.assertIsNotNone(r["VRd_s"])
        self.assertAlmostEqual(r["Asw_s"], uls2023.detailing.calc_asw_min_s(FCK, FYK, 0.3))
        off = self.s23.shear_reinforcement(20.0, 0.0, As_long=0.0015, min_shear=False)
        self.assertEqual(off["Asw_s"], 0.0)

    def test_axial_force_enters_the_shear_resistance_with_the_moment(self):
        kw = {"As_long": 0.0015}
        self.assertEqual(self.s23.shear_reinforcement(95.0, 0.0, **kw)["mode"], "stirrups")
        # compression with the moment (shear span, 8.2.2(3)-(4)) raises VRd,c: no stirrups
        comp = self.s23.shear_reinforcement(95.0, -400.0, Med=40.0, **kw)
        self.assertEqual(comp["mode"], "no_shear_reinf")
        # without the moment the axial force is ignored (as in 2004)
        self.assertEqual(self.s23.shear_reinforcement(95.0, -400.0, **kw)["mode"], "stirrups")
        # tension lowers VRd,c
        self.assertEqual(self.s23.shear_reinforcement(72.0, 0.0, **kw)["mode"], "no_shear_reinf")
        self.assertEqual(self.s23.shear_reinforcement(72.0, 800.0, Med=40.0, **kw)["mode"],
                         "stirrups")

    def test_torsion_is_clause_8_3(self):
        t23 = self.s23.torsion_reinforcement(30.0, 2.0)
        ref = uls2023.calc_torsion(30.0, 0.3, 0.5, FCK, 1.5, FYK, 1.15, 2.0, cover=0.05,
                                   alpha_cc=1.0)
        self.assertAlmostEqual(t23["TRd_max"], ref["TRd_max"])
        self.assertIn("nu", t23)
        t04 = self.s04.torsion_reinforcement(30.0, 2.0)
        self.assertNotIn("nu", t04)
        self.assertNotAlmostEqual(t23["TRd_max"], t04["TRd_max"], places=3)

    def test_shear_torsion_is_the_2023_combination(self):
        r = self.s23.shear_torsion_reinforcement(150.0, 15.0, As_long=0.0015)
        ref = uls2023.eurocode2_shear_torsion_check(uls2023.ShearTorsionInput(
            b=0.3, h=0.5, cover=0.05, fck=FCK, fyk=FYK, as_long=0.0015), 150.0, 15.0)
        self.assertAlmostEqual(r["Asw_s"], ref.asw_total_s)
        self.assertAlmostEqual(r["interaction"], ref.interaction)
        self.assertEqual(r["cot"], ref.cot)
        self.assertTrue(r["passed"])
        self.assertFalse(self.s23.shear_torsion_reinforcement(
            150.0, 30.0, As_long=0.0015)["passed"])    # Annex G ν: T/TRd,max = 0.93


class TestDesignConcreteSections2023(unittest.TestCase):
    def _struc(self, domain="plane"):
        s = Structure2D(domain=domain)
        s.add_node("n1", 0.0, 0.0)
        s.add_node("n2", 5.0, 0.0)
        _material(s)
        s.add_section("CONC", "C", b=0.3, h=0.5)
        s.add_bar_element("E1", "n1", "n2", "CONC")
        return s

    @staticmethod
    def _results(t, v, m):
        return {"element_forces": {}, "combinations": {"ULS": {"element_forces": {
            "E1": {"i": [t, v, m], "j": [-t, -v, -m]}}}}}

    def test_rows_carry_the_edition_and_differ_from_2004(self):
        s, res = self._struc(), self._results(0.0, 450.0, 100.0)
        r04 = design_concrete_sections(s, res)
        r23 = design_concrete_sections(s, res, edition="2023")
        self.assertTrue(all(r["edition"] == "2004" for r in r04))
        self.assertTrue(all(r["edition"] == "2023" for r in r23))
        g04 = next(r for r in r04 if r["governing"])
        g23 = next(r for r in r23 if r["governing"])
        self.assertNotAlmostEqual(g23["VRd_max"], g04["VRd_max"], places=3)
        self.assertAlmostEqual(g23["As_bot"], g04["As_bot"], places=9)     # C30: same flexure

    def test_default_is_unchanged_2004(self):
        s, res = self._struc(), self._results(0.0, 150.0, 100.0)
        self.assertEqual(design_concrete_sections(s, res),
                         design_concrete_sections(s, res, edition="2004"))

    def test_grillage_uses_the_2023_shear_torsion(self):
        s = self._struc(domain="plate")
        res = self._results(t=15.0, v=100.0, m=60.0)
        rows = design_concrete_sections(s, res, edition="2023")
        gov = next(r for r in rows if r["governing"])
        self.assertEqual(gov["kind"], "grillage")
        self.assertEqual(gov["edition"], "2023")
        cover = s.sections["CONC"].rc_cover
        ref = RCSection(0.3, 0.5, cover, FCK, FYK, edition="2023")
        fl = ref.flexural_reinforcement(gov["M_Ed"], 0.0)
        st = ref.shear_torsion_reinforcement(gov["V_Ed"], gov["T_Ed"],
                                             As_long=fl["As_tension"])
        self.assertAlmostEqual(gov["Asw_s"], st["Asw_s"])
        self.assertAlmostEqual(gov["TRd_max"], st["TRd_max"])

    def test_report_title_and_meta_name_the_edition(self):
        s, res = self._struc(), self._results(0.0, 150.0, 100.0)
        rows = design_concrete_sections(s, res, edition="2023", with_reports=True)
        gov = next(r for r in rows if r["governing"])
        self.assertIn("2023", gov["report"]["title"])
        self.assertEqual(gov["report"]["meta"]["edition"], "2023")

    def test_envelope_follows_the_edition(self):
        s = self._struc()
        res = {"combo_distribution": {"ULS": {"E1": {
            "x": [0.0, 2.5, 5.0], "N": [0.0, 0.0, 0.0],
            "V": [450.0, 0.0, -450.0], "M": [0.0, 100.0, 0.0]}}}}
        e04 = reinforcement_envelope(s, res)
        e23 = reinforcement_envelope(s, res, edition="2023")
        self.assertNotAlmostEqual(float(e23["E1"]["Asw_s"][0]),
                                  float(e04["E1"]["Asw_s"][0]), places=7)

    def test_envelope_records_the_governing_combination(self):
        s = self._struc()
        d = lambda M, V: {"E1": {"x": [0.0, 2.5, 5.0], "N": [0.0] * 3,
                                 "V": [V, 0.0, -V], "M": [0.0, M, 0.0]}}
        res = {"combo_distribution": {"LOW": d(50.0, 100.0),
                                      "HIGH": d(150.0, 450.0)}}
        e = reinforcement_envelope(s, res)["E1"]
        self.assertEqual(e["combo_bot"][1], "HIGH")
        self.assertEqual(e["combo_asw"][0], "HIGH")
        self.assertEqual(len(e["combo_top"]), 3)

    def test_crack_envelope_records_the_governing_combination(self):
        from xdfem2d.rc_design import crack_width_envelope
        s = self._struc()
        d = lambda M: {"E1": {"x": [0.0, 2.5, 5.0], "N": [0.0] * 3,
                              "V": [0.0] * 3, "M": [0.0, M, 0.0]}}
        res = {"combo_distribution": {"LOW": d(60.0), "HIGH": d(160.0)}}
        env = reinforcement_envelope(s, res)
        ce = crack_width_envelope(s, res, [], k_t=0.4, rc_envelope=env)["E1"]
        self.assertGreater(ce["wk"][1], 0.0)
        self.assertEqual(ce["combo"][1], "HIGH")
        self.assertEqual(len(ce["combo"]), 3)


def _tri_struc(thickness=0.2, domain="plane", fck=FCK):
    s = Structure2D(domain=domain)
    _material(s, fck)
    s.add_tri_section("T", "C", thickness=thickness,
                      formulation="CST" if domain == "plane" else "DKT")
    for nid, (x, y) in {"A": (0, 0), "B": (1, 0), "C": (0, 1)}.items():
        s.add_node(nid, x, y)
    s.add_tri_element("T1", "A", "B", "C", "T")
    return s


class TestMembranes2023(unittest.TestCase):
    @staticmethod
    def _results(nxx, nyy, nxy, t=0.2):
        return {"combinations": {"ULS": {"tri_stress": {"T1": {
            "sx": nxx / t, "sy": nyy / t, "txy": nxy / t, "formulation": "CST"}}}}}

    def test_nu_and_the_limit_are_reported(self):
        s = _tri_struc()
        res = self._results(-1500.0, -200.0, 800.0)
        r04 = design_concrete_planes(s, res)[0]
        r23 = design_concrete_planes(s, res, edition="2023")[0]
        self.assertAlmostEqual(r04["nu"], 0.6 * (1 - 30.0 / 250.0))
        self.assertAlmostEqual(r23["nu"], 0.4)
        self.assertAlmostEqual(r23["f_lim"], 0.4 * r23["fcd"])
        self.assertFalse(r04["crushing"])                     # 9.6 MPa < 0.528·20
        self.assertTrue(r23["crushing"])                      # 9.6 MPa > 0.4·20
        self.assertEqual((r04["edition"], r23["edition"]), ("2004", "2023"))

    def test_reinforcement_follows_table_g1(self):
        r = design_concrete_planes(_tri_struc(), self._results(100.0, 50.0, 20.0),
                                   edition="2023")[0]
        fyd = FYK * 1e3 / 1.15
        self.assertAlmostEqual(r["Asx"], (100.0 + 20.0) / fyd)
        self.assertAlmostEqual(r["Asy"], (50.0 + 20.0) / fyd)
        self.assertAlmostEqual(r["theta"], 1.0)

    def test_strain_nu_mode_and_no_reinforcement_case(self):
        s = _tri_struc()
        strain = design_concrete_planes(s, self._results(-1500.0, -200.0, 800.0),
                                        edition="2023", nu_mode="strain")[0]
        self.assertGreater(strain["nu"], 0.4)
        both = design_concrete_planes(s, self._results(-400.0, -300.0, 50.0),
                                      edition="2023")[0]
        self.assertEqual((both["Asx"], both["Asy"], both["nu"]), (0.0, 0.0, 1.0))
        self.assertIsNone(both["theta"])                      # undefined without reinforcement

    def test_2004_strut_model_option(self):
        s = _tri_struc()
        res = self._results(-1500.0, -200.0, 800.0)
        a = design_concrete_planes(s, res)[0]
        b = design_concrete_planes(s, res, strut_model="6.109")[0]
        self.assertAlmostEqual(a["nu"], 0.528)                  # EN 1992-1-1 Annex F.1(4)
        self.assertAlmostEqual(b["nu"], 0.4837, places=3)       # EN 1992-2 §6.109, Δθ = 2.6°
        self.assertEqual(a["Asy"], b["Asy"])                    # the reinforcement is the same
        pay = design_tri_and_store(s, res, prefs={"ec2_strut_model": "6.109"})
        self.assertAlmostEqual(pay["rows"][0]["nu"], 0.4837, places=3)

    def test_theta_is_none_where_it_is_undefined_in_2004_too(self):
        r = design_concrete_planes(_tri_struc(), self._results(-400.0, -300.0, 50.0))[0]
        self.assertIsNone(r["theta"])
        self.assertTrue(math.isfinite(design_concrete_planes(
            _tri_struc(), self._results(100.0, 50.0, 20.0))[0]["theta"]))

    def test_report_for_the_governing_row(self):
        r = design_concrete_planes(_tri_struc(), self._results(100.0, 50.0, 20.0),
                                   edition="2023", with_reports=True)[0]
        self.assertIn("Annex G", r["report"]["title"])

    def test_design_tri_and_store_reads_the_edition_from_prefs(self):
        s = _tri_struc()
        res = self._results(-1500.0, -200.0, 800.0)
        payload = design_tri_and_store(s, res, prefs={"ec2_edition": "2023"})
        self.assertEqual(payload["rows"][0]["edition"], "2023")
        self.assertTrue(payload["rows"][0]["crushing"])
        payload = design_tri_and_store(s, res, prefs={"ec2_edition": "2023",
                                                      "ec2_nu_mode": "strain"})
        self.assertGreater(payload["rows"][0]["nu"], 0.4)


class TestSlabs2023(unittest.TestCase):
    @staticmethod
    def _results(mx, my, mxy, **wa):
        d = {"mx": mx, "my": my, "mxy": mxy, "formulation": "DKT",
             "mx_bot": wa.get("mx_bot", 0.0), "my_bot": wa.get("my_bot", 0.0),
             "mx_top": wa.get("mx_top", 0.0), "my_top": wa.get("my_top", 0.0),
             "vx": wa.get("vx", 0.0), "vy": wa.get("vy", 0.0)}
        return {"combinations": {"ULS": {"tri_stress": {"T1": d}}}}

    def test_2023_uses_tables_g2_g3_on_the_element_moments(self):
        s = _tri_struc(0.2, domain="plate")
        row = design_concrete_slabs(s, self._results(50.0, 30.0, 10.0), edition="2023")[0]
        self.assertEqual(row["edition"], "2023")
        self.assertAlmostEqual(row["mx_bot"], 60.0)          # mx + |mxy|
        self.assertAlmostEqual(row["my_bot"], 40.0)
        self.assertEqual((row["mx_top"], row["my_top"]), (0.0, 0.0))
        covers = s.tri_sections["T"].resolved_covers()
        ref = uls2023.slab_design.eurocode2_slab_check(uls2023.slab_design.SlabInput(
            m_x=50.0, m_y=30.0, m_xy=10.0, fck=FCK, fyk=FYK, thickness=0.2,
            cover_bot_x=covers["bot_x"], cover_bot_y=covers["bot_y"],
            cover_top_x=covers["top_x"], cover_top_y=covers["top_y"]))
        self.assertAlmostEqual(row["Asx_bot"], ref.asx_bot)
        self.assertAlmostEqual(row["Asy_top"], ref.asy_top)
        self.assertEqual(row["note"], "")

    def test_2004_still_uses_the_wood_armer_moments(self):
        s = _tri_struc(0.2, domain="plate")
        res = self._results(50.0, 30.0, 10.0, mx_bot=60.0, my_bot=40.0)
        row = design_concrete_slabs(s, res)[0]
        self.assertEqual((row["mx_bot"], row["my_bot"]), (60.0, 40.0))
        self.assertEqual(row["edition"], "2004")
        self.assertAlmostEqual(row["Asx_bot"], design_concrete_slabs(
            s, res, edition="2023")[0]["Asx_bot"], places=9)  # same strips, same d

    def test_hogging_gives_top_steel_and_a_face_without_tension_gets_zero(self):
        s = _tri_struc(0.2, domain="plate")
        row = design_concrete_slabs(s, self._results(-40.0, -20.0, 5.0), edition="2023")[0]
        self.assertAlmostEqual(row["mx_top"], -45.0)
        self.assertGreater(row["Asx_top"], 0.0)
        self.assertEqual((row["Asx_bot"], row["Asy_bot"]), (0.0, 0.0))     # no minimum

    def test_warnings_of_the_checks_go_to_the_row_note(self):
        s = _tri_struc(0.2, domain="plate")
        with warnings.catch_warnings():
            warnings.simplefilter("error")                     # nothing leaks as a Python warning
            row = design_concrete_slabs(s, self._results(0.0, 0.0, 60.0), edition="2023",
                                        slab_method="optimum")[0]
        self.assertIn("0,07", row["note"])
        over = design_concrete_slabs(
            s, self._results(0.0, 0.0, 0.0, mx_bot=200.0, my_bot=10.0))[0]
        self.assertIn("over-reinforced", over["note"])
        none = design_concrete_slabs(
            s, self._results(0.0, 0.0, 0.0, mx_bot=400.0, my_bot=10.0))[0]
        self.assertIn("exceeds the capacity", none["note"])
        self.assertTrue(math.isnan(none["Asx_bot"]))

    def test_2004_informative_strut_note_does_not_change_the_steel(self):
        s = _tri_struc(0.2, domain="plate")
        # Wood-Armer moments of mx = my = 0, mxy = 60 (m* = |mxy|)
        wa = dict(mx_bot=60.0, my_bot=60.0, mx_top=-60.0, my_top=-60.0)
        heavy = design_concrete_slabs(s, self._results(0.0, 0.0, 60.0, **wa))[0]
        self.assertTrue(heavy["strut_warning"])
        self.assertIn("Wood-Armer does not check the concrete strut", heavy["note"])
        self.assertIn("informative", heavy["note"])
        self.assertFalse(heavy["crushing"])                  # reserved for the normative check
        self.assertEqual(heavy["method"], "wood-armer")
        # the steel is exactly the Wood-Armer design, with or without the note
        from eurocodepy.ec2.uls import SlabInput, eurocode2_slab_check
        c = s.tri_sections["T"].resolved_covers()
        ref = eurocode2_slab_check(SlabInput(
            mx_bot=60.0, my_bot=60.0, mx_top=-60.0, my_top=-60.0, fck=FCK, fyk=FYK,
            thickness=0.2, cover_bot_x=c["bot_x"], cover_bot_y=c["bot_y"],
            cover_top_x=c["top_x"], cover_top_y=c["top_y"]))
        self.assertAlmostEqual(heavy["Asx_bot"], ref.asx_bot, places=12)
        self.assertAlmostEqual(heavy["Asy_top"], ref.asy_top, places=12)
        light = design_concrete_slabs(s, self._results(0.0, 0.0, 10.0, mx_bot=10.0, my_bot=10.0,
                                                       mx_top=-10.0, my_top=-10.0))[0]
        self.assertFalse(light["strut_warning"])
        self.assertEqual(light["note"], "")
        none = design_concrete_slabs(s, self._results(5.0, 5.0, 0.0, mx_bot=5.0, my_bot=5.0))[0]
        self.assertFalse(none["strut_warning"])
        # 2023 handles it normatively instead (sandwich + crushing), no informative flag
        e23 = design_concrete_slabs(s, self._results(0.0, 0.0, 60.0), edition="2023")[0]
        self.assertFalse(e23["strut_warning"])

    def test_sandwich_model_where_the_tables_do_not_apply(self):
        s = _tri_struc(0.2, domain="plate")
        res = self._results(50.0, 30.0, 45.0)               # |mxy| ≥ 0.07·d²·fcd
        row = design_concrete_slabs(s, res, edition="2023")[0]
        self.assertEqual(row["method"], "sandwich")
        t = 2 * sum(s.tri_sections["T"].resolved_covers().values()) / 4
        z = 0.2 - t
        fyd = FYK * 1e3 / 1.15
        # the sandwich layers carry mRd = ftd·t·z: Tables G.2/G.3 values for nEd = 0
        self.assertAlmostEqual(row["mx_bot"], 50.0 + 45.0, places=6)
        self.assertAlmostEqual(row["Asx_bot"], row["mx_bot"] / z / fyd, places=9)
        self.assertIn("crushing", row)
        # forced tables / forced sandwich, and the inside-the-range default
        forced = design_concrete_slabs(s, self._results(50.0, 30.0, 10.0), edition="2023",
                                       slab_method="sandwich")[0]
        self.assertEqual(forced["method"], "sandwich")
        self.assertEqual(design_concrete_slabs(s, self._results(50.0, 30.0, 10.0),
                                               edition="2023")[0]["method"], "optimum")
        self.assertEqual(design_concrete_slabs(s, self._results(50.0, 30.0, 10.0,
                                               mx_bot=60.0, my_bot=40.0))[0]["method"],
                         "wood-armer")

    def test_sandwich_strut_crushing_is_flagged_and_noted(self):
        s = _tri_struc(0.2, domain="plate")
        row = design_concrete_slabs(s, self._results(60.0, 60.0, 190.0), edition="2023")[0]
        self.assertTrue(row["crushing"])
        self.assertIn("exceeds ν·f_cd", row["note"])
        strain = design_concrete_slabs(s, self._results(60.0, 60.0, 190.0), edition="2023",
                                       nu_mode="strain")[0]
        self.assertEqual(strain["Asx_bot"], row["Asx_bot"])  # ν changes the verdict, not the steel

    def test_prefs_choose_the_method_and_nu(self):
        s = _tri_struc(0.2, domain="plate")
        res = self._results(50.0, 30.0, 10.0)
        pay = design_tri_and_store(s, res, prefs={"ec2_edition": "2023",
                                                  "ec2_slab_method": "sandwich"})
        self.assertEqual(pay["rows"][0]["method"], "sandwich")

    def test_out_of_plane_shear_2004_inner_layer_of_annex_ll(self):
        s = _tri_struc(0.2, domain="plate")
        wa = dict(mx_bot=30.0, my_bot=10.0)
        base = design_concrete_slabs(s, self._results(30.0, 10.0, 0.0, **wa))[0]
        self.assertEqual((base["shear_mode"], base["asw_s"], base["v_ed"]), ("none", 0.0, 0.0))
        low = design_concrete_slabs(s, self._results(30.0, 10.0, 0.0, vx=40.0, **wa))[0]
        self.assertEqual((low["shear_mode"], low["asw_s"]), ("no_shear_reinf", 0.0))
        r = design_concrete_slabs(s, self._results(30.0, 10.0, 0.0, vx=250.0, vy=0.0, **wa))[0]
        self.assertEqual((r["shear_mode"], r["v_ed"]), ("stirrups", 250.0))
        self.assertGreater(r["asw_s"], 0.0)
        self.assertIn("chord force", r["note"])
        self.assertAlmostEqual(r["Asx_bot"], base["Asx_bot"], places=12)       # Wood-Armer bars unchanged
        self.assertFalse(r["crushing"])
        crush = design_concrete_slabs(s, self._results(30.0, 10.0, 0.0, vx=900.0, **wa))[0]
        self.assertTrue(crush["crushing"])
        self.assertEqual(crush["shear_mode"], "strut_crushing")
        # the reinforcement of the shear is that of the beam of unit width (6.2.3), d = h − larger cover
        from eurocodepy.ec2.uls import ShearInput, eurocode2_shear_check
        c = s.tri_sections["T"].resolved_covers()
        d = 0.2 - max(c.values())
        ref = eurocode2_shear_check(ShearInput(b=1.0, d=d, fck=FCK, fyk=FYK,
                                               as_long=base["Asx_bot"] / d * d, min_shear=False), 250.0)
        self.assertAlmostEqual(r["asw_s"], ref.asw_s, places=10)

    def test_out_of_plane_shear_2023_with_the_chord_force(self):
        s = _tri_struc(0.2, domain="plate")
        base = design_concrete_slabs(s, self._results(30.0, 10.0, 0.0), edition="2023")[0]
        r = design_concrete_slabs(s, self._results(30.0, 10.0, 0.0, vx=250.0), edition="2023")[0]
        self.assertEqual((r["shear_mode"], r["method"]), ("stirrups", "sandwich"))
        self.assertGreater(r["asw_s"], 0.0)
        self.assertGreater(r["Asx_top"], base["Asx_top"])           # the truss force n_c/2 in both layers
        self.assertGreater(r["Asx_bot"], base["Asx_bot"])
        self.assertEqual(base["shear_mode"], "none")
        crush = design_concrete_slabs(s, self._results(0.0, 0.0, 0.0, vx=900.0), edition="2023")[0]
        self.assertTrue(crush["crushing"])
        self.assertNotIn("chord force", r["note"])

    def test_design_tri_and_store_dispatches_the_edition(self):
        s = _tri_struc(0.2, domain="plate")
        payload = design_tri_and_store(s, self._results(50.0, 30.0, 10.0),
                                       prefs={"ec2_edition": "2023"})
        self.assertEqual(payload["rows"][0]["edition"], "2023")


class TestPunching2023(unittest.TestCase):
    def _slab(self, n=4, L=6.0, t=0.25):
        s = Structure2D(domain="plate")
        _material(s)
        s.add_quad_section("S", "C", thickness=t, formulation="DKT4",
                           rc_cover_bot_x=0.035, rc_cover_bot_y=0.045,
                           rc_cover_top_x=0.035, rc_cover_top_y=0.045)
        ids = {}
        for j in range(n + 1):
            for i in range(n + 1):
                nid = f"N{i}_{j}"
                ids[(i, j)] = nid
                s.add_node(nid, L * i / n, L * j / n)
        e = 0
        for j in range(n):
            for i in range(n):
                s.add_quad_element(f"Q{e}", ids[(i, j)], ids[(i + 1, j)],
                                   ids[(i + 1, j + 1)], ids[(i, j + 1)], "S")
                e += 1
        s.add_support("COL", w=True)
        s.assign_support(ids[(2, 2)], "COL")
        s.add_load_case("G", self_weight_factor=0.0)
        for qid in list(s.quad_elements_by_id):
            s.add_area_load(qid, "G", pz=-20.0)
        s.add_load_combination("ULS", {"G": 1.35})
        s.add_punch_column("C1", ids[(2, 2)], shape="rectangular", bx=0.4, by=0.4,
                           position="center")
        return s

    def test_2023_parameters_come_from_the_edition_data_and_the_annex(self):
        s = self._slab()
        res = s.calculate()
        base = design_punching(s, res, edition="2023", gamma_v=None, dmax=None,
                               eta_sys=None)[0]
        explicit = design_punching(s, res, edition="2023")[0]
        self.assertAlmostEqual(base["v_Rdc"], explicit["v_Rdc"])
        ec.register_annex({"country": "XP", "editions": {"ec2": {"2023": {
            "partial_factors": {"materials": {"persistent": {"gamma_v": 1.6}}},
            "punch_params": {"dmax": 16.0, "eta_sys": 1.2}}}}})
        try:
            pat = design_punching(s, res, edition="2023", gamma_v=None, dmax=None,
                                  eta_sys=None, annex="XP")[0]
        finally:
            ec.unregister_annex("XP")
        self.assertLess(pat["v_Rdc"], base["v_Rdc"])          # γv 1.6 and d_dg 32 mm
        self.assertAlmostEqual(pat["v_Rd_max"], 1.2 * pat["v_Rdc"])

    def test_2004_ignores_the_none_defaults(self):
        s = self._slab()
        res = s.calculate()
        a = design_punching(s, res)[0]
        b = design_punching(s, res, gamma_v=None, dmax=None, eta_sys=None)[0]
        self.assertEqual(a["v_Rdc"], b["v_Rdc"])


if __name__ == "__main__":
    unittest.main()

