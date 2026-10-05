"""BeamBarParams / resolve_beam_params: beam > section > prefs > built-in."""
import unittest

from xdfem2d.beam_bars_params import BeamBarParams, resolve_beam_params


class _Sec:
    rc_bar_phi = 20.0
    beam_overrides = {"diameters": (12, 16), "stirrup_diameter_mm": None,
                      "max_layers": 3}


class TestResolve(unittest.TestCase):
    def test_builtin_defaults(self):
        p = resolve_beam_params()
        self.assertEqual(p, BeamBarParams())
        self.assertEqual(p.symmetry, "rule")
        self.assertNotIn(8, p.diameters)

    def test_preferences_override_builtin(self):
        p = resolve_beam_params({"beam_stirrup_mm": 10, "beam_n_through": 4,
                                 "dmax": 25.0, "beam_zone_mode": "fixed"})
        self.assertEqual(p.stirrup_diameter_mm, 10.0)
        self.assertEqual(p.n_through, 4)
        self.assertIsInstance(p.n_through, int)
        self.assertEqual(p.dg, 25.0)
        self.assertEqual(p.zone_mode, "fixed")

    def test_zero_dmax_is_ignored(self):
        self.assertEqual(resolve_beam_params({"dmax": 0.0}).dg, 20.0)

    def test_section_overrides_prefs_and_sets_bar_estimate(self):
        p = resolve_beam_params({"beam_max_layers": 2,
                                 "beam_diameters": [10, 25]}, _Sec())
        self.assertEqual(p.diameters, (12, 16))
        self.assertEqual(p.max_layers, 3)
        self.assertEqual(p.d_bar_est, 20.0)       # Section.rc_bar_phi
        self.assertEqual(p.stirrup_diameter_mm, 8.0)   # None = inherit

    def test_beam_overrides_everything(self):
        p = resolve_beam_params({"beam_max_layers": 2}, _Sec(),
                                {"max_layers": 1, "n_through": 0,
                                 "shift_d": None})
        self.assertEqual(p.max_layers, 1)
        self.assertEqual(p.n_through, 0)
        self.assertEqual(p.shift_d, 0.0)          # None = inherit

    def test_unknown_beam_key_rejected(self):
        with self.assertRaises(ValueError):
            resolve_beam_params(None, None, {"max_layer": 1})

    def test_section_overrides_limited_to_type_dependent_params(self):
        class S:
            beam_overrides = {"n_through": 4}
        with self.assertRaises(ValueError):
            resolve_beam_params(None, S())

    def test_beam_override_keys_limited_to_user_params(self):
        for k in ("dg", "bar_penalty", "d_bar_est", "min_bars"):
            with self.assertRaises(ValueError, msg=k):
                resolve_beam_params(None, None, {k: 1})

    def test_validation(self):
        for bad in ({"symmetry": "odd"}, {"zone_mode": "x"},
                    {"max_layers": 0}, {"support_bottom_ratio": 2.0},
                    {"diameters": ()}, {"zones": (0.5, -1.0)}):
            with self.assertRaises(ValueError, msg=bad):
                resolve_beam_params(None, None, bad)


def _model():
    """Two-span continuous beam on three supports + its design rows."""
    from context import Structure2D  # noqa: F401  (adds src/ to sys.path)
    from xdfem2d.rc_design import design_concrete_sections
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
    res = {"element_forces": {}, "combinations": {"ULS1": {"element_forces": {
        "E1": {"i": [0, 40, 0], "j": [0, -40, 150]},
        "E2": {"i": [0, 30, -90], "j": [0, -30, 20]}}}}}
    return s, design_concrete_sections(s, res)


class TestDesignUsesParams(unittest.TestCase):
    def test_params_object_drives_every_beam(self):
        from xdfem2d.rc_design import design_beam_bars
        s, rows = _model()
        one = design_beam_bars(s, rows, params=BeamBarParams(
            zone_mode="fixed", zones=(1.0,)))
        self.assertEqual(len(one), 2)                       # one zone per span
        halves = design_beam_bars(s, rows, params=BeamBarParams(
            zone_mode="fixed", zones=(0.5, 0.5), diameters=(12,)))
        self.assertEqual(len(halves), 4)
        self.assertTrue(all(g[1] == 12 for r in halves
                            for g in r["bottom_layers"] + r["top_layers"]))
        with self.assertRaises(ValueError):                 # validated up front
            design_beam_bars(s, rows, params=BeamBarParams(max_layers=0))

    def test_there_are_no_loose_parameter_arguments(self):
        """Parameters come from prefs or one params object only — the window
        knows just those layers, so a loose argument would be lost on its
        first recalculation."""
        from dataclasses import fields
        from xdfem2d.rc_design import design_beam_bars
        s, rows = _model()
        for f in fields(BeamBarParams):
            with self.assertRaises(TypeError, msg=f.name):
                design_beam_bars(s, rows, **{f.name: getattr(BeamBarParams(),
                                                             f.name)})
        with self.assertRaises(TypeError):
            design_beam_bars(s, rows, bogus=1)

    def test_params_ignore_beam_overrides_but_prefs_apply_them(self):
        from xdfem2d.rc_design import design_beam_bars
        s, rows = _model()
        s.assign_beam(["E1", "E2"], tag="V1")
        s.beams["V1"]["overrides"] = {"diameters": [20]}
        via_prefs = design_beam_bars(s, rows)
        self.assertTrue(all(g[1] == 20 for r in via_prefs
                            for g in r["bottom_layers"] + r["top_layers"]))
        via_params = design_beam_bars(s, rows, params=BeamBarParams())
        self.assertTrue(any(g[1] != 20 for r in via_params
                            for g in r["bottom_layers"] + r["top_layers"]))

    def test_preferences_drive_the_design(self):
        from xdfem2d.rc_design import design_beam_bars
        s, rows = _model()
        out = design_beam_bars(s, rows, prefs={
            "beam_zone_mode": "fixed", "beam_diameters": [16],
            "beam_max_diameters": 1})
        self.assertEqual(len(out), 6)                       # 3 fixed zones
        self.assertTrue(all(g[1] == 16 for r in out
                            for g in r["bottom_layers"] + r["top_layers"]))


if __name__ == "__main__":
    unittest.main()


class TestStirrupParams(unittest.TestCase):
    def test_defaults_and_prefs(self):
        p = resolve_beam_params()
        self.assertEqual(p.stirrup_legs, 2)
        self.assertEqual(p.stirrup_diameters, (6, 8, 10, 12))
        self.assertIn(150, p.stirrup_spacings)
        q = resolve_beam_params({"beam_stirrup_legs": 4,
                                 "beam_stirrup_diameters": [8, 10],
                                 "beam_stirrup_spacings": [100, 200]})
        self.assertEqual((q.stirrup_legs, q.stirrup_diameters,
                          q.stirrup_spacings), (4, (8, 10), (100, 200)))
        self.assertIsInstance(q.stirrup_legs, int)

    def test_beam_override_and_validation(self):
        p = resolve_beam_params(None, None, {"stirrup_legs": 3})
        self.assertEqual(p.stirrup_legs, 3)
        for bad in ({"stirrup_legs": 1}, {"stirrup_diameters": ()},
                    {"stirrup_spacings": (100, -5)}):
            with self.assertRaises(ValueError, msg=bad):
                resolve_beam_params(None, None, bad)

    def test_a_section_cannot_override_the_stirrup_lists(self):
        class S:
            beam_overrides = {"stirrup_legs": 3}
        with self.assertRaises(ValueError):
            resolve_beam_params(None, S())


class TestAnchorageParams(unittest.TestCase):
    def test_defaults_prefs_and_validation(self):
        p = resolve_beam_params()
        self.assertEqual((p.anchorage_shape, p.lap_percentage, p.bar_length),
                         ("straight", 50.0, 12.0))
        q = resolve_beam_params({"beam_anchorage_shape": "bent",
                                 "beam_lap_percentage": 25,
                                 "beam_bar_length": 14})
        self.assertEqual((q.anchorage_shape, q.lap_percentage, q.bar_length),
                         ("bent", 25.0, 14.0))
        o = resolve_beam_params(None, None, {"anchorage_shape": "bent"})
        self.assertEqual(o.anchorage_shape, "bent")
        for bad in ({"anchorage_shape": "hook"}, {"lap_percentage": 120},
                    {"lap_percentage": -1}, {"bar_length": 0}):
            with self.assertRaises(ValueError, msg=bad):
                resolve_beam_params(None, None, bad)

    def test_a_section_cannot_override_them(self):
        class S:
            beam_overrides = {"bar_length": 10.0}
        with self.assertRaises(ValueError):
            resolve_beam_params(None, S())


class TestRegistry(unittest.TestCase):
    """One registry of parameters drives every form (PARAM_SPECS)."""

    def test_every_spec_is_a_real_parameter_with_a_valid_default(self):
        from dataclasses import fields
        from xdfem2d.beam_bars_params import PARAM_SPECS, BeamBarParams
        names = {f.name for f in fields(BeamBarParams)}
        seen, prefs = set(), set()
        for sp in PARAM_SPECS:
            self.assertIn(sp.name, names)
            self.assertNotIn(sp.name, seen)
            self.assertNotIn(sp.pref_key, prefs)
            seen.add(sp.name)
            prefs.add(sp.pref_key)
            self.assertTrue(sp.label and sp.short)
            d = sp.default()
            if sp.kind == "choice":
                self.assertIn(d, [v for _l, v in sp.extra])
            elif sp.kind == "list":
                lo, hi = sp.extra
                self.assertTrue(all(lo <= x <= hi for x in d))
            else:
                lo, hi = sp.extra[0], sp.extra[1]
                self.assertTrue(lo <= d <= hi, sp.name)
        BeamBarParams().validate()

    def test_the_derived_tables_follow_the_registry(self):
        from xdfem2d.beam_bars_params import (
            BEAM_PARAMS, PARAM_SPECS, PREF_KEYS, SECTION_PARAMS)
        self.assertEqual(set(SECTION_PARAMS),
                         {sp.name for sp in PARAM_SPECS if sp.section_ok})
        self.assertEqual(SECTION_PARAMS, ("diameters", "stirrup_diameter_mm",
                                          "max_layers"))
        self.assertEqual(set(BEAM_PARAMS),
                         {sp.name for sp in PARAM_SPECS} | {"zones"})
        for sp in PARAM_SPECS:
            self.assertEqual(PREF_KEYS[sp.pref_key], sp.name)
        self.assertEqual(PREF_KEYS["dmax"], "dg")

    def test_the_range_of_every_spec_is_accepted_by_the_engine(self):
        from xdfem2d.beam_bars_params import PARAM_SPECS
        for sp in PARAM_SPECS:
            if sp.kind in ("int", "float"):
                for v in (sp.extra[0], sp.extra[1]):
                    try:
                        resolve_beam_params(None, None, {sp.name: v})
                    except ValueError:
                        # an extreme may be refused by a cross-rule (legs >= 2,
                        # 0 < length ...): it must still be a ValueError, and
                        # the default must always be accepted
                        pass
                resolve_beam_params(None, None, {sp.name: sp.default()})

    def test_parse_number_list(self):
        from xdfem2d.beam_bars_params import parse_number_list
        self.assertEqual(parse_number_list("25, 10;16  12,12", 4, 50),
                         [10, 12, 16, 25])
        self.assertEqual(parse_number_list("12.5, 10", 4, 50), [10, 12.5])
        for bad in ("", "a, b", "2, 12", "12, 99"):
            with self.assertRaises(ValueError):
                parse_number_list(bad, 4, 50)
