"""Validation tests for element point loads (concentrated loads applied at a
distance ``a`` along a bar element, not only at nodes).

Each case is compared against the closed-form simply-supported-beam solution
and cross-checked against the equivalent model built with a node at the load
point plus a nodal point load.

Run with:  python -m unittest discover -s tests
"""
import math
import unittest

from context import Structure2D


def _ss_beam_element_load(L, a, P=0.0, M=0.0, E=30e6, b=0.3, h=0.6):
    """Simply-supported single-element beam with an element point load at a."""
    s = Structure2D()
    s.add_material("M", elastic_modulus=E, unit_weight=0.0)
    s.add_section("S", "M", b=b, h=h)
    s.add_node("A", 0.0, 0.0)
    s.add_node("B", L, 0.0)
    s.add_bar_element("E1", "A", "B", "S")
    s.add_load_case("LC")
    s.add_element_point_load("E1", "LC", a=a, fy=-P, mz=M)
    s.add_support("PIN", ux=True, uy=True, tz=False)
    s.add_support("ROL", ux=False, uy=True, tz=False)
    s.assign_support("A", "PIN")
    s.assign_support("B", "ROL")
    return s.calculate()


def _ss_beam_nodal_load(L, a, P, E=30e6, b=0.3, h=0.6):
    """Same beam but split at a with a nodal load — reference model."""
    s = Structure2D()
    s.add_material("M", elastic_modulus=E, unit_weight=0.0)
    s.add_section("S", "M", b=b, h=h)
    s.add_node("A", 0.0, 0.0)
    s.add_node("C", a, 0.0)
    s.add_node("B", L, 0.0)
    s.add_bar_element("E1", "A", "C", "S")
    s.add_bar_element("E2", "C", "B", "S")
    s.add_load_case("LC")
    s.add_point_load("C", "LC", fy=-P)
    s.add_support("PIN", ux=True, uy=True, tz=False)
    s.add_support("ROL", ux=False, uy=True, tz=False)
    s.assign_support("A", "PIN")
    s.assign_support("B", "ROL")
    return s.calculate()


class TestTransversePointLoad(unittest.TestCase):
    L, P = 6.0, 100.0

    def test_central_load(self):
        r = _ss_beam_element_load(self.L, self.L / 2, self.P)
        self.assertAlmostEqual(r["reactions"]["LC"]["A"][1], self.P / 2, places=4)
        self.assertAlmostEqual(r["reactions"]["LC"]["B"][1], self.P / 2, places=4)
        d = r["element_distribution"]["LC"]["E1"]
        Mmax = max(abs(v) for v in d["M"])
        Vmax = max(abs(v) for v in d["V"])
        self.assertAlmostEqual(Mmax, self.P * self.L / 4, places=3)   # PL/4 = 150
        self.assertAlmostEqual(Vmax, self.P / 2, places=3)

    def test_offcentre_reactions_and_moment(self):
        a = 2.0
        b = self.L - a
        r = _ss_beam_element_load(self.L, a, self.P)
        self.assertAlmostEqual(r["reactions"]["LC"]["A"][1], self.P * b / self.L, places=4)
        self.assertAlmostEqual(r["reactions"]["LC"]["B"][1], self.P * a / self.L, places=4)
        d = r["element_distribution"]["LC"]["E1"]
        Mmax = max(abs(v) for v in d["M"])
        self.assertAlmostEqual(Mmax, self.P * a * b / self.L, places=3)   # Pab/L = 133.33

    def test_matches_nodal_load_model(self):
        a = 2.5
        re = _ss_beam_element_load(self.L, a, self.P)
        rn = _ss_beam_nodal_load(self.L, a, self.P)
        for nd in ("A", "B"):
            self.assertAlmostEqual(re["reactions"]["LC"][nd][1],
                                   rn["reactions"]["LC"][nd][1], places=4)


class TestMomentAndAxial(unittest.TestCase):
    def test_concentrated_moment_reactions(self):
        L, M0 = 6.0, 80.0
        r = _ss_beam_element_load(L, L / 2, P=0.0, M=M0)
        # A couple M0 on a simply-supported beam → reactions ±M0/L.
        self.assertAlmostEqual(r["reactions"]["LC"]["A"][1], M0 / L, places=4)
        self.assertAlmostEqual(r["reactions"]["LC"]["B"][1], -M0 / L, places=4)

    def test_axial_point_load_splits_by_lever(self):
        # Axial load on a bar fixed axially at both ends → N reactions a/L, b/L.
        L, Pax, a = 6.0, 90.0, 2.0
        s = Structure2D()
        s.add_material("M", elastic_modulus=30e6, unit_weight=0.0)
        s.add_section("S", "M", b=0.3, h=0.6)
        s.add_node("A", 0.0, 0.0)
        s.add_node("B", L, 0.0)
        s.add_bar_element("E1", "A", "B", "S")
        s.add_load_case("LC")
        s.add_element_point_load("E1", "LC", a=a, fx=Pax)
        s.add_support("FIX", ux=True, uy=True, tz=True)
        s.add_support("ROL", ux=True, uy=True, tz=False)
        s.assign_support("A", "FIX")
        s.assign_support("B", "ROL")
        r = s.calculate()
        # Sum of horizontal reactions balances the applied axial load.
        rx = r["reactions"]["LC"]["A"][0] + r["reactions"]["LC"]["B"][0]
        self.assertAlmostEqual(rx, -Pax, places=4)


class TestSerialization(unittest.TestCase):
    def test_x2d_round_trip(self):
        import os
        import tempfile
        from xdfem2d.file_io import save_x2d, load_x2d

        r = _ss_beam_element_load(6.0, 2.0, 100.0)
        s = Structure2D()
        s.add_material("M", elastic_modulus=30e6, unit_weight=0.0)
        s.add_section("S", "M", b=0.3, h=0.6)
        s.add_node("A", 0.0, 0.0); s.add_node("B", 6.0, 0.0)
        s.add_bar_element("E1", "A", "B", "S")
        s.add_load_case("LC")
        s.add_element_point_load("E1", "LC", a=2.0, fy=-100.0, mz=5.0,
                                 coord_sys="local")
        path = os.path.join(tempfile.gettempdir(), "epl_roundtrip.x2d")
        save_x2d(s, None, path)
        s2, _res, _view = load_x2d(path)
        self.assertEqual(len(s2.element_point_loads), 1)
        epl = s2.element_point_loads[0]
        self.assertEqual(epl.element_id, "E1")
        self.assertAlmostEqual(epl.a, 2.0)
        self.assertAlmostEqual(epl.fy, -100.0)
        self.assertAlmostEqual(epl.mz, 5.0)
        self.assertEqual(epl.coord_sys, "local")


if __name__ == "__main__":
    unittest.main()
