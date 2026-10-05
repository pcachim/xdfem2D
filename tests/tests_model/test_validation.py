"""Validation tests: compare xdfem2D against closed-form analytical solutions.

Each test models a textbook case whose displacements, reactions and internal
forces are known exactly (or to a tight series tolerance), then checks the FEM
output against the hand formula. These are the tests that assess *calculation
quality* — if the solver regresses, they fail.

Conventions used below
----------------------
E   = 30e6 kN/m²  (≈ concrete)
I   = b·h³/12     (rectangular section)
A   = b·h
Run with:  python -m unittest discover -s tests
"""
import math
import unittest

from context import Structure2D, rect_inertia, rect_area, assert_close, midspan_moment

E = 30e6  # kN/m²


def _base(struc, b=0.3, h=0.6, e=E, gamma=0.0):
    """Attach a standard material + rectangular section ('S') to a structure."""
    struc.add_material("M", elastic_modulus=e, unit_weight=gamma)
    struc.add_section("S", "M", b=b, h=h)
    return rect_inertia(b, h), rect_area(b, h)


def _simple_supports(struc, n_left, n_right):
    """Pin at n_left, roller at n_right (statically determinate simply-supported)."""
    struc.add_support("PIN", ux=True, uy=True, tz=False)
    struc.add_support("ROLLER", ux=False, uy=True, tz=False)
    struc.assign_support(n_left, "PIN")
    struc.assign_support(n_right, "ROLLER")


def _encastre(struc, name):
    struc.add_support("FIX", ux=True, uy=True, tz=True)
    struc.assign_support(name, "FIX")


class TestSimplySupportedUDL(unittest.TestCase):
    """Simply supported beam, uniform load w over span L (single element)."""

    def setUp(self):
        self.w, self.L = 10.0, 5.0
        s = Structure2D()
        self.I, self.A = _base(s)
        s.add_node("N1", 0.0, 0.0)
        s.add_node("N2", self.L, 0.0)
        s.add_bar_element("E1", "N1", "N2", "S")
        _simple_supports(s, "N1", "N2")
        s.add_load_case("LC")
        s.add_distributed_load("E1", "LC", fye=-self.w, fyd=-self.w)
        self.r = s.calculate()

    def test_reactions(self):
        R = self.w * self.L / 2.0  # 25 kN each
        assert_close(self, self.r["reactions"]["LC"]["N1"][1], R, msg="Ry @ N1")
        assert_close(self, self.r["reactions"]["LC"]["N2"][1], R, msg="Ry @ N2")

    def test_no_horizontal_reaction(self):
        assert_close(self, self.r["reactions"]["LC"]["N1"][0], 0.0, msg="Rx @ N1")

    def test_midspan_moment(self):
        M = self.w * self.L ** 2 / 8.0  # 31.25 kNm
        assert_close(self, midspan_moment(self.r, "LC", "E1"), M, msg="M_max")

    def test_end_rotation(self):
        theta = self.w * self.L ** 3 / (24.0 * E * self.I)  # wL³/24EI
        assert_close(self, abs(self.r["displacements"]["LC"]["N1"][2]), theta,
                     msg="end rotation")

    def test_end_shear(self):
        V = self.w * self.L / 2.0
        assert_close(self, abs(self.r["element_forces"]["LC"]["E1"]["i"][1]), V,
                     msg="shear @ i")


class TestSimplySupportedPointLoad(unittest.TestCase):
    """Central point load P on a simply supported beam (two elements, node at mid)."""

    def setUp(self):
        self.P, self.L = 20.0, 6.0
        s = Structure2D()
        self.I, self.A = _base(s)
        s.add_node("N1", 0.0, 0.0)
        s.add_node("NM", self.L / 2.0, 0.0)
        s.add_node("N2", self.L, 0.0)
        s.add_bar_element("E1", "N1", "NM", "S")
        s.add_bar_element("E2", "NM", "N2", "S")
        _simple_supports(s, "N1", "N2")
        s.add_load_case("LC")
        s.add_point_load("NM", "LC", fy=-self.P)
        self.r = s.calculate()

    def test_reactions(self):
        assert_close(self, self.r["reactions"]["LC"]["N1"][1], self.P / 2.0, msg="Ry @ N1")
        assert_close(self, self.r["reactions"]["LC"]["N2"][1], self.P / 2.0, msg="Ry @ N2")

    def test_midspan_deflection(self):
        delta = self.P * self.L ** 3 / (48.0 * E * self.I)  # PL³/48EI
        assert_close(self, abs(self.r["displacements"]["LC"]["NM"][1]), delta,
                     msg="mid deflection")

    def test_midspan_moment(self):
        M = self.P * self.L / 4.0  # PL/4
        assert_close(self, midspan_moment(self.r, "LC", "E1"), M, msg="M_mid")


class TestCantileverPointLoad(unittest.TestCase):
    """Cantilever fixed at the base, transverse point load P at the free tip."""

    def setUp(self):
        self.P, self.L = 10.0, 4.0
        s = Structure2D()
        self.I, self.A = _base(s)
        s.add_node("NA", 0.0, 0.0)
        s.add_node("NB", self.L, 0.0)
        s.add_bar_element("E1", "NA", "NB", "S")
        _encastre(s, "NA")
        s.add_load_case("LC")
        s.add_point_load("NB", "LC", fy=-self.P)
        self.r = s.calculate()

    def test_tip_deflection(self):
        delta = self.P * self.L ** 3 / (3.0 * E * self.I)  # PL³/3EI
        assert_close(self, abs(self.r["displacements"]["LC"]["NB"][1]), delta,
                     msg="tip deflection")

    def test_tip_rotation(self):
        theta = self.P * self.L ** 2 / (2.0 * E * self.I)  # PL²/2EI
        assert_close(self, abs(self.r["displacements"]["LC"]["NB"][2]), theta,
                     msg="tip rotation")

    def test_base_reaction(self):
        assert_close(self, self.r["reactions"]["LC"]["NA"][1], self.P, msg="Ry @ base")

    def test_base_moment(self):
        M = self.P * self.L  # PL
        assert_close(self, abs(self.r["reactions"]["LC"]["NA"][2]), M, msg="Mz @ base")


class TestCantileverUDL(unittest.TestCase):
    """Cantilever fixed at the base, uniform load w over the whole span."""

    def setUp(self):
        self.w, self.L = 8.0, 3.0
        s = Structure2D()
        self.I, self.A = _base(s)
        s.add_node("NA", 0.0, 0.0)
        s.add_node("NB", self.L, 0.0)
        s.add_bar_element("E1", "NA", "NB", "S")
        _encastre(s, "NA")
        s.add_load_case("LC")
        s.add_distributed_load("E1", "LC", fye=-self.w, fyd=-self.w)
        self.r = s.calculate()

    def test_tip_deflection(self):
        delta = self.w * self.L ** 4 / (8.0 * E * self.I)  # wL⁴/8EI
        assert_close(self, abs(self.r["displacements"]["LC"]["NB"][1]), delta,
                     msg="tip deflection")

    def test_base_moment(self):
        M = self.w * self.L ** 2 / 2.0  # wL²/2
        assert_close(self, abs(self.r["reactions"]["LC"]["NA"][2]), M, msg="Mz @ base")

    def test_base_reaction(self):
        assert_close(self, self.r["reactions"]["LC"]["NA"][1], self.w * self.L,
                     msg="Ry @ base")


class TestFixedFixedUDL(unittest.TestCase):
    """Clamped-clamped beam under uniform load: exact for one Euler element."""

    def setUp(self):
        self.w, self.L = 12.0, 6.0
        s = Structure2D()
        self.I, self.A = _base(s)
        s.add_node("NA", 0.0, 0.0)
        s.add_node("NB", self.L, 0.0)
        s.add_bar_element("E1", "NA", "NB", "S")
        s.add_support("FIX", ux=True, uy=True, tz=True)
        s.assign_support("NA", "FIX")
        s.assign_support("NB", "FIX")
        s.add_load_case("LC")
        s.add_distributed_load("E1", "LC", fye=-self.w, fyd=-self.w)
        self.r = s.calculate()

    def test_end_moment(self):
        M = self.w * self.L ** 2 / 12.0  # wL²/12
        assert_close(self, abs(self.r["reactions"]["LC"]["NA"][2]), M, msg="end moment")

    def test_reactions(self):
        assert_close(self, self.r["reactions"]["LC"]["NA"][1], self.w * self.L / 2.0,
                     msg="Ry @ NA")


class TestAxialBar(unittest.TestCase):
    """Pure axial extension of a bar pinned at one end, axial load at the other."""

    def setUp(self):
        self.P, self.L = 100.0, 5.0
        s = Structure2D()
        self.I, self.A = _base(s)
        s.add_node("N1", 0.0, 0.0)
        s.add_node("N2", self.L, 0.0)
        s.add_bar_element("E1", "N1", "N2", "S")
        # Fully fix N1; restrain N2 vertically so only axial DOF is free.
        s.add_support("FIX", ux=True, uy=True, tz=True)
        s.add_support("VY", ux=False, uy=True, tz=True)
        s.assign_support("N1", "FIX")
        s.assign_support("N2", "VY")
        s.add_load_case("LC")
        s.add_point_load("N2", "LC", fx=self.P)
        self.r = s.calculate()

    def test_elongation(self):
        delta = self.P * self.L / (E * self.A)  # PL/EA
        assert_close(self, self.r["displacements"]["LC"]["N2"][0], delta,
                     msg="axial elongation")

    def test_axial_force(self):
        assert_close(self, abs(self.r["element_forces"]["LC"]["E1"]["i"][0]), self.P,
                     msg="axial force")


class TestStaticEquilibrium(unittest.TestCase):
    """Global equilibrium on an asymmetric portal frame: ΣReactions = ΣLoads."""

    def setUp(self):
        s = Structure2D()
        _base(s)
        # Portal: two columns (3 m) + beam (5 m)
        s.add_node("A", 0.0, 0.0)
        s.add_node("B", 0.0, 3.0)
        s.add_node("C", 5.0, 3.0)
        s.add_node("D", 5.0, 0.0)
        s.add_bar_element("col1", "A", "B", "S")
        s.add_bar_element("beam", "B", "C", "S")
        s.add_bar_element("col2", "C", "D", "S")
        s.add_support("FIX", ux=True, uy=True, tz=True)
        s.assign_support("A", "FIX")
        s.assign_support("D", "FIX")
        s.add_load_case("LC")
        s.add_distributed_load("beam", "LC", fye=-15.0, fyd=-15.0)   # 15 kN/m down over 5 m = 75 kN
        s.add_point_load("B", "LC", fx=20.0)          # 20 kN horizontal
        self.r = s.calculate()

    def test_vertical_equilibrium(self):
        ry = sum(self.r["reactions"]["LC"][n][1] for n in ("A", "D"))
        assert_close(self, ry, 15.0 * 5.0, msg="ΣRy = applied vertical")

    def test_horizontal_equilibrium(self):
        rx = sum(self.r["reactions"]["LC"][n][0] for n in ("A", "D"))
        assert_close(self, rx, -20.0, msg="ΣRx = -applied horizontal")

    def test_moment_equilibrium(self):
        # ΣM about origin A: reactions + applied loads must balance to zero.
        R = self.r["reactions"]["LC"]
        nodes = {"A": (0.0, 0.0), "D": (5.0, 0.0)}
        m = 0.0
        for n, (x, y) in nodes.items():
            rx, ry, mz = R[n]
            m += mz + x * ry - y * rx
        # Applied: UDL resultant 75 kN down at x=2.5; H load 20 kN at (0,3)
        m += (-75.0) * 2.5      # vertical load moment about A
        m += -(20.0) * 3.0      # horizontal load: +Fx at height y -> -Fx*y
        assert_close(self, m, 0.0, abs_tol=1e-6, msg="ΣM about A")


if __name__ == "__main__":
    unittest.main(verbosity=2)
