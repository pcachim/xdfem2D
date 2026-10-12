"""Special-feature tests: end-release hinges, support springs, thermal loads.

These cover modelling features beyond the plain stiffness assembly, each checked
against a hand calculation or an exact physical invariant.
"""
import unittest

from context import Structure2D, rect_area, assert_close

E = 30e6  # kN/m²
ALPHA = 1.0e-5  # default Material thermal-expansion coefficient


def _mat_sec(s, b=0.3, h=0.6):
    s.add_material("M", elastic_modulus=E, unit_weight=0.0)  # alpha defaults to 1e-5
    s.add_section("S", "M", b=b, h=h)
    return rect_area(b, h)


class TestHingeRelease(unittest.TestCase):
    """A moment hinge at one clamped end turns a fixed-fixed beam into a
    propped cantilever. The hinged support must carry zero moment, and the
    far clamped end must develop the propped-cantilever moment wL²/8."""

    def setUp(self):
        self.w, self.L = 12.0, 6.0
        s = Structure2D()
        _mat_sec(s)
        s.add_node("NA", 0.0, 0.0)
        s.add_node("NB", self.L, 0.0)
        # Hinge at the i-end (NA): releases bending moment there.
        s.add_bar_element("E1", "NA", "NB", "S", hinge_i=True)
        s.add_support("FIX", ux=True, uy=True, tz=True)
        s.assign_support("NA", "FIX")
        s.assign_support("NB", "FIX")
        s.add_load_case("LC")
        s.add_distributed_load("E1", "LC", fye=-self.w, fyd=-self.w)
        self.r = s.calculate()

    def test_hinged_end_zero_moment(self):
        assert_close(self, self.r["reactions"]["LC"]["NA"][2], 0.0,
                     abs_tol=1e-6, msg="moment at hinged end")

    def test_fixed_end_moment(self):
        M = self.w * self.L ** 2 / 8.0  # propped-cantilever fixed-end moment
        assert_close(self, abs(self.r["reactions"]["LC"]["NB"][2]), M,
                     msg="moment at clamped end")

    def test_reactions_split(self):
        # Propped cantilever: 3wL/8 at the prop, 5wL/8 at the clamp.
        assert_close(self, self.r["reactions"]["LC"]["NA"][1], 3.0 * self.w * self.L / 8.0,
                     msg="vertical reaction at prop")
        assert_close(self, self.r["reactions"]["LC"]["NB"][1], 5.0 * self.w * self.L / 8.0,
                     msg="vertical reaction at clamp")


class TestHingeVariants(unittest.TestCase):
    """Moment releases at the j-end and at both ends."""

    def _beam(self, hinge_i, hinge_j):
        w, L = 12.0, 6.0
        s = Structure2D()
        _mat_sec(s)
        s.add_node("NA", 0.0, 0.0)
        s.add_node("NB", L, 0.0)
        s.add_bar_element("E1", "NA", "NB", "S", hinge_i=hinge_i, hinge_j=hinge_j)
        # Clamp NA fully; NB held vertically (and against rotation) so the model
        # is stable regardless of which ends are released.
        s.add_support("FIX", ux=True, uy=True, tz=True)
        s.add_support("VYR", ux=False, uy=True, tz=True)
        s.assign_support("NA", "FIX")
        s.assign_support("NB", "VYR")
        s.add_load_case("LC")
        s.add_distributed_load("E1", "LC", fye=-w, fyd=-w)
        return s.calculate(), w, L

    def test_hinge_j_releases_moment_there(self):
        r, w, L = self._beam(hinge_i=False, hinge_j=True)
        assert_close(self, r["element_forces"]["LC"]["E1"]["j"][2], 0.0,
                     abs_tol=1e-6, msg="moment at hinged j-end")

    def test_both_hinges_zero_end_moments(self):
        r, w, L = self._beam(hinge_i=True, hinge_j=True)
        assert_close(self, r["element_forces"]["LC"]["E1"]["i"][2], 0.0,
                     abs_tol=1e-6, msg="moment at i-end")
        assert_close(self, r["element_forces"]["LC"]["E1"]["j"][2], 0.0,
                     abs_tol=1e-6, msg="moment at j-end")


class TestElementSpring(unittest.TestCase):
    """Element (Winkler foundation) springs carry part of the load and settle."""

    def _build(self, ky):
        s = Structure2D()
        _mat_sec(s)
        s.add_node("N1", 0.0, 0.0)
        s.add_node("N2", 5.0, 0.0)
        s.add_bar_element("E1", "N1", "N2", "S")
        s.add_support("PIN", ux=True, uy=True)
        s.assign_support("N1", "PIN")
        s.add_element_spring("E1", ky=ky)
        s.add_load_case("LC")
        s.add_distributed_load("E1", "LC", fye=-10.0, fyd=-10.0)
        return s.calculate()

    def test_unsupported_end_settles_downward(self):
        r = self._build(ky=5000.0)
        self.assertLess(r["displacements"]["LC"]["N2"][1], 0.0,
                        "free end on a foundation should settle downward")

    def test_stiffer_foundation_settles_less(self):
        soft = self._build(ky=2000.0)
        stiff = self._build(ky=2.0e5)
        self.assertLess(abs(stiff["displacements"]["LC"]["N2"][1]),
                        abs(soft["displacements"]["LC"]["N2"][1]),
                        "stiffer foundation should reduce settlement")


class TestSupportSpring(unittest.TestCase):
    """Replace the roller of a simply-supported beam with a vertical spring.

    By statics (moments about the pin) the spring must carry exactly wL/2,
    independent of its stiffness; a stiffer spring must settle less.
    """

    def _build(self, ky):
        s = Structure2D()
        _mat_sec(s)
        self.L, self.w = 5.0, 10.0
        s.add_node("N1", 0.0, 0.0)
        s.add_node("N2", self.L, 0.0)
        s.add_bar_element("E1", "N1", "N2", "S")
        s.add_support("PIN", ux=True, uy=True)
        s.assign_support("N1", "PIN")
        s.add_node_spring("N2", ky=ky)
        s.add_load_case("LC")
        s.add_distributed_load("E1", "LC", fye=-self.w, fyd=-self.w)
        return s

    def test_spring_carries_half_load(self):
        r = self._build(ky=1.0e5).calculate()
        force = abs(r["spring_forces"]["LC"]["N2"][1])
        assert_close(self, force, self.w * self.L / 2.0, rel=1e-6,
                     msg="spring vertical force")

    def test_spring_force_consistency(self):
        ky = 1.0e5
        r = self._build(ky=ky).calculate()
        uy = r["displacements"]["LC"]["N2"][1]
        force = r["spring_forces"]["LC"]["N2"][1]
        assert_close(self, force, ky * uy, msg="F = k·u identity")

    def test_stiffer_spring_settles_less(self):
        soft = self._build(ky=1.0e4).calculate()
        stiff = self._build(ky=1.0e6).calculate()
        self.assertLess(
            abs(stiff["displacements"]["LC"]["N2"][1]),
            abs(soft["displacements"]["LC"]["N2"][1]),
            "stiffer spring should settle less",
        )


class TestThermalLoad(unittest.TestCase):
    def test_restrained_uniform_heating_axial_force(self):
        """Fully restrained bar, uniform ΔT: N = E·A·α·ΔT (compression)."""
        dT, L = 20.0, 5.0
        s = Structure2D()
        A = _mat_sec(s)
        s.add_node("N1", 0.0, 0.0)
        s.add_node("N2", L, 0.0)
        s.add_bar_element("E1", "N1", "N2", "S")
        s.add_support("FIX", ux=True, uy=True, tz=True)
        s.assign_support("N1", "FIX")
        s.assign_support("N2", "FIX")
        s.add_load_case("LC")
        s.add_temperature_load("E1", "LC", delta_t_uniform=dT)
        r = s.calculate()
        N = E * A * ALPHA * dT
        assert_close(self, abs(r["element_forces"]["LC"]["E1"]["i"][0]), N,
                     msg="restrained thermal axial force")

    @staticmethod
    def _free_expansion(dT=30.0, L=5.0):
        s = Structure2D()
        _mat_sec(s)
        s.add_node("N1", 0.0, 0.0)
        s.add_node("N2", L, 0.0)
        s.add_bar_element("E1", "N1", "N2", "S")
        s.add_support("FIX", ux=True, uy=True, tz=True)
        s.add_support("VY", ux=False, uy=True, tz=True)  # free to slide axially
        s.assign_support("N1", "FIX")
        s.assign_support("N2", "VY")
        s.add_load_case("LC")
        s.add_temperature_load("E1", "LC", delta_t_uniform=dT)
        return s, dT, L

    def test_free_uniform_heating_elongation(self):
        """Axially free bar, uniform ΔT: δ = α·ΔT·L, and N = 0 (stress-free)."""
        s, dT, L = self._free_expansion()
        r = s.calculate()
        assert_close(self, r["displacements"]["LC"]["N2"][0], ALPHA * dT * L,
                     msg="free thermal elongation")
        assert_close(self, r["element_forces"]["LC"]["E1"]["i"][0], 0.0,
                     abs_tol=1e-6, msg="free thermal axial force should be zero")

    def test_restrained_heating_is_compression(self):
        """A restrained heated bar must be in compression (negative axial force)."""
        dT, L = 20.0, 5.0
        s = Structure2D()
        _mat_sec(s)
        s.add_node("N1", 0.0, 0.0)
        s.add_node("N2", L, 0.0)
        s.add_bar_element("E1", "N1", "N2", "S")
        s.add_support("FIX", ux=True, uy=True, tz=True)
        s.assign_support("N1", "FIX")
        s.assign_support("N2", "FIX")
        s.add_load_case("LC")
        s.add_temperature_load("E1", "LC", delta_t_uniform=dT)
        r = s.calculate()
        self.assertLess(r["element_forces"]["LC"]["E1"]["i"][0], 0.0,
                        "restrained heating should produce compression")


class TestThermalGradient(unittest.TestCase):
    """Through-depth thermal gradient: curvature κ = α·ΔT_grad/h."""

    def _build(self, restrained, dT=20.0, b=0.3, h=0.6, L=5.0):
        s = Structure2D()
        _mat_sec(s, b=b, h=h)
        s.add_node("N1", 0.0, 0.0)
        s.add_node("N2", L, 0.0)
        s.add_bar_element("E1", "N1", "N2", "S")
        if restrained:  # clamped-clamped
            s.add_support("FIX", ux=True, uy=True, tz=True)
            s.assign_support("N1", "FIX")
            s.assign_support("N2", "FIX")
        else:           # simply supported, free to curve
            s.add_support("PIN", ux=True, uy=True)
            s.add_support("ROL", ux=False, uy=True)
            s.assign_support("N1", "PIN")
            s.assign_support("N2", "ROL")
        s.add_load_case("LC")
        s.add_temperature_load("E1", "LC", delta_t_gradient=dT)
        self.EI = E * (b * h ** 3 / 12.0)
        self.kappa = ALPHA * dT / h
        return s

    def test_simply_supported_zero_moment(self):
        """Free to curve → statically determinate → zero bending moment."""
        r = self._build(restrained=False).calculate()
        assert_close(self, r["element_forces"]["LC"]["E1"]["i"][2], 0.0,
                     abs_tol=1e-6, msg="M_i simply supported")
        assert_close(self, r["element_forces"]["LC"]["E1"]["j"][2], 0.0,
                     abs_tol=1e-6, msg="M_j simply supported")

    def test_clamped_constant_moment(self):
        """Clamped-clamped → constant restraint moment EI·κ throughout."""
        r = self._build(restrained=True).calculate()
        assert_close(self, abs(r["element_forces"]["LC"]["E1"]["i"][2]),
                     self.EI * self.kappa, msg="restraint moment")


if __name__ == "__main__":
    unittest.main(verbosity=2)
