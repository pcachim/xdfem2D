"""Phase-4 deferred effects: imposed strain (shrinkage) via the thermal path.

Cross-checks the imposed-strain helper against (a) an equivalent TemperatureLoad
and (b) the closed-form restrained-bar axial force N = -EA·eps.
"""
import unittest

from context import Structure2D, assert_close
from xdfem2d.phasing import add_imposed_strain


def _restrained_bar(alpha=1e-5):
    """Axially restrained bar (both ends pinned in x) for shrinkage checks."""
    s = Structure2D()
    s.add_material("M", elastic_modulus=30e6, unit_weight=0.0, alpha=alpha)
    s.add_section("S", "M", b=0.3, h=0.6)
    s.add_node("N1", 0.0, 0.0)
    s.add_node("N2", 5.0, 0.0)
    s.add_bar_element("E1", "N1", "N2", "S")
    s.add_support("PINX", ux=True, uy=True)
    s.add_support("ROLLX", ux=True, uy=True)   # both ends hold x → axially restrained
    s.assign_support("N1", "PINX")
    s.assign_support("N2", "ROLLX")
    return s


class TestImposedStrain(unittest.TestCase):
    def test_matches_equivalent_temperature(self):
        eps = -3.0e-4                      # shrinkage (shortening)
        alpha = 1e-5
        # Imposed-strain model.
        s1 = _restrained_bar(alpha)
        s1.add_load_case("CS")
        add_imposed_strain(s1, ["E1"], eps, "CS")
        r1 = s1.calculate()
        # Equivalent explicit temperature load ΔT = eps/alpha.
        s2 = _restrained_bar(alpha)
        s2.add_load_case("CS")
        s2.add_temperature_load("E1", "CS", delta_t_uniform=eps / alpha)
        r2 = s2.calculate()
        for end in ("i", "j"):
            for k in range(3):
                assert_close(self, r1["element_forces"]["CS"]["E1"][end][k],
                             r2["element_forces"]["CS"]["E1"][end][k],
                             rel=1e-9, abs_tol=1e-6)

    def test_restrained_axial_force_closed_form(self):
        eps = -2.0e-4
        s = _restrained_bar()
        s.add_load_case("CS")
        add_imposed_strain(s, ["E1"], eps, "CS")
        r = s.calculate()
        sec = s.sections["S"]
        mat = s.materials["M"]
        EA = mat.elastic_modulus * (sec.b * sec.h)
        expected_N = -EA * eps            # shrinkage → tension for restrained bar
        N_i = r["element_forces"]["CS"]["E1"]["i"][0]
        assert_close(self, N_i, expected_N, rel=1e-6, abs_tol=1e-3)


if __name__ == "__main__":
    unittest.main()
